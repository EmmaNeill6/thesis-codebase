"""Configuration, membership, labels, shared prediction, and mask evaluation."""

import csv
import json
import math
import random
import re
import subprocess
from os.path import abspath
from pathlib import Path

import yaml


def load_config(config_path):
    """Load a dictionary and resolve filesystem paths relative to its YAML file."""
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("Pipeline configuration must be a YAML mapping")
    for section in (
        "dataset", "model", "sam3", "evaluation", "splitting", "training", "runtime"
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section {section!r} must be a mapping")

    def resolve_path(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Configured filesystem paths must be nonempty strings")
        path = Path(value).expanduser()
        # Normalize relative paths without dereferencing the virtualenv's Python symlink.
        return abspath(config_path.parent / path)

    for section, key in (
        ("dataset", "images_dir"),
        ("model", "pretrained_checkpoint"),
        ("sam3", "generator_script"),
        ("training", "trainer_script"),
        ("runtime", "python"),
        ("runtime", "output_dir"),
    ):
        config[section][key] = resolve_path(config[section][key])
    bank = config["model"]["bank_checkpoints"]
    if not isinstance(bank, list):
        raise ValueError("model.bank_checkpoints must be a list")
    config["model"]["bank_checkpoints"] = [resolve_path(path) for path in bank]
    if config["sam3"].get("existing_masks_dir") is not None:
        config["sam3"]["existing_masks_dir"] = resolve_path(config["sam3"]["existing_masks_dir"])
    if config["sam3"].get("cache_dir") is not None:
        config["sam3"]["cache_dir"] = resolve_path(config["sam3"]["cache_dir"])
    return config


def prepare_dataset(config):
    """Select images, split whole sub-trips or filename sequences, and save membership."""
    dataset, settings = config["dataset"], config["splitting"]
    selection_mode = dataset.get("selection_mode", "frame_range")
    grouping = settings["grouping"]
    if selection_mode not in {"frame_range", "all"}:
        raise ValueError("selection_mode must be frame_range or all")
    if grouping not in {"sub_trip", "sequence"}:
        raise ValueError("grouping must be sub_trip or sequence")
    size = settings.get("group_size", 10)
    if grouping == "sub_trip" and (type(size) is not int or size <= 0):
        raise ValueError("group_size must be a positive integer")
    roles = ("train", "validation", "test")
    fractions = [settings[role] for role in roles]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 < value < 1 for value in fractions):
        raise ValueError("Split fractions must be finite numbers between 0 and 1")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1")

    source_dir = Path(dataset["images_dir"])
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {source_dir}")
    paths = sorted(path for path in source_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if selection_mode == "frame_range":
        first, last, step = (dataset[key] for key in ("first_frame", "last_frame", "frame_step"))
        if any(type(value) is not int for value in (first, last, step)):
            raise ValueError("Frame bounds and frame_step must be integers")
        if first < 0 or last < first or step <= 0:
            raise ValueError("Invalid frame range or frame_step")
        requested = set(range(first, last + 1, step))
        selected = {}
        for path in paths:
            match = re.search(r"(\d+)$", path.stem)
            if match is None or int(match.group(1)) not in requested:
                continue
            frame = int(match.group(1))
            if frame in selected:
                raise ValueError(f"Duplicate requested frame {frame}: {selected[frame]}, {path}")
            selected[frame] = path
        missing = sorted(requested - selected.keys())
        if missing:
            raise ValueError(f"Missing {len(missing)} requested frames; first missing: {missing[:10]}")
        selected_items = sorted(selected.items())
        image_selection = {key: dataset[key] for key in
                           ("images_dir", "first_frame", "last_frame", "frame_step")}
    else:
        if not paths:
            raise ValueError(f"No images found in: {source_dir}")
        selected_items = []
        for index, path in enumerate(paths):
            match = re.search(r"(\d+)$", path.stem)
            selected_items.append((int(match.group(1)) if match else index, path))
        image_selection = {"images_dir": str(source_dir), "selection_mode": "all"}

    records, grouped = [], {}
    for index, (frame, path) in enumerate(selected_items):
        if grouping == "sequence":
            match = re.fullmatch(r"(.+)_(\d+)", path.stem)
            if match is None:
                raise ValueError(
                    f"Sequence grouping requires <sequence>_<frame> filenames: {path.name}. "
                    "For individual numbered images, set splitting.grouping: sub_trip and group_size: 1."
                )
            group_id = match.group(1)
        else:
            group_id = f"subtrip_{index // size:04d}"
        record = {"image_id": path.stem, "image_path": str(path.resolve()),
                  "frame_number": frame, "subtrip_id": group_id}
        records.append(record)
        grouped.setdefault(group_id, []).append(record)
    if len({record["image_id"] for record in records}) != len(records):
        raise ValueError("Selected image stems must be unique")
    groups = list(grouped.values())
    random.Random(settings["seed"]).shuffle(groups)
    requested_counts = [len(groups) * fraction for fraction in fractions]
    group_counts = [math.floor(count) for count in requested_counts]
    # Stable ties use train/validation/test order.
    remainder_order = sorted(range(len(roles)),
                             key=lambda index: -(requested_counts[index] - group_counts[index]))
    for index in remainder_order[:len(groups) - sum(group_counts)]:
        group_counts[index] += 1

    splits, offset = {}, 0
    for role, count in zip(roles, group_counts):
        splits[role] = sorted(
            [record for group in groups[offset:offset + count] for record in group],
            key=lambda record: record["frame_number"],
        )
        offset += count
        if not splits[role]:
            raise ValueError(f"Empty {role} split; select more groups or adjust fractions")

    seen_images, group_owners = set(), {}
    for role, members in splits.items():
        for record in members:
            path, group_id = record["image_path"], record["subtrip_id"]
            if path in seen_images:
                raise ValueError(f"Image appears more than once: {path}")
            seen_images.add(path)
            if group_id in group_owners and group_owners[group_id] != role:
                raise ValueError(f"Sub-trip appears in multiple roles: {group_id}")
            group_owners[group_id] = role
    if (seen_images != {record["image_path"] for record in records}
            or sum(map(len, splits.values())) != len(records)
            or set(group_owners) != {record["subtrip_id"] for record in records}):
        raise ValueError("Split membership does not cover every selected image and group")

    manifest = {
        "dataset_name": dataset["name"],
        "image_selection": image_selection,
        "split_settings": dict(settings),
        "counts": {
            "total": {"images": len(records), "groups": len(groups)},
            **{role: {"images": len(splits[role]), "groups": count}
               for role, count in zip(roles, group_counts)},
        },
        "splits": splits,
    }
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "splits.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def generate_pseudo_labels(config):
    """Reuse reference masks or call the original SAM3 generator for missing ones."""
    output_dir = Path(config["runtime"]["output_dir"])
    manifest_path = output_dir / "splits.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found; run --prepare-data first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_dir = Path(manifest.get("image_selection", {}).get("images_dir", config["dataset"]["images_dir"])).resolve()
    if source_dir != Path(config["dataset"]["images_dir"]).resolve():
        raise ValueError("The saved split belongs to another image folder; update config.yaml before generating its labels")
    records = sorted(
        [record for role in ("train", "validation", "test")
         for record in manifest["splits"][role]],
        key=lambda record: record["frame_number"],
    )
    sam3 = config["sam3"]
    label_dir = output_dir / "pseudo_labels"
    if sam3.get("cache_dir"):
        import hashlib
        cache_identity = {
            "images_dir": str(source_dir),
            **{key: sam3[key] for key in ("model_id", "confidence_threshold", "padding_mode",
                                        "padding_percent", "prompts")},
        }
        cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:16]
        label_dir = Path(sam3["cache_dir"]) / cache_key
    mask_dirs = [label_dir / "masks", output_dir / "pseudo_labels" / "masks"]
    if sam3.get("existing_masks_dir") is not None:
        existing_dir = Path(sam3["existing_masks_dir"])
        mask_dirs[:0] = [existing_dir / "masks", existing_dir]
    masks, missing = {}, []
    for record in records:
        image_id = record["image_id"]
        mask = next((directory / f"{image_id}.png" for directory in mask_dirs
                     if (directory / f"{image_id}.png").is_file()), None)
        if mask is None:
            missing.append(record)
        else:
            masks[image_id] = mask.resolve()

    if missing:
        selection_path = output_dir / "sam3_selection.csv"
        with selection_path.open("w", newline="", encoding="utf-8") as source:
            writer = csv.DictWriter(source, fieldnames=["image"])
            writer.writeheader()
            writer.writerows({"image": str(Path(record["image_path"]).resolve())}
                             for record in missing)
        subprocess.run(
            [config["runtime"]["python"], sam3["generator_script"],
             "--selection-manifest", str(selection_path),
             "--output-dir", str(label_dir),
             "--model-id", sam3["model_id"],
             "--confidence-threshold", str(sam3["confidence_threshold"]),
             "--padding-percent", str(sam3["padding_percent"])],
            check=True,
        )
        for record in missing:
            masks[record["image_id"]] = (label_dir / "masks" / f"{record['image_id']}.png").resolve()

    from PIL import Image
    for record in records:
        mask = masks[record["image_id"]]
        if not mask.is_file():
            raise FileNotFoundError(f"Reference mask missing for {record['image_id']}: {mask}")
        with Image.open(mask) as reference, Image.open(record["image_path"]) as image:
            if reference.mode != "L" or reference.size != image.size:
                raise ValueError(f"Expected an original-resolution single-channel class mask: {mask}")
            minimum, maximum = reference.getextrema()
            if minimum < 0 or maximum > 2:
                raise ValueError(f"Reference mask must contain only class IDs 0, 1 or 2: {mask}")
        record["mask_path"] = str(mask)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {"total": len(records), "reused": len(records) - len(missing),
               "generated": len(missing), "cache_dir": str(label_dir.resolve())}
    (output_dir / "pseudo_labels.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def predict_ewasr(checkpoint_path, image_records, output_dir, input_width, input_height):
    """Predict original-resolution class-ID masks with any compatible checkpoint."""
    if any(type(value) is not int or value <= 0 for value in (input_width, input_height)):
        raise ValueError("Prediction input dimensions must be positive integers")
    if not image_records:
        raise ValueError("No images supplied for prediction")
    if len({record["image_id"] for record in image_records}) != len(image_records):
        raise ValueError("Prediction image IDs must be unique")

    # Keep model imports lazy and locate reusable scripts independently of cwd.
    import sys
    module_dir = str(Path(__file__).resolve().parent.parent / "eWaSR_Head_Fine_Tuning")
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from ewasr_mlx.model import _resize, create_head_finetuning_model
    from evaluate_native_resolution import prepare_input, resize_class_scores
    from visualize_test_predictions import colour_overlay

    model = create_head_finetuning_model(checkpoint_path)
    model.set_head_training_mode(False)
    output_dir = Path(output_dir).expanduser().resolve()
    masks_dir, overlays_dir = output_dir / "masks", output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for record in image_records:
        image_path = Path(record["image_path"]).expanduser().resolve()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        input_array, (height, width) = prepare_input(image, (input_width, input_height))
        model_input = mx.array(input_array)
        scores = model(model_input)
        full_scores = _resize(scores, model_input.shape[1:3], "linear")
        mx.eval(full_scores)
        cropped_scores = np.asarray(full_scores)[0, :height, :width]
        original_scores = resize_class_scores(cropped_scores, image.size)
        mask = np.argmax(original_scores, axis=-1).astype(np.uint8)
        prediction_path = masks_dir / f"{record['image_id']}.png"
        Image.fromarray(mask).save(prediction_path)
        colour_overlay(image, mask, 0.35).save(overlays_dir / f"{record['image_id']}.png")
        predictions.append({"image_id": record["image_id"], "image_path": str(image_path),
                            "prediction_path": str(prediction_path)})
    return predictions


def evaluate_masks(prediction_records, reference_records, output_dir, modes, metadata=None):
    """Score explicit image-ID pairs at saved resolution; no inference or resizing."""
    if not modes or not set(modes).issubset({"binary", "three_class"}):
        raise ValueError("Evaluation modes must contain binary and/or three_class")
    if not prediction_records:
        raise ValueError("No prediction records to evaluate")
    references = {record["image_id"]: record for record in reference_records}
    if len(references) != len(reference_records):
        raise ValueError("Reference image IDs must be unique")
    if len({record["image_id"] for record in prediction_records}) != len(prediction_records):
        raise ValueError("Prediction image IDs must be unique")
    pairs = []
    for record in prediction_records:
        image_id = record["image_id"]
        if image_id not in references:
            raise ValueError(f"No reference record for image_id {image_id!r}")
        reference = references[image_id]
        if not reference.get("mask_path"):
            raise ValueError(f"Missing reference mask_path for {image_id}; run --generate-labels first")
        if not record.get("prediction_path"):
            raise ValueError(f"Missing prediction_path for {image_id}; run prediction first")
        prediction_path, reference_path = (Path(value).expanduser().resolve() for value in
                                           (record["prediction_path"], reference["mask_path"]))
        for kind, path in (("Prediction", prediction_path), ("Reference", reference_path)):
            if not path.is_file():
                raise FileNotFoundError(f"{kind} mask missing for {image_id}: {path}")
        pairs.append((image_id, prediction_path, reference_path))

    import sys
    project_dir = str(Path(__file__).resolve().parents[2])
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    import numpy as np
    from evaluation.three_class.class_metrics import (
        METRIC_NAMES, evaluate_pair, metrics_from_confusion, finite_mean, json_safe,
    )
    if "binary" in modes:
        from evaluation.binary.water_metrics import metrics_from_counts

    def binary_metrics(confusion):
        tp = int(confusion[1, 1])
        fp = int(confusion[0, 1] + confusion[2, 1])
        fn = int(confusion[1, 0] + confusion[1, 2])
        tn = int(confusion.sum()) - tp - fp - fn
        names = ("true_positive", "false_positive", "false_negative", "true_negative")
        water = metrics_from_counts(dict(zip(names, (tp, fp, fn, tn))))
        non_water = metrics_from_counts(dict(zip(names, (tn, fn, fp, tp))))
        result = {
            f"{label}_{metric}": scores["water_iou" if metric == "iou" else metric]
            for label, scores in (("water", water), ("non_water", non_water))
            for metric in ("iou", "dice", "precision", "recall")
        }
        result["mean_iou"] = finite_mean((result["water_iou"], result["non_water_iou"]))
        result["pixel_accuracy"] = water["pixel_accuracy"]
        return result, [[tn, fp], [fn, tp]]

    total = np.zeros((3, 3), dtype=np.int64)
    rows = []
    for image_id, prediction_path, reference_path in pairs:
        confusion, valid_pixels, _ = evaluate_pair(prediction_path, reference_path)
        total += confusion
        row = {"image_id": image_id, "prediction_path": str(prediction_path),
               "reference_path": str(reference_path), "valid_pixel_count": valid_pixels}
        if "binary" in modes:
            scores, _ = binary_metrics(confusion)
            row.update({f"binary_{key}": value for key, value in scores.items()})
        if "three_class" in modes:
            scores = metrics_from_confusion(confusion)
            row.update({f"three_class_{key}": scores[key] for key in METRIC_NAMES})
        rows.append(row)

    report = {**(metadata or {}), "evaluated_count": len(rows),
              "reference_provenance": "SAM3 pseudo-label agreement"}
    for mode in dict.fromkeys(modes):
        if mode == "binary":
            scores, confusion = binary_metrics(total)
            order = ["non-water", "water"]
        else:
            scores = metrics_from_confusion(total)
            confusion, order = total.tolist(), ["obstacle", "water", "sky"]
        metric_keys = [key.removeprefix(f"{mode}_") for key in rows[0]
                       if key.startswith(f"{mode}_")]
        report[mode] = {
            "global_metrics": scores,
            "macro_metrics": {key: finite_mean(row[f"{mode}_{key}"] for row in rows)
                              for key in metric_keys},
            "confusion_matrix": confusion,
            "class_order": order,
            "confusion_matrix_axes": "rows = reference, columns = prediction",
        }
    report = json_safe(report)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as source:
        writer = csv.DictWriter(source, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))
    return report


def write_prediction_manifest(checkpoint_path, predictions, output_dir,
                              input_width, input_height, available_validation_count, *, split="validation"):
    """Record exact membership; keep validation fields compatible with earlier readers."""
    manifest = {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "split": split,
        "input_width": input_width, "input_height": input_height,
        "prediction_policy": {
            "input_resize": "bilinear", "normalization": "float32 / 255, ImageNet mean/std",
            "padding": "bottom/right zeros to multiples of 64 after normalization",
            "score_resize": "linear to padded input, crop padding, bilinear per class to original size",
            "classification": "argmax after all score interpolation",
            "class_ids": {"obstacle": 0, "water": 1, "sky": 2},
            "overlay_opacity": 0.35,
        },
        f"available_{split}_count": available_validation_count,
        "processed_count": len(predictions),
        "partial_run": len(predictions) < available_validation_count,
        "predictions": predictions,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def assess_checkpoint(config, checkpoint_path, validation_records, candidate_id, output_dir):
    """Recompute and score a checkpoint on the complete validation membership."""
    evaluation = config["evaluation"]
    width, height = evaluation["input_width"], evaluation["input_height"]
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    output_dir = Path(output_dir).resolve()
    try:
        predictions = predict_ewasr(checkpoint_path, validation_records, output_dir, width, height)
        if (not validation_records or len(predictions) != len(validation_records)
                or {record["image_id"] for record in predictions}
                != {record["image_id"] for record in validation_records}):
            raise ValueError("Assessment predictions must cover every validation image exactly once")
        manifest = write_prediction_manifest(
            checkpoint_path, predictions, output_dir, width, height, len(validation_records)
        )
        metadata = {key: value for key, value in manifest.items() if key != "predictions"}
        report = evaluate_masks(predictions, validation_records, output_dir, evaluation["modes"], metadata)
        score = report["binary"]["global_metrics"]["water_iou"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Global binary water IoU is undefined or non-finite")
    except Exception as error:
        raise RuntimeError(f"Assessment failed for {candidate_id} ({checkpoint_path}): {error}") from error
    return {
        "candidate_id": candidate_id,
        "source": candidate_id if candidate_id in {"pretrained", "fine_tuned"} else "bank",
        "checkpoint_path": checkpoint_path,
        "validation_water_iou": score,
        "accepted": score >= evaluation["acceptance_threshold"],
        "metrics_path": str(output_dir / "metrics.json"),
    }


def _acceptance_threshold(config, stage):
    """Validate the shared acceptance settings before starting model work."""
    evaluation = config["evaluation"]
    if evaluation["acceptance_metric"] != "binary_water_iou":
        raise ValueError("Model selection requires acceptance_metric: binary_water_iou")
    if "binary" not in evaluation["modes"]:
        raise ValueError("Model selection requires binary evaluation to be enabled")
    threshold = evaluation["acceptance_threshold"]
    if threshold is None:
        raise ValueError(f"Set evaluation.acceptance_threshold in config.yaml before {stage}; it is currently null")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("acceptance_threshold must be an explicit finite number between 0 and 1")
    return threshold


def write_candidate_scores(candidates, output_dir):
    """Use the same candidate CSV columns for selection and training results."""
    with (Path(output_dir) / "candidate_scores.csv").open("w", newline="", encoding="utf-8") as source:
        writer = csv.DictWriter(source, fieldnames=[
            "candidate_id", "source", "checkpoint_path", "validation_water_iou", "accepted", "metrics_path",
        ])
        writer.writeheader()
        writer.writerows(candidates)


def select_model(config):
    """Select pretrained weights first, otherwise rank the bank or record a handoff."""
    evaluation = config["evaluation"]
    threshold = _acceptance_threshold(config, "--select-model")
    output_dir = Path(config["runtime"]["output_dir"])
    split_path = output_dir / "splits.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing {split_path}; run --prepare-data first")
    validation = json.loads(split_path.read_text(encoding="utf-8"))["splits"]["validation"]
    if not validation or len({record["image_id"] for record in validation}) != len(validation):
        raise ValueError("Selection requires nonempty validation membership with unique image IDs")
    for record in validation:
        if not record.get("mask_path") or not Path(record["mask_path"]).is_file():
            raise ValueError(f"Missing reference mask for {record['image_id']}; run --generate-labels first")

    pretrained = assess_checkpoint(config, config["model"]["pretrained_checkpoint"], validation,
                                   "pretrained", output_dir / "pretrained" / "validation")
    candidates = [pretrained]
    selected = pretrained if pretrained["accepted"] else None
    if selected is None:
        for index, checkpoint in enumerate(config["model"]["bank_checkpoints"], start=1):
            path = Path(checkpoint)
            name = re.sub(r"[^A-Za-z0-9_-]+", "_", path.parent.name or path.stem)
            candidate_id = f"bank_{index:03d}_{name}"
            candidates.append(assess_checkpoint(
                config, checkpoint, validation, candidate_id,
                output_dir / "bank" / candidate_id / "validation",
            ))
        accepted = [candidate for candidate in candidates if candidate["accepted"]]
        # max() preserves configuration-list order for exact score ties.
        selected = max(accepted, key=lambda candidate: candidate["validation_water_iou"], default=None)

    choice = {
        "status": "selected" if selected else "requires_training",
        **{key: selected[key] if selected else None for key in
           ("source", "candidate_id", "checkpoint_path", "validation_water_iou")},
        "acceptance_metric": evaluation["acceptance_metric"],
        "acceptance_threshold": threshold,
        "validation_count": len(validation),
        "requires_training": selected is None,
        "selection_reason": (
            "Pretrained checkpoint meets the threshold; bank assessment skipped"
            if selected is pretrained else
            "Highest-scoring accepted bank checkpoint; exact ties use configuration order"
            if selected else "No assessed checkpoint meets the threshold; training is required"
        ),
        "input_width": evaluation["input_width"], "input_height": evaluation["input_height"],
        "reference_provenance": "SAM3 pseudo-label agreement",
        "assessed_candidates": candidates,
    }
    write_candidate_scores(candidates, output_dir)
    (output_dir / "model_choice.json").write_text(
        json.dumps(choice, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return choice


def train_head_candidate(config):
    """Consume a selection handoff, train the existing head, and assess its best weights."""
    output_dir = Path(config["runtime"]["output_dir"])
    choice_path = output_dir / "model_choice.json"
    summary_path = output_dir / "training" / "training_summary.json"
    if not choice_path.is_file():
        raise FileNotFoundError(f"Missing {choice_path}; run --select-model first")
    choice = json.loads(choice_path.read_text(encoding="utf-8"))
    if summary_path.is_file() and (choice.get("source") == "fine_tuned" or choice["status"] == "requires_training"):
        return {**json.loads(summary_path.read_text(encoding="utf-8")),
                "message": "Training already completed; reporting the recorded result"}
    if choice["status"] == "selected":
        return {"message": "Training unnecessary: a model is already selected",
                "checkpoint_path": choice["checkpoint_path"]}
    if choice["status"] != "requires_training":
        raise ValueError(f"Training requires a requires_training handoff; recorded status is {choice['status']}")
    threshold = _acceptance_threshold(config, "--train-head")
    training, evaluation = config["training"], config["evaluation"]
    if training["optimizer"] != "AdamW":
        raise ValueError("The existing head trainer requires training.optimizer: AdamW")
    if (evaluation["input_width"], evaluation["input_height"]) != (512, 384):
        raise ValueError("Head training requires the existing 512×384 input recipe")
    if any(type(training[key]) is not int or training[key] <= 0 for key in ("epochs", "batch_size")):
        raise ValueError("Training epochs and batch_size must be positive integers")
    if (not math.isfinite(training["learning_rate"]) or training["learning_rate"] <= 0
            or not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0):
        raise ValueError("Training learning_rate must be positive and weight_decay nonnegative")
    if not isinstance(training["class_weights"], list) or len(training["class_weights"]) != 3:
        raise ValueError("Training class_weights must be a list of three class weights")
    split_path = output_dir / "splits.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing {split_path}; run --prepare-data and --generate-labels first")
    splits = json.loads(split_path.read_text(encoding="utf-8"))["splits"]
    records = {role: splits[role] for role in ("train", "validation")}
    seen_ids = set()
    for role, members in records.items():
        if not members:
            raise ValueError(f"Training requires a nonempty {role} split")
        for record in members:
            if record["image_id"] in seen_ids:
                raise ValueError(f"Duplicate or overlapping train/validation image ID: {record['image_id']}")
            seen_ids.add(record["image_id"])
            if not record.get("mask_path"):
                raise ValueError(f"Missing mask_path for {record['image_id']}; run --generate-labels first")
            for key in ("image_path", "mask_path"):
                if not Path(record[key]).is_file():
                    raise FileNotFoundError(f"Missing {key} for {record['image_id']}: {record[key]}")

    # Load the configured script directly, without its CLI or automatic test evaluation.
    import argparse
    import importlib.util
    import sys
    trainer_path = Path(training["trainer_script"])
    if str(trainer_path.parent) not in sys.path:
        sys.path.insert(0, str(trainer_path.parent))
    spec = importlib.util.spec_from_file_location("_pipeline_head_trainer", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    samples = {role: [trainer.Sample(image_path=Path(record["image_path"]),
                                    mask_path=Path(record["mask_path"]), image_id=record["image_id"])
                      for record in members] for role, members in records.items()}
    args = argparse.Namespace(
        weights=Path(config["model"]["pretrained_checkpoint"]), output_dir=output_dir / "training",
        learning_rate=training["learning_rate"], weight_decay=training["weight_decay"],
        epochs=training["epochs"], batch_size=training["batch_size"],
        class_weights=",".join(str(value) for value in training["class_weights"]),
        width=evaluation["input_width"], height=evaluation["input_height"],
        seed=config["splitting"]["seed"], selection_metric="binary_water_iou",
    )
    result = trainer.train(args, samples["train"], samples["validation"])
    best_checkpoint = Path(result["best_checkpoint"]).resolve()
    if not best_checkpoint.is_file() or result["best_epoch"] is None:
        raise RuntimeError("Trainer did not return a validation-selected best checkpoint")
    candidate = assess_checkpoint(config, best_checkpoint, records["validation"],
                                  "fine_tuned", output_dir / "fine_tuned" / "validation")
    summary = {
        "starting_checkpoint": str(args.weights),
        "training_settings": {"optimizer": training["optimizer"],
                              **{key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()}},
        "train_count": len(samples["train"]), "validation_count": len(samples["validation"]),
        "best_checkpoint": str(best_checkpoint), "best_epoch": result["best_epoch"],
        "trainer_head_grid_validation_water_iou": result["best_score"],
        "trainer_validation_resolution": {"width": 128, "height": 96},
        "pipeline_original_resolution_validation_water_iou": candidate["validation_water_iou"],
        "pipeline_validation_resolution": "original",
        "acceptance_threshold": threshold, "accepted": candidate["accepted"],
        "training_elapsed_seconds": result["training_elapsed_seconds"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    choice.update({
        **{key: candidate[key] for key in
           ("source", "candidate_id", "checkpoint_path", "validation_water_iou", "accepted")},
        "status": "selected" if candidate["accepted"] else "candidate_below_threshold",
        "requires_training": False, "training_completed": True,
        "acceptance_metric": evaluation["acceptance_metric"], "acceptance_threshold": threshold,
        "validation_count": len(records["validation"]),
        "input_width": args.width, "input_height": args.height,
        "reference_provenance": "SAM3 pseudo-label agreement",
        "selection_reason": ("Fine-tuned best checkpoint meets the original-resolution validation threshold"
                             if candidate["accepted"] else
                             "Training completed, but the best checkpoint is below the original-resolution validation threshold"),
    })
    choice["assessed_candidates"].append(candidate)
    write_candidate_scores(choice["assessed_candidates"], output_dir)
    choice_path.write_text(json.dumps(choice, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {**summary, "message": "Head training and original-resolution assessment completed"}


def _file_sha256(path):
    """Stream checkpoint contents for the final report's small reuse signature."""
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_evaluate(config):
    """Evaluate the recorded choice on every reserved test image, preserving acceptance."""
    output_dir = Path(config["runtime"]["output_dir"])
    choice_path = output_dir / "model_choice.json"
    if not choice_path.is_file():
        raise FileNotFoundError(f"Missing {choice_path}; run --select-model first")
    choice = json.loads(choice_path.read_text(encoding="utf-8"))
    if choice["status"] == "requires_training":
        raise ValueError("Training handoff is unfinished; run --train-head before --final-evaluate")
    if (choice["status"] not in {"selected", "candidate_below_threshold"}
            or choice["status"] == "candidate_below_threshold" and choice["source"] != "fine_tuned"):
        raise ValueError("Final evaluation requires a selected model or recorded below-threshold fine-tuned candidate")
    checkpoint = Path(choice["checkpoint_path"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Recorded checkpoint is missing: {checkpoint}")
    evaluation = config["evaluation"]
    width, height = choice["input_width"], choice["input_height"]
    if (width, height) != (evaluation["input_width"], evaluation["input_height"]) or (width, height) != (512, 384):
        raise ValueError("Current input dimensions must agree with the recorded choice's 512×384 recipe")
    if not {"binary", "three_class"}.issubset(evaluation["modes"]):
        raise ValueError("Final evaluation requires both binary and three_class modes")
    if (evaluation["reference_mask_resolution"] != "original"
            or evaluation["interpolate_class_scores_before_argmax"] is not True):
        raise ValueError("Final evaluation requires original-resolution references and score interpolation before argmax")
    split_path = output_dir / "splits.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing {split_path}; run --prepare-data and --generate-labels first")
    membership = json.loads(split_path.read_text(encoding="utf-8"))
    splits, test = membership["splits"], membership["splits"]["test"]
    test_ids = {record["image_id"] for record in test}
    if not test or len(test_ids) != len(test):
        raise ValueError("Final evaluation requires a nonempty test split with unique image IDs")
    other_records = splits["train"] + splits["validation"]
    other_ids = {record["image_id"] for record in other_records}
    other_images = {str(Path(record["image_path"]).resolve()) for record in other_records}
    test_images = set()
    for record in test:
        if not record.get("mask_path"):
            raise ValueError(f"Missing test mask_path for {record['image_id']}; run --generate-labels first")
        image = str(Path(record["image_path"]).resolve())
        if record["image_id"] in other_ids or image in other_images or image in test_images:
            raise ValueError(f"Test image duplicates or overlaps train/validation: {record['image_id']}")
        test_images.add(image)
        for key in ("image_path", "mask_path"):
            if not Path(record[key]).is_file():
                raise FileNotFoundError(f"Missing test {key} for {record['image_id']}: {record[key]}")

    candidates = [candidate for candidate in choice["assessed_candidates"]
                  if candidate["candidate_id"] == choice["candidate_id"]
                  and Path(candidate["checkpoint_path"]).resolve() == checkpoint]
    if len(candidates) != 1:
        raise ValueError("Recorded choice must match exactly one assessed candidate by ID and checkpoint")
    candidate = candidates[0]
    accepted = choice.get("accepted", candidate["accepted"])
    if accepted != (choice["status"] == "selected"):
        raise ValueError("Recorded selection status and validation acceptance disagree")
    validation_path = Path(candidate["metrics_path"])
    if not validation_path.is_file():
        raise FileNotFoundError(f"Recorded validation metrics are missing: {validation_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (validation.get("split") != "validation" or validation.get("partial_run") is not False
            or validation.get("evaluated_count") != len(splits["validation"])
            or (validation.get("input_width"), validation.get("input_height")) != (width, height)
            or Path(validation.get("checkpoint_path", "")).resolve() != checkpoint
            or not {"binary", "three_class"}.issubset(validation)):
        raise ValueError("Recorded validation metrics must contain complete binary and three-class validation results")
    training_path = output_dir / "training" / "training_summary.json"
    training = None
    if choice["source"] == "fine_tuned":
        if not training_path.is_file():
            raise FileNotFoundError(f"Fine-tuned training summary is missing: {training_path}")
        training = json.loads(training_path.read_text(encoding="utf-8"))

    # Only checkpoint contents are hashed; membership and settings are small JSON records.
    import hashlib
    dataset = {"name": membership["dataset_name"],
               "counts": {role: len(splits[role]) for role in ("train", "validation", "test")}}
    signature_inputs = {
        "choice": choice, "checkpoint_sha256": _file_sha256(checkpoint), "dataset": dataset,
        "test_records": sorted(test, key=lambda record: record["image_id"]),
        "inference_settings": {key: evaluation[key] for key in
                               ("input_width", "input_height", "modes", "reference_mask_resolution",
                                "interpolate_class_scores_before_argmax")},
        "validation": validation, "training": training,
    }
    signature = hashlib.sha256(json.dumps(signature_inputs, sort_keys=True, allow_nan=False).encode()).hexdigest()
    results_path = output_dir / "results.json"
    if results_path.is_file():
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        if saved.get("evaluation_signature") == signature:
            return saved
        raise ValueError("Existing results.json belongs to another evaluation; use a separate runtime.output_dir with its recorded splits and model choice")

    test_dir = output_dir / "test"
    predictions = predict_ewasr(checkpoint, test, test_dir, width, height)
    if len(predictions) != len(test) or {record["image_id"] for record in predictions} != test_ids:
        raise ValueError("Final predictions must cover every test image exactly once")
    manifest = write_prediction_manifest(checkpoint, predictions, test_dir, width, height, len(test), split="test")
    metadata = {key: value for key, value in manifest.items() if key != "predictions"}
    test_results = evaluate_masks(predictions, test, test_dir, evaluation["modes"], metadata)
    report = {
        "dataset": dataset,
        "selection": {**{key: choice[key] for key in
                         ("checkpoint_path", "candidate_id", "source", "status", "selection_reason",
                          "acceptance_metric", "acceptance_threshold", "validation_water_iou")},
                      "accepted": accepted},
        "validation": validation, "training": training, "test": test_results,
        "input_width": width, "input_height": height,
        "prediction_policy": manifest["prediction_policy"],
        "reference_provenance": "SAM3 pseudo-label agreement",
        "artifacts": {"test_masks": str(test_dir / "masks"), "test_overlays": str(test_dir / "overlays"),
                      "prediction_manifest": str(test_dir / "prediction_manifest.json"),
                      "test_metrics": str(test_dir / "metrics.json"), "per_image_csv": str(test_dir / "per_image.csv"),
                      "validation_metrics": str(validation_path), "results": str(results_path)},
        "evaluation_signature": signature,
    }
    results_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
