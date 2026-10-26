"""Isolated eWaSR experiment: train decoder.sim4 and decoder.seg_head together.

The completed Pohang V3 resolved manifest fixes membership. Its source YAML is
read for dataset provenance and pair coverage, but its later edited split
boundaries do not replace the recorded V3 roles. The unseen split is reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from train_head import (
    DEFAULT_WEIGHTS,
    Sample,
    batches,
    configured_path,
    evaluate,
    find_samples,
    flat_parameter_summary,
    validate_dataset,
    validation_selection_score,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "pohang_canal_pier_head_sim4_v1.yaml"
ROLES = ("train", "validation", "test", "unseen")
TRAINABLE_PREFIXES = ("decoder.sim4.", "decoder.seg_head.")
SIM4_BATCHNORMS = ("bn_lc", "bn_gc", "bn_gc1")
FIXED_COUNTS = {"train": 359, "validation": 119, "test": 90, "unseen": 30}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_locked_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} changed: expected SHA-256 {expected_sha256}, got {actual}")


def load_experiment_config(path: Path) -> tuple[dict, dict[str, Path]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Experiment YAML not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected a schema_version: 1 experiment YAML")
    source = config["split_source"]
    paths = {
        "config": path,
        "definition": configured_path(source["definition_yaml"], path),
        "manifest": configured_path(source["v3_resolved_manifest"], path),
        "checkpoint": configured_path(config["model"]["checkpoint"], path),
        "output": configured_path(config["experiment"]["output_dir"], path),
    }
    require_locked_file(paths["definition"], source["definition_sha256"], "Pohang split YAML")
    require_locked_file(paths["manifest"], source["manifest_sha256"], "V3 resolved split manifest")
    if paths["checkpoint"] != DEFAULT_WEIGHTS.resolve() or not paths["checkpoint"].is_file():
        raise ValueError("This experiment must start from the original pretrained non-IMU checkpoint")
    if paths["output"] == paths["manifest"].parent or paths["output"] == paths["definition"].parent:
        raise ValueError("The new output directory must be separate from the source experiments")
    if source.get("expected_counts") != FIXED_COUNTS:
        raise ValueError(f"The V3 split counts must be {FIXED_COUNTS}")
    model = config["model"]
    if (model.get("architecture") != "ewasr_resnet18" or model.get("imu") is not False
        or model.get("trainable_modules") != ["decoder.sim4", "decoder.seg_head"]
        or model.get("sim4_batchnorm_mode") != "eval"
        or model.get("seg_head_batchnorm_mode") != "train"):
        raise ValueError("Model must train only sim4 and seg_head with fixed sim4 BatchNorm statistics")
    preprocessing = config["preprocessing"]
    required_preprocessing = {
        "input_width": 512, "input_height": 384,
        "mask_width": 128, "mask_height": 96,
        "mask_interpolation": "nearest", "image_normalization": "imagenet",
        "image_mean": [.485, .456, .406], "image_std": [.229, .224, .225],
    }
    if any(preprocessing.get(key) != value for key, value in required_preprocessing.items()):
        raise ValueError("Preprocessing must match the existing 512x384 head-only experiment")
    training = config["training"]
    required_training = {
        "optimizer": "adamw",
        "learning_rates": {"decoder.seg_head": .001, "decoder.sim4": .00001},
        "weight_decay": .001, "epochs": 20, "batch_size": 2,
        "class_weights": [1, 1, 1], "loss": "cross_entropy",
        "checkpoint_selection_metric": "validation_binary_water_iou",
    }
    if any(training.get(key) != value for key, value in required_training.items()):
        raise ValueError("Training settings must match the fixed head + sim4 recipe")
    return config, paths


def ensure_empty_output_dir(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output}")


def load_v3_splits(config: dict, paths: dict[str, Path]) -> tuple[dict[str, list[Sample]], dict[Path, str], dict]:
    """Use recorded V3 membership; verify all pairs against the source YAML."""
    source_config = yaml.safe_load(paths["definition"].read_text(encoding="utf-8"))
    if (source_config.get("schema_version") != 1
        or source_config.get("labels") != {"obstacle": 0, "water": 1, "sky": 2}):
        raise ValueError("Invalid read-only Pohang split definition")
    source_checkpoint = configured_path(source_config["model"]["checkpoint"], paths["definition"])
    if source_checkpoint != paths["checkpoint"]:
        raise ValueError("Pohang source YAML uses a different pretrained checkpoint")

    available: dict[tuple[Path, Path], str] = {}
    scanned: set[tuple[Path, Path]] = set()
    for role in ROLES:
        for source in source_config["data"][role].get("sources", []):
            source_id = source.get("source_id", "")
            domain = source_id.split("_", 1)[0]
            if domain not in {"canal", "pier"}:
                raise ValueError(f"Unknown Pohang source domain: {source_id!r}")
            images_dir = configured_path(source["images_dir"], paths["definition"])
            masks_dir = configured_path(source["masks_dir"], paths["definition"])
            roots = (images_dir, masks_dir)
            if roots in scanned:
                continue
            scanned.add(roots)
            for sample in find_samples(images_dir, masks_dir):
                key = (sample.image_path.resolve(), sample.mask_path.resolve())
                if key in available and available[key] != domain:
                    raise ValueError(f"Pair appears in both Pohang domains: {key}")
                available[key] = domain

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != set(ROLES):
        raise ValueError("V3 manifest must contain exactly train, validation, test and unseen")
    splits: dict[str, list[Sample]] = {}
    domains: dict[Path, str] = {}
    used: set[tuple[Path, Path]] = set()
    for role in ROLES:
        entries = manifest[role]
        if not isinstance(entries, list) or len(entries) != config["split_source"]["expected_counts"][role]:
            raise ValueError(f"Unexpected V3 {role} count")
        selected = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"image", "mask"}:
                raise ValueError(f"Malformed V3 {role} manifest entry")
            image, mask = Path(entry["image"]), Path(entry["mask"])
            if not image.is_absolute():
                image = HERE.parents[1] / image
            if not mask.is_absolute():
                mask = HERE.parents[1] / mask
            key = (image.resolve(), mask.resolve())
            if key not in available or key in used:
                raise ValueError(f"V3 {role} pair is missing, changed or duplicated: {key}")
            used.add(key)
            domains[key[0]] = available[key]
            selected.append(Sample(*key))
        splits[role] = selected
    if used != set(available):
        raise ValueError("V3 manifest no longer covers the Pohang source universe exactly")
    # Check only development image/mask pixels. Test is validated at final
    # evaluation; unseen pixels remain untouched throughout this experiment.
    validate_dataset(splits["train"])
    validate_dataset(splits["validation"])
    return splits, domains, manifest


def expected_trainable_names(all_names: list[str]) -> set[str]:
    return {name for name in all_names
            if name.startswith(TRAINABLE_PREFIXES)
            and not name.endswith(("running_mean", "running_var"))}


def check_trainable_names(actual_names: list[str], all_names: list[str]) -> None:
    expected = expected_trainable_names(all_names)
    actual = set(actual_names)
    if len(actual) != len(actual_names) or actual != expected:
        raise RuntimeError(f"Incorrect trainable tensors: missing={sorted(expected - actual)}, "
                           f"unexpected={sorted(actual - expected)}")
    if not all(any(name.startswith(prefix) for name in actual) for prefix in TRAINABLE_PREFIXES):
        raise RuntimeError("Both sim4 and seg_head must have trainable tensors")


def create_head_sim4_model(weights_path: Path):
    """Load the full checkpoint and expose exactly sim4 + seg_head to autograd."""
    from ewasr_mlx import create_head_finetuning_model

    model = create_head_finetuning_model(weights_path)
    model.decoder.sim4.unfreeze()
    for name in SIM4_BATCHNORMS:
        getattr(model.decoder.sim4, name).freeze(
            keys=["running_mean", "running_var"], recurse=False, strict=True)
    model.set_head_training_mode(True)  # Head BN trains; sim4 BN stays in eval.
    check_batchnorm_modes(model, head_training=True)
    return model


def check_batchnorm_modes(model, *, head_training: bool) -> None:
    if bool(model.decoder.seg_head.bn.training) != head_training:
        raise RuntimeError("Segmentation-head BatchNorm mode changed")
    if any(getattr(model.decoder.sim4, name).training for name in SIM4_BATCHNORMS):
        raise RuntimeError("sim4 BatchNorm must use fixed pretrained running statistics")


def check_gradients(gradients, expected_names: set[str]) -> None:
    from mlx.utils import tree_flatten

    flattened = tree_flatten(gradients)
    names = [name for name, _ in flattened]
    if set(names) != expected_names or len(names) != len(expected_names):
        raise RuntimeError("Gradients reached tensors outside sim4 and seg_head")
    magnitude = {prefix: 0.0 for prefix in TRAINABLE_PREFIXES}
    for name, value in flattened:
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite gradient: {name}")
        for prefix in TRAINABLE_PREFIXES:
            if name.startswith(prefix):
                magnitude[prefix] += float(np.abs(array).sum())
    if any(value == 0 for value in magnitude.values()):
        raise RuntimeError(f"Both trainable modules must receive nonzero gradients: {magnitude}")


def audit_frozen_checkpoint(original_path: Path, trained_path: Path) -> dict[str, object]:
    """Check the complete full-model checkpoint, including fixed sim4 BN state."""
    with np.load(original_path) as original, np.load(trained_path) as trained:
        if set(original.files) != set(trained.files):
            raise RuntimeError("Checkpoint tensor names differ from the pretrained model")
        changed, unexpected = [], []
        for name in original.files:
            if np.array_equal(original[name], trained[name]):
                continue
            changed.append(name)
            allowed = name.startswith("decoder.seg_head.") or (
                name.startswith("decoder.sim4.")
                and not name.endswith(("running_mean", "running_var")))
            if not allowed:
                unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"Frozen pretrained tensors changed: {unexpected}")
    if not all(any(name.startswith(prefix) for name in changed) for prefix in TRAINABLE_PREFIXES):
        raise RuntimeError("Both sim4 and seg_head must change from the pretrained weights")
    return {"changed_tensor_names": changed, "frozen_tensors_unchanged": True,
            "sim4_running_statistics_unchanged": True}


def add_directional_metrics(metrics: dict) -> dict:
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    water_total = int(confusion[1].sum())
    obstacle_total = int(confusion[0].sum())
    return {
        **metrics,
        "water_predicted_as_obstacle_count": int(confusion[1, 0]),
        "water_predicted_as_obstacle_rate": (
            float(confusion[1, 0] / water_total) if water_total else None),
        "obstacle_missed_as_water_count": int(confusion[0, 1]),
        "obstacle_missed_as_water_rate": (
            float(confusion[0, 1] / obstacle_total) if obstacle_total else None),
    }


def evaluate_test_groups(model, splits: dict[str, list[Sample]], domains: dict[Path, str], args):
    validate_dataset(splits["test"])
    groups = {"aggregate": splits["test"]}
    for domain in ("canal", "pier"):
        groups[domain] = [sample for sample in splits["test"]
                          if domains[sample.image_path.resolve()] == domain]
        if not groups[domain]:
            raise ValueError(f"No {domain} test samples")
    results = {}
    for name, samples in groups.items():
        results[name] = {"pairs": len(samples),
                         "metrics": add_directional_metrics(evaluate(model, samples, args))}
        check_batchnorm_modes(model, head_training=True)
    return results


def format_test_summary(name: str, entry: dict) -> list[str]:
    metrics = entry["metrics"]
    return [f"{name}: {entry['pairs']} pairs",
            f"  water IoU: {metrics['binary_water_iou']:.6f}",
            f"  water predicted as obstacle: {metrics['water_predicted_as_obstacle_count']} "
            f"({metrics['water_predicted_as_obstacle_rate']:.6%} of water pixels)",
            f"  obstacle missed as water: {metrics['obstacle_missed_as_water_count']} "
            f"({metrics['obstacle_missed_as_water_rate']:.6%} of obstacle pixels)"]


def train_and_evaluate(config: dict, paths: dict[str, Path], splits, domains, manifest) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    if not mx.metal.is_available():
        raise RuntimeError("MLX cannot access an Apple Metal GPU")
    ensure_empty_output_dir(paths["output"])
    started_at = datetime.now(timezone.utc)
    total_started = time.perf_counter()
    model = create_head_sim4_model(paths["checkpoint"])
    with np.load(paths["checkpoint"]) as original:
        all_names = list(original.files)
    trainable_names, trainable_count = flat_parameter_summary(model.trainable_parameters())
    check_trainable_names(trainable_names, all_names)
    expected_names = set(trainable_names)
    training = config["training"]
    optimizer = optim.MultiOptimizer(
        [optim.AdamW(learning_rate=training["learning_rates"]["decoder.sim4"],
                     weight_decay=training["weight_decay"], bias_correction=True),
         optim.AdamW(learning_rate=training["learning_rates"]["decoder.seg_head"],
                     weight_decay=training["weight_decay"], bias_correction=True)],
        [lambda name, _: name.startswith("decoder.sim4.")],
    )

    def loss_fn(images, targets):
        logits = model(images)
        return mx.mean(nn.losses.cross_entropy(logits, targets, reduction="none"))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    args = SimpleNamespace(batch_size=2, width=512, height=384)
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_splits.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    settings = {
        "experiment": config["experiment"], "training": training,
        "preprocessing": config["preprocessing"], "model": config["model"],
        "config_path": str(paths["config"]),
        "split_definition_yaml": str(paths["definition"]),
        "split_definition_sha256": config["split_source"]["definition_sha256"],
        "v3_manifest": str(paths["manifest"]),
        "v3_manifest_sha256": config["split_source"]["manifest_sha256"],
        "pretrained_checkpoint": str(paths["checkpoint"]),
        "pretrained_checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "split_counts": {role: len(samples) for role, samples in splits.items()},
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "started_at_utc": started_at.isoformat(),
    }
    (output / "run_config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Training pairs: {len(splits['train'])}; validation pairs: {len(splits['validation'])}")
    print(f"Trainable parameters: {trainable_count:,}")
    print("Trainable tensors:", *trainable_names, sep="\n  ")

    rng = np.random.default_rng(config["experiment"]["seed"])
    history_path = output / "training_history.jsonl"
    best_path = output / "best_head_sim4.npz"
    best_score, best_epoch, best_record = -np.inf, None, None
    checked_gradients = False
    training_started = time.perf_counter()
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, training["epochs"] + 1):
            epoch_started = time.perf_counter()
            model.set_head_training_mode(True)
            check_batchnorm_modes(model, head_training=True)
            losses = []
            for images, masks in batches(splits["train"], batch_size=2, width=512,
                                          height=384, rng=rng):
                loss, gradients = loss_and_grad(mx.array(images), mx.array(masks))
                if not checked_gradients:
                    mx.eval(loss, gradients)
                    check_gradients(gradients, expected_names)
                    checked_gradients = True
                optimizer.update(model, gradients)
                mx.eval(model.parameters(), optimizer.state, loss)
                losses.append(float(loss.item()))
            validation = add_directional_metrics(evaluate(model, splits["validation"], args))
            check_batchnorm_modes(model, head_training=True)
            record = {"epoch": epoch, "training_loss": float(np.mean(losses)),
                      "validation": validation}
            score = validation_selection_score(record, "binary_water_iou")
            if score > best_score:
                best_score, best_epoch, best_record = score, epoch, record
                model.save_weights(str(best_path))
            record["epoch_elapsed_seconds"] = time.perf_counter() - epoch_started
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            print(json.dumps(record), flush=True)
    training_seconds = time.perf_counter() - training_started
    if best_epoch is None:
        raise RuntimeError("No validation checkpoint was selected")
    frozen_audit = audit_frozen_checkpoint(paths["checkpoint"], best_path)
    (output / "frozen_state_audit.json").write_text(
        json.dumps(frozen_audit, indent=2) + "\n", encoding="utf-8")

    # Reload through the two-module setup rather than the head-only factory.
    selected_model = create_head_sim4_model(best_path)
    reloaded_names, _ = flat_parameter_summary(selected_model.trainable_parameters())
    check_trainable_names(reloaded_names, all_names)
    tests = evaluate_test_groups(selected_model, splits, domains, args)
    results = {
        "best_epoch": best_epoch,
        "selection_metric": "validation_binary_water_iou",
        "best_validation": best_record["validation"],
        "test": tests,
        "unseen": "reserved; pixels not read or evaluated",
        "best_checkpoint": str(best_path),
        "training_elapsed_seconds": training_seconds,
        "total_elapsed_seconds": time.perf_counter() - total_started,
        "trainable_parameter_names": trainable_names,
        "frozen_state_audit": frozen_audit,
    }
    (output / "experiment_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = ["eWaSR decoder.sim4 + decoder.seg_head fine-tuning",
             f"Started (UTC): {started_at.isoformat()}",
             f"Finished (UTC): {datetime.now(timezone.utc).isoformat()}",
             f"Split source: {paths['manifest']}",
             f"Split counts: {settings['split_counts']}",
             f"Original pretrained checkpoint: {paths['checkpoint']}",
             "Optimizer: AdamW; head LR 0.001; sim4 LR 0.00001; decay 0.001",
             "Epochs: 20; batch size: 2; class weights: 1/1/1",
             "Input: 512x384; nearest-neighbour 128x96 masks; cross-entropy",
             "sim4 BatchNorm: fixed pretrained running statistics; head BatchNorm: training statistics",
             f"Trainable parameters ({trainable_count}): {', '.join(trainable_names)}",
             f"Best epoch: {best_epoch}; validation water IoU: {best_score:.6f}",
             f"Best checkpoint: {best_path}",
             f"Training time: {training_seconds:.2f} seconds",
             f"Total time: {results['total_elapsed_seconds']:.2f} seconds",
             "Metrics measure agreement with SAM3 pseudo-labels.", ""]
    for name in ("aggregate", "canal", "pier"):
        lines.extend(format_test_summary(name, tests[name]) + [""])
    lines.append("Unseen split: reserved; not evaluated.")
    (output / "evaluation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluation report: {output / 'evaluation.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-data", action="store_true",
                        help="Validate exact V3 split reuse without MLX or output writes")
    args = parser.parse_args()
    config, paths = load_experiment_config(args.config)
    ensure_empty_output_dir(paths["output"])
    splits, domains, manifest = load_v3_splits(config, paths)
    counts = {role: len(samples) for role, samples in splits.items()}
    print("Validated exact V3 split counts:", counts)
    print("V3 test domains:", {domain: sum(domains[s.image_path] == domain for s in splits["test"])
                               for domain in ("canal", "pier")})
    print("Unseen pixels were not read.")
    if args.check_data:
        return
    train_and_evaluate(config, paths, splits, domains, manifest)


if __name__ == "__main__":
    main()
