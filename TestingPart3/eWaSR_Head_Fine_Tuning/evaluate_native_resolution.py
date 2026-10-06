"""Evaluate a recorded split at native or specified inference resolution.

Training, checkpoint selection, and earlier results are untouched. Both models
use TestingPart1's normalization, bottom/right padding to multiples of 64, and
linear score upsampling before argmax. References are full-resolution masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluation.three_class.class_metrics import read_mask  # noqa: E402
from evaluation.three_class.three_class_evaluator import main as evaluate_masks  # noqa: E402


MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_native(image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
    """Keep RGB pixels at native resolution, then normalize and zero-pad."""
    return prepare_input(image)


def prepare_input(image: Image.Image, input_size=None) -> tuple[np.ndarray, tuple[int, int]]:
    """Optionally resize the RGB input; references always stay full-resolution."""
    if input_size is not None:
        image = image.resize(tuple(input_size), Image.Resampling.BILINEAR)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = array.shape[:2]
    array = (array - MEAN) / STD
    array = np.pad(array, ((0, (-height) % 64), (0, (-width) % 64), (0, 0)))
    return array[None, ...], (height, width)


def resize_class_scores(scores: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize continuous scores rather than interpolating discrete class IDs."""
    if scores.ndim != 3 or scores.shape[-1] != 3:
        raise ValueError("Expected HxWx3 class scores")
    if (scores.shape[1], scores.shape[0]) == target_size:
        return scores
    return np.stack([
        np.asarray(Image.fromarray(scores[..., index].astype(np.float32)).resize(
            target_size, Image.Resampling.BILINEAR
        )) for index in range(3)
    ], axis=-1)


def load_recorded_split(experiment: Path, split_name="test"):
    """Read persisted membership rather than regenerating random assignments."""
    splits = json.loads((experiment / "resolved_splits.json").read_text())
    test = splits[split_name]
    if not test:
        raise ValueError(f"Recorded {split_name} split is empty")
    image_paths, mask_paths, keys = set(), set(), set()
    pairs = []
    for sample in test:
        image, mask = Path(sample["image"]).resolve(), Path(sample["mask"]).resolve()
        if image in image_paths or mask in mask_paths or mask.name in keys:
            raise ValueError(f"Duplicate image, mask or evaluation key in {split_name} split")
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing {split_name} pair: {image}, {mask}")
        reference = read_mask(mask)
        if not set(np.unique(reference)).issubset({0, 1, 2}):
            raise ValueError(f"Reference must use class IDs 0, 1, 2: {mask}")
        with Image.open(image) as source:
            if source.size != (reference.shape[1], reference.shape[0]):
                raise ValueError(f"Native image/reference dimensions differ: {image}")
        image_paths.add(image)
        mask_paths.add(mask)
        keys.add(mask.name)
        pairs.append((image, mask))
    for role in ("train", "validation", "test"):
        if role == split_name:
            continue
        if any(Path(sample["image"]).resolve() in image_paths
               or Path(sample["mask"]).resolve() in mask_paths for sample in splits[role]):
            raise ValueError(f"Recorded {split_name} files overlap with {role}")
    return pairs


def load_recorded_test(experiment: Path):
    return load_recorded_split(experiment, "test")


def write_comparison(output, manifest, reports, previous):
    split_name = manifest["split"]
    previous_pretrained = previous["baseline_test" if split_name == "test" else "baseline_validation"]
    previous_adapted = previous["adapted_test"] if split_name == "test" else previous["selected"]["validation_metrics"]
    metrics = ("mean_iou", "mean_dice", "mean_precision", "mean_recall", "pixel_accuracy",
               "obstacle_iou", "water_iou", "sky_iou")
    comparison = {
        "inference_manifest": str(output / "inference_manifest.json"),
        "split": split_name,
        "image_count": manifest["image_count"],
        "inference_input_size": manifest["inference_input_size"],
        "reference_role": "SAM3 pseudo-label agreement, not independent human ground truth",
        "models": {
            name: {"evaluation_dir": str(report_dir),
                   "macro_metrics": report["macro_metrics"],
                   "global_metrics": report["global_metrics"]}
            for name, (report_dir, report) in reports.items()
        },
        "original_head_resolution_metrics": {
            "pretrained": previous_pretrained, "fine_tuned": previous_adapted,
            "input_resolution": "512x384", "evaluation_resolution": "128x96",
        },
    }
    lines = [
        f"Full-resolution mask evaluation — recorded {split_name} split",
        f"Evaluated: {manifest['image_count']} images per model",
        f"Selected learning rate: {manifest['learning_rate']}",
        f"Selected epoch: {manifest['best_epoch']}",
        "References: full-resolution SAM3 pseudo-labels (5% reflection padding)",
        f"Inference input: {manifest['inference_input_size']}; normalize then pad to multiples of 64",
        "Output: interpolate scores to input, crop padding, rescale scores to original size, then argmax",
        "No model selection or retraining was performed.", "",
        "Global metrics (summed pixel confusion; percentages):",
        "Metric | Pretrained | Fine-tuned",
    ]
    for metric in metrics:
        values = [reports[name][1]["global_metrics"][metric] for name in ("pretrained", "fine_tuned")]
        lines.append(f"{metric} | {100 * values[0]:.4f}% | {100 * values[1]:.4f}%")
    lines += ["", "Per-image macro metrics (percentages):", "Metric | Pretrained | Fine-tuned"]
    for metric in metrics:
        values = [reports[name][1]["macro_metrics"][metric] for name in ("pretrained", "fine_tuned")]
        lines.append(f"{metric} | {100 * values[0]:.4f}% | {100 * values[1]:.4f}%")
    lines += ["", f"Earlier head-resolution {split_name} mean IoU (128x96):",
              f"Pretrained: {100 * previous_pretrained['mean_iou']:.4f}%",
              f"Fine-tuned: {100 * previous_adapted['mean_iou']:.4f}%",
              "Resolution comparisons also differ in input resizing and score interpolation.",
              "These scores quantify SAM3 agreement; they are not human-ground-truth accuracy.", "",
              *(f"{name} full results: {directory}" for name, (directory, _) in reports.items())]
    (output / "comparison_summary.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    (output / "summary.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=HERE / "outputs/tampere_dock_head_only_v1")
    parser.add_argument("--output-dir", type=Path, help="Must be a new directory")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--input-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Optional inference resize; both dimensions must be positive multiples of 64")
    args = parser.parse_args()
    if args.input_size is not None and any(value <= 0 or value % 64 for value in args.input_size):
        parser.error("--input-size dimensions must be positive multiples of 64")
    experiment = args.experiment_dir.resolve()
    pairs = load_recorded_split(experiment, args.split)
    config = yaml.safe_load((experiment / "resolved_experiment.yaml").read_text())
    selection = json.loads((experiment / "selected_model.json").read_text())
    previous = json.loads((experiment / "experiment_results.json").read_text())
    checkpoints = {"pretrained": Path(config["model"]["checkpoint"]).resolve(),
                   "fine_tuned": Path(selection["checkpoint"]).resolve()}
    preserved = [experiment / name for name in ("selected_model.json", "experiment_results.json",
                  "experiment_summary.txt", "resolved_splits.json", "resolved_experiment.yaml")]
    protected_hashes = {str(path): fingerprint(path) for path in [*preserved, *checkpoints.values()]}
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    resolution_name = "native" if args.input_size is None else "x".join(map(str, args.input_size))
    output = args.output_dir.resolve() if args.output_dir else experiment / f"{resolution_name}_resolution_{args.split}" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    references = {path.parent for _, path in pairs}
    if len(references) != 1:
        raise ValueError("This runner requires a single reference-mask directory")
    reference_dir = references.pop()
    selection_path = output / f"{args.split}_selection.txt"
    selection_path.write_text("\n".join(mask.name for _, mask in pairs) + "\n")
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_dir": str(experiment), "split": args.split, "image_count": len(pairs),
        "inference_input_size": args.input_size or "native",
        "learning_rate": selection["learning_rate"], "best_epoch": selection["best_epoch"],
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "protected_file_sha256": protected_hashes,
        "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
        "padding": "zero after normalization; bottom/right only; alignment 64",
        "prediction_policy": "linear score upsampling to input; crop padding; bilinear scores to original size; argmax last",
        "reference_policy": "native-resolution SAM3 class-ID masks; unchanged",
        "sample_fingerprints": [{"image": str(image), "mask": str(mask),
                                 "image_sha256": fingerprint(image), "mask_sha256": fingerprint(mask)}
                                for image, mask in pairs],
        "model_outputs": {},
    }
    (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    import mlx.core as mx
    from ewasr_mlx.model import _resize, create_head_finetuning_model
    from visualize_test_predictions import colour_overlay, comparison_image, create_contact_sheet

    reports = {}
    for name, checkpoint in checkpoints.items():
        model = create_head_finetuning_model(checkpoint)
        model.eval()
        masks, overlays = output / name / "masks", output / name / "overlays"
        masks.mkdir(parents=True)
        overlays.mkdir()
        dimensions = []
        for index, (image_path, reference) in enumerate(pairs, start=1):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            native_input, (height, width) = prepare_input(image, args.input_size)
            model_input = mx.array(native_input)
            scores = model(model_input)
            grid = list(scores.shape[1:3])
            full_scores = _resize(scores, model_input.shape[1:3], "linear")
            mx.eval(full_scores)
            original_scores = resize_class_scores(np.asarray(full_scores)[0, :height, :width], image.size)
            mask = np.argmax(original_scores, axis=-1).astype(np.uint8)
            if mask.shape != (image.height, image.width):
                raise ValueError("Prediction shape does not match native reference")
            Image.fromarray(mask).save(masks / reference.name)
            colour_overlay(image, mask, 0.35).save(overlays / reference.name, format="PNG")
            dimensions.append({"image": image_path.name, "input_width": width, "input_height": height,
                               "original_width": image.width, "original_height": image.height,
                               "padded_width": native_input.shape[2], "padded_height": native_input.shape[1],
                               "grid_width": grid[1], "grid_height": grid[0]})
            print(f"{name}: {index}/{len(pairs)} {image_path.name} — {width}x{height}", flush=True)
            del model_input, scores, full_scores, original_scores
        del model
        mx.clear_cache()
        manifest["model_outputs"][name] = {"masks": str(masks), "overlays": str(overlays), "dimensions": dimensions}
        report_dir = evaluate_masks([
            "--prediction-dir", str(masks), "--ground-truth-dir", str(reference_dir),
            "--selection-file", str(selection_path), "--model-name", f"eWaSR_{name}",
            "--dataset-name", f"Tampere Dock {args.split} — full-resolution SAM3 5% pseudo-labels",
            "--run-name", f"{experiment.name}_{resolution_name}_{args.split}", "--results-dir", str(PROJECT / "evaluation/three_class/results"),
        ])
        report = json.loads((report_dir / "summary.json").read_text())
        assert report["number_of_images"] == len(pairs)
        assert report["valid_pixel_count"] == sum(item["original_width"] * item["original_height"] for item in dimensions)
        reports[name] = (report_dir, report)
        manifest["model_outputs"][name]["evaluation_dir"] = str(report_dir)
        (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    comparisons = output / "comparisons"
    comparisons.mkdir()
    comparison_paths = []
    for image_path, reference in pairs:
        with Image.open(image_path) as source:
            original = source.convert("RGB")
        with Image.open(output / "pretrained/overlays" / reference.name) as source:
            pretrained = source.convert("RGB")
        with Image.open(output / "fine_tuned/overlays" / reference.name) as source:
            adapted = source.convert("RGB")
        pseudo = colour_overlay(original, read_mask(reference), 0.35)
        destination = comparisons / f"{image_path.stem}_comparison.jpg"
        comparison_image(original, pretrained, adapted, pseudo).save(destination, quality=95)
        comparison_paths.append(destination)
    create_contact_sheet(comparison_paths, output / "contact_sheet.jpg")
    for path, digest in protected_hashes.items():
        assert fingerprint(Path(path)) == digest, f"Protected input changed: {path}"
    for item in manifest["sample_fingerprints"]:
        assert fingerprint(Path(item["image"])) == item["image_sha256"]
        assert fingerprint(Path(item["mask"])) == item["mask_sha256"]
    manifest["protected_inputs_unchanged"] = True
    manifest["complete"] = True
    (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_comparison(output, manifest, reports, previous)
    print(f"Resolution evaluation complete: {output / 'summary.txt'}", flush=True)


if __name__ == "__main__":
    main()
