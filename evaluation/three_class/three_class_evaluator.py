"""Evaluate saved obstacle/water/sky masks independently of model inference."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from .class_metrics import (
        CLASS_NAMES, json_safe, evaluate_pair, macro_average, metrics_from_confusion,
    )
else:
    from class_metrics import (
        CLASS_NAMES, json_safe, evaluate_pair, macro_average, metrics_from_confusion,
    )


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from dataset_specs import load_dataset  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("custom", "mastr1325"), default="custom")
    parser.add_argument("--dataset-name", help="Display name for a custom reference set")
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--ground-truth-dir", type=Path,
                        help="Required for custom data; matching relative PNG paths")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-name", help="Defaults to the prediction directory's parent name")
    parser.add_argument("--selection-file", type=Path,
                        help="Optional list of relative ground-truth PNG paths")
    parser.add_argument("--ground-truth-class-values", type=int, nargs=3, default=(0, 1, 2),
                        metavar=("OBSTACLE", "WATER", "SKY"),
                        help="Raw ground-truth values mapped to obstacle/water/sky")
    parser.add_argument("--ignore-value", type=int,
                        help="GT ignore label; defaults to 4 for MaSTR1325, none for custom")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    if args.dataset == "custom" and args.ground_truth_dir is None:
        parser.error("--ground-truth-dir is required for --dataset custom")
    if args.dataset == "mastr1325" and args.ground_truth_dir is not None:
        parser.error("Use --dataset custom when supplying --ground-truth-dir")
    if args.dataset == "mastr1325" and args.ignore_value is None:
        args.ignore_value = 4
    args.ground_truth_class_values = tuple(args.ground_truth_class_values)
    if len(set(args.ground_truth_class_values)) != 3:
        parser.error("--ground-truth-class-values must contain three distinct values")
    if args.ignore_value in args.ground_truth_class_values:
        parser.error("--ignore-value must differ from all ground-truth class values")
    return args


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not value:
        raise ValueError("Dataset, model and run names cannot be empty")
    return value


def index_masks(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {directory}")
    masks = {
        path.relative_to(directory).as_posix(): path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.casefold() == ".png"
    }
    if not masks:
        raise ValueError(f"No PNG masks found in: {directory}")
    return masks


def load_requests(args) -> tuple[str, dict[str, tuple[Path, Path | None]], dict[str, str]]:
    """Return canonical keys, references, and supported prediction-name aliases."""
    if args.dataset == "mastr1325":
        dataset = load_dataset("mastr1325", HERE)
        requests = {
            request.output_path.as_posix(): (request.ground_truth_path, request.image_path)
            for request in dataset.images
        }
        aliases = {
            request.image_path.with_suffix(".png").name: request.output_path.as_posix()
            for request in dataset.images
        }
        return dataset.name, requests, aliases
    references = index_masks(args.ground_truth_dir)
    return args.dataset_name or "Custom three-class dataset", {
        key: (path, None) for key, path in references.items()
    }, {}


def selected_keys(requests: dict, selection_file: Path | None) -> list[str]:
    if selection_file is None:
        return list(requests)
    keys = list(dict.fromkeys(
        line.strip() for line in selection_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ))
    if not keys:
        raise ValueError(f"Selection file is empty: {selection_file}")
    unknown = [key for key in keys if key not in requests]
    if unknown:
        raise ValueError("Unknown evaluation keys: " + ", ".join(unknown[:5]))
    return keys


def pair_predictions(raw_predictions: dict, requests: dict, aliases: dict) -> tuple[dict, list[str]]:
    paired, extras = {}, []
    for raw_key, path in raw_predictions.items():
        key = raw_key if raw_key in requests else aliases.get(raw_key)
        if key is None:
            extras.append(raw_key)
        elif key in paired:
            raise ValueError(f"Multiple predicted masks match {key}: {paired[key]} and {path}")
        else:
            paired[key] = path
    return paired, extras


def write_pairing_report(path, dataset_name, predictions, keys, extras, missing):
    lines = [
        "Three-class mask pairing report",
        f"Dataset: {dataset_name}",
        f"Prediction directory: {predictions.resolve()}",
        f"Expected predictions: {len(keys)}",
        f"Matched predictions: {len(keys) - len(missing)}",
        "", f"Predictions without ground truth ({len(extras)}):",
        *(extras or ["None"]),
        "", f"Missing predictions ({len(missing)}):", *(missing or ["None"]),
        "", "Metrics are withheld if any selected prediction is missing or invalid.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results(output, args, dataset_name, run_name, rows, confusion, extras):
    macro = macro_average(rows)
    global_metrics = metrics_from_confusion(confusion)
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_type": "three_class",
        "number_of_images": len(rows),
        "configuration": {
            "dataset": args.dataset,
            "dataset_name": dataset_name,
            "model_name": args.model_name,
            "run_name": run_name,
            "prediction_dir": str(args.prediction_dir.resolve()),
            "ground_truth_dir": str(args.ground_truth_dir.resolve()) if args.ground_truth_dir else str(PROJECT / "data/MaSTR1325/masks"),
            "selection_file": str(args.selection_file.resolve()) if args.selection_file else None,
            "prediction_class_ids": dict(zip(CLASS_NAMES, range(3))),
            "ground_truth_class_values": dict(zip(CLASS_NAMES, args.ground_truth_class_values)),
            "ground_truth_ignore_value": args.ignore_value,
            "resolution_policy": "native resolution; mismatched masks rejected",
        },
        "definitions": {
            "confusion_matrix": "Rows: ground truth; columns: prediction; order: obstacle, water, sky",
            "macro_metrics": "Equal-weight average across images; undefined values excluded per metric",
            "global_metrics": "Computed from the summed pixel confusion matrix",
            "mean_iou": "Equal-weight average of class IoUs with nonzero union",
            "undefined_metrics": "NaN in CSV; null in JSON; never replaced with zero",
            "dice": "Per-class Dice equals per-class F1",
        },
        "pairing": {"expected": len(rows), "evaluated": len(rows),
                    "predictions_without_ground_truth": len(extras), "missing_predictions": 0},
        "valid_pixel_count": int(confusion.sum()),
        "ignored_pixel_count": sum(row["ignored_pixel_count"] for row in rows),
        "confusion_matrix": confusion.tolist(),
        "macro_metrics": macro,
        "global_metrics": global_metrics,
    }
    with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["ground_truth/prediction", *CLASS_NAMES])
        writer.writerows([name, *values] for name, values in zip(CLASS_NAMES, confusion.tolist()))
    lines = [
        "Three-class evaluation summary", f"Dataset: {dataset_name}",
        f"Model: {args.model_name}", f"Run: {run_name}", f"Evaluated: {len(rows)}",
        f"Valid pixels: {int(confusion.sum())}", f"Predictions without ground truth: {len(extras)}",
        "", "Classes: obstacle=0, water=1, sky=2",
        "Undefined metrics exclude absent denominators; missing class IoU is not zero.",
        "", "Macro metrics (mean across images):",
        *(f"  {name}: {value:.6f}" for name, value in macro.items()),
        "", "Global metrics (all valid pixels):",
        *(f"  {name}: {value}" if isinstance(value, int) else f"  {name}: {value:.6f}"
          for name, value in global_metrics.items()),
        "", "Confusion matrix (GT rows, prediction columns; obstacle/water/sky):",
        *("  " + " ".join(str(value) for value in row) for row in confusion.tolist()),
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    dataset_name, requests, aliases = load_requests(args)
    keys = selected_keys(requests, args.selection_file)
    predictions, extras = pair_predictions(index_masks(args.prediction_dir), requests, aliases)
    missing = [key for key in keys if key not in predictions]
    run_name = args.run_name or args.prediction_dir.resolve().parent.name
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output = args.results_dir / safe_name(args.model_name) / args.dataset / safe_name(run_name) / timestamp
    output.mkdir(parents=True)
    write_pairing_report(output / "mask_pairing_report.txt", dataset_name,
                         args.prediction_dir, keys, extras, missing)
    if missing:
        raise ValueError(f"{len(missing)} predictions are missing. Metrics withheld; see {output / 'mask_pairing_report.txt'}")
    rows = []
    confusion = np.zeros((3, 3), dtype=np.int64)
    for key in keys:
        reference, image_path = requests[key]
        counts, valid_pixels, (height, width) = evaluate_pair(
            predictions[key], reference, ground_truth_class_values=args.ground_truth_class_values,
            ignore_value=args.ignore_value,
        )
        if image_path is not None:
            with Image.open(image_path) as image:
                if image.size != (width, height):
                    raise ValueError(f"Image/mask size mismatch for {key}")
        confusion += counts
        rows.append({
            "dataset_key": args.dataset, "dataset_name": dataset_name,
            "model_name": args.model_name, "run_name": run_name,
            "image_id": image_path.stem if image_path else reference.stem,
            "evaluation_key": key, "prediction_path": str(predictions[key].resolve()),
            "ground_truth_path": str(reference.resolve()),
            "image_width_px": width, "image_height_px": height,
            "image_pixel_count": width * height, "image_resolution": f"{width}x{height}",
            "valid_pixel_count": valid_pixels, "ignored_pixel_count": width * height - valid_pixels,
            **metrics_from_confusion(counts),
        })
    write_results(output, args, dataset_name, run_name, rows, confusion, extras)
    print(f"Evaluated {len(rows)} images. Global mean IoU: {metrics_from_confusion(confusion)['mean_iou']:.6f}")
    print(f"Results: {output}")
    return output


if __name__ == "__main__":
    main()
