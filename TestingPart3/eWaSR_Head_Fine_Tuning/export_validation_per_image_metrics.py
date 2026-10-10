"""Export per-image validation metrics from saved head-only eWaSR runs.

The predictions are evaluated at the 128x96 head resolution used by the
training reports. The pooled confusion matrix must reproduce evaluation.txt
before any CSV is written.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from ewasr_mlx.model import create_head_finetuning_model
from train_head import Sample, confusion_metrics, load_sample


FIELDS = (
    "model", "split", "image_id", "image_path", "reference_mask_path",
    "valid_pixels", "water_iou", "three_class_mean_iou", "pixel_accuracy",
    "obstacle_precision", "obstacle_recall", "water_labelled_as_obstacle_rate",
)


def report_confusion(path: Path) -> np.ndarray:
    report = path.read_text(encoding="utf-8")
    validation = report.split("Best validation epoch:", 1)[1].split(
        "Selected checkpoint on test split:", 1
    )[0]
    match = re.search(r"^  confusion_matrix \(rows=true, columns=predicted\): (.+)$", validation, re.M)
    if match is None:
        raise ValueError(f"Missing validation confusion matrix in {path}")
    return np.asarray(ast.literal_eval(match.group(1)), dtype=np.int64)


def score_run(model_name: str, run_dir: Path) -> list[dict[str, object]]:
    run_dir = run_dir.expanduser().resolve()
    splits = json.loads((run_dir / "resolved_splits.json").read_text(encoding="utf-8"))
    samples = splits["validation"]
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    width, height = int(config["width"]), int(config["height"])
    checkpoint = run_dir / "best_head_only.npz"
    model = create_head_finetuning_model(checkpoint)
    model.set_head_training_mode(False)

    rows: list[dict[str, object]] = []
    pooled = np.zeros((3, 3), dtype=np.int64)
    for index, entry in enumerate(samples, 1):
        sample = Sample(Path(entry["image"]), Path(entry["mask"]))
        image, reference = load_sample(sample, width=width, height=height)
        prediction = mx.argmax(model(mx.array(image[None, ...])), axis=-1)
        mx.eval(prediction)
        predicted = np.asarray(prediction, dtype=np.int32)[0]
        if predicted.shape != reference.shape:
            raise ValueError(f"Prediction/reference size mismatch for {sample.image_path}")
        confusion = np.bincount(
            (reference.ravel() * 3 + predicted.ravel()), minlength=9
        ).reshape(3, 3)
        pooled += confusion
        metrics = confusion_metrics(confusion)
        rows.append({
            "model": model_name,
            "split": "validation",
            "image_id": sample.image_path.stem,
            "image_path": str(sample.image_path),
            "reference_mask_path": str(sample.mask_path),
            "valid_pixels": int(confusion.sum()),
            "water_iou": metrics["binary_water_iou"],
            "three_class_mean_iou": metrics["mean_iou"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "obstacle_precision": metrics["obstacle_precision"],
            "obstacle_recall": metrics["obstacle_recall"],
            "water_labelled_as_obstacle_rate": metrics["water_as_obstacle_rate"],
        })
        if index % 20 == 0 or index == len(samples):
            print(f"{model_name}: {index}/{len(samples)} validation images", flush=True)

    expected = report_confusion(run_dir / "evaluation.txt")
    if not np.array_equal(pooled, expected):
        raise ValueError(
            f"Pooled confusion mismatch for {model_name}: "
            f"computed {pooled.tolist()}, report {expected.tolist()}"
        )
    print(f"{model_name}: pooled confusion matches evaluation.txt", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs=2, action="append", metavar=("NAME", "RUN_DIR"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.model]
    if len(names) != len(set(names)):
        parser.error("model names must be unique")
    rows = [row for name, run_dir in args.model for row in score_run(name, Path(run_dir))]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
