"""Metrics for obstacle=0, water=1, sky=2 class-index masks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = ("obstacle", "water", "sky")
NUM_CLASSES = len(CLASS_NAMES)
CONFUSION_NAMES = tuple(
    f"true_{actual}_pred_{predicted}"
    for actual in CLASS_NAMES for predicted in CLASS_NAMES
)
METRIC_NAMES = (
    "pixel_accuracy", "mean_iou", "mean_dice", "mean_precision", "mean_recall",
    *(f"{name}_{metric}" for name in CLASS_NAMES
      for metric in ("iou", "dice", "precision", "recall")),
)


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def finite_mean(values) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else math.nan


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float | int]:
    """Rows are ground truth; columns are predictions. Absent ratios are NaN."""
    confusion = np.asarray(confusion)
    if (confusion.shape != (NUM_CLASSES, NUM_CLASSES)
            or not np.issubdtype(confusion.dtype, np.integer)
            or np.any(confusion < 0)):
        raise ValueError("Confusion matrix must be a nonnegative integer 3x3 array")
    total = int(confusion.sum())
    actual = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    result = {
        name: int(value) for name, value in zip(CONFUSION_NAMES, confusion.flat)
    }
    result["pixel_accuracy"] = divide(int(np.trace(confusion)), total)
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        fp, fn = int(predicted[index]) - tp, int(actual[index]) - tp
        result.update({
            f"{name}_true_positive": tp,
            f"{name}_false_positive": fp,
            f"{name}_false_negative": fn,
            f"{name}_true_negative": total - tp - fp - fn,
            f"{name}_support": int(actual[index]),
            f"{name}_predicted_pixels": int(predicted[index]),
            f"{name}_iou": divide(tp, tp + fp + fn),
            f"{name}_dice": divide(2 * tp, 2 * tp + fp + fn),
            f"{name}_precision": divide(tp, tp + fp),
            f"{name}_recall": divide(tp, tp + fn),
        })
    for metric in ("iou", "dice", "precision", "recall"):
        result[f"mean_{metric}"] = finite_mean(
            result[f"{name}_{metric}"] for name in CLASS_NAMES
        )
    return result


def read_mask(path: Path) -> np.ndarray:
    """Preserve palette indices; never turn visualisation colours into labels."""
    with Image.open(path) as source:
        mask = np.asarray(source)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Expected a single-channel class-index mask: {path}")
    return mask


def evaluate_pair(
    prediction_path: Path,
    ground_truth_path: Path,
    *,
    ground_truth_class_values: tuple[int, int, int] = (0, 1, 2),
    ignore_value: int | None = None,
) -> tuple[np.ndarray, int, tuple[int, int]]:
    """Validate and score at native resolution, masking only GT ignore pixels."""
    if len(ground_truth_class_values) != 3 or len(set(ground_truth_class_values)) != 3:
        raise ValueError("Ground-truth class values must be three distinct integers")
    if ignore_value in ground_truth_class_values:
        raise ValueError("Ignore value must differ from all ground-truth class values")
    prediction = read_mask(prediction_path)
    ground_truth = read_mask(ground_truth_path)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Mask size mismatch: {prediction_path} {prediction.shape} versus "
            f"{ground_truth_path} {ground_truth.shape}. No automatic resizing is performed."
        )
    invalid_prediction = np.setdiff1d(np.unique(prediction), np.arange(NUM_CLASSES))
    if invalid_prediction.size:
        raise ValueError(
            f"Prediction must contain only class IDs 0, 1, 2: {prediction_path} "
            f"contains unsupported values {invalid_prediction.tolist()}"
        )
    allowed = list(ground_truth_class_values)
    if ignore_value is not None:
        allowed.append(ignore_value)
    invalid_ground_truth = np.setdiff1d(np.unique(ground_truth), allowed)
    if invalid_ground_truth.size:
        raise ValueError(
            f"Unsupported ground-truth values {invalid_ground_truth.tolist()}: "
            f"{ground_truth_path}; expected {allowed}"
        )
    valid = ground_truth != ignore_value if ignore_value is not None else np.ones(
        ground_truth.shape, dtype=bool
    )
    valid_pixels = int(valid.sum())
    if not valid_pixels:
        raise ValueError(f"Ground truth has no valid pixels: {ground_truth_path}")
    targets = np.zeros(valid_pixels, dtype=np.int64)
    for index, raw_value in enumerate(ground_truth_class_values):
        targets[ground_truth[valid] == raw_value] = index
    predictions = prediction[valid].astype(np.int64)
    confusion = np.bincount(
        targets * NUM_CLASSES + predictions,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    return confusion, valid_pixels, ground_truth.shape


def macro_average(rows: list[dict]) -> dict[str, float]:
    return {name: finite_mean(row[name] for row in rows) for name in METRIC_NAMES}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value
