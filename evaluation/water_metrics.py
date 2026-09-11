"""Metric calculations used by the shared water-mask evaluator."""

import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt


COUNT_NAMES = (
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
)
METRIC_NAMES = (
    "water_iou",
    "dice",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "pixel_accuracy",
    "area_error_percent",
    "mean_symmetric_boundary_distance_px",
    "normalized_boundary_error_percent",
)
BOUNDARY_KERNEL = np.ones((3, 3), dtype=bool)


def divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp = counts["true_positive"], counts["false_positive"]
    fn, tn = counts["false_negative"], counts["true_negative"]
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    balanced_terms = [x for x in (recall, specificity) if not math.isnan(x)]
    true_water, predicted_water = tp + fn, tp + fp
    return {
        **counts,
        "water_iou": divide(tp, tp + fp + fn),
        "dice": divide(2 * tp, 2 * tp + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (
            sum(balanced_terms) / len(balanced_terms)
            if balanced_terms
            else math.nan
        ),
        "pixel_accuracy": divide(tp + tn, tp + fp + fn + tn),
        "area_error_percent": divide(
            100 * (predicted_water - true_water), true_water
        ),
    }


def water_boundary(water: np.ndarray) -> np.ndarray:
    """Return every internal water/non-water boundary, excluding image edges."""
    interior = binary_erosion(
        np.ones(water.shape, dtype=bool),
        structure=BOUNDARY_KERNEL,
        border_value=0,
    )
    eroded = binary_erosion(
        water,
        structure=BOUNDARY_KERNEL,
        border_value=1,
    )
    return water & ~eroded & interior


def boundary_metrics(
    predicted_water: np.ndarray,
    actual_water: np.ndarray,
) -> dict[str, float | int | str]:
    predicted, actual = map(water_boundary, (predicted_water, actual_water))
    predicted_count, actual_count = int(predicted.sum()), int(actual.sum())
    height, width = actual_water.shape
    diagonal = math.hypot(width, height)

    if not predicted_count or not actual_count:
        masks_match = np.array_equal(predicted_water, actual_water)
        distance = 0.0 if masks_match else diagonal
        status = "empty_match" if masks_match else "missing_boundary"
    else:
        combined = predicted | actual
        rows, columns = np.nonzero(combined)
        area = np.s_[
            rows.min() : rows.max() + 1,
            columns.min() : columns.max() + 1,
        ]
        predicted, actual = predicted[area], actual[area]
        forward = distance_transform_edt(~actual)[predicted].mean()
        backward = distance_transform_edt(~predicted)[actual].mean()
        distance, status = float((forward + backward) / 2), "ok"

    return {
        "mean_symmetric_boundary_distance_px": distance,
        "normalized_boundary_error_percent": 100 * distance / diagonal,
        "predicted_boundary_pixels": predicted_count,
        "ground_truth_boundary_pixels": actual_count,
        "boundary_status": status,
    }


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def evaluate_pair(prediction_path: Path, ground_truth_path: Path, encoding):
    prediction, ground_truth = map(load_mask, (prediction_path, ground_truth_path))
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: {prediction_path} {prediction.shape}, "
            f"{ground_truth_path} {ground_truth.shape}"
        )

    valid = np.ones(ground_truth.shape, dtype=bool)
    if encoding.ignore_value is not None:
        valid &= ground_truth != encoding.ignore_value
    actual_water = (
        ground_truth >= encoding.water_threshold
        if encoding.water_threshold is not None
        else ground_truth == encoding.water_value
    )
    predicted_water = prediction >= 128
    scored_prediction, scored_actual = predicted_water & valid, actual_water & valid
    counts = {
        "true_positive": int(np.sum(scored_prediction & scored_actual)),
        "false_positive": int(np.sum(scored_prediction & ~scored_actual & valid)),
        "false_negative": int(np.sum(~scored_prediction & scored_actual & valid)),
        "true_negative": int(np.sum(~scored_prediction & ~scored_actual & valid)),
    }
    return (
        counts,
        boundary_metrics(predicted_water, scored_actual),
        int(valid.sum()),
        ground_truth.shape,
    )


def macro_average(rows: list[dict]) -> dict[str, float]:
    averages = {}
    for name in METRIC_NAMES:
        values = [float(row[name]) for row in rows]
        values = [value for value in values if not math.isnan(value)]
        averages[name] = sum(values) / len(values) if values else math.nan
    return averages


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value
