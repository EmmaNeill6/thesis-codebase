"""Evaluate any model's 0/255 binary water masks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from water_metrics import (
    COUNT_NAMES,
    evaluate_pair,
    json_safe,
    macro_average,
    metrics_from_counts,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SPECS_DIR = PROJECT
RESOLUTION_CSV = PROJECT / "dataset_metadata/dataset_image_resolutions.csv"
sys.path.insert(0, str(SPECS_DIR))
from dataset_specs import DATASETS, load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate binary water masks.")
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument(
        "--prediction-dir",
        required=True,
        type=Path,
        help="The predicted_masks directory for one model run",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--run-name",
        help="Defaults to the directory containing predicted_masks",
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="Optional list of evaluation keys; all are used by default",
    )
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not value:
        raise ValueError("Model and run names cannot be empty")
    return value


def load_resolutions() -> dict[tuple[str, str], dict[str, str]]:
    if not RESOLUTION_CSV.is_file():
        raise FileNotFoundError(
            f"Resolution metadata not found: {RESOLUTION_CSV}. Run "
            "dataset_resolution_audit.py first."
        )
    with RESOLUTION_CSV.open(encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source)
        return {
            (row["dataset_key"], row["evaluation_key"]): row for row in rows
        }


def select_requests(dataset, selection_file: Path | None):
    requests = list(dataset.images)
    if selection_file is None:
        return requests
    if not selection_file.is_file():
        raise FileNotFoundError(f"Selection file not found: {selection_file}")

    keys = [
        line.strip()
        for line in selection_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not keys:
        raise ValueError(f"Selection file is empty: {selection_file}")
    by_key = {request.output_path.as_posix(): request for request in requests}
    unknown = [key for key in keys if key not in by_key]
    if unknown:
        raise ValueError("Unknown evaluation keys: " + ", ".join(unknown[:5]))
    return [by_key[key] for key in dict.fromkeys(keys)]


def index_predictions(prediction_dir: Path) -> dict[str, Path]:
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    predictions = {
        path.relative_to(prediction_dir).as_posix(): path
        for path in sorted(prediction_dir.rglob("*.png"))
        if path.is_file()
    }
    if not predictions:
        raise FileNotFoundError(f"No PNG predictions found in: {prediction_dir}")
    return predictions


def write_pairing_report(
    path: Path,
    dataset_name: str,
    prediction_dir: Path,
    expected: int,
    matched: int,
    extras: list[str],
    missing: list[str],
) -> None:
    lines = [
        "Mask pairing report",
        f"Dataset: {dataset_name}",
        f"Prediction directory: {prediction_dir.resolve()}",
        f"Expected predictions: {expected}",
        f"Matched predictions: {matched}",
        "",
        f"Predicted masks with no corresponding ground truth ({len(extras)}):",
        *((f"  {item}" for item in extras) if extras else ["  None"]),
        "",
        f"Ground truths with no corresponding prediction ({len(missing)}):",
        *((f"  {item}" for item in missing) if missing else ["  None"]),
        "",
        "Only matched and validated pairs contribute to metrics.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def result_directory(args, run_name: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    path = (
        args.results_dir
        / safe_name(args.model_name)
        / args.dataset
        / safe_name(run_name)
        / timestamp
    )
    path.mkdir(parents=True)
    return path


def evaluate_requests(args, dataset, requests, predictions, resolutions):
    rows = []
    global_counts = dict.fromkeys(COUNT_NAMES, 0)
    run_name = args.run_name or args.prediction_dir.resolve().parent.name
    for request in requests:
        key = request.output_path.as_posix()
        resolution = resolutions.get((args.dataset, key))
        if resolution is None:
            raise KeyError(f"Resolution metadata missing for {args.dataset}/{key}")

        counts, boundaries, valid_pixels, shape = evaluate_pair(
            predictions[key],
            request.ground_truth_path,
            dataset.ground_truth_encoding,
        )
        height, width = shape
        audited_size = (
            int(resolution["image_width_px"]),
            int(resolution["image_height_px"]),
        )
        if (width, height) != audited_size:
            raise ValueError(f"Resolution audit mismatch for {args.dataset}/{key}")
        for name in COUNT_NAMES:
            global_counts[name] += counts[name]

        rows.append(
            {
                "dataset_key": args.dataset,
                "dataset_name": dataset.name,
                "model_name": args.model_name,
                "run_name": run_name,
                "image_id": request.image_path.stem,
                "evaluation_key": key,
                "image_width_px": width,
                "image_height_px": height,
                "image_pixel_count": resolution["image_pixel_count"],
                "image_megapixels": resolution["image_megapixels"],
                "image_aspect_ratio": resolution["image_aspect_ratio"],
                "image_resolution": resolution["image_resolution"],
                "valid_pixel_count": valid_pixels,
                **metrics_from_counts(counts),
                **boundaries,
            }
        )
    return rows, global_counts


def write_results(output_dir: Path, args, dataset, run_name, rows, extras):
    macro = macro_average(rows)
    global_metrics = metrics_from_counts(
        {name: sum(int(row[name]) for row in rows) for name in COUNT_NAMES}
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "number_of_images": len(rows),
        "configuration": {
            "dataset": args.dataset,
            "model_name": args.model_name,
            "run_name": run_name,
            "prediction_dir": str(args.prediction_dir.resolve()),
            "selection_file": str(args.selection_file) if args.selection_file else None,
            "prediction_water_rule": "pixel >= 128",
            "ground_truth_ignore_value": dataset.ground_truth_encoding.ignore_value,
            "boundary_metric": (
                "full symmetric water-boundary distance divided by image diagonal"
            ),
        },
        "pairing": {
            "expected": len(rows),
            "evaluated": len(rows),
            "predictions_without_ground_truth": len(extras),
            "missing_predictions": 0,
        },
        "macro_metrics": macro,
        "global_metrics": global_metrics,
    }

    with (output_dir / "per_image.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "Water mask evaluation summary",
        f"Dataset: {dataset.name}",
        f"Model: {args.model_name}",
        f"Run: {run_name}",
        f"Evaluated: {len(rows)}",
        f"Predictions without ground truth: {len(extras)}",
        "",
        "Macro metrics:",
        *(f"  {name}: {value:.4f}" for name, value in macro.items()),
        "",
        "Global metrics:",
        *(f"  {name}: {value:.4f}" for name, value in global_metrics.items()),
    ]
    (output_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, PROJECT / "TestingPart1/output_eWaSR")
    requests = select_requests(dataset, args.selection_file)
    predictions = index_predictions(args.prediction_dir)
    all_keys = {item.output_path.as_posix() for item in dataset.images}
    selected_keys = {item.output_path.as_posix() for item in requests}
    extras = sorted(set(predictions) - all_keys)
    missing = sorted(selected_keys - set(predictions))
    run_name = args.run_name or args.prediction_dir.resolve().parent.name
    output_dir = result_directory(args, run_name)
    pairing_path = output_dir / "mask_pairing_report.txt"
    write_pairing_report(
        pairing_path,
        dataset.name,
        args.prediction_dir,
        len(requests),
        len(requests) - len(missing),
        extras,
        missing,
    )
    if missing:
        raise RuntimeError(
            f"{len(missing)} predictions are missing. Metrics were withheld; "
            f"see {pairing_path}."
        )

    rows, _ = evaluate_requests(
        args, dataset, requests, predictions, load_resolutions()
    )
    write_results(output_dir, args, dataset, run_name, rows, extras)
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()
