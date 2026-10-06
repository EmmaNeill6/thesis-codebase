"""Compare paired eWaSR and SAM 3 timing and water-mask evaluation results.

Only timing-protocol-v2 observations are accepted. By default the two timing
files must contain exactly the same evaluation keys; --allow-intersection is
available for an explicitly labelled partial comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import fmean, median

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from timing_comparison import (  # noqa: E402
    TIMING_DEFINITIONS,
    TIMING_PROTOCOL_VERSION,
    timing_stats,
)


QUALITY_METRICS = (
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
HEADLINE_QUALITY = ("water_iou", "dice", "precision", "recall")
TIMING_METRICS = ("inference_ms", "final_mask_latency_ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ewasr-run-dir", required=True, type=Path)
    parser.add_argument("--sam3-run-dir", required=True, type=Path)
    parser.add_argument(
        "--ewasr-evaluation", required=True, type=Path,
        help="Evaluator result directory or its per_image.csv",
    )
    parser.add_argument(
        "--sam3-evaluation", required=True, type=Path,
        help="Evaluator result directory or its per_image.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="New directory for the comparison; defaults under evaluation/binary/results/comparison",
    )
    parser.add_argument(
        "--allow-intersection", action="store_true",
        help="Compare only shared timing keys and report exclusions",
    )
    return parser.parse_args()


def read_rows(path: Path, required: set[str], label: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"{label} lacks columns: {sorted(missing_columns)}")
        rows = {}
        for line_number, row in enumerate(reader, start=2):
            key = (row["evaluation_key"] or "").strip()
            if not key:
                raise ValueError(f"{label} has an empty evaluation_key at line {line_number}")
            if key in rows:
                raise ValueError(f"{label} has duplicate evaluation_key {key!r}")
            rows[key] = row
    if not rows:
        raise ValueError(f"{label} contains no image rows: {path}")
    return rows


def finite_number(value: str | None, label: str, *, allow_undefined: bool = False) -> float | None:
    if value is None or not value.strip():
        if allow_undefined:
            return None
        raise ValueError(f"Missing numeric value for {label}")
    number = float(value)
    if math.isnan(number) and allow_undefined:
        return None
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value for {label}: {value!r}")
    return number


def load_timing(run_dir: Path, label: str) -> dict[str, dict[str, float]]:
    summary_path = run_dir / "comparable_timing_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"{label} timing summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("timing_protocol_version") != TIMING_PROTOCOL_VERSION:
        raise ValueError(f"{label} timing protocol is not version {TIMING_PROTOCOL_VERSION}")
    if summary.get("definitions") != TIMING_DEFINITIONS:
        raise ValueError(f"{label} timing definitions differ from the shared protocol")

    raw = read_rows(
        run_dir / "comparable_timings.csv",
        {"evaluation_key", *TIMING_METRICS},
        f"{label} timing CSV",
    )
    if summary.get("sample_count") != len(raw):
        raise ValueError(f"{label} timing CSV and summary have different sample counts")
    timings = {}
    for key, row in raw.items():
        inference = finite_number(row["inference_ms"], f"{label}/{key}/inference_ms")
        latency = finite_number(
            row["final_mask_latency_ms"], f"{label}/{key}/final_mask_latency_ms"
        )
        if inference < 0 or latency < inference:
            raise ValueError(f"Invalid timing boundaries for {label}/{key}")
        timings[key] = {"inference_ms": inference, "final_mask_latency_ms": latency}
    return timings


def evaluation_path(path: Path) -> Path:
    return path / "per_image.csv" if path.is_dir() else path


def load_evaluation(path: Path, label: str) -> dict[str, dict]:
    raw = read_rows(
        evaluation_path(path),
        {
            "evaluation_key", "dataset_key", "model_name", "run_name",
            "image_width_px", "image_height_px", "valid_pixel_count",
            *QUALITY_METRICS,
        },
        f"{label} evaluation CSV",
    )
    evaluation = {}
    for key, row in raw.items():
        metadata = {}
        for name in ("image_width_px", "image_height_px", "valid_pixel_count"):
            value = finite_number(row[name], f"{label}/{key}/{name}")
            if value < 0 or not value.is_integer():
                raise ValueError(f"Invalid {name} for {label}/{key}")
            metadata[name] = int(value)
        evaluation[key] = {
            "dataset_key": row["dataset_key"],
            "model_name": row["model_name"],
            "run_name": row["run_name"],
            **metadata,
            "quality": {
                metric: finite_number(
                    row[metric], f"{label}/{key}/{metric}", allow_undefined=True
                )
                for metric in QUALITY_METRICS
            },
        }
    signatures = {
        (row["dataset_key"], row["model_name"], row["run_name"])
        for row in evaluation.values()
    }
    if len(signatures) != 1:
        raise ValueError(f"{label} evaluation CSV mixes datasets, models, or runs")
    return evaluation


def verify_evaluation_source(
    path: Path, run_dir: Path, label: str, evaluation: dict[str, dict]
) -> None:
    """Require evaluator provenance to point at this predictor's masks."""

    summary_path = evaluation_path(path).with_name("summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{label} evaluator summary is required to verify mask provenance: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source = summary.get("configuration", {}).get("prediction_dir")
    expected = (run_dir / "predicted_masks").resolve()
    if not source or Path(source).resolve() != expected:
        raise ValueError(
            f"{label} evaluator scored {source!r}, but this timing run uses "
            f"{str(expected)!r}. Re-evaluate this predictor's masks."
        )
    if summary.get("number_of_images") != len(evaluation):
        raise ValueError(f"{label} evaluator CSV and summary have different image counts")
    dataset = next(iter(evaluation.values()))["dataset_key"]
    if summary.get("configuration", {}).get("dataset") != dataset:
        raise ValueError(f"{label} evaluator summary has a different dataset")


def _difference(sam3: float | None, ewasr: float | None) -> float | None:
    return None if sam3 is None or ewasr is None else sam3 - ewasr


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return fmean(defined) if defined else None


def _delta_stats(values: list[float | None]) -> dict[str, float | int | None]:
    defined = [value for value in values if value is not None]
    return {
        "defined_images": len(defined),
        "mean": fmean(defined) if defined else None,
        "median": median(defined) if defined else None,
    }


def build_comparison(
    ewasr_timing: dict[str, dict[str, float]],
    sam3_timing: dict[str, dict[str, float]],
    ewasr_eval: dict[str, dict],
    sam3_eval: dict[str, dict],
    *,
    allow_intersection: bool = False,
) -> tuple[list[dict], dict]:
    ewasr_keys, sam3_keys = set(ewasr_timing), set(sam3_timing)
    ewasr_only, sam3_only = sorted(ewasr_keys - sam3_keys), sorted(sam3_keys - ewasr_keys)
    if (ewasr_only or sam3_only) and not allow_intersection:
        raise ValueError(
            "Timing image sets differ; rerun on the same images or use "
            "--allow-intersection for a labelled partial comparison. "
            f"eWaSR-only={len(ewasr_only)} {ewasr_only[:5]}; "
            f"SAM3-only={len(sam3_only)} {sam3_only[:5]}"
        )
    keys = sorted(ewasr_keys & sam3_keys)
    if not keys:
        raise ValueError("The timing files have no shared evaluation keys")
    for label, evaluation in (("eWaSR", ewasr_eval), ("SAM 3", sam3_eval)):
        missing = sorted(set(keys) - set(evaluation))
        if missing:
            raise ValueError(f"{label} evaluation lacks {len(missing)} paired keys: {missing[:5]}")

    rows = []
    datasets = set()
    for key in keys:
        e, s = ewasr_eval[key], sam3_eval[key]
        datasets.update((e["dataset_key"], s["dataset_key"]))
        for name in ("image_width_px", "image_height_px", "valid_pixel_count"):
            if e[name] != s[name]:
                raise ValueError(f"Evaluation metadata differs for {key}: {name}")
        row = {
            "evaluation_key": key,
            "dataset_key": e["dataset_key"],
            "image_width_px": e["image_width_px"],
            "image_height_px": e["image_height_px"],
            "valid_pixel_count": e["valid_pixel_count"],
        }
        for metric in TIMING_METRICS:
            row[f"ewasr_{metric}"] = ewasr_timing[key][metric]
            row[f"sam3_{metric}"] = sam3_timing[key][metric]
            row[f"sam3_minus_ewasr_{metric}"] = _difference(
                sam3_timing[key][metric], ewasr_timing[key][metric]
            )
        for metric in QUALITY_METRICS:
            row[f"ewasr_{metric}"] = e["quality"][metric]
            row[f"sam3_{metric}"] = s["quality"][metric]
            row[f"sam3_minus_ewasr_{metric}"] = _difference(
                s["quality"][metric], e["quality"][metric]
            )
        rows.append(row)
    if len(datasets) != 1 or not next(iter(datasets)):
        raise ValueError(f"Evaluation files mix datasets: {sorted(datasets)}")

    models = {}
    for model in ("ewasr", "sam3"):
        inference = [row[f"{model}_inference_ms"] for row in rows]
        mean_inference = fmean(inference)
        models[model] = {
            "evaluation_model_name": (ewasr_eval if model == "ewasr" else sam3_eval)[keys[0]]["model_name"],
            "evaluation_run_name": (ewasr_eval if model == "ewasr" else sam3_eval)[keys[0]]["run_name"],
            "inference_ms": timing_stats(inference),
            "final_mask_latency_ms": timing_stats(
                [row[f"{model}_final_mask_latency_ms"] for row in rows]
            ),
            "inference_only_fps_from_mean": 1000 / mean_inference if mean_inference else None,
            "macro_quality_on_paired_images": {
                metric: _mean_defined([row[f"{model}_{metric}"] for row in rows])
                for metric in QUALITY_METRICS
            },
        }

    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_key": next(iter(datasets)),
        "timing_protocol_version": TIMING_PROTOCOL_VERSION,
        "paired_images": len(rows),
        "pairing": {
            "policy": "intersection" if allow_intersection else "exact_timing_key_match",
            "ewasr_timing_images": len(ewasr_keys),
            "sam3_timing_images": len(sam3_keys),
            "ewasr_only_excluded_count": len(ewasr_only),
            "sam3_only_excluded_count": len(sam3_only),
            "ewasr_only_excluded_examples": ewasr_only[:10],
            "sam3_only_excluded_examples": sam3_only[:10],
        },
        "models": models,
        "paired_sam3_minus_ewasr": {
            metric: _delta_stats([row[f"sam3_minus_ewasr_{metric}"] for row in rows])
            for metric in (*TIMING_METRICS, *QUALITY_METRICS)
        },
        "interpretation": (
            "Negative SAM 3 minus eWaSR timing means SAM 3 is faster; positive "
            "water IoU/Dice difference means SAM 3 has better overlap. "
            "Inference-only FPS is not actual batch throughput."
        ),
    }
    return rows, summary


def memory_snapshot(
    run_dir: Path, paired_images: int, *, model: str, dataset: str
) -> dict:
    """Use memory only when a single invocation covers the paired images."""

    path = run_dir / "memory_usage.jsonl"
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 1:
        return {"status": "multiple_invocations_or_empty", "record_count": len(records), "path": str(path)}
    record = records[0]
    if record.get("model") != model or record.get("dataset") != dataset:
        return {"status": "model_or_dataset_mismatch", "path": str(path)}
    if (
        record.get("successful_images_this_invocation") != paired_images
        or record.get("attempted_images_this_invocation") != paired_images
    ):
        return {
            "status": "image_count_mismatch",
            "attempted_images_this_invocation": record.get("attempted_images_this_invocation"),
            "successful_images_this_invocation": record.get("successful_images_this_invocation"),
            "paired_images": paired_images,
            "path": str(path),
        }
    peak = record.get("peak_process_rss_mib")
    if not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
        raise ValueError(f"Invalid peak process RSS in {path}")
    return {
        "status": "single_invocation_same_count",
        "peak_process_rss_mib": peak,
        "recorded_at": record.get("recorded_at"),
        "path": str(path),
    }


def _display(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_outputs(output_dir: Path, rows: list[dict], summary: dict) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Comparison output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    with (output_dir / "paired_per_image.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Paired eWaSR–SAM 3 performance comparison",
        "",
        f"Dataset: `{summary['dataset_key']}` · Paired images: **{summary['paired_images']}**",
        "",
        "| Model | Mean water IoU | Mean Dice | Median final-mask latency (ms) | p95 final-mask latency (ms) | Inference-only FPS | Peak process RSS (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, label in (("ewasr", "eWaSR"), ("sam3", "SAM 3")):
        data = summary["models"][model]
        memory = summary["memory"][model]
        peak = memory.get("peak_process_rss_mib") if memory["status"] == "single_invocation_same_count" else None
        lines.append(
            f"| {label} | "
            f"{_display(data['macro_quality_on_paired_images']['water_iou'])} | "
            f"{_display(data['macro_quality_on_paired_images']['dice'])} | "
            f"{_display(data['final_mask_latency_ms']['median'], 1)} | "
            f"{_display(data['final_mask_latency_ms']['p95'], 1)} | "
            f"{_display(data['inference_only_fps_from_mean'], 2)} | "
            f"{_display(peak, 1)} |"
        )
    lines.extend(
        [
            "",
            "The rows are paired by `evaluation_key`; timing statistics and macro quality are recalculated **on those paired images only**.",
            "Inference-only FPS is derived from mean inference time, not whole-run throughput.",
            "Peak RSS is shown only when one logged invocation has the same successful-image count as the paired set; it is process memory, not GPU-only memory.",
            f"Pairing policy: `{summary['pairing']['policy']}`; eWaSR-only excluded: {summary['pairing']['ewasr_only_excluded_count']}; SAM 3-only excluded: {summary['pairing']['sam3_only_excluded_count']}.",
            "",
        ]
    )
    (output_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ewasr_timing = load_timing(args.ewasr_run_dir, "eWaSR")
    sam3_timing = load_timing(args.sam3_run_dir, "SAM 3")
    ewasr_eval = load_evaluation(args.ewasr_evaluation, "eWaSR")
    sam3_eval = load_evaluation(args.sam3_evaluation, "SAM 3")
    verify_evaluation_source(
        args.ewasr_evaluation, args.ewasr_run_dir, "eWaSR", ewasr_eval
    )
    verify_evaluation_source(
        args.sam3_evaluation, args.sam3_run_dir, "SAM 3", sam3_eval
    )
    rows, summary = build_comparison(
        ewasr_timing, sam3_timing, ewasr_eval, sam3_eval,
        allow_intersection=args.allow_intersection,
    )
    summary["inputs"] = {
        "ewasr_run_dir": str(args.ewasr_run_dir.resolve()),
        "sam3_run_dir": str(args.sam3_run_dir.resolve()),
        "ewasr_evaluation": str(evaluation_path(args.ewasr_evaluation).resolve()),
        "sam3_evaluation": str(evaluation_path(args.sam3_evaluation).resolve()),
    }
    if (
        summary["pairing"]["ewasr_only_excluded_count"]
        or summary["pairing"]["sam3_only_excluded_count"]
    ):
        summary["memory"] = {
            "ewasr": {"status": "partial_key_intersection"},
            "sam3": {"status": "partial_key_intersection"},
        }
    else:
        summary["memory"] = {
            "ewasr": memory_snapshot(
                args.ewasr_run_dir, len(rows), model="eWaSR", dataset=summary["dataset_key"]
            ),
            "sam3": memory_snapshot(
                args.sam3_run_dir, len(rows), model="SAM 3", dataset=summary["dataset_key"]
            ),
        }
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_dir = args.output_dir or HERE / "results" / "comparison" / summary["dataset_key"] / timestamp
    write_outputs(output_dir, rows, summary)
    print(f"Paired images: {len(rows)}")
    print(f"Comparison: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
