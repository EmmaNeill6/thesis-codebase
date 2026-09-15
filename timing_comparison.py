"""Shared timing definitions and reports for binary-water predictors."""

from __future__ import annotations

import csv
import io
import json
import math
import resource
import sys
from datetime import datetime
from pathlib import Path
from statistics import fmean, median


TIMING_PROTOCOL_VERSION = 2
TIMING_DEFINITIONS = {
    "inference_ms": (
        "After model-specific image preprocessing, through model execution and "
        "materialization of model outputs as NumPy arrays; excludes final-mask "
        "construction and file writes."
    ),
    "final_mask_latency_ms": (
        "From opening the source image through construction of the final "
        "0/255 uint8 water mask in memory; excludes output file writes."
    ),
}
MEMORY_DEFINITION = (
    "Peak resident memory of this Python process since launch, including model "
    "loading, warm-up, inference, and output writing; not GPU-only memory."
)


def percentile(values: list[float], fraction: float) -> float:
    """Linearly interpolate between adjacent sorted observations."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": fmean(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_timing_comparison(output_dir: Path, rows: list[dict]) -> dict:
    """Write identical, versioned latency reports for both model runners."""

    output_dir.mkdir(parents=True, exist_ok=True)
    keys = [str(row["evaluation_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Timing comparison contains duplicate evaluation keys")

    inference = [float(row["inference_ms"]) for row in rows]
    latency = [float(row["final_mask_latency_ms"]) for row in rows]
    for inference_ms, latency_ms in zip(inference, latency, strict=True):
        if not (math.isfinite(inference_ms) and math.isfinite(latency_ms)):
            raise ValueError("Timing comparison requires finite measurements")
        if inference_ms < 0 or latency_ms < inference_ms:
            raise ValueError("Final-mask latency must include inference latency")

    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(("evaluation_key", "inference_ms", "final_mask_latency_ms"))
    for key, inference_ms, latency_ms in zip(keys, inference, latency, strict=True):
        writer.writerow((key, f"{inference_ms:.6f}", f"{latency_ms:.6f}"))

    mean_inference_ms = fmean(inference) if inference else None
    summary = {
        "timing_protocol_version": TIMING_PROTOCOL_VERSION,
        "definitions": TIMING_DEFINITIONS,
        "sample_count": len(rows),
        "inference_ms": timing_stats(inference),
        "final_mask_latency_ms": timing_stats(latency),
        "inference_only_fps_from_mean": (
            1000 / mean_inference_ms if mean_inference_ms else None
        ),
    }
    _atomic_text(output_dir / "comparable_timings.csv", table.getvalue())
    _atomic_text(
        output_dir / "comparable_timing_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    return summary


def peak_process_rss_mib() -> float:
    """Return peak process RSS, converting macOS bytes or Linux KiB to MiB."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    if sys.platform.startswith("linux"):
        return peak / 1024
    raise RuntimeError(f"Unsupported peak RSS unit on {sys.platform}")


def append_memory_usage(
    output_dir: Path,
    *,
    model: str,
    dataset: str,
    attempted_images: int,
    successful_images: int,
) -> dict:
    """Append one process-level memory measurement per predictor invocation."""

    if not 0 <= successful_images <= attempted_images:
        raise ValueError("Successful image count must not exceed attempts")
    report = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
        "model": model,
        "dataset": dataset,
        "attempted_images_this_invocation": attempted_images,
        "successful_images_this_invocation": successful_images,
        "peak_process_rss_mib": peak_process_rss_mib(),
        "definition": MEMORY_DEFINITION,
        "platform": sys.platform,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "memory_usage.jsonl").open("a", encoding="utf-8") as target:
        target.write(json.dumps(report, allow_nan=False) + "\n")
    return report
