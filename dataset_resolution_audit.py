"""Record image resolutions for every dataset image with a matching mask.

The per-image CSV is designed to be joined into later evaluation results using
``dataset_key`` and ``evaluation_key``. The dataset summary JSON records the
resolution distribution and any image/mask dimension mismatches.

Run with no arguments to audit every dataset declared in
``dataset_specs.py``::

    python dataset_resolution_audit.py

Optionally select one or more datasets::

    python dataset_resolution_audit.py --datasets mastr1325 tampere
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
DATASET_SPECS_PATH = ROOT / "dataset_specs.py"
DEFAULT_OUTPUT_DIR = ROOT / "dataset_metadata"
DATASET_OUTPUT_ROOT = ROOT / "TestingPart1/output_eWaSR"
SCHEMA_VERSION = "1.0"

# The project-root module is shared by every model and the evaluator.
sys.path.insert(0, str(ROOT))
from dataset_specs import DATASETS, ImageRequest, load_dataset  # noqa: E402


@dataclass(frozen=True)
class ResolutionRecord:
    """Machine-readable metadata for one paired image and ground-truth mask."""

    schema_version: str
    dataset_key: str
    dataset_name: str
    evaluation_key: str
    image_path: str
    ground_truth_path: str
    image_filename: str
    image_extension: str
    image_format: str
    image_mode: str
    image_width_px: int
    image_height_px: int
    image_pixel_count: int
    image_megapixels: float
    image_aspect_ratio: float
    image_orientation: str
    image_resolution: str
    mask_format: str
    mask_mode: str
    mask_width_px: int
    mask_height_px: int
    mask_pixel_count: int
    mask_resolution: str
    dimensions_match: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record image and mask resolutions for paired samples selected by "
            "dataset_specs.py."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=tuple(DATASETS),
        help="Datasets to audit (default: every configured dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def relative_path(path: Path) -> str:
    """Return a project-relative POSIX path where possible."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def orientation(width: int, height: int) -> str:
    if width > height:
        return "landscape"
    if width < height:
        return "portrait"
    return "square"


def image_metadata(path: Path) -> tuple[str, str, int, int]:
    """Read metadata without decoding the complete pixel array."""
    with Image.open(path) as image:
        return image.format or "unknown", image.mode, *image.size


def make_record(
    dataset_key: str,
    dataset_name: str,
    request: ImageRequest,
) -> ResolutionRecord:
    if not request.image_path.is_file():
        raise FileNotFoundError(f"Image not found: {request.image_path}")
    ground_truth_path = request.ground_truth_path
    if ground_truth_path is None:
        raise ValueError(
            f"{dataset_name} image has no ground-truth mask: {request.image_path}"
        )
    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth mask not found: {ground_truth_path}"
        )

    image_format, image_mode, image_width, image_height = image_metadata(
        request.image_path
    )
    mask_format, mask_mode, mask_width, mask_height = image_metadata(
        ground_truth_path
    )
    image_pixels = image_width * image_height
    mask_pixels = mask_width * mask_height

    return ResolutionRecord(
        schema_version=SCHEMA_VERSION,
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        # This is also the relative output path used for predicted masks.
        # It is therefore a stable join key for future evaluation records.
        evaluation_key=request.output_path.as_posix(),
        image_path=relative_path(request.image_path),
        ground_truth_path=relative_path(ground_truth_path),
        image_filename=request.image_path.name,
        image_extension=request.image_path.suffix.casefold(),
        image_format=image_format,
        image_mode=image_mode,
        image_width_px=image_width,
        image_height_px=image_height,
        image_pixel_count=image_pixels,
        image_megapixels=round(image_pixels / 1_000_000, 6),
        image_aspect_ratio=round(image_width / image_height, 8),
        image_orientation=orientation(image_width, image_height),
        image_resolution=f"{image_width}x{image_height}",
        mask_format=mask_format,
        mask_mode=mask_mode,
        mask_width_px=mask_width,
        mask_height_px=mask_height,
        mask_pixel_count=mask_pixels,
        mask_resolution=f"{mask_width}x{mask_height}",
        dimensions_match=(image_width, image_height)
        == (mask_width, mask_height),
    )


def resolution_counts(
    records: list[ResolutionRecord],
    width_field: str,
    height_field: str,
) -> list[dict[str, int | str]]:
    counts = Counter(
        (getattr(record, width_field), getattr(record, height_field))
        for record in records
    )
    return [
        {
            "resolution": f"{width}x{height}",
            "width_px": width,
            "height_px": height,
            "pixel_count": width * height,
            "number_of_samples": count,
        }
        for (width, height), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def summarise_dataset(
    dataset_key: str,
    dataset_name: str,
    records: list[ResolutionRecord],
) -> dict[str, object]:
    widths = [record.image_width_px for record in records]
    heights = [record.image_height_px for record in records]
    pixel_counts = [record.image_pixel_count for record in records]
    mismatches = [
        record.evaluation_key for record in records if not record.dimensions_match
    ]

    return {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "number_of_paired_samples": len(records),
        "number_of_image_resolutions": len(set(zip(widths, heights))),
        "minimum_image_width_px": min(widths),
        "maximum_image_width_px": max(widths),
        "minimum_image_height_px": min(heights),
        "maximum_image_height_px": max(heights),
        "minimum_image_pixel_count": min(pixel_counts),
        "maximum_image_pixel_count": max(pixel_counts),
        "image_resolutions": resolution_counts(
            records, "image_width_px", "image_height_px"
        ),
        "mask_resolutions": resolution_counts(
            records, "mask_width_px", "mask_height_px"
        ),
        "number_of_dimension_mismatches": len(mismatches),
        "dimension_mismatch_evaluation_keys": mismatches,
    }


def write_csv(path: Path, records: list[ResolutionRecord]) -> None:
    fieldnames = list(asdict(records[0]))
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def main() -> None:
    args = parse_args()
    records: list[ResolutionRecord] = []
    summaries: list[dict[str, object]] = []

    for dataset_key in args.datasets:
        dataset = load_dataset(dataset_key, DATASET_OUTPUT_ROOT)
        paired_requests = [
            request
            for request in dataset.images
            if request.ground_truth_path is not None
        ]
        if not paired_requests:
            print(
                f"Skipping {dataset.name}: no ground-truth masks are configured.",
                file=sys.stderr,
            )
            continue
        dataset_records = [
            make_record(dataset_key, dataset.name, request)
            for request in paired_requests
        ]
        records.extend(dataset_records)
        summaries.append(
            summarise_dataset(dataset_key, dataset.name, dataset_records)
        )

    if not records:
        raise ValueError("No paired dataset samples were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "dataset_image_resolutions.csv"
    json_path = args.output_dir / "dataset_resolution_summary.json"
    write_csv(csv_path, records)

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_source": relative_path(DATASET_SPECS_PATH),
        "records_csv": relative_path(csv_path),
        "join_columns": ["dataset_key", "evaluation_key"],
        "number_of_paired_samples": len(records),
        "datasets": summaries,
    }
    json_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Recorded {len(records)} paired image/mask samples.")
    for summary in summaries:
        print(
            f"  {summary['dataset_name']}: "
            f"{summary['number_of_paired_samples']} samples, "
            f"{summary['number_of_image_resolutions']} image resolutions, "
            f"{summary['number_of_dimension_mismatches']} dimension mismatches"
        )
    print(f"Per-image CSV: {csv_path}")
    print(f"Summary JSON:  {json_path}")


if __name__ == "__main__":
    main()
