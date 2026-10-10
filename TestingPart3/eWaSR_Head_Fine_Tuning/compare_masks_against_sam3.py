"""Color pixel disagreements between SAM3 masks and one or two predictions.

Examples:
    python compare_masks_against_sam3.py \
        --sam3-masks /path/to/sam3/masks \
        --prediction /path/to/run/test_visualizations/predicted_masks \
        --output-dir /path/to/comparison

    python compare_masks_against_sam3.py \
        --sam3-masks /path/to/sam3/masks \
        --prediction /path/to/run_a \
        --prediction /path/to/run_b \
        --images /path/to/original/images \
        --output-dir /path/to/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sam3-masks", type=Path, required=True,
        help="Directory containing SAM3 class-ID PNG masks",
    )
    parser.add_argument(
        "--prediction", type=Path, action="append", required=True,
        help="Prediction mask directory or run directory; supply once or twice",
    )
    parser.add_argument(
        "--images", type=Path,
        help="Original image directory; inferred from visualization manifests if omitted",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Folder in which to write failure_masks, failure_overlays, and summaries",
    )
    args = parser.parse_args()
    if len(args.prediction) not in (1, 2):
        parser.error("supply --prediction once or twice")
    return args


def resolve_prediction_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = (
        path,
        path / "predicted_masks",
        path / "test_visualizations" / "predicted_masks",
        path / "test_visualization" / "predicted_masks",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.png")):
            return candidate
    raise FileNotFoundError(f"No predicted_masks PNGs found under {path}")


def pngs_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = {path.stem: path for path in directory.glob("*.png") if path.is_file()}
    if not files:
        raise ValueError(f"No PNG masks found in {directory}")
    return files


def images_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Multiple original images have stem {path.stem!r}")
            result[path.stem] = path
    return result


def images_from_manifests(prediction_dirs: list[Path]) -> dict[str, Path]:
    """Find original images recorded when predicted masks were visualized."""
    mappings: list[dict[str, Path]] = []
    for mask_dir in prediction_dirs:
        manifest = mask_dir.parent / "visualization_manifest.json"
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [Path(value).expanduser().resolve() for value in data.get("test_images", [])]
        mapping = {path.stem: path for path in paths}
        if len(mapping) != len(paths):
            raise ValueError(f"Duplicate image stems in {manifest}")
        if mapping:
            mappings.append(mapping)
    if not mappings:
        raise ValueError(
            "Cannot locate original images from visualization manifests. "
            "Supply --images to produce overlays."
        )
    originals = mappings[0]
    for mapping in mappings[1:]:
        for name in originals.keys() & mapping.keys():
            if originals[name] != mapping[name]:
                raise ValueError(f"Original image paths disagree for {name}; supply --images")
    return originals


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        if source.mode not in {"L", "P", "I", "I;16"}:
            raise ValueError(f"Expected single-channel class-ID mask: {path} ({source.mode})")
        mask = np.asarray(source)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Expected 2D integer class IDs: {path}")
    return mask


def check_existing_outputs(directory: Path, names: set[str]) -> None:
    """Avoid leaving old comparison images in a reused output folder."""
    for subdir in ("failure_masks", "failure_overlays"):
        folder = directory / subdir
        if folder.is_dir():
            stale = {p.stem for p in folder.glob("*.png")} - names
            if stale:
                raise ValueError(
                    f"{folder} contains {len(stale)} images from another comparison. "
                    "Choose an empty output folder."
                )


def main() -> None:
    args = parse_args()
    reference_dir = args.sam3_masks.expanduser().resolve()
    prediction_dirs = [resolve_prediction_dir(path) for path in args.prediction]
    output_dir = args.output_dir.expanduser().resolve()
    references = pngs_by_stem(reference_dir)
    predictions = [pngs_by_stem(path) for path in prediction_dirs]
    names = set(predictions[0])
    if len(predictions) == 2 and names != set(predictions[1]):
        only_first = len(names - set(predictions[1]))
        only_second = len(set(predictions[1]) - names)
        raise ValueError(
            "Prediction folders must contain the same image names "
            f"({only_first} only in first, {only_second} only in second)."
        )
    missing_references = names - set(references)
    if missing_references:
        example = sorted(missing_references)[0]
        raise ValueError(
            f"SAM3 directory lacks {len(missing_references)} prediction masks; "
            f"for example {example}.png"
        )
    originals = (
        images_by_stem(args.images.expanduser().resolve())
        if args.images is not None else images_from_manifests(prediction_dirs)
    )
    missing_images = names - set(originals)
    if missing_images:
        example = sorted(missing_images)[0]
        raise ValueError(
            f"Original image source lacks {len(missing_images)} images; "
            f"for example {example}"
        )
    missing_files = [name for name in names if not originals[name].is_file()]
    if missing_files:
        example = sorted(missing_files)[0]
        raise FileNotFoundError(f"Original image not found: {originals[example]}")
    if output_dir in [reference_dir, *prediction_dirs]:
        raise ValueError("Output folder must differ from every input mask folder")
    check_existing_outputs(output_dir, names)
    masks_dir = output_dir / "failure_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "failure_overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    two_predictions = len(predictions) == 2
    fields = ["image", "total_pixels"]
    if two_predictions:
        fields += [
            "both_agree_pixels", "prediction_1_only_failed_blue",
            "prediction_2_only_failed_purple", "both_failed_red",
            "prediction_1_failed_percent", "prediction_2_failed_percent",
        ]
        palette = np.array([BLACK, BLUE, PURPLE, RED], dtype=np.uint8)
    else:
        fields += ["agree_pixels", "prediction_1_failed_red", "prediction_1_failed_percent"]
        palette = np.array([BLACK, RED], dtype=np.uint8)

    totals = {field: 0 for field in fields if field != "image" and not field.endswith("percent")}
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, name in enumerate(sorted(names), 1):
            reference = read_mask(references[name])
            first = read_mask(predictions[0][name])
            if first.shape != reference.shape:
                raise ValueError(f"Mask dimensions differ for {name}: first prediction")
            category = (first != reference).astype(np.uint8)
            if two_predictions:
                second = read_mask(predictions[1][name])
                if second.shape != reference.shape:
                    raise ValueError(f"Mask dimensions differ for {name}: second prediction")
                category += 2 * (second != reference).astype(np.uint8)
            counts = np.bincount(category.ravel(), minlength=len(palette))
            rgb = palette[category]
            Image.fromarray(rgb, mode="RGB").save(masks_dir / f"{name}.png")
            with Image.open(originals[name]) as source:
                image = np.asarray(source.convert("RGB"))
            if image.shape[:2] != reference.shape:
                raise ValueError(f"Original image dimensions differ for {name}")
            overlay = image.copy()
            failed = category != 0
            overlay[failed] = np.rint(
                0.25 * image[failed] + 0.75 * rgb[failed]
            ).astype(np.uint8)
            Image.fromarray(overlay, mode="RGB").save(overlays_dir / f"{name}.png")
            row: dict[str, object] = {"image": name, "total_pixels": int(category.size)}
            if two_predictions:
                row.update({
                    "both_agree_pixels": int(counts[0]),
                    "prediction_1_only_failed_blue": int(counts[1]),
                    "prediction_2_only_failed_purple": int(counts[2]),
                    "both_failed_red": int(counts[3]),
                    "prediction_1_failed_percent": round(100 * (counts[1] + counts[3]) / category.size, 3),
                    "prediction_2_failed_percent": round(100 * (counts[2] + counts[3]) / category.size, 3),
                })
            else:
                row.update({
                    "agree_pixels": int(counts[0]),
                    "prediction_1_failed_red": int(counts[1]),
                    "prediction_1_failed_percent": round(100 * counts[1] / category.size, 3),
                })
            writer.writerow(row)
            for key in totals:
                totals[key] += int(row[key])
            if index % 10 == 0 or index == len(names):
                print(f"Compared {index}/{len(names)} images", flush=True)

    manifest = {
        "sam3_masks": str(reference_dir),
        "prediction_masks": [str(path) for path in prediction_dirs],
        "original_images": (
            str(args.images.expanduser().resolve())
            if args.images else "visualization_manifest.json"
        ),
        "image_count": len(names),
        "compared_images": sorted(names),
        "comparison": "Exact class-ID equality at each pixel against the SAM3 pseudo-label",
        "colors_rgb": (
            {"agree": BLACK, "prediction_1_failed": RED}
            if not two_predictions else
            {"both_agree": BLACK, "prediction_1_only_failed": BLUE,
             "prediction_2_only_failed": PURPLE, "both_failed": RED}
        ),
        "totals": totals,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
