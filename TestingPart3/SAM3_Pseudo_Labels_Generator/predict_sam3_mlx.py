"""Three-class maritime image segmentation with SAM 3 on MLX.

The output mask follows the eWaSR label convention:

    0 = obstacle, 1 = water, 2 = sky

SAM 3 returns instance masks for text prompts rather than one dense semantic
mask.  This script prompts it once per class (while sharing the MLX vision
backbone), gives each covered pixel to the highest-confidence instance, and
uses obstacle as the fallback for pixels that SAM 3 leaves uncovered.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data/Tampere-WaterSeg"
DEFAULT_OUTPUT = SCRIPT_DIR / "output/Tampere-WaterSeg/open"

CLASS_PROMPTS = ("obstacle", "navigable water", "sky")
CLASS_IDS = {prompt: class_id for class_id, prompt in enumerate(CLASS_PROMPTS)}

# The same three-class palette used by this repository's eWaSR/MaSTr tooling.
CLASS_COLOURS = np.array(
    [
        [247, 195, 37],  # obstacle
        [41, 167, 224],  # water
        [90, 75, 164],  # sky
    ],
    dtype=np.uint8,
)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def create_predictor(model_id: str, confidence_threshold: float) -> Any:
    """Load the MLX SAM 3 model and processor."""

    import mlx.core as mx
    from mlx_vlm.models.sam3.generate import Sam3Predictor
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
    from mlx_vlm.utils import get_model_path, load_model

    if not mx.metal.is_available():
        raise RuntimeError("MLX cannot access an Apple Metal GPU")

    model_path = Path(get_model_path(model_id)).resolve()
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    return Sam3Predictor(
        model,
        processor,
        score_threshold=confidence_threshold,
    )


def semantic_mask_from_instances(
    labels: Iterable[str],
    masks: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Merge scored SAM 3 instances into a dense eWaSR-style class mask.

    Overlapping instances are resolved by confidence. Pixels with no accepted
    instance remain class 0 (obstacle), ensuring that every output pixel has
    exactly one of the three required class IDs.
    """

    height, width = image_shape
    labels = list(labels)
    masks = np.asarray(masks)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    if not labels:
        return np.zeros((height, width), dtype=np.uint8)
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError(
            f"SAM 3 mask shape {masks.shape} does not match image {(height, width)}"
        )
    if not (len(labels) == len(masks) == len(scores)):
        raise ValueError("SAM 3 returned inconsistent labels, masks and scores")

    # Confidence evidence for each semantic class. np.argmax breaks exact ties
    # toward the lowest ID, which is the conservative obstacle class.
    class_evidence = np.zeros((len(CLASS_PROMPTS), height, width), dtype=np.float32)
    for label, instance_mask, score in zip(labels, masks, scores):
        try:
            class_id = CLASS_IDS[label]
        except KeyError as error:
            raise ValueError(f"Unexpected SAM 3 class label: {label!r}") from error
        covered = np.asarray(instance_mask, dtype=bool)
        class_evidence[class_id, covered] = np.maximum(
            class_evidence[class_id, covered], score
        )

    return np.argmax(class_evidence, axis=0).astype(np.uint8)


def predict_image(
    predictor: Any,
    image: Image.Image,
    confidence_threshold: float,
    padding_percent: float,
) -> np.ndarray:
    """Run MLX SAM 3 on a reflected border, then crop to the input size."""

    class_mask, _ = _predict_image(
        predictor, image, confidence_threshold, padding_percent, collect_scores=False
    )
    return class_mask


def predict_image_with_scores(
    predictor: Any,
    image: Image.Image,
    confidence_threshold: float,
    padding_percent: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the class mask and accepted instance scores for inspection."""

    class_mask, report = _predict_image(
        predictor, image, confidence_threshold, padding_percent, collect_scores=True
    )
    assert report is not None
    return class_mask, report


def _predict_image(
    predictor: Any,
    image: Image.Image,
    confidence_threshold: float,
    padding_percent: float,
    collect_scores: bool,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Run inference, optionally collecting per-instance score metadata."""

    from mlx_vlm.models.sam3.generate import predict_multi

    image = image.convert("RGB")
    original_shape = (image.height, image.width)
    padded_image, pad_y, pad_x = add_reflection_padding(image, padding_percent)
    result = predict_multi(
        predictor,
        padded_image,
        prompts=list(CLASS_PROMPTS),
        score_threshold=confidence_threshold,
    )
    padded_mask = semantic_mask_from_instances(
        labels=result.labels or [],
        masks=result.masks,
        scores=result.scores,
        image_shape=(padded_image.height, padded_image.width),
    )
    class_mask = crop_padding(
        padded_mask,
        original_shape=original_shape,
        pad_y=pad_y,
        pad_x=pad_x,
    )
    if not collect_scores:
        return class_mask, None

    class_scores: dict[str, list[float]] = {prompt: [] for prompt in CLASS_PROMPTS}
    if result.labels:
        cropped_masks = np.asarray(result.masks).astype(bool)[
            :, pad_y : pad_y + original_shape[0], pad_x : pad_x + original_shape[1]
        ]
        for label, instance_mask, score in zip(
            result.labels, cropped_masks, np.asarray(result.scores).reshape(-1)
        ):
            if instance_mask.any():
                class_scores[label].append(float(score))
        uncovered_pixels = int((~np.any(cropped_masks, axis=0)).sum())
    else:
        uncovered_pixels = int(class_mask.size)

    report = {
        "score_meaning": "One score per accepted SAM 3 instance visible in the saved image, not per pixel",
        "confidence_threshold": confidence_threshold,
        "instance_scores_by_class": class_scores,
        "uncovered_pixels_assigned_obstacle": uncovered_pixels,
    }
    return class_mask, report


def add_reflection_padding(
    image: Image.Image,
    padding_percent: float,
) -> tuple[Image.Image, int, int]:
    """Mirror a percentage of each dimension around an RGB image."""

    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = image_array.shape[:2]
    if padding_percent == 0:
        return Image.fromarray(image_array, mode="RGB"), 0, 0
    if height < 2 or width < 2:
        raise ValueError("Reflection padding requires image dimensions of at least 2x2")

    pad_y = max(1, round(height * padding_percent / 100.0))
    pad_x = max(1, round(width * padding_percent / 100.0))
    padded = np.pad(
        image_array,
        ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
        mode="reflect",
    )
    return Image.fromarray(padded, mode="RGB"), pad_y, pad_x


def crop_padding(
    padded_mask: np.ndarray,
    original_shape: tuple[int, int],
    pad_y: int,
    pad_x: int,
) -> np.ndarray:
    """Remove the reflected border and verify the original mask dimensions."""

    height, width = original_shape
    expected_shape = (height + 2 * pad_y, width + 2 * pad_x)
    if padded_mask.shape != expected_shape:
        raise ValueError(
            f"Padded mask shape {padded_mask.shape} does not match {expected_shape}"
        )
    return padded_mask[pad_y : pad_y + height, pad_x : pad_x + width].copy()


def create_overlay(
    image: np.ndarray,
    class_mask: np.ndarray,
    mask_opacity: float = 0.3,
) -> np.ndarray:
    """Blend the eWaSR three-class colours over an RGB image."""

    if image.shape[:2] != class_mask.shape:
        raise ValueError("Image and class mask dimensions differ")
    if class_mask.size and int(class_mask.max()) >= len(CLASS_COLOURS):
        raise ValueError("Class mask contains an unsupported class ID")
    colours = CLASS_COLOURS[class_mask]
    overlay = image.astype(np.float32) * (1.0 - mask_opacity)
    overlay += colours.astype(np.float32) * mask_opacity
    return np.clip(overlay, 0, 255).astype(np.uint8)


def resolve_image_root(input_path: Path, subset: str) -> Path:
    """Accept either an image folder or a Tampere-WaterSeg dataset root."""

    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if input_path.is_file():
        return input_path

    tampere_subset = input_path / "images" / subset
    return tampere_subset if tampere_subset.is_dir() else input_path


def find_images(input_path: Path, recursive: bool) -> list[Path]:
    """Find input images in stable filename order."""

    if input_path.is_file():
        if input_path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]

    paths = input_path.rglob("*") if recursive else input_path.iterdir()
    images = sorted(
        path
        for path in paths
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found in: {input_path}")
    return images


def images_from_manifest(manifest_path: Path) -> tuple[Path, list[Path]]:
    """Select exactly the source images listed in a label manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    images: list[Path] = []
    seen: set[Path] = set()
    with manifest_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            raise ValueError(f"Manifest has no image column: {manifest_path}")
        for row in reader:
            image_value = row["image"]
            if not image_value:
                raise ValueError(f"Empty image path in {manifest_path}")
            image = Path(image_value)
            if not image.is_absolute():
                image = PROJECT_DIR / image
            image = image.resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            if image.suffix.casefold() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported image extension: {image}")
            if image in seen:
                raise ValueError(f"Duplicate image in manifest: {image}")
            seen.add(image)
            images.append(image)

    if not images:
        raise ValueError(f"No images in manifest: {manifest_path}")
    image_root = Path(os.path.commonpath([str(image.parent) for image in images]))
    return image_root, images


def relative_output_path(image_path: Path, image_root: Path) -> Path:
    """Return a collision-resistant PNG path below an output directory."""

    relative = Path(image_path.name) if image_root.is_file() else image_path.relative_to(image_root)
    return relative.with_suffix(".png")


def save_outputs(
    image: Image.Image,
    class_mask: np.ndarray,
    mask_path: Path,
    overlay_path: Path,
) -> None:
    """Save the raw class-ID mask and its RGB visualization overlay."""

    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    overlay = create_overlay(image_array, class_mask)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(class_mask, mode="L").save(mask_path)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)


def save_score_report(report: dict[str, Any], path: Path) -> None:
    """Save the accepted instance scores alongside one output mask."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def export_scores_csv(output_dir: Path) -> Path:
    """Rebuild one dataset CSV from saved per-image score reports."""

    score_dir = output_dir / "scores"
    score_paths = sorted(score_dir.rglob("*.json"))
    if not score_paths:
        raise FileNotFoundError(f"No score reports found in {score_dir}")

    csv_path = output_dir / "instance_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "image",
                "class_id",
                "class_prompt",
                "instance_id",
                "score",
                "instance_count",
                "uncovered_pixels_assigned_obstacle",
                "confidence_threshold",
            )
        )
        for score_path in score_paths:
            report = json.loads(score_path.read_text(encoding="utf-8"))
            image = score_path.relative_to(score_dir).with_suffix("").as_posix()
            class_scores = report["instance_scores_by_class"]
            for class_prompt in CLASS_PROMPTS:
                scores = class_scores[class_prompt]
                for instance_id, score in enumerate(scores or [None], start=1):
                    writer.writerow(
                        (
                            image,
                            CLASS_IDS[class_prompt],
                            class_prompt,
                            instance_id if score is not None else "",
                            score if score is not None else "",
                            len(scores),
                            report["uncovered_pixels_assigned_obstacle"],
                            report["confidence_threshold"],
                        )
                    )
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment images into obstacle, water and sky with SAM 3 on MLX."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Image, image directory, or Tampere-WaterSeg root "
            f"(default: {DEFAULT_INPUT})"
        ),
    )
    parser.add_argument(
        "--subset",
        default="open",
        choices=("open", "channel", "dock"),
        help="Tampere subset when --input points at its dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--padding-percent",
        type=float,
        default=5.0,
        help=(
            "Reflection padding added to each image side as a percentage of "
            "that dimension; use 0 to disable (default: 5)"
        ),
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Process only images in a manifest's image column",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Process only the first N images (useful for an initial test)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute images whose mask and overlay already exist",
    )
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Save per-image score JSON and rebuild instance_scores.csv",
    )
    parser.add_argument(
        "--scores-only",
        action="store_true",
        help="Save score JSON and CSV without writing new masks or overlays",
    )
    parser.add_argument(
        "--scores-csv-only",
        action="store_true",
        help="Build instance_scores.csv from existing score JSON without loading SAM 3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if args.scores_csv_only:
        print(f"Dataset scores: {export_scores_csv(output_dir)}")
        return

    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")
    if not 0.0 <= args.padding_percent <= 50.0:
        raise ValueError("--padding-percent must be between 0 and 50")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    collect_scores = args.save_scores or args.scores_only

    if args.selection_manifest:
        image_root, images = images_from_manifest(args.selection_manifest)
    else:
        image_root = resolve_image_root(args.input, args.subset)
        images = find_images(image_root, args.recursive)
    if args.max_images is not None:
        images = images[: args.max_images]

    pending: list[tuple[Path, Path, Path, Path]] = []
    for image_path in images:
        relative_path = relative_output_path(image_path, image_root)
        mask_path = output_dir / "masks" / relative_path
        overlay_path = output_dir / "overlays" / relative_path
        score_path = output_dir / "scores" / relative_path.with_suffix(".json")
        if args.scores_only:
            complete = score_path.is_file()
        else:
            complete = mask_path.is_file() and overlay_path.is_file()
            if collect_scores:
                complete = complete and score_path.is_file()
        if args.overwrite or not complete:
            pending.append((image_path, mask_path, overlay_path, score_path))

    print(f"Input: {image_root}")
    if args.selection_manifest:
        print(f"Selection manifest: {args.selection_manifest.expanduser().resolve()}")
    print(f"Images selected: {len(images)}")
    print(f"Images to process: {len(pending)}")
    print(f"Reflection padding per side: {args.padding_percent:g}%")
    if not pending:
        print(f"All selected outputs already exist in: {output_dir}")
        if collect_scores:
            print(f"Dataset scores: {export_scores_csv(output_dir)}")
        return

    print(f"Loading {args.model_id} with MLX...")
    predictor = create_predictor(args.model_id, args.confidence_threshold)

    for index, (image_path, mask_path, overlay_path, score_path) in enumerate(
        pending, start=1
    ):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if collect_scores:
            class_mask, report = predict_image_with_scores(
                predictor,
                image,
                confidence_threshold=args.confidence_threshold,
                padding_percent=args.padding_percent,
            )
        else:
            class_mask = predict_image(
                predictor,
                image,
                confidence_threshold=args.confidence_threshold,
                padding_percent=args.padding_percent,
            )
        if not args.scores_only:
            save_outputs(image, class_mask, mask_path, overlay_path)
        if collect_scores:
            save_score_report(report, score_path)
            print(f"  scores: {report['instance_scores_by_class']}")
        print(f"[{index}/{len(pending)}] {image_path.name}")

    if not args.scores_only:
        print(f"Masks: {output_dir / 'masks'}")
        print(f"Overlays: {output_dir / 'overlays'}")
    if collect_scores:
        print(f"Scores: {output_dir / 'scores'}")
        print(f"Dataset scores: {export_scores_csv(output_dir)}")


if __name__ == "__main__":
    main()
