"""Render held-out test predictions from the selected fine-tuned eWaSR model.

The script reads the test split from the experiment YAML, the original
pretrained checkpoint from the YAML, and the chosen fine-tuned checkpoint from
``selected_model.json``. It writes class-index masks, full-resolution overlays,
side-by-side comparisons with the SAM3 pseudo-labels, and a small contact sheet.
It intentionally does not benchmark inference time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ewasr_mlx.model import create_head_finetuning_model
from run_experiment import DEFAULT_CONFIG, load_config, load_split, validate_configuration
from train_head import NORMALIZE_MEAN, NORMALIZE_STD, Sample


CLASS_COLOURS = np.array(
    [
        [247, 195, 37],  # obstacle
        [41, 167, 224],  # water
        [90, 75, 164],  # sky
    ],
    dtype=np.uint8,
)
DEFAULT_OPACITY = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render overlays for the YAML-defined held-out test split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Experiment YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=None,
        help="Selected-model JSON (default: <experiment output>/selected_model.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Visualization directory (default: <experiment output>/test_visualizations)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=DEFAULT_OPACITY,
        help="Mask overlay opacity between 0 and 1 (default: %(default)s)",
    )
    return parser.parse_args()


def load_selected_checkpoint(selection_path: Path) -> tuple[Path, dict[str, object]]:
    selection_path = selection_path.expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(f"Selected-model record not found: {selection_path}")
    with selection_path.open("r", encoding="utf-8") as source:
        selection = json.load(source)
    checkpoint_value = selection.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise ValueError(f"No checkpoint path in: {selection_path}")
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint}")
    return checkpoint, selection


def prepare_image(image: Image.Image, width: int, height: int) -> np.ndarray:
    resized = image.convert("RGB").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return (array - NORMALIZE_MEAN) / NORMALIZE_STD


def colour_overlay(
    image: Image.Image, mask: np.ndarray, opacity: float
) -> Image.Image:
    if mask.ndim != 2:
        raise ValueError(f"Expected a two-dimensional mask, got shape {mask.shape}")
    values = np.unique(mask)
    if not set(values.tolist()).issubset({0, 1, 2}):
        raise ValueError(f"Mask contains unsupported class IDs: {values.tolist()}")

    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_image = Image.fromarray(mask.astype(np.uint8), mode="L").resize(
        image.size, Image.Resampling.NEAREST
    )
    full_mask = np.asarray(mask_image, dtype=np.uint8)
    colours = CLASS_COLOURS[full_mask].astype(np.float32)
    blended = image_array * (1.0 - opacity) + colours * opacity
    return Image.fromarray(np.rint(blended).clip(0, 255).astype(np.uint8), mode="RGB")


def labelled_panel(image: Image.Image, label: str, width: int = 384) -> Image.Image:
    height = round(image.height * width / image.width)
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    label_height = 34
    panel = Image.new("RGB", (width, height + label_height), "white")
    panel.paste(resized, (0, label_height))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=16)
    draw.text((10, 8), label, fill="black", font=font)
    return panel


def comparison_image(
    original: Image.Image,
    pretrained_overlay: Image.Image,
    fine_tuned_overlay: Image.Image,
    pseudo_label_overlay: Image.Image,
) -> Image.Image:
    panels = [
        labelled_panel(original, "Original test image"),
        labelled_panel(pretrained_overlay, "Pretrained eWaSR prediction"),
        labelled_panel(fine_tuned_overlay, "Fine-tuned eWaSR prediction"),
        labelled_panel(pseudo_label_overlay, "SAM3 pseudo-label reference"),
    ]
    comparison = Image.new(
        "RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels))
    )
    x = 0
    for panel in panels:
        comparison.paste(panel, (x, 0))
        x += panel.width
    return comparison


def create_contact_sheet(comparisons: list[Path], output_path: Path) -> None:
    if not comparisons:
        return
    selected_indices = np.linspace(
        0, len(comparisons) - 1, min(10, len(comparisons)), dtype=int
    )
    thumbnails = []
    for index in selected_indices:
        with Image.open(comparisons[index]) as source:
            thumbnail = source.convert("RGB")
            target_width = 768
            target_height = round(thumbnail.height * target_width / thumbnail.width)
            thumbnails.append(
                thumbnail.resize((target_width, target_height), Image.Resampling.LANCZOS)
            )

    columns = 2
    rows = (len(thumbnails) + columns - 1) // columns
    cell_width = max(image.width for image in thumbnails)
    cell_height = max(image.height for image in thumbnails)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for index, thumbnail in enumerate(thumbnails):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(thumbnail, (x, y))
    sheet.save(output_path, quality=92)


def write_manifest(
    output_path: Path,
    *,
    config_path: Path,
    selection_path: Path,
    pretrained_checkpoint: Path,
    fine_tuned_checkpoint: Path,
    selection: dict[str, object],
    samples: list[Sample],
    opacity: float,
) -> None:
    manifest = {
        "config": str(config_path),
        "selection_record": str(selection_path),
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "fine_tuned_checkpoint": str(fine_tuned_checkpoint),
        "learning_rate": selection.get("learning_rate"),
        "best_epoch": selection.get("best_epoch"),
        "test_sample_count": len(samples),
        "test_images": [str(sample.image_path) for sample in samples],
        "class_ids": {"obstacle": 0, "water": 1, "sky": 2},
        "class_colours_rgb": {
            "obstacle": CLASS_COLOURS[0].tolist(),
            "water": CLASS_COLOURS[1].tolist(),
            "sky": CLASS_COLOURS[2].tolist(),
        },
        "overlay_opacity": opacity,
        "interpretation": (
            "The fourth panel is a SAM3 pseudo-label reference, not independent "
            "human ground truth. Both sets of raw masks contain class IDs 0, 1, 2."
        ),
    }
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.opacity <= 1.0:
        raise ValueError("--opacity must be between 0 and 1")

    config = load_config(args.config)
    validate_configuration(config)
    samples = load_split(config, "test")
    experiment_output = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    selection_path = (
        args.selection.expanduser().resolve()
        if args.selection is not None
        else experiment_output / "selected_model.json"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else experiment_output / "test_visualizations"
    )
    fine_tuned_checkpoint, selection = load_selected_checkpoint(selection_path)
    pretrained_checkpoint = Path(config["model"]["checkpoint"]).expanduser().resolve()
    if not pretrained_checkpoint.is_file():
        raise FileNotFoundError(
            f"Pretrained eWaSR checkpoint not found: {pretrained_checkpoint}"
        )

    predicted_masks_dir = output_dir / "predicted_masks"
    overlays_dir = output_dir / "predicted_overlays"
    pretrained_masks_dir = output_dir / "pretrained_masks"
    pretrained_overlays_dir = output_dir / "pretrained_overlays"
    comparisons_dir = output_dir / "comparisons"
    for directory in (
        predicted_masks_dir,
        overlays_dir,
        pretrained_masks_dir,
        pretrained_overlays_dir,
        comparisons_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    fine_tuned_model = create_head_finetuning_model(fine_tuned_checkpoint)
    fine_tuned_model.eval()
    pretrained_model = create_head_finetuning_model(pretrained_checkpoint)
    pretrained_model.eval()
    width = int(config["preprocessing"]["input_width"])
    height = int(config["preprocessing"]["input_height"])
    comparison_paths = []

    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as source:
            original = source.convert("RGB")
        input_array = prepare_image(original, width, height)[None, ...]
        model_input = mx.array(input_array)
        fine_tuned_prediction = mx.argmax(fine_tuned_model(model_input), axis=-1)
        pretrained_prediction = mx.argmax(pretrained_model(model_input), axis=-1)
        mx.eval(fine_tuned_prediction, pretrained_prediction)
        prediction_array = np.asarray(fine_tuned_prediction, dtype=np.uint8)[0]
        pretrained_array = np.asarray(pretrained_prediction, dtype=np.uint8)[0]

        raw_mask = Image.fromarray(prediction_array, mode="L").resize(
            original.size, Image.Resampling.NEAREST
        )
        raw_mask.save(predicted_masks_dir / f"{sample.image_path.stem}.png")

        prediction_overlay = colour_overlay(original, prediction_array, args.opacity)
        prediction_overlay.save(
            overlays_dir / f"{sample.image_path.stem}_overlay.jpg", quality=95
        )

        pretrained_mask = Image.fromarray(pretrained_array, mode="L").resize(
            original.size, Image.Resampling.NEAREST
        )
        pretrained_mask.save(pretrained_masks_dir / f"{sample.image_path.stem}.png")
        pretrained_overlay = colour_overlay(original, pretrained_array, args.opacity)
        pretrained_overlay.save(
            pretrained_overlays_dir / f"{sample.image_path.stem}_overlay.jpg",
            quality=95,
        )

        with Image.open(sample.mask_path) as source:
            pseudo_mask = np.asarray(source.convert("L"), dtype=np.uint8)
        pseudo_label_overlay = colour_overlay(original, pseudo_mask, args.opacity)
        comparison = comparison_image(
            original,
            pretrained_overlay,
            prediction_overlay,
            pseudo_label_overlay,
        )
        comparison_path = comparisons_dir / f"{sample.image_path.stem}_comparison.jpg"
        comparison.save(comparison_path, quality=92)
        comparison_paths.append(comparison_path)
        print(f"[{index:02d}/{len(samples):02d}] {sample.image_path.name}")

    create_contact_sheet(comparison_paths, output_dir / "contact_sheet.jpg")
    write_manifest(
        output_dir / "visualization_manifest.json",
        config_path=Path(config["_config_path"]),
        selection_path=selection_path,
        pretrained_checkpoint=pretrained_checkpoint,
        fine_tuned_checkpoint=fine_tuned_checkpoint,
        selection=selection,
        samples=samples,
        opacity=args.opacity,
    )
    print(f"Saved {len(samples)} test visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
