"""Run the pretrained MLX eWaSR model on arbitrary image folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
EWASR_DIR = PROJECT_DIR / "TestingPart1/eWaSR"
DEFAULT_WEIGHTS = EWASR_DIR / "Pretrained/ewasr_resnet18_mlx.npz"

if str(EWASR_DIR) not in sys.path:
    sys.path.insert(0, str(EWASR_DIR))

from wasr.mlx_model import Predictor  # noqa: E402


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
INPUT_ALIGNMENT = 64
NORMALIZE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
NORMALIZE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
CLASS_COLOURS = np.array(
    [
        [247, 195, 37],  # obstacle
        [41, 167, 224],  # water
        [90, 75, 164],  # sky
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate three-class masks with pretrained eWaSR on MLX."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    paths = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = sorted(
        path
        for path in paths
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found in: {input_dir}")
    return images


def predict_image(predictor: Predictor, image: Image.Image) -> np.ndarray:
    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = image_array.shape[:2]
    image_tensor = image_array.astype(np.float32) / 255.0
    image_tensor = (image_tensor - NORMALIZE_MEAN) / NORMALIZE_STD

    pad_bottom = (-height) % INPUT_ALIGNMENT
    pad_right = (-width) % INPUT_ALIGNMENT
    if pad_bottom or pad_right:
        image_tensor = np.pad(
            image_tensor,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="constant",
        )

    prediction = np.asarray(predictor.predict(image_tensor[None, ...]))[0]
    prediction = prediction[:height, :width].astype(np.uint8, copy=False)
    if prediction.shape != (height, width):
        raise ValueError(
            f"Prediction shape {prediction.shape} differs from image {(height, width)}"
        )
    if prediction.size and int(prediction.max()) >= len(CLASS_COLOURS):
        raise ValueError("Prediction contains an unsupported class ID")
    return prediction


def save_outputs(
    image: Image.Image,
    mask: np.ndarray,
    mask_path: Path,
    overlay_path: Path,
    opacity: float = 0.3,
) -> None:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    colours = CLASS_COLOURS[mask].astype(np.float32)
    overlay = np.clip(
        image_array * (1.0 - opacity) + colours * opacity, 0, 255
    ).astype(np.uint8)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(mask_path)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)


def main() -> None:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")

    images = find_images(input_dir, args.recursive)
    if args.max_images is not None:
        images = images[: args.max_images]

    pending: list[tuple[Path, Path, Path]] = []
    for image_path in images:
        relative_path = image_path.relative_to(input_dir).with_suffix(".png")
        mask_path = output_dir / "masks" / relative_path
        overlay_path = output_dir / "overlays" / relative_path
        if args.overwrite or not (mask_path.is_file() and overlay_path.is_file()):
            pending.append((image_path, mask_path, overlay_path))

    print(f"Input: {input_dir}", flush=True)
    print(f"Images selected: {len(images)}", flush=True)
    print(f"Images to process: {len(pending)}", flush=True)
    print(f"Weights: {weights}", flush=True)
    if not pending:
        print(f"All outputs already exist in: {output_dir}", flush=True)
        return

    predictor = Predictor(weights)
    print(f"Device: {predictor.device}", flush=True)
    for index, (image_path, mask_path, overlay_path) in enumerate(pending, start=1):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        mask = predict_image(predictor, image)
        save_outputs(image, mask, mask_path, overlay_path)
        if index == 1 or index % 25 == 0 or index == len(pending):
            print(f"[{index}/{len(pending)}] {image_path}", flush=True)

    print(f"Masks: {output_dir / 'masks'}", flush=True)
    print(f"Overlays: {output_dir / 'overlays'}", flush=True)


if __name__ == "__main__":
    main()
