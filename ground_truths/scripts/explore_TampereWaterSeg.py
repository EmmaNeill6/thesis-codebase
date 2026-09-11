"""Create visual overlays for the selected Tampere-WaterSeg ground truths.

The three ``ImageTest_Tam_*.txt`` files select images from the channel,
open-water, and dock subsets. In the binary masks, white (255) is water and
black (0) is non-water.
"""

from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_FOLDER = Path(__file__).resolve().parents[2]
DATASET_FOLDER = PROJECT_FOLDER / "data" / "Tampere-WaterSeg"
OUTPUT_FOLDER = (
    PROJECT_FOLDER / "ground_truths" / "output" / "TampereWaterSeg_Out"
)

# Subset name, image-list filename, image directory, and mask directory.
SUBSETS = (
    ("channel", "ImageTest_Tam_Channel.txt", "channel", "channel_mask"),
    ("open", "ImageTest_Tam_Open.txt", "open", "open_mask"),
    ("dock", "ImageTest_Tam_Dock.txt", "dock", "dock_mask"),
)

# Match the class colours and blend strength used by eWaSR's visualisations.
# Tampere-WaterSeg is binary, so sky and obstacles are both non-water.
NON_WATER_COLOUR = np.array([247, 195, 37], dtype=np.float32)
WATER_COLOUR = np.array([41, 167, 224], dtype=np.float32)
IMAGE_STRENGTH = 0.7
MASK_STRENGTH = 0.3


def read_image_ids(list_path: Path) -> list[str]:
    """Read unique image IDs, preserving their order in the list file."""
    if not list_path.is_file():
        raise FileNotFoundError(f"Missing image list: {list_path}")

    image_ids: list[str] = []
    seen: set[str] = set()
    for line in list_path.read_text(encoding="utf-8").splitlines():
        image_id = line.strip()
        if not image_id or image_id.startswith("#") or image_id in seen:
            continue
        image_ids.append(image_id)
        seen.add(image_id)

    if not image_ids:
        raise ValueError(f"The image list is empty: {list_path}")

    return image_ids


def create_overlay(
    image_id: str,
    subset: str,
    images_folder: Path,
    masks_folder: Path,
) -> Path:
    """Blend one binary ground-truth mask over its RGB image."""
    expected_prefix = f"{subset}_image_"
    if not image_id.startswith(expected_prefix):
        raise ValueError(
            f"Unexpected image ID {image_id!r} for the {subset!r} subset; "
            f"expected a name starting with {expected_prefix!r}"
        )

    image_number = image_id.removeprefix(expected_prefix)
    image_path = images_folder / f"{image_id}.jpg"
    mask_path = masks_folder / f"{subset}_mask_{image_number}.png"

    missing_files = [
        str(path) for path in (image_path, mask_path) if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing_files))

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)

    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"Image and mask dimensions do not match for {image_id}: "
            f"{image.shape[:2]} and {mask.shape}"
        )

    # A few masks contain anti-aliased boundary pixels between black and white.
    # Thresholding keeps their intended binary meaning: white-side values are
    # water and black-side values are non-water.
    water_pixels = mask >= 128
    colours = np.empty_like(image)
    colours[~water_pixels] = NON_WATER_COLOUR
    colours[water_pixels] = WATER_COLOUR
    overlay = image * IMAGE_STRENGTH + colours * MASK_STRENGTH

    output_path = OUTPUT_FOLDER / f"{image_id}_overlay.png"
    Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(output_path)
    return output_path


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    print(f"Reading Tampere-WaterSeg from: {DATASET_FOLDER}")
    for subset, list_filename, image_dirname, mask_dirname in SUBSETS:
        list_path = DATASET_FOLDER / list_filename
        image_ids = read_image_ids(list_path)
        images_folder = DATASET_FOLDER / image_dirname
        masks_folder = DATASET_FOLDER / mask_dirname

        print(f"{subset}: {len(image_ids)} unique image(s) from {list_filename}")
        for image_id in image_ids:
            output_path = create_overlay(
                image_id, subset, images_folder, masks_folder
            )
            output_paths.append(output_path)
            print(f"  {output_path}")

    print(f"Saved {len(output_paths)} overlay(s) to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
