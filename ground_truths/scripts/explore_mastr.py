from pathlib import Path

import numpy as np
from PIL import Image


# Shared project paths. The dataset and image list are used by this script,
# eWaSR inference, and the FiftyOne evaluation.
project_folder = Path(__file__).resolve().parents[2]
ground_truths_folder = project_folder / "ground_truths"
dataset_folder = project_folder / "data" / "MaSTR1325"
images_folder = dataset_folder / "images"
masks_folder = dataset_folder / "masks"
output_folder = ground_truths_folder / "output" / "MaSTR_Out"
image_list_path = project_folder / "evaluation_images.txt"

# These are the same class colours and blend strength used by eWaSR/predict.py.
# 0 = obstacle/environment, 1 = water, 2 = sky
segmentation_colours = np.array(
    [
        [247, 195, 37],
        [41, 167, 224],
        [90, 75, 164],
    ],
    dtype=np.uint8,
)
image_strength = 0.7
mask_strength = 0.3


def read_image_ids():
    if not image_list_path.is_file():
        raise FileNotFoundError(f"Missing image list: {image_list_path}")

    image_ids = [
        line.strip()
        for line in image_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    if not image_ids:
        raise ValueError(f"The image list is empty: {image_list_path}")

    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"The image list contains duplicate IDs: {image_list_path}")

    return image_ids


def create_ground_truth_overlay(image_id):
    image_path = images_folder / f"{image_id}.jpg"
    mask_path = masks_folder / f"{image_id}m.png"

    missing_files = [
        str(path)
        for path in (image_path, mask_path)
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing_files))

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path))

    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"Image and mask dimensions do not match for {image_id}: "
            f"{image.shape[:2]} and {mask.shape}"
        )

    # MaSTR1325 class 4 is an ignored/uncertain region, not a model class.
    # Leave those pixels unchanged because they are excluded during evaluation.
    valid_pixels = mask < len(segmentation_colours)
    overlay = image.copy()
    overlay[valid_pixels] = (
        image[valid_pixels] * image_strength
        + segmentation_colours[mask[valid_pixels]] * mask_strength
    )

    output_path = output_folder / mask_path.name
    Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(output_path)
    return output_path


def main():
    output_folder.mkdir(parents=True, exist_ok=True)
    requested_image_ids = read_image_ids()

    print(f"Reading image IDs from: {image_list_path}")
    print("Saving MaSTR1325 ground-truth overlays:")
    for image_id in requested_image_ids:
        output_path = create_ground_truth_overlay(image_id)
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
