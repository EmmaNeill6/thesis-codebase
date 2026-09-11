"""Create mask overlays for the WaterNet images selected in its test list.

``ImageTest_WaterNet.txt`` is grouped by dataset. A line ending in ``:`` is
the subfolder name and the following lines are image stems in that subfolder.
The WaterNet masks are binary: white is water and black is non-water.
"""

from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_FOLDER = Path(__file__).resolve().parents[2]
DATASET_FOLDER = PROJECT_FOLDER / "data" / "water_v2"
IMAGES_FOLDER = DATASET_FOLDER / "JPEGImages"
MASKS_FOLDER = DATASET_FOLDER / "Annotations"
IMAGE_LIST_PATH = DATASET_FOLDER / "ImageTest_WaterNet.txt"
OUTPUT_FOLDER = PROJECT_FOLDER / "ground_truths" / "output" / "WaterNet_Out"

# ADE20K is deliberately not part of this visualisation set. Keeping the
# exclusion here also prevents it from being processed if it is added to the
# text file later.
EXCLUDED_SUBSETS = {"ade20k"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Match the class colours and blend strength used by the other ground-truth
# exploration scripts in this project.
NON_WATER_COLOUR = np.array([247, 195, 37], dtype=np.float32)
WATER_COLOUR = np.array([41, 167, 224], dtype=np.float32)
IMAGE_STRENGTH = 0.7
MASK_STRENGTH = 0.3


def read_image_requests(list_path: Path) -> list[tuple[str, str]]:
    """Return ``(subfolder, image_stem)`` pairs from a grouped list file."""
    if not list_path.is_file():
        raise FileNotFoundError(f"Missing image list: {list_path}")

    requests: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    current_subset: str | None = None

    for line_number, raw_line in enumerate(
        list_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.endswith(":"):
            current_subset = line[:-1].strip()
            if not current_subset:
                raise ValueError(f"Empty folder heading on line {line_number}")
            if Path(current_subset).name != current_subset:
                raise ValueError(
                    f"Invalid folder heading {current_subset!r} on line {line_number}"
                )
            continue

        if current_subset is None:
            raise ValueError(
                f"Image name {line!r} on line {line_number} has no folder heading"
            )
        if Path(line).name != line:
            raise ValueError(f"Invalid image name {line!r} on line {line_number}")
        if current_subset.casefold() in EXCLUDED_SUBSETS:
            continue

        request = (current_subset, line)
        if request not in seen:
            requests.append(request)
            seen.add(request)

    if not requests:
        raise ValueError(f"The image list contains no usable entries: {list_path}")

    return requests


def find_file(folder: Path, image_stem: str, file_kind: str) -> Path:
    """Find one image or mask by stem, allowing the dataset's mixed formats."""
    matches = sorted(
        path
        for path in folder.glob(f"{image_stem}.*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    )
    if not matches:
        raise FileNotFoundError(
            f"Missing {file_kind} for {image_stem!r} in: {folder}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {file_kind} files found for {image_stem!r}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def create_overlay(subset: str, image_stem: str) -> Path:
    """Blend one binary WaterNet mask over its corresponding RGB image."""
    image_path = find_file(IMAGES_FOLDER / subset, image_stem, "image")
    mask_path = find_file(MASKS_FOLDER / subset, image_stem, "mask")

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)

    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"Image and mask dimensions do not match for {subset}/{image_stem}: "
            f"{image.shape[:2]} and {mask.shape}"
        )

    # Thresholding also handles any anti-aliased pixels around mask boundaries.
    water_pixels = mask >= 128
    colours = np.empty_like(image)
    colours[water_pixels] = WATER_COLOUR
    colours[~water_pixels] = NON_WATER_COLOUR
    overlay = image * IMAGE_STRENGTH + colours * MASK_STRENGTH

    subset_output_folder = OUTPUT_FOLDER / subset
    subset_output_folder.mkdir(parents=True, exist_ok=True)
    output_path = subset_output_folder / f"{image_stem}_overlay.png"
    Image.fromarray(overlay.astype(np.uint8)).save(output_path)
    return output_path


def main() -> None:
    requests = read_image_requests(IMAGE_LIST_PATH)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    created_count = 0
    missing_by_subset: dict[str, int] = {}

    print(f"Reading WaterNet selections from: {IMAGE_LIST_PATH}")
    print(f"Checking {len(requests)} requested ground-truth overlay(s):")
    for subset, image_stem in requests:
        try:
            output_path = create_overlay(subset, image_stem)
        except FileNotFoundError as error:
            missing_by_subset[subset] = missing_by_subset.get(subset, 0) + 1
            print(f"  Skipping {subset}/{image_stem}: {error}")
            continue

        created_count += 1
        print(f"  {output_path}")

    if created_count == 0:
        raise FileNotFoundError("None of the requested images has a matching mask")

    print(f"Saved {created_count} overlay(s) to: {OUTPUT_FOLDER}")
    if missing_by_subset:
        skipped_count = sum(missing_by_subset.values())
        missing_summary = ", ".join(
            f"{subset}: {count}"
            for subset, count in sorted(missing_by_subset.items())
        )
        print(
            f"Skipped {skipped_count} request(s) without a matching file "
            f"({missing_summary})"
        )


if __name__ == "__main__":
    main()
