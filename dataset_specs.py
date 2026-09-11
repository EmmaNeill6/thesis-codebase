"""Dataset-specific image selection, optional mask pairing, and output naming.

The same :class:`ImageRequest` objects are shared by prediction and dataset
auditing so that every downstream step refers to exactly the same samples.
Unlabelled datasets can be used for qualitative prediction with no ground truth.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class ImageRequest:
    image_path: Path
    output_path: Path
    ground_truth_path: Path | None


@dataclass(frozen=True)
class MaskEncoding:
    """Rules for converting a dataset's ground truth to water/non-water."""

    water_value: int
    water_threshold: int | None
    ignore_value: int | None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    output_dir: Path
    images: tuple[ImageRequest, ...]
    ground_truth_encoding: MaskEncoding | None


def _images(folder: Path, recursive: bool = False) -> list[Path]:
    """Return supported image files in a stable order."""
    paths = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        path
        for path in paths
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    )


def _tampere(output_root: Path) -> DatasetSpec:
    root = DATA_DIR / "Tampere-WaterSeg"
    images = []
    for subset in ("channel", "open", "dock"):
        prefix = f"{subset}_image_"
        for image_path in _images(root / "images" / subset):
            image_number = image_path.stem.removeprefix(prefix)
            if not image_path.stem.startswith(prefix) or not image_number.isdigit():
                continue
            output_path = Path(
                f"{subset}_mask/{subset}_mask_{image_number}.png"
            )
            ground_truth_path = root / "masks" / output_path
            if ground_truth_path.is_file():
                images.append(
                    ImageRequest(image_path, output_path, ground_truth_path)
                )
    return DatasetSpec(
        "Tampere-WaterSeg",
        output_root / "binary_water/tampere_waterseg",
        tuple(images),
        MaskEncoding(water_value=255, water_threshold=128, ignore_value=None),
    )


def _waternet(output_root: Path) -> DatasetSpec:
    root = DATA_DIR / "water_v2"
    image_root = root / "JPEGImages"
    annotation_root = root / "Annotations"
    images = []
    for image_path in _images(image_root, recursive=True):
        relative_mask = image_path.relative_to(image_root).with_suffix(".png")
        ground_truth_path = annotation_root / relative_mask
        if ground_truth_path.is_file():
            images.append(
                ImageRequest(image_path, relative_mask, ground_truth_path)
            )
    return DatasetSpec(
        "WaterNet",
        output_root / "binary_water/waternet",
        tuple(images),
        MaskEncoding(water_value=255, water_threshold=128, ignore_value=None),
    )


def _mastr1325(output_root: Path) -> DatasetSpec:
    root = DATA_DIR / "MaSTR1325"
    images = []
    for image_path in _images(root / "images"):
        output_path = Path(f"{image_path.stem}m.png")
        ground_truth_path = root / "masks" / output_path
        if ground_truth_path.is_file():
            images.append(
                ImageRequest(image_path, output_path, ground_truth_path)
            )
    return DatasetSpec(
        "MaSTR1325",
        output_root / "binary_water/mastr1325",
        tuple(images),
        MaskEncoding(water_value=1, water_threshold=None, ignore_value=4),
    )


def _pohang(output_root: Path) -> DatasetSpec:
    """Load one unlabelled Pohang image per stereo camera folder."""
    root = DATA_DIR / "pohang00"
    images = []
    for scene_path in sorted(path for path in root.iterdir() if path.is_dir()):
        for camera in ("left_images", "right_images"):
            image_root = scene_path / camera
            if not image_root.is_dir():
                continue
            for image_path in _images(image_root)[:1]:
                output_path = image_path.relative_to(root).with_suffix(".png")
                images.append(ImageRequest(image_path, output_path, None))
    return DatasetSpec(
        "Pohang",
        output_root / "binary_water/pohang",
        tuple(images),
        # This local Pohang subset has RGB stereo frames but no segmentation masks.
        None,
    )


DATASETS: dict[str, Callable[[Path], DatasetSpec]] = {
    "tampere": _tampere,
    "waternet": _waternet,
    "mastr1325": _mastr1325,
    "pohang": _pohang,
}


def load_dataset(name: str, output_root: Path) -> DatasetSpec:
    """Load one named dataset and validate that it selects some images."""
    try:
        dataset = DATASETS[name.casefold()](output_root)
    except KeyError as error:
        choices = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset {name!r}; choose from: {choices}") from error
    if not dataset.images:
        raise ValueError(f"{dataset.name} selected no images")
    return dataset
