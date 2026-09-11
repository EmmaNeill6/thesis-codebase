"""Predict binary water masks and record CPU-clock inference times."""

import sys
from pathlib import Path
from statistics import fmean, median
from time import perf_counter_ns

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DATASET_SPECS_PATH = PROJECT_DIR / "dataset_specs.py"
OUTPUT_ROOT = SCRIPT_DIR.parent / "output_eWaSR"

if not DATASET_SPECS_PATH.is_file():
    raise FileNotFoundError(f"Shared dataset specification not found: {DATASET_SPECS_PATH}")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset_specs import load_dataset  # noqa: E402
from wasr.inference import Predictor  # noqa: E402
import wasr.models as models  # noqa: E402
from wasr.utils import load_weights  # noqa: E402

# Change only this line to select: "tampere", "waternet", or "mastr1325".
DATASET = "pohang"

MODEL = "ewasr_resnet18"
WEIGHTS = SCRIPT_DIR / "Pretrained/ewasr_resnet18.pth"
WATER_CLASS_ID = 1
NON_WATER_COLOUR = np.array([247, 195, 37], dtype=np.float32)
WATER_COLOUR = np.array([41, 167, 224], dtype=np.float32)


def create_predictor() -> Predictor:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS}")
    model = models.get_model(MODEL, num_classes=3, pretrained=False)
    model.load_state_dict(load_weights(str(WEIGHTS)))
    return Predictor(model, half_precision=False)


def normalizer():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def predict_image(predictor, normalize, image_path: Path):
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image, dtype=np.uint8)
    image_tensor = normalize(image).unsqueeze(0)

    start_ns = perf_counter_ns()
    prediction = predictor.predict_batch({"image": image_tensor})[0]
    inference_ms = (perf_counter_ns() - start_ns) / 1_000_000

    binary_mask = np.where(prediction == WATER_CLASS_ID, 255, 0).astype(np.uint8)
    if binary_mask.shape != image_array.shape[:2]:
        raise ValueError(f"Prediction size differs from image: {image_path}")
    return image_array, binary_mask, inference_ms


def save_outputs(image, mask, mask_path: Path, overlay_path: Path) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(mask_path)

    colours = np.where(
        (mask == 255)[..., None], WATER_COLOUR, NON_WATER_COLOUR
    )
    overlay = (image * 0.7 + colours * 0.3).astype(np.uint8)
    Image.fromarray(overlay).save(overlay_path)


def main() -> None:
    dataset = load_dataset(DATASET, OUTPUT_ROOT)
    predictor = create_predictor()
    normalize = normalizer()
    mask_dir = dataset.output_dir / "predicted_masks"
    overlay_dir = dataset.output_dir / "prediction_overlays"
    timing_path = dataset.output_dir / "inference_times.txt"
    dataset.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset.name} ({len(dataset.images)} images)")
    print(f"Device: {predictor.device}")
    times = []
    with timing_path.open("w", encoding="utf-8") as timing_file:
        timing_file.write("image\tinference_ms\n")
        for request in tqdm(dataset.images, desc="Predicting"):
            image, mask, elapsed_ms = predict_image(
                predictor, normalize, request.image_path
            )
            save_outputs(
                image,
                mask,
                mask_dir / request.output_path,
                overlay_dir / request.output_path,
            )
            times.append(elapsed_ms)
            timing_file.write(f"{request.image_path}\t{elapsed_ms:.3f}\n")
            timing_file.flush()

        timing_file.write(f"# mean_ms\t{fmean(times):.3f}\n")
        timing_file.write(f"# median_ms\t{median(times):.3f}\n")
        timing_file.write(f"# fps_from_mean\t{1000 / fmean(times):.3f}\n")

    print(f"Binary masks: {mask_dir}")
    print(f"Per-image timing: {timing_path}")
    print(f"Mean inference: {fmean(times):.3f} ms ({1000 / fmean(times):.2f} FPS)")


if __name__ == "__main__":
    main()
