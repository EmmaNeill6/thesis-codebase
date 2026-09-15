"""Predict binary water masks with eWaSR on MLX and record execution times."""

import sys
from pathlib import Path
from statistics import fmean, median
from time import perf_counter_ns

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DATASET_SPECS_PATH = PROJECT_DIR / "dataset_specs.py"
OUTPUT_ROOT = SCRIPT_DIR.parent / "output_eWaSR"

if not DATASET_SPECS_PATH.is_file():
    raise FileNotFoundError(f"Shared dataset specification not found: {DATASET_SPECS_PATH}")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset_specs import load_dataset  # noqa: E402
from timing_comparison import append_memory_usage, write_timing_comparison  # noqa: E402
from wasr.mlx_model import Predictor  # noqa: E402

# Select "tampere", "waternet", "mastr1325", or "pohang".
DATASET = "mastr1325"

WEIGHTS = SCRIPT_DIR / "Pretrained/ewasr_resnet18_mlx.npz"
WATER_CLASS_ID = 1
INPUT_ALIGNMENT = 64
NORMALIZE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
NORMALIZE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
NON_WATER_COLOUR = np.array([247, 195, 37], dtype=np.float32)
WATER_COLOUR = np.array([41, 167, 224], dtype=np.float32)


def create_predictor() -> Predictor:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS}")
    return Predictor(WEIGHTS)


def normalizer():
    def normalize(image):
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return (pixels - NORMALIZE_MEAN) / NORMALIZE_STD

    return normalize


def predict_image(predictor, normalize, image_path: Path):
    latency_started_ns = perf_counter_ns()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image_array = np.asarray(image, dtype=np.uint8)
    height, width = image_array.shape[:2]
    image_tensor = normalize(image)

    # Align all encoder feature maps so pyramid pooling has integer kernels.
    pad_bottom = (-height) % INPUT_ALIGNMENT
    pad_right = (-width) % INPUT_ALIGNMENT
    if pad_bottom or pad_right:
        image_tensor = np.pad(
            image_tensor,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="constant",
        )
    image_tensor = image_tensor[None, ...]

    start_ns = perf_counter_ns()
    prediction = np.array(predictor.predict(image_tensor))[0]
    inference_ms = (perf_counter_ns() - start_ns) / 1_000_000
    prediction = prediction[:height, :width]

    binary_mask = np.where(prediction == WATER_CLASS_ID, 255, 0).astype(np.uint8)
    if binary_mask.shape != image_array.shape[:2]:
        raise ValueError(f"Prediction size differs from image: {image_path}")
    final_mask_latency_ms = (perf_counter_ns() - latency_started_ns) / 1_000_000
    return image_array, binary_mask, inference_ms, final_mask_latency_ms


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
    print("Warming up MLX graph...")
    predict_image(predictor, normalize, dataset.images[0].image_path)

    times = []
    comparison_rows = []
    with timing_path.open("w", encoding="utf-8") as timing_file:
        timing_file.write("image\tinference_ms\n")
        for request in tqdm(dataset.images, desc="Predicting"):
            image, mask, elapsed_ms, final_mask_latency_ms = predict_image(
                predictor, normalize, request.image_path
            )
            save_outputs(
                image,
                mask,
                mask_dir / request.output_path,
                overlay_dir / request.output_path,
            )
            times.append(elapsed_ms)
            comparison_rows.append(
                {
                    "evaluation_key": request.output_path.as_posix(),
                    "inference_ms": elapsed_ms,
                    "final_mask_latency_ms": final_mask_latency_ms,
                }
            )
            timing_file.write(f"{request.image_path}\t{elapsed_ms:.3f}\n")
            timing_file.flush()

        timing_file.write(f"# mean_ms\t{fmean(times):.3f}\n")
        timing_file.write(f"# median_ms\t{median(times):.3f}\n")
        timing_file.write(f"# fps_from_mean\t{1000 / fmean(times):.3f}\n")

    comparison = write_timing_comparison(dataset.output_dir, comparison_rows)
    memory = append_memory_usage(
        dataset.output_dir,
        model="eWaSR",
        dataset=DATASET,
        attempted_images=len(dataset.images),
        successful_images=len(comparison_rows),
    )
    print(f"Binary masks: {mask_dir}")
    print(f"Per-image timing: {timing_path}")
    print(f"Mean inference: {fmean(times):.3f} ms ({1000 / fmean(times):.2f} FPS)")
    print(
        "Median final-mask latency: "
        f"{comparison['final_mask_latency_ms']['median']:.3f} ms"
    )
    print(
        "Comparable timing report: "
        f"{dataset.output_dir / 'comparable_timing_summary.json'}"
    )
    print(f"Peak process RSS: {memory['peak_process_rss_mib']:.1f} MiB")
    print(f"Memory report: {dataset.output_dir / 'memory_usage.jsonl'}")


if __name__ == "__main__":
    main()
