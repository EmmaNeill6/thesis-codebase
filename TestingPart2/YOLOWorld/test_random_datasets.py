"""Run YOLO-World on a deterministic random sample from every dataset.

The outputs are qualitative object-detection results. The source datasets have
water segmentation masks rather than bounding-box annotations, so detection
counts must not be described as accuracy, precision, recall or mAP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Any

from PIL import Image, ImageDraw, ImageOps
import torch
import ultralytics
from ultralytics import YOLOWorld
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "configs/maritime_random5.yaml"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset_specs import DATASETS, load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test YOLO-World on random samples from project datasets."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    unknown = [name for name in datasets if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")
    if len(datasets) != len(set(datasets)):
        raise ValueError("datasets must not contain duplicates")

    prompts = config.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a non-empty list")
    prompts = [str(prompt).strip() for prompt in prompts]
    if any(not prompt for prompt in prompts) or len(prompts) != len(set(prompts)):
        raise ValueError("prompts must be non-empty and unique")
    config["prompts"] = prompts

    sample_count = int(config.get("sample_count_per_dataset", 5))
    if sample_count <= 0:
        raise ValueError("sample_count_per_dataset must be positive")
    config["sample_count_per_dataset"] = sample_count
    config["selection_seed"] = int(config.get("selection_seed", 4022))

    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be a YAML mapping")
    model["checkpoint"] = str((path.parent / model["checkpoint"]).resolve())
    model["device"] = str(model.get("device", "mps"))
    model["image_size"] = int(model.get("image_size", 1280))
    model["confidence_threshold"] = float(
        model.get("confidence_threshold", 0.15)
    )
    model["nms_iou_threshold"] = float(model.get("nms_iou_threshold", 0.7))
    model["agnostic_nms"] = bool(model.get("agnostic_nms", True))
    if model["device"] not in {"mps", "cpu"}:
        raise ValueError("model.device must be mps or cpu")
    if model["image_size"] <= 0:
        raise ValueError("model.image_size must be positive")
    for name in ("confidence_threshold", "nms_iou_threshold"):
        if not 0.0 <= model[name] <= 1.0:
            raise ValueError(f"model.{name} must be between 0 and 1")
    if not Path(model["checkpoint"]).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model['checkpoint']}")

    output_root = config.get("output_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ValueError("output_root must be a path string")
    config["output_root"] = str((path.parent / output_root).resolve())
    config["config_path"] = str(path)
    return config


def select_samples(dataset_key: str, images: tuple, count: int, seed: int) -> list:
    """Select stable pseudo-random samples by ranking salted SHA-256 hashes."""

    if len(images) < count:
        raise ValueError(
            f"Dataset {dataset_key} contains {len(images)} images, fewer than {count}"
        )

    def rank(request: Any) -> str:
        value = f"{seed}:{dataset_key}:{request.output_path.as_posix()}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    chosen = sorted(images, key=rank)[:count]
    return sorted(chosen, key=lambda request: request.output_path.as_posix())


def synchronise(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sample_slug(index: int, request: Any) -> str:
    key = request.output_path.with_suffix("").as_posix().replace("/", "__")
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in key)
    return f"{index:02d}_{safe}"


def detection_records(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    records = []
    for coordinates, confidence, class_id in zip(
        boxes.xyxy.detach().cpu().tolist(),
        boxes.conf.detach().cpu().tolist(),
        boxes.cls.detach().cpu().to(torch.int64).tolist(),
        strict=True,
    ):
        records.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": confidence,
                "xyxy_pixels": coordinates,
            }
        )
    return records


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_contact_sheet(annotated_paths: list[Path], output_path: Path) -> None:
    """Create a compact visual index for one dataset's annotated samples."""

    columns = 2
    cell_width, cell_height = 720, 500
    margin, label_height = 20, 30
    rows = (len(annotated_paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * cell_width, rows * cell_height), (242, 242, 242)
    )
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(annotated_paths):
        row, column = divmod(index, columns)
        left = column * cell_width + margin
        top = row * cell_height + margin
        with Image.open(path) as source:
            preview = ImageOps.contain(
                source.convert("RGB"),
                (cell_width - 2 * margin, cell_height - label_height - 2 * margin),
            )
        image_left = left + (cell_width - 2 * margin - preview.width) // 2
        sheet.paste(preview, (image_left, top + label_height))
        draw.text((left, top), path.stem, fill=(20, 20, 20))
    sheet.save(output_path, quality=92)


def write_dataset_summary(
    dataset_dir: Path, dataset_key: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    class_counts = Counter(
        detection["class_name"]
        for record in records
        for detection in record["detections"]
    )
    inference_ms = [record["ultralytics_speed_ms"]["inference"] for record in records]
    wall_seconds = [record["prediction_wall_seconds"] for record in records]
    summary = {
        "dataset": dataset_key,
        "sample_count": len(records),
        "images_with_detections": sum(bool(record["detections"]) for record in records),
        "images_without_detections": sum(not record["detections"] for record in records),
        "total_detections": sum(len(record["detections"]) for record in records),
        "detections_by_class": dict(sorted(class_counts.items())),
        "mean_model_inference_ms": fmean(inference_ms),
        "median_model_inference_ms": median(inference_ms),
        "mean_prediction_wall_seconds": fmean(wall_seconds),
        "note": (
            "Detection counts are qualitative outputs, not accuracy metrics; "
            "these datasets do not provide bounding-box ground truth."
        ),
    }
    atomic_json(dataset_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    model_config = config["model"]

    if model_config["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable. Run with host GPU access; "
            f"PyTorch MPS build support={torch.backends.mps.is_built()}."
        )

    selected: dict[str, list[Any]] = {}
    for dataset_key in config["datasets"]:
        dataset = load_dataset(dataset_key, output_root)
        selected[dataset_key] = select_samples(
            dataset_key,
            dataset.images,
            config["sample_count_per_dataset"],
            config["selection_seed"],
        )

    print(f"Loading {model_config['checkpoint']}...")
    model_started = perf_counter()
    model = YOLOWorld(model_config["checkpoint"])
    model.set_classes(config["prompts"])
    model_load_seconds = perf_counter() - model_started

    first_request = selected[config["datasets"][0]][0]
    print("Warming up MPS with the first selected image...")
    synchronise(model_config["device"])
    warmup_started = perf_counter()
    model.predict(
        source=str(first_request.image_path),
        device=model_config["device"],
        conf=model_config["confidence_threshold"],
        iou=model_config["nms_iou_threshold"],
        imgsz=model_config["image_size"],
        agnostic_nms=model_config["agnostic_nms"],
        save=False,
        verbose=False,
    )
    synchronise(model_config["device"])
    warmup_seconds = perf_counter() - warmup_started

    summaries = {}
    all_records = []
    for dataset_key in config["datasets"]:
        dataset_dir = output_root / dataset_key
        originals_dir = dataset_dir / "originals"
        annotated_dir = dataset_dir / "annotated"
        detections_dir = dataset_dir / "detections"
        for directory in (originals_dir, annotated_dir, detections_dir):
            directory.mkdir(parents=True, exist_ok=True)

        selection = []
        dataset_records = []
        annotated_paths = []
        for index, request in enumerate(selected[dataset_key], start=1):
            slug = sample_slug(index, request)
            original_path = originals_dir / f"{slug}{request.image_path.suffix.lower()}"
            annotated_path = annotated_dir / f"{slug}.jpg"
            record_path = detections_dir / f"{slug}.json"
            selection.append(
                {
                    "sample_number": index,
                    "evaluation_key": request.output_path.as_posix(),
                    "source": str(request.image_path.resolve()),
                }
            )

            shutil.copy2(request.image_path, original_path)
            synchronise(model_config["device"])
            started = perf_counter()
            results = model.predict(
                source=str(request.image_path),
                device=model_config["device"],
                conf=model_config["confidence_threshold"],
                iou=model_config["nms_iou_threshold"],
                imgsz=model_config["image_size"],
                agnostic_nms=model_config["agnostic_nms"],
                save=False,
                verbose=False,
            )
            synchronise(model_config["device"])
            wall_seconds = perf_counter() - started
            if len(results) != 1:
                raise RuntimeError(f"Expected one result, received {len(results)}")
            result = results[0]
            detections = detection_records(result)
            plotted_bgr = result.plot()
            Image.fromarray(plotted_bgr[..., ::-1]).save(annotated_path, quality=95)
            annotated_paths.append(annotated_path)

            record = {
                "dataset": dataset_key,
                "sample_number": index,
                "evaluation_key": request.output_path.as_posix(),
                "source": str(request.image_path.resolve()),
                "original_copy": relative(original_path, output_root),
                "annotated_image": relative(annotated_path, output_root),
                "original_shape_height_width": list(result.orig_shape),
                "prediction_wall_seconds": wall_seconds,
                "ultralytics_speed_ms": result.speed,
                "detection_count": len(detections),
                "detections": detections,
            }
            atomic_json(record_path, record)
            dataset_records.append(record)
            all_records.append(record)
            labels = ", ".join(item["class_name"] for item in detections) or "none"
            print(
                f"[{dataset_key} {index}/{len(selected[dataset_key])}] "
                f"{request.image_path.name}: {len(detections)} ({labels})"
            )

        atomic_json(
            dataset_dir / "selection.json",
            {
                "selection_seed": config["selection_seed"],
                "method": "lowest salted SHA-256 ranks, then sorted by evaluation key",
                "samples": selection,
            },
        )
        write_contact_sheet(annotated_paths, dataset_dir / "contact_sheet.jpg")
        summaries[dataset_key] = write_dataset_summary(
            dataset_dir, dataset_key, dataset_records
        )

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete",
        "task": "qualitative open-vocabulary object detection",
        "run_name": config.get("run_name"),
        "configuration_file": config["config_path"],
        "datasets": config["datasets"],
        "sample_count_per_dataset": config["sample_count_per_dataset"],
        "selection_seed": config["selection_seed"],
        "prompts": config["prompts"],
        "model": model_config,
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "total_images": len(all_records),
        "total_detections": sum(len(record["detections"]) for record in all_records),
        "dataset_summaries": summaries,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "interpretation_warning": (
            "No bounding-box ground truth is available. These results support "
            "qualitative inspection and runtime measurement only."
        ),
    }
    atomic_json(output_root / "run_manifest.json", manifest)
    shutil.copy2(config["config_path"], output_root / "experiment_config.yaml")
    print(f"Completed {len(all_records)} images. Output: {output_root}")


if __name__ == "__main__":
    main()
