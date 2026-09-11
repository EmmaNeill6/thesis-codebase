"""Run a reproducible YOLO-World open-vocabulary detection smoke test.

This is deliberately an object-detection runner: it saves bounding boxes and
an annotated image, but it does not rasterise boxes into segmentation masks.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PIL import Image
import torch
import ultralytics
from ultralytics import YOLOWorld


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_mps/smoke_test"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "weights/yolov8s-worldv2.pt"
DEFAULT_PROMPTS = ("object")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO-World open-vocabulary object detection."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Ultralytics YOLO-World checkpoint name or local path",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=list(DEFAULT_PROMPTS),
        help="Object categories to detect",
    )
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.source = args.source.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.source.is_file():
        raise FileNotFoundError(f"Source image not found: {args.source}")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    args.prompts = list(dict.fromkeys(item.strip() for item in args.prompts))
    if not args.prompts or any(not item for item in args.prompts):
        raise ValueError("--prompts must contain non-empty category names")
    if args.device == "mps" and not torch.backends.mps.is_available():
        built = torch.backends.mps.is_built()
        raise RuntimeError(
            "MPS was requested but is unavailable "
            f"(PyTorch built with MPS support: {built})."
        )


def synchronise(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def output_stem(source: Path) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{source.stem}_{timestamp}"


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    load_started = perf_counter()
    model = YOLOWorld(args.model)
    model.set_classes(args.prompts)
    model_load_seconds = perf_counter() - load_started

    synchronise(args.device)
    inference_started = perf_counter()
    results = model.predict(
        source=str(args.source),
        device=args.device,
        conf=args.confidence,
        iou=args.iou,
        imgsz=args.image_size,
        save=False,
        verbose=False,
    )
    synchronise(args.device)
    prediction_wall_seconds = perf_counter() - inference_started

    if len(results) != 1:
        raise RuntimeError(f"Expected one result, received {len(results)}")
    result = results[0]
    boxes = result.boxes
    detections = []
    if boxes is not None:
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().to(torch.int64).tolist()
        for coordinates, confidence, class_id in zip(
            xyxy, confidences, class_ids, strict=True
        ):
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": confidence,
                    "xyxy_pixels": coordinates,
                }
            )

    stem = output_stem(args.source)
    annotated_path = args.output_dir / f"{stem}_annotated.jpg"
    record_path = args.output_dir / f"{stem}_detections.json"

    # Ultralytics plot() returns a BGR array; Pillow expects RGB.
    plotted_bgr = result.plot()
    Image.fromarray(plotted_bgr[..., ::-1]).save(annotated_path, quality=95)

    record = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "open-vocabulary object detection",
        "source": str(args.source),
        "annotated_image": str(annotated_path),
        "model": args.model,
        "prompts": args.prompts,
        "device": args.device,
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.iou,
        "image_size": args.image_size,
        "original_shape_height_width": list(result.orig_shape),
        "model_load_seconds": model_load_seconds,
        "prediction_wall_seconds": prediction_wall_seconds,
        "ultralytics_speed_ms": result.speed,
        "detection_count": len(detections),
        "detections": detections,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"Device: {args.device}")
    print(f"Prompts: {', '.join(args.prompts)}")
    print(f"Detections: {len(detections)}")
    for detection in detections:
        print(
            f"  {detection['class_name']}: "
            f"{detection['confidence']:.3f} {detection['xyxy_pixels']}"
        )
    print(f"Annotated image: {annotated_path}")
    print(f"JSON record: {record_path}")


if __name__ == "__main__":
    main()
