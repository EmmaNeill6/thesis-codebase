"""Fine-tune only the eWaSR segmentation head using three-class masks.

Masks must use the shared maritime convention:
    0 = obstacle, 1 = water, 2 = sky

This runner intentionally contains no inference-runtime benchmarking.  It
records training loss and segmentation quality for the fine-tuning study.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_WEIGHTS = (
    PROJECT_DIR / "TestingPart1/eWaSR/Pretrained/ewasr_resnet18_mlx.npz"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs/head_only"

CLASS_NAMES = ("obstacle", "water", "sky")
NUM_CLASSES = len(CLASS_NAMES)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NORMALIZE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
NORMALIZE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path


def find_samples(
    images_dir: Path,
    masks_dir: Path,
    *,
    ignore_masks_without_images: bool = False,
) -> list[Sample]:
    """Pair each available PNG mask with an image of the same relative stem.

    Masks are authoritative so a pseudo-labelled subset can be trained without
    requiring a mask for every image in the larger source directory.
    """

    images_dir = images_dir.expanduser().resolve()
    masks_dir = masks_dir.expanduser().resolve()
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {images_dir}")
    if not masks_dir.is_dir():
        raise NotADirectoryError(f"Mask directory does not exist: {masks_dir}")

    mask_paths = sorted(path for path in masks_dir.rglob("*.png") if path.is_file())
    if not mask_paths:
        raise ValueError(f"No PNG masks found in: {masks_dir}")
    samples = []
    missing = []
    ambiguous = []
    for mask_path in mask_paths:
        relative_stem = mask_path.relative_to(masks_dir).with_suffix("")
        candidates = [
            (images_dir / relative_stem).with_suffix(extension)
            for extension in sorted(SUPPORTED_IMAGE_EXTENSIONS)
        ]
        matches = [candidate for candidate in candidates if candidate.is_file()]
        if len(matches) == 1:
            samples.append(Sample(image_path=matches[0], mask_path=mask_path))
        elif not matches:
            missing.append(relative_stem)
        else:
            ambiguous.append(relative_stem)

    if missing and not ignore_masks_without_images:
        preview = ", ".join(str(path) for path in missing[:5])
        remainder = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise FileNotFoundError(
            f"Missing {len(missing)} image(s) below {images_dir} for masks: "
            f"{preview}{remainder}"
        )
    if ambiguous:
        preview = ", ".join(str(path) for path in ambiguous[:5])
        raise ValueError(f"Multiple image extensions match these masks: {preview}")
    return samples


def validate_mask(mask_path: Path) -> None:
    """Reject visualisation masks and unsupported pseudo-label values."""

    with Image.open(mask_path) as source:
        mask = np.asarray(source.convert("L"), dtype=np.uint8)
    values = np.unique(mask)
    if not set(values.tolist()).issubset(range(NUM_CLASSES)):
        raise ValueError(
            f"Mask must contain only class IDs 0, 1 and 2: {mask_path} "
            f"contains {values.tolist()}"
        )


def validate_dataset(samples: list[Sample]) -> None:
    """Check every image/mask pair before a training run starts."""

    for sample in samples:
        with Image.open(sample.image_path) as image_source:
            image_size = image_source.size
        with Image.open(sample.mask_path) as mask_source:
            mask_size = mask_source.size
        if image_size != mask_size:
            raise ValueError(
                f"Image and mask sizes differ for {sample.image_path.name}: "
                f"{image_size} versus {mask_size}"
            )
        validate_mask(sample.mask_path)


def load_sample(
    sample: Sample,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize and normalize one image and downsample its mask to head resolution."""

    with Image.open(sample.image_path) as source:
        image = source.convert("RGB").resize(
            (width, height), Image.Resampling.BILINEAR
        )
    with Image.open(sample.mask_path) as source:
        mask = source.convert("L").resize(
            (width // 4, height // 4), Image.Resampling.NEAREST
        )

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = (image_array - NORMALIZE_MEAN) / NORMALIZE_STD
    mask_array = np.asarray(mask, dtype=np.int32)
    return image_array, mask_array


def batches(
    samples: list[Sample],
    *,
    batch_size: int,
    width: int,
    height: int,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield fixed-resolution NHWC image batches and class-index masks."""

    indices = np.arange(len(samples))
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        loaded = [
            load_sample(samples[index], width=width, height=height)
            for index in selected
        ]
        images, masks = zip(*loaded)
        yield np.stack(images), np.stack(masks)


def confusion_metrics(confusion: np.ndarray) -> dict[str, object]:
    """Calculate pixel accuracy and per-class/mean intersection over union."""

    true_pixels = confusion.sum(axis=1)
    predicted_pixels = confusion.sum(axis=0)
    intersection = np.diag(confusion)
    union = true_pixels + predicted_pixels - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
        where=union > 0,
    )
    total = int(confusion.sum())
    pixel_accuracy = float(intersection.sum() / total) if total else float("nan")
    water_intersection = int(confusion[1, 1])
    water_union = int(true_pixels[1] + predicted_pixels[1] - water_intersection)
    non_water_intersection = int(confusion[np.ix_([0, 2], [0, 2])].sum())
    non_water_true = int(true_pixels[[0, 2]].sum())
    non_water_predicted = int(predicted_pixels[[0, 2]].sum())
    non_water_union = non_water_true + non_water_predicted - non_water_intersection
    water_iou = water_intersection / water_union if water_union else float("nan")
    non_water_iou = (
        non_water_intersection / non_water_union
        if non_water_union
        else float("nan")
    )
    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": float(np.nanmean(iou)),
        "class_iou": {
            name: None if np.isnan(value) else float(value)
            for name, value in zip(CLASS_NAMES, iou)
        },
        "binary_water_iou": float(water_iou),
        "binary_non_water_iou": float(non_water_iou),
        "binary_mean_iou": float(np.nanmean([water_iou, non_water_iou])),
    }


def parse_class_weights(value: str | None) -> np.ndarray | None:
    if value is None:
        return None
    parsed = np.asarray([float(item) for item in value.split(",")], dtype=np.float32)
    if parsed.shape != (NUM_CLASSES,) or np.any(parsed <= 0):
        raise argparse.ArgumentTypeError(
            "--class-weights must be three positive comma-separated values"
        )
    return parsed


def flat_parameter_summary(parameters) -> tuple[list[str], int]:
    from mlx.utils import tree_flatten

    flattened = tree_flatten(parameters)
    return [name for name, _ in flattened], sum(value.size for _, value in flattened)


def evaluate(model, samples, args) -> dict[str, object]:
    import mlx.core as mx
    import mlx.nn as nn

    model.set_head_training_mode(False)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    losses = []
    for images, masks in batches(
        samples,
        batch_size=args.batch_size,
        width=args.width,
        height=args.height,
    ):
        targets = mx.array(masks)
        logits = model(mx.array(images))
        loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
        prediction = mx.argmax(logits, axis=-1)
        mx.eval(loss, prediction)
        losses.append(float(loss.item()))

        target_np = masks.reshape(-1)
        prediction_np = np.asarray(prediction).reshape(-1)
        confusion += np.bincount(
            target_np * NUM_CLASSES + prediction_np,
            minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    result = confusion_metrics(confusion)
    result["loss"] = float(np.mean(losses))
    model.set_head_training_mode(True)
    return result


def train(args: argparse.Namespace, train_samples, validation_samples) -> dict[str, object]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from ewasr_mlx import create_head_finetuning_model

    if not mx.metal.is_available():
        raise RuntimeError("MLX cannot access an Apple Metal GPU")

    model = create_head_finetuning_model(args.weights)
    trainable_names, trainable_count = flat_parameter_summary(
        model.trainable_parameters()
    )
    unexpected = [
        name
        for name in trainable_names
        if not name.startswith("decoder.seg_head")
    ]
    if unexpected:
        raise RuntimeError(f"Non-head parameters are trainable: {unexpected}")
    trainable_running_stats = [
        name
        for name in trainable_names
        if name.endswith(("running_mean", "running_var"))
    ]
    if trainable_running_stats:
        raise RuntimeError(
            f"BatchNorm running statistics must not receive gradients: {trainable_running_stats}"
        )
    if not trainable_names:
        raise RuntimeError("No trainable segmentation-head parameters were found")

    class_weights = parse_class_weights(args.class_weights)
    class_weights_mx = None if class_weights is None else mx.array(class_weights)

    def loss_fn(images, targets):
        logits = model(images)
        per_pixel = nn.losses.cross_entropy(logits, targets, reduction="none")
        if class_weights_mx is None:
            return mx.mean(per_pixel)
        pixel_weights = mx.take(class_weights_mx, targets)
        return mx.sum(per_pixel * pixel_weights) / mx.sum(pixel_weights)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bias_correction=True,
    )
    rng = np.random.default_rng(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "training_history.jsonl"
    latest_path = args.output_dir / "latest_head_only.npz"
    best_path = args.output_dir / "best_head_only.npz"

    config = vars(args).copy()
    config.update(
        {
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": trainable_count,
            "class_names": CLASS_NAMES,
        }
    )
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"Training pairs: {len(train_samples)}")
    print(f"Validation pairs: {len(validation_samples)}")
    print(f"Trainable parameters: {trainable_count:,}")
    print("Trainable tensors:")
    for name in trainable_names:
        print(f"  {name}")

    best_score = -np.inf
    best_epoch = None
    best_record = None
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, args.epochs + 1):
            model.set_head_training_mode(True)
            training_losses = []
            for images, masks in batches(
                train_samples,
                batch_size=args.batch_size,
                width=args.width,
                height=args.height,
                rng=rng,
            ):
                loss, gradients = loss_and_grad(mx.array(images), mx.array(masks))
                optimizer.update(model, gradients)
                mx.eval(model.parameters(), optimizer.state, loss)
                training_losses.append(float(loss.item()))

            record: dict[str, object] = {
                "epoch": epoch,
                "training_loss": float(np.mean(training_losses)),
            }
            if validation_samples:
                record["validation"] = evaluate(model, validation_samples, args)
                score = float(record["validation"]["mean_iou"])
            else:
                score = -float(record["training_loss"])

            model.save_weights(str(latest_path))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_record = record
                model.save_weights(str(best_path))

            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            print(json.dumps(record))

    print(f"Latest checkpoint: {latest_path}")
    print(f"Best checkpoint: {best_path}")
    print(f"Training history: {history_path}")
    return {
        "best_checkpoint": best_path,
        "latest_checkpoint": latest_path,
        "history": history_path,
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "best_record": best_record,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune only the eWaSR segmentation head with MLX."
    )
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-masks", type=Path, required=True)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--val-masks", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weights", help="Obstacle,water,sky weights, e.g. 2,1,1")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=4022)
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Validate image/mask pairing and class IDs without loading MLX",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if args.width % 64 or args.height % 64:
        raise ValueError("--width and --height must both be divisible by 64")
    if (args.val_images is None) != (args.val_masks is None):
        raise ValueError("Provide both --val-images and --val-masks, or neither")

    train_samples = find_samples(args.train_images, args.train_masks)
    validation_samples = (
        find_samples(args.val_images, args.val_masks) if args.val_images else []
    )
    validate_dataset(train_samples)
    validate_dataset(validation_samples)
    print(f"Validated {len(train_samples)} training pairs")
    print(f"Validated {len(validation_samples)} validation pairs")
    if args.check_data:
        return

    args.weights = args.weights.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    train(args, train_samples, validation_samples)


if __name__ == "__main__":
    main()
