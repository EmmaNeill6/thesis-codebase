"""Fine-tune only the eWaSR segmentation head using three-class masks.

Masks must use the shared maritime convention:
    0 = obstacle, 1 = water, 2 = sky

This runner intentionally contains no inference-runtime benchmarking.  It
records training loss and segmentation quality for the fine-tuning study.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
FIXED_LEARNING_RATE = 0.001
FIXED_WEIGHT_DECAY = 0.001
FIXED_EPOCHS = 20
FIXED_BATCH_SIZE = 2
FIXED_CLASS_WEIGHTS = "1,1,1"

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


def configured_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def select_configured_samples(samples: list[Sample], selection: dict, role: str) -> list[Sample]:
    """Apply the filename selection recorded in a DataSource YAML file."""
    kind = selection.get("kind")
    if kind == "all":
        return sorted(samples, key=lambda sample: str(sample.image_path))
    if kind == "filename_regex":
        pattern = re.compile(str(selection["pattern"]))
        return sorted((sample for sample in samples if pattern.fullmatch(sample.image_path.stem)),
                      key=lambda sample: str(sample.image_path))
    if kind == "sequence_ids":
        pattern = re.compile(str(selection["pattern"]))
        ids = list(selection["ids"])
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{role} sequence_ids must be nonempty and unique")
        chosen = []
        found = set()
        for sample in samples:
            match = pattern.fullmatch(sample.image_path.stem)
            if match is None or len(match.groups()) != 1:
                raise ValueError(f"{role} filename does not match its sequence pattern: {sample.image_path.name}")
            if match.group(1) in ids:
                chosen.append(sample)
                found.add(match.group(1))
        if found != set(ids):
            raise ValueError(f"{role} missing sequence IDs: {sorted(set(ids) - found)}")
        return sorted(chosen, key=lambda sample: str(sample.image_path))
    if kind not in {"inclusive_range", "every_nth"}:
        raise ValueError(f"Unsupported {role} selection kind: {kind!r}")
    pattern = re.compile(str(selection["index_pattern"]))
    start, end = int(selection["start"]), int(selection["end"])
    step = int(selection.get("step", 1)) if kind == "every_nth" else 1
    if start > end or step <= 0:
        raise ValueError(f"Invalid {role} selection range or step")
    chosen = []
    for sample in samples:
        match = pattern.fullmatch(sample.image_path.stem)
        if match is None or len(match.groups()) != 1:
            raise ValueError(f"{role} filename does not match its index pattern: {sample.image_path.name}")
        index = int(match.group(1))
        if start <= index <= end and (index - start) % step == 0:
            chosen.append((index, sample))
    return [sample for _, sample in sorted(chosen, key=lambda pair: pair[0])]


def load_configured_splits(config_path: Path) -> tuple[dict, dict[str, list[Sample]]]:
    """Resolve YAML paths relative to the YAML and validate all selected pairs."""
    import yaml

    config_path = config_path.expanduser().resolve()
    if config_path.parent != (SCRIPT_DIR.parent / "DataSource").resolve():
        raise ValueError("--config must be a YAML file in TestingPart3/DataSource")
    with config_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected a schema_version: 1 DataSource YAML")
    if config.get("labels") != {"obstacle": 0, "water": 1, "sky": 2}:
        raise ValueError("Expected labels obstacle=0, water=1, sky=2")
    model = config["model"]
    if model.get("imu") is not False or model.get("trainable_module") != "decoder.seg_head":
        raise ValueError("Expected non-IMU eWaSR with only decoder.seg_head trainable")
    data = config["data"]
    roles = ("train", "validation", "test", "unseen")
    splits = {}
    for role in roles:
        if role not in data:
            if role == "unseen":
                continue
            raise ValueError(f"Missing data.{role} split")
        split = data[role]
        sources = split.get("sources", [split])
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"data.{role}.sources must be a nonempty list")
        selected = []
        for source in sources:
            images = configured_path(source["images_dir"], config_path)
            masks = configured_path(source["masks_dir"], config_path)
            available = find_samples(images, masks,
                                     ignore_masks_without_images=bool(source.get("shared_masks_dir", False)))
            subset = select_configured_samples(available, source["selection"], role)
            if "expected_count" in source and len(subset) != int(source["expected_count"]):
                raise ValueError(f"{role} source selected {len(subset)} pairs; expected {source['expected_count']}")
            if role != "unseen":
                validate_dataset(subset)
            selected.extend(subset)
        if "expected_count" in split and len(selected) != int(split["expected_count"]):
            raise ValueError(f"{role} selected {len(selected)} pairs; expected {split['expected_count']}")
        if not selected:
            raise ValueError(f"{role} has no selected pairs")
        splits[role] = selected
    image_owners, mask_owners = set(), set()
    for role, samples in splits.items():
        for sample in samples:
            image, mask = sample.image_path.resolve(), sample.mask_path.resolve()
            if image in image_owners or mask in mask_owners:
                raise ValueError(f"Image or mask appears in multiple YAML splits: {role}: {image}")
            image_owners.add(image)
            mask_owners.add(mask)
    return config, splits


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
        "obstacle_recall": float(intersection[0] / true_pixels[0]) if true_pixels[0] else None,
        "obstacle_precision": float(intersection[0] / predicted_pixels[0]) if predicted_pixels[0] else None,
        "obstacle_miss_rate": float(1 - intersection[0] / true_pixels[0]) if true_pixels[0] else None,
        "water_as_obstacle_rate": float(confusion[1, 0] / true_pixels[1]) if true_pixels[1] else None,
        "confusion_matrix": confusion.tolist(),
    }


def validation_selection_score(record: dict[str, object], metric: str) -> float:
    """Choose checkpoints by an explicit validation metric, never test data."""
    if metric not in {"mean_iou", "binary_water_iou"}:
        raise ValueError(f"Unsupported checkpoint selection metric: {metric}")
    value = record["validation"][metric]
    if value is None or not np.isfinite(value):
        raise ValueError(f"Undefined validation {metric}")
    return float(value)


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

    training_started = time.perf_counter()
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
    selection_metric = getattr(args, "selection_metric", "mean_iou")
    if selection_metric not in {"mean_iou", "binary_water_iou"}:
        raise ValueError(f"Unsupported checkpoint selection metric: {selection_metric}")
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
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
                score = validation_selection_score(record, selection_metric)
            else:
                score = -float(record["training_loss"])

            model.save_weights(str(latest_path))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_record = record
                model.save_weights(str(best_path))

            record["epoch_elapsed_seconds"] = time.perf_counter() - epoch_started
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            print(json.dumps(record))

    print(f"Latest checkpoint: {latest_path}")
    print(f"Best checkpoint: {best_path}")
    print(f"Training history: {history_path}")
    training_elapsed_seconds = time.perf_counter() - training_started
    print(f"Training elapsed: {training_elapsed_seconds:.2f} seconds")
    return {
        "best_checkpoint": best_path,
        "latest_checkpoint": latest_path,
        "history": history_path,
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "best_record": best_record,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "training_elapsed_seconds": training_elapsed_seconds,
    }


def format_metrics(metrics: dict[str, object], label: str) -> list[str]:
    lines = [label]
    for key in ("loss", "pixel_accuracy", "mean_iou", "binary_water_iou",
                "binary_non_water_iou", "binary_mean_iou", "obstacle_recall",
                "obstacle_precision", "obstacle_miss_rate", "water_as_obstacle_rate"):
        value = metrics.get(key)
        lines.append(f"  {key}: {value:.6f}" if value is not None else f"  {key}: unavailable")
    for name, value in metrics["class_iou"].items():
        lines.append(f"  {name}_iou: {value:.6f}" if value is not None else f"  {name}_iou: unavailable")
    lines.append(f"  confusion_matrix (rows=true, columns=predicted): {metrics['confusion_matrix']}")
    return lines


def write_evaluation_text(output_dir: Path, result: dict[str, object], *,
                          args: argparse.Namespace,
                          config_path: Path | None, counts: dict[str, int],
                          test_metrics: dict[str, object] | None,
                          started_at: datetime, total_elapsed_seconds: float,
                          selection_metric: str) -> Path:
    lines = [
        "eWaSR segmentation-head fine-tuning evaluation",
        f"Started (UTC): {started_at.isoformat()}",
        f"Finished (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Dataset YAML: {config_path if config_path else 'direct image/mask directories'}",
        f"Split counts: {counts}",
        "Optimizer: AdamW",
        f"Learning rate: {args.learning_rate}",
        f"Weight decay: {args.weight_decay}",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch_size}",
        f"Class weights (obstacle, water, sky): {args.class_weights or 'unweighted'}",
        f"Best epoch: {result['best_epoch']}",
        f"Checkpoint selection: validation {selection_metric}" if counts.get("validation") else "Checkpoint selection: lowest training loss",
        f"Best checkpoint: {result['best_checkpoint']}",
        f"Training elapsed: {result['training_elapsed_seconds']:.2f} seconds",
        f"Total elapsed (data loading, training and evaluation): {total_elapsed_seconds:.2f} seconds",
        "",
        "Metrics against pseudo-label masks measure agreement with those labels.",
        "",
    ]
    if counts.get("validation"):
        lines.extend(format_metrics(result["best_record"]["validation"], "Best validation epoch:"))
    if test_metrics is not None:
        lines.extend([""] + format_metrics(test_metrics, "Selected checkpoint on test split:"))
    if counts.get("unseen"):
        lines.extend(["", "Unseen split: reserved; no evaluation performed."])
    path = output_dir / "evaluation.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune only the eWaSR segmentation head with MLX."
    )
    parser.add_argument("--config", type=Path, help="DataSource YAML defining image/mask paths and splits")
    parser.add_argument("--train-images", type=Path)
    parser.add_argument("--train-masks", type=Path)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--val-masks", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=FIXED_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FIXED_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=FIXED_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=FIXED_WEIGHT_DECAY)
    parser.add_argument("--class-weights", default=FIXED_CLASS_WEIGHTS,
                        help="Obstacle,water,sky weights (default: 1,1,1)")
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
    total_started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if args.width % 64 or args.height % 64:
        raise ValueError("--width and --height must both be divisible by 64")
    if args.config is not None:
        requested = {"learning_rate": FIXED_LEARNING_RATE,
                     "weight_decay": FIXED_WEIGHT_DECAY, "epochs": FIXED_EPOCHS,
                     "batch_size": FIXED_BATCH_SIZE, "class_weights": FIXED_CLASS_WEIGHTS}
        for name, fixed_value in requested.items():
            if getattr(args, name) != fixed_value:
                raise ValueError(f"--config uses fixed {name}={fixed_value}; do not override it")
        if any(value is not None for value in (args.train_images, args.train_masks,
                                               args.val_images, args.val_masks)):
            raise ValueError("Use --config or direct image/mask directories, not both")
        if args.weights is not None:
            raise ValueError("The YAML model.checkpoint supplies --weights")
        config_path = args.config.expanduser().resolve()
        config, splits = load_configured_splits(config_path)
        preprocessing = config["preprocessing"]
        if (int(preprocessing["input_width"]), int(preprocessing["input_height"])) != (512, 384):
            raise ValueError("This head-only recipe requires 512 x 384 input")
        if (int(preprocessing["mask_width"]), int(preprocessing["mask_height"])) != (128, 96):
            raise ValueError("This head-only recipe requires 128 x 96 masks")
        if preprocessing.get("mask_interpolation") != "nearest":
            raise ValueError("Mask interpolation must be nearest")
        args.width, args.height = 512, 384
        args.weights = configured_path(config["model"]["checkpoint"], config_path)
        if not args.weights.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.weights}")
        yaml_output = configured_path(config["experiment"]["output_dir"], config_path)
        args.output_dir = (args.output_dir or yaml_output / "head_only_lr1e-3_wd1e-3_20ep").expanduser().resolve()
        args.seed = int(config["experiment"].get("seed", args.seed))
        configured_metric = config.get("training", {}).get("checkpoint_selection_metric", "validation_mean_iou")
        metric_names = {"validation_mean_iou": "mean_iou",
                        "validation_binary_water_iou": "binary_water_iou"}
        if configured_metric not in metric_names:
            raise ValueError(f"Unsupported validation selection metric: {configured_metric}")
        args.selection_metric = metric_names[configured_metric]
        # This single-run recipe uses the requested fixed settings; the YAML's
        # training block belongs to other experiment runners.
        args.learning_rate = FIXED_LEARNING_RATE
        args.weight_decay = FIXED_WEIGHT_DECAY
        args.epochs = FIXED_EPOCHS
        args.batch_size = FIXED_BATCH_SIZE
        args.class_weights = FIXED_CLASS_WEIGHTS
        args.config = config_path
        train_samples, validation_samples = splits["train"], splits["validation"]
        print("Validated split counts:", {role: len(samples) for role, samples in splits.items()})
    else:
        if args.train_images is None or args.train_masks is None:
            raise ValueError("Provide --config or both --train-images and --train-masks")
        if (args.val_images is None) != (args.val_masks is None):
            raise ValueError("Provide both --val-images and --val-masks, or neither")
        train_samples = find_samples(args.train_images, args.train_masks)
        validation_samples = (find_samples(args.val_images, args.val_masks)
                              if args.val_images else [])
        validate_dataset(train_samples)
        validate_dataset(validation_samples)
        splits = {"train": train_samples, "validation": validation_samples}
        args.weights = (args.weights or DEFAULT_WEIGHTS).expanduser().resolve()
        args.output_dir = (args.output_dir or DEFAULT_OUTPUT).expanduser().resolve()
        args.selection_metric = "mean_iou"
        config_path = None
        print(f"Validated {len(train_samples)} training pairs")
        print(f"Validated {len(validation_samples)} validation pairs")
    if args.check_data:
        return

    if config_path is not None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        manifest = {role: [{"image": str(sample.image_path), "mask": str(sample.mask_path)}
                           for sample in samples] for role, samples in splits.items()}
        (args.output_dir / "resolved_splits.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = train(args, train_samples, validation_samples)
    test_metrics = None
    if "test" in splits:
        from ewasr_mlx import create_head_finetuning_model

        selected_model = create_head_finetuning_model(result["best_checkpoint"])
        test_metrics = evaluate(selected_model, splits["test"], args)
    report = write_evaluation_text(
        args.output_dir, result, args=args, config_path=config_path,
        counts={role: len(samples) for role, samples in splits.items()},
        test_metrics=test_metrics, started_at=started_at,
        total_elapsed_seconds=time.perf_counter() - total_started,
        selection_metric=args.selection_metric,
    )
    print(f"Evaluation report: {report}")


if __name__ == "__main__":
    main()
