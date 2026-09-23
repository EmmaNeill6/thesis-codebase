"""Run the YAML-defined head-only eWaSR learning-rate experiment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from train_head import Sample, evaluate, find_samples, train, validate_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR.parent / "DataSource/head_only_finetuning_v1.yaml"


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Experiment YAML not found: {path}")
    with path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    config = require_mapping(config, "configuration")
    if config.get("schema_version") != 1:
        raise ValueError("Only schema_version: 1 is supported")
    config["_config_path"] = path
    return config


def extract_index(sample: Sample, pattern: re.Pattern[str], split_name: str) -> int:
    match = pattern.fullmatch(sample.image_path.stem)
    if match is None or len(match.groups()) != 1:
        raise ValueError(
            f"{split_name} filename does not match its one-group index_pattern: "
            f"{sample.image_path.name}"
        )
    return int(match.group(1))


def select_samples(
    samples: list[Sample], selection: dict[str, Any], split_name: str
) -> list[Sample]:
    kind = selection.get("kind")
    if kind == "all":
        return sorted(samples, key=lambda sample: str(sample.image_path))

    pattern = re.compile(str(selection["index_pattern"]))
    indexed = [(extract_index(sample, pattern, split_name), sample) for sample in samples]
    start = int(selection["start"])
    end = int(selection["end"])
    if start > end:
        raise ValueError(f"{split_name} selection start must not exceed end")

    if kind == "inclusive_range":
        selected = [sample for index, sample in indexed if start <= index <= end]
    elif kind == "every_nth":
        step = int(selection["step"])
        if step <= 0:
            raise ValueError(f"{split_name} every_nth step must be positive")
        selected = [
            sample
            for index, sample in indexed
            if start <= index <= end and (index - start) % step == 0
        ]
    else:
        raise ValueError(f"Unsupported {split_name} selection kind: {kind!r}")

    selected.sort(key=lambda sample: extract_index(sample, pattern, split_name))
    return selected


def load_split(config: dict[str, Any], split_name: str) -> list[Sample]:
    split = require_mapping(config["data"][split_name], f"data.{split_name}")
    # ``sources`` supports an explicit collection of separate recordings.  The
    # single-source form remains supported for the first Tampere experiment.
    sources = split.get("sources")
    if sources is None:
        sources = [split]
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"data.{split_name}.sources must be a non-empty list")

    selected: list[Sample] = []
    for source_index, source_value in enumerate(sources):
        source = require_mapping(
            source_value, f"data.{split_name}.sources[{source_index}]"
        )
        source_name = str(source.get("source_id", f"source_{source_index + 1}"))
        samples = find_samples(
            Path(source["images_dir"]),
            Path(source["masks_dir"]),
            # LaRS keeps masks from three source folders in one flat directory.
            # In that case the images directory defines the source membership.
            ignore_masks_without_images=bool(source.get("shared_masks_dir", False)),
        )
        source_selected = select_samples(
            samples,
            require_mapping(
                source["selection"],
                f"data.{split_name}.{source_name}.selection",
            ),
            f"{split_name}.{source_name}",
        )
        expected_count = int(source.get("expected_count", len(source_selected)))
        if len(source_selected) != expected_count:
            raise ValueError(
                f"{split_name}.{source_name} selected {len(source_selected)} pairs; "
                f"expected {expected_count}"
            )
        validate_dataset(source_selected)
        selected.extend(source_selected)

    split_expected_count = split.get("expected_count")
    if split_expected_count is not None and len(selected) != int(split_expected_count):
        raise ValueError(
            f"{split_name} selected {len(selected)} pairs; "
            f"expected {int(split_expected_count)}"
        )
    return selected


def validate_configuration(config: dict[str, Any]) -> None:
    labels = require_mapping(config.get("labels"), "labels")
    if labels != {"obstacle": 0, "water": 1, "sky": 2}:
        raise ValueError("labels must be obstacle=0, water=1, sky=2")

    model = require_mapping(config.get("model"), "model")
    if model.get("imu") is not False:
        raise ValueError("This experiment requires the non-IMU model")
    if model.get("trainable_module") != "decoder.seg_head":
        raise ValueError("Only decoder.seg_head may be trainable")

    preprocessing = require_mapping(config.get("preprocessing"), "preprocessing")
    width = int(preprocessing["input_width"])
    height = int(preprocessing["input_height"])
    if (width, height) != (512, 384):
        raise ValueError("This first experiment requires a 512 x 384 input")
    if (int(preprocessing["mask_width"]), int(preprocessing["mask_height"])) != (
        width // 4,
        height // 4,
    ):
        raise ValueError("Mask size must be one quarter of the input dimensions")
    if preprocessing.get("mask_interpolation") != "nearest":
        raise ValueError("Mask interpolation must be nearest")

    training = require_mapping(config.get("training"), "training")
    if str(training.get("optimizer")).casefold() != "adamw":
        raise ValueError("Only AdamW is supported by this experiment runner")
    if training.get("loss") != "cross_entropy":
        raise ValueError("This first experiment requires standard cross-entropy")
    if training.get("class_weights") is not None:
        raise ValueError("class_weights must be null for standard cross-entropy")
    learning_rates = [float(value) for value in training["learning_rates"]]
    if float(training["initial_learning_rate"]) not in learning_rates:
        raise ValueError("initial_learning_rate must be included in learning_rates")


def split_manifest(samples: list[Sample]) -> list[dict[str, str]]:
    return [
        {"image": str(sample.image_path), "mask": str(sample.mask_path)}
        for sample in samples
    ]


def ensure_disjoint(splits: dict[str, list[Sample]]) -> None:
    names = list(splits)
    for index, left_name in enumerate(names):
        left = {sample.image_path.resolve() for sample in splits[left_name]}
        for right_name in names[index + 1 :]:
            right = {sample.image_path.resolve() for sample in splits[right_name]}
            overlap = left & right
            if overlap:
                preview = ", ".join(path.name for path in sorted(overlap)[:5])
                raise ValueError(
                    f"{left_name} and {right_name} overlap: {preview}"
                )


def training_arguments(
    config: dict[str, Any], learning_rate: float, output_dir: Path
) -> SimpleNamespace:
    training_config = config["training"]
    preprocessing = config["preprocessing"]
    model = config["model"]
    data = config["data"]

    def first_source(split_name: str) -> dict[str, Any]:
        split = data[split_name]
        return split.get("sources", [split])[0]

    train_source = first_source("train")
    validation_source = first_source("validation")
    return SimpleNamespace(
        # The actual source lists are passed directly to train(); these paths
        # remain in the saved run arguments for provenance and CLI parity.
        train_images=Path(train_source["images_dir"]),
        train_masks=Path(train_source["masks_dir"]),
        val_images=Path(validation_source["images_dir"]),
        val_masks=Path(validation_source["masks_dir"]),
        weights=Path(model["checkpoint"]).expanduser().resolve(),
        output_dir=output_dir,
        epochs=int(training_config["epochs"]),
        batch_size=int(training_config["batch_size"]),
        learning_rate=float(learning_rate),
        weight_decay=float(training_config["weight_decay"]),
        class_weights=None,
        width=int(preprocessing["input_width"]),
        height=int(preprocessing["input_height"]),
        seed=int(config["experiment"]["seed"]),
        check_data=False,
    )


def metric_text(metrics: dict[str, Any], indent: str = "") -> list[str]:
    class_iou = metrics["class_iou"]
    return [
        f"{indent}loss: {metrics['loss']:.6f}",
        f"{indent}pixel accuracy: {metrics['pixel_accuracy']:.6f}",
        f"{indent}three-class mean IoU: {metrics['mean_iou']:.6f}",
        f"{indent}obstacle IoU: {class_iou['obstacle']:.6f}",
        f"{indent}water IoU: {class_iou['water']:.6f}",
        f"{indent}sky IoU: {class_iou['sky']:.6f}",
        f"{indent}binary water IoU: {metrics['binary_water_iou']:.6f}",
        f"{indent}binary non-water IoU: {metrics['binary_non_water_iou']:.6f}",
        f"{indent}binary mean IoU: {metrics['binary_mean_iou']:.6f}",
    ]


def serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def write_summary(
    output_dir: Path,
    config: dict[str, Any],
    splits: dict[str, list[Sample]],
    baseline_validation: dict[str, Any],
    trials: list[dict[str, Any]],
    selected: dict[str, Any],
    baseline_test: dict[str, Any],
    adapted_test: dict[str, Any],
) -> None:
    lines = [
        "HEAD-ONLY EWASR FINE-TUNING — EXPERIMENT SUMMARY",
        "=================================================",
        "",
        f"Configuration: {config['_config_path']}",
        f"Training pairs: {len(splits['train'])}",
        f"Validation pairs: {len(splits['validation'])}",
        f"Test pairs: {len(splits['test'])}",
        "",
        "Important interpretation:",
        "Validation and test masks are SAM3 pseudo-labels. These metrics measure",
        "agreement with SAM3 and are not a substitute for final human-ground-truth",
        "evaluation.",
        "",
        "Pretrained baseline on validation:",
    ]
    lines.extend(metric_text(baseline_validation, "  "))
    lines.extend(["", "Learning-rate trials (best validation epoch):"])
    for trial in trials:
        lines.extend(
            [
                "",
                f"  learning rate: {trial['learning_rate']:.8g}",
                f"  best epoch: {trial['best_epoch']}",
            ]
        )
        lines.extend(metric_text(trial["best_validation"], "    "))
    lines.extend(
        [
            "",
            f"Selected learning rate: {selected['learning_rate']:.8g}",
            f"Selected best epoch: {selected['best_epoch']}",
            f"Selected checkpoint: {selected['best_checkpoint']}",
            "Selection criterion: validation three-class mean IoU",
            "",
            "Pretrained baseline on held-out test:",
        ]
    )
    lines.extend(metric_text(baseline_test, "  "))
    lines.extend(["", "Selected adapted model on held-out test:"])
    lines.extend(metric_text(adapted_test, "  "))
    lines.extend(
        [
            "",
            "Test deltas (adapted minus pretrained):",
            f"  three-class mean IoU: {adapted_test['mean_iou'] - baseline_test['mean_iou']:+.6f}",
            f"  binary water IoU: {adapted_test['binary_water_iou'] - baseline_test['binary_water_iou']:+.6f}",
            f"  pixel accuracy: {adapted_test['pixel_accuracy'] - baseline_test['pixel_accuracy']:+.6f}",
            "",
            "Do not use these held-out test results to revise hyperparameters. Run the",
            "chosen model once more against independent human ground truth for the",
            "final performance claim.",
        ]
    )
    (output_dir / "experiment_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(config: dict[str, Any], splits: dict[str, list[Sample]]) -> None:
    import mlx.core as mx

    from ewasr_mlx import create_head_finetuning_model

    if not mx.metal.is_available():
        raise RuntimeError("MLX cannot access an Apple Metal GPU")

    output_dir = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {name: split_manifest(samples) for name, samples in splits.items()}
    (output_dir / "resolved_splits.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    reference_args = training_arguments(
        config,
        float(config["training"]["initial_learning_rate"]),
        output_dir / "baseline",
    )
    baseline_model = create_head_finetuning_model(reference_args.weights)
    baseline_validation = evaluate(
        baseline_model, splits["validation"], reference_args
    )
    del baseline_model

    trials = []
    for learning_rate in [float(value) for value in config["training"]["learning_rates"]]:
        trial_name = f"lr_{learning_rate:.0e}".replace("+", "")
        args = training_arguments(config, learning_rate, output_dir / trial_name)
        print(f"\n=== Learning-rate trial: {learning_rate:.8g} ===")
        result = train(args, splits["train"], splits["validation"])
        best_validation = result["best_record"]["validation"]
        trials.append(
            {
                "learning_rate": learning_rate,
                "best_epoch": result["best_epoch"],
                "best_checkpoint": result["best_checkpoint"],
                "best_validation": best_validation,
                "trainable_parameter_count": result["trainable_parameter_count"],
            }
        )

    selected = max(trials, key=lambda trial: trial["best_validation"]["mean_iou"])

    baseline_model = create_head_finetuning_model(reference_args.weights)
    baseline_test = evaluate(baseline_model, splits["test"], reference_args)
    del baseline_model

    selected_args = training_arguments(
        config,
        float(selected["learning_rate"]),
        Path(selected["best_checkpoint"]).parent,
    )
    adapted_model = create_head_finetuning_model(Path(selected["best_checkpoint"]))
    adapted_test = evaluate(adapted_model, splits["test"], selected_args)
    del adapted_model

    selected_model = {
        "learning_rate": selected["learning_rate"],
        "best_epoch": selected["best_epoch"],
        "checkpoint": selected["best_checkpoint"],
        "selection_metric": "validation_mean_iou",
        "validation_metrics": selected["best_validation"],
        "test_metrics_against_sam3_pseudo_labels": adapted_test,
    }
    (output_dir / "selected_model.json").write_text(
        json.dumps(serializable(selected_model), indent=2), encoding="utf-8"
    )
    complete_results = {
        "baseline_validation": baseline_validation,
        "trials": trials,
        "selected": selected_model,
        "baseline_test": baseline_test,
        "adapted_test": adapted_test,
    }
    (output_dir / "experiment_results.json").write_text(
        json.dumps(serializable(complete_results), indent=2), encoding="utf-8"
    )
    write_summary(
        output_dir,
        config,
        splits,
        baseline_validation,
        trials,
        selected,
        baseline_test,
        adapted_test,
    )
    print(f"\nExperiment summary: {output_dir / 'experiment_summary.txt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the YAML-defined head-only eWaSR experiment."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Validate configuration, split selection, pairs and masks only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_configuration(config)
    splits = {
        "train": load_split(config, "train"),
        "validation": load_split(config, "validation"),
        "test": load_split(config, "test"),
    }
    ensure_disjoint(splits)
    for name, samples in splits.items():
        print(
            f"{name}: {len(samples)} pairs "
            f"({samples[0].image_path.name} ... {samples[-1].image_path.name})"
        )
    if args.check_data:
        print("Configuration and all selected data passed validation.")
        return
    run(config, splits)


if __name__ == "__main__":
    main()
