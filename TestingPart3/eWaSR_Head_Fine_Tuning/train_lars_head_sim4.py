"""Train eWaSR decoder.sim4 and decoder.seg_head on the matched LaSR Channel split."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from train_head import (
    DEFAULT_WEIGHTS,
    Sample,
    batches,
    configured_path,
    evaluate,
    flat_parameter_summary,
    load_configured_splits,
    validation_selection_score,
)
from train_head_sim4 import (
    add_directional_metrics,
    audit_frozen_checkpoint,
    check_batchnorm_modes,
    check_gradients,
    check_trainable_names,
    create_head_sim4_model,
    ensure_empty_output_dir,
    require_locked_file,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "lars_channel_head_sim4_v1.yaml"
SOURCE_YAML = (HERE / "../DataSource/head_only_lars_channel_staged_v1.yaml").resolve()
V2_MANIFEST = HERE.parents[1] / "results/manifests/lars_channel.json"
OUTPUT_DIR = HERE / "outputs/lars_channel_head_sim4_v1"
ROLES = ("train", "validation", "test", "unseen")
COUNTS = {"train": 175, "validation": 58, "test": 44, "unseen": 15}


def load_experiment_config(path: Path) -> tuple[dict, dict[str, Path]]:
    path = path.expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected a schema_version: 1 experiment YAML")
    source = config["split_source"]
    paths = {
        "config": path,
        "definition": configured_path(source["definition_yaml"], path),
        "manifest": configured_path(source["v2_resolved_manifest"], path),
        "checkpoint": configured_path(config["model"]["checkpoint"], path),
        "output": configured_path(config["experiment"]["output_dir"], path),
    }
    if (paths["definition"] != SOURCE_YAML or paths["manifest"] != V2_MANIFEST
        or paths["checkpoint"] != DEFAULT_WEIGHTS.resolve() or paths["output"] != OUTPUT_DIR):
        raise ValueError("LaSR source, V2 manifest, original checkpoint and new output path are fixed")
    require_locked_file(paths["definition"], source["definition_sha256"], "LaSR split YAML")
    require_locked_file(paths["manifest"], source["manifest_sha256"], "LaSR V2 resolved manifest")
    if not paths["checkpoint"].is_file():
        raise FileNotFoundError(f"Original pretrained checkpoint missing: {paths['checkpoint']}")
    if source.get("expected_counts") != COUNTS:
        raise ValueError(f"LaSR split counts must be {COUNTS}")
    model = config["model"]
    if (model.get("architecture") != "ewasr_resnet18" or model.get("imu") is not False
        or model.get("trainable_modules") != ["decoder.sim4", "decoder.seg_head"]
        or model.get("sim4_batchnorm_mode") != "eval"
        or model.get("seg_head_batchnorm_mode") != "train"):
        raise ValueError("Model must train only sim4 and seg_head with fixed sim4 BatchNorm statistics")
    preprocessing = config["preprocessing"]
    required_preprocessing = {
        "input_width": 512, "input_height": 384,
        "mask_width": 128, "mask_height": 96,
        "mask_interpolation": "nearest", "image_normalization": "imagenet",
        "image_mean": [.485, .456, .406], "image_std": [.229, .224, .225],
    }
    if any(preprocessing.get(key) != value for key, value in required_preprocessing.items()):
        raise ValueError("Preprocessing must match the 512x384 LaSR head-only run")
    training = config["training"]
    required_training = {
        "optimizer": "adamw",
        "learning_rates": {"decoder.seg_head": .001, "decoder.sim4": .00001},
        "weight_decay": .001, "epochs": 20, "batch_size": 2,
        "class_weights": [1, 1, 1], "loss": "cross_entropy",
        "checkpoint_selection_metric": "validation_binary_water_iou",
    }
    if any(training.get(key) != value for key, value in required_training.items()):
        raise ValueError("Training settings must match the completed Pohang head + sim4 recipe")
    return config, paths


def assert_matching_v2_manifest(
    splits: dict[str, list[Sample]], manifest: dict, expected_counts: dict[str, int] = COUNTS
) -> None:
    """Reject any difference between the YAML's role membership and V2's."""
    if not isinstance(manifest, dict) or set(splits) != set(manifest) or set(splits) != set(ROLES):
        raise ValueError("LaSR and V2 must both contain train, validation, test and unseen")
    for role in ROLES:
        entries = manifest[role]
        samples = splits[role]
        if (not isinstance(entries, list) or len(entries) != expected_counts[role]
            or len(samples) != expected_counts[role]):
            raise ValueError(f"LaSR {role} count differs from the expected V2 count")
        recorded = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"image", "mask"}:
                raise ValueError(f"Malformed V2 {role} manifest entry")
            recorded.append(tuple(
                (Path(entry[key]) if Path(entry[key]).is_absolute() else
                 HERE.parents[1] / entry[key]).resolve() for key in ("image", "mask")
            ))
        selected = [(sample.image_path.resolve(), sample.mask_path.resolve()) for sample in samples]
        if (len(set(recorded)) != len(recorded) or len(set(selected)) != len(selected)
            or set(recorded) != set(selected)):
            raise ValueError(f"LaSR {role} membership does not match head-only V2")


def load_matching_splits(paths: dict[str, Path]):
    _, splits = load_configured_splits(paths["definition"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert_matching_v2_manifest(splits, manifest)
    return splits, manifest


def train_and_evaluate(config: dict, paths: dict[str, Path], splits, manifest) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    if not mx.metal.is_available():
        raise RuntimeError("MLX cannot access an Apple Metal GPU")
    ensure_empty_output_dir(paths["output"])
    started_at = datetime.now(timezone.utc)
    total_started = time.perf_counter()
    model = create_head_sim4_model(paths["checkpoint"])
    with np.load(paths["checkpoint"]) as original:
        all_names = list(original.files)
    trainable_names, trainable_count = flat_parameter_summary(model.trainable_parameters())
    check_trainable_names(trainable_names, all_names)
    expected_names = set(trainable_names)
    training = config["training"]
    optimizer = optim.MultiOptimizer(
        [optim.AdamW(learning_rate=training["learning_rates"]["decoder.sim4"],
                     weight_decay=training["weight_decay"], bias_correction=True),
         optim.AdamW(learning_rate=training["learning_rates"]["decoder.seg_head"],
                     weight_decay=training["weight_decay"], bias_correction=True)],
        [lambda name, _: name.startswith("decoder.sim4.")],
    )

    def loss_fn(images, targets):
        return mx.mean(nn.losses.cross_entropy(model(images), targets, reduction="none"))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    args = SimpleNamespace(batch_size=2, width=512, height=384)
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_splits.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    settings = {
        "experiment": config["experiment"], "training": training,
        "preprocessing": config["preprocessing"], "model": config["model"],
        "config_path": str(paths["config"]),
        "split_definition_yaml": str(paths["definition"]),
        "split_definition_sha256": config["split_source"]["definition_sha256"],
        "v2_manifest": str(paths["manifest"]),
        "v2_manifest_sha256": config["split_source"]["manifest_sha256"],
        "pretrained_checkpoint": str(paths["checkpoint"]),
        "pretrained_checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "split_counts": {role: len(samples) for role, samples in splits.items()},
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "started_at_utc": started_at.isoformat(),
    }
    (output / "run_config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Training pairs: {len(splits['train'])}; validation pairs: {len(splits['validation'])}", flush=True)
    print(f"Trainable parameters: {trainable_count:,}", flush=True)

    rng = np.random.default_rng(config["experiment"]["seed"])
    history_path = output / "training_history.jsonl"
    best_path = output / "best_head_sim4.npz"
    best_score, best_epoch, best_record = -np.inf, None, None
    checked_gradients = False
    training_started = time.perf_counter()
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, training["epochs"] + 1):
            epoch_started = time.perf_counter()
            model.set_head_training_mode(True)
            check_batchnorm_modes(model, head_training=True)
            losses = []
            for images, masks in batches(splits["train"], batch_size=2, width=512,
                                          height=384, rng=rng):
                loss, gradients = loss_and_grad(mx.array(images), mx.array(masks))
                if not checked_gradients:
                    mx.eval(loss, gradients)
                    check_gradients(gradients, expected_names)
                    checked_gradients = True
                optimizer.update(model, gradients)
                mx.eval(model.parameters(), optimizer.state, loss)
                losses.append(float(loss.item()))
            validation = add_directional_metrics(evaluate(model, splits["validation"], args))
            check_batchnorm_modes(model, head_training=True)
            record = {"epoch": epoch, "training_loss": float(np.mean(losses)),
                      "validation": validation}
            score = validation_selection_score(record, "binary_water_iou")
            if score > best_score:
                best_score, best_epoch, best_record = score, epoch, record
                model.save_weights(str(best_path))
            record["epoch_elapsed_seconds"] = time.perf_counter() - epoch_started
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            print(json.dumps(record), flush=True)
    training_seconds = time.perf_counter() - training_started
    if best_epoch is None:
        raise RuntimeError("No validation checkpoint was selected")
    frozen_audit = audit_frozen_checkpoint(paths["checkpoint"], best_path)
    (output / "frozen_state_audit.json").write_text(
        json.dumps(frozen_audit, indent=2) + "\n", encoding="utf-8")

    selected_model = create_head_sim4_model(best_path)
    reloaded_names, _ = flat_parameter_summary(selected_model.trainable_parameters())
    check_trainable_names(reloaded_names, all_names)
    test_metrics = add_directional_metrics(evaluate(selected_model, splits["test"], args))
    check_batchnorm_modes(selected_model, head_training=True)
    results = {
        "best_epoch": best_epoch,
        "selection_metric": "validation_binary_water_iou",
        "best_validation": best_record["validation"],
        "test": {"pairs": len(splits["test"]), "metrics": test_metrics},
        "unseen": "reserved; pixels not read or evaluated",
        "best_checkpoint": str(best_path),
        "training_elapsed_seconds": training_seconds,
        "total_elapsed_seconds": time.perf_counter() - total_started,
        "trainable_parameter_names": trainable_names,
        "frozen_state_audit": frozen_audit,
    }
    (output / "experiment_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "LaSR Channel eWaSR decoder.sim4 + decoder.seg_head fine-tuning",
        f"Started (UTC): {started_at.isoformat()}",
        f"Finished (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Split source: {paths['definition']}",
        f"Matched head-only V2 manifest: {paths['manifest']}",
        f"Split counts: {settings['split_counts']}",
        f"Original pretrained checkpoint: {paths['checkpoint']}",
        "Optimizer: AdamW; head LR 0.001; sim4 LR 0.00001; decay 0.001",
        "Epochs: 20; batch size: 2; class weights: 1/1/1",
        "Input: 512x384; nearest-neighbour 128x96 masks; cross-entropy",
        "sim4 BatchNorm: fixed pretrained running statistics; head BatchNorm: training statistics",
        f"Trainable parameters ({trainable_count}): {', '.join(trainable_names)}",
        f"Best epoch: {best_epoch}",
        f"Training time: {training_seconds:.2f} seconds",
        f"Total time: {results['total_elapsed_seconds']:.2f} seconds",
        "Metrics measure agreement with SAM3 pseudo-labels.",
        "",
    ]
    for label, metrics in (("Best validation", best_record["validation"]),
                           ("Selected checkpoint on test", test_metrics)):
        lines.extend([
            f"{label}:",
            f"  water IoU: {metrics['binary_water_iou']:.6f}",
            f"  three-class mean IoU: {metrics['mean_iou']:.6f}",
            f"  water predicted as obstacle: {metrics['water_predicted_as_obstacle_count']} "
            f"({metrics['water_predicted_as_obstacle_rate']:.6%} of water pixels)",
            f"  obstacle missed as water: {metrics['obstacle_missed_as_water_count']} "
            f"({metrics['obstacle_missed_as_water_rate']:.6%} of obstacle pixels)",
            f"  confusion matrix (rows=true, columns=predicted): {metrics['confusion_matrix']}",
            "",
        ])
    lines.append("Unseen split: reserved; not evaluated.")
    (output / "evaluation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluation report: {output / 'evaluation.txt'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-data", action="store_true",
                        help="Check exact LaSR/V2 membership without MLX or output writes")
    args = parser.parse_args()
    config, paths = load_experiment_config(args.config)
    ensure_empty_output_dir(paths["output"])
    splits, manifest = load_matching_splits(paths)
    print("Validated LaSR/V2 membership:", {role: len(splits[role]) for role in ROLES}, flush=True)
    print("Unseen pixels were not read.", flush=True)
    if args.check_data:
        return
    train_and_evaluate(config, paths, splits, manifest)


if __name__ == "__main__":
    main()
