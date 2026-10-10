"""Evaluate preselected staged checkpoints once on their recorded test splits.

This script reads saved stage selections. It never chooses a model using test
metrics and never touches the separately reserved ``unseen`` split.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from train_head import Sample, evaluate, validate_dataset


HERE = Path(__file__).resolve().parent
DATASETS = {
    "catarob_staged_v1": "CataRob",
    "lars_channel_staged_v1": "LaSR Channel",
    "pohang_canal_pier_staged_v1": "Pohang Canal + Pier",
    "tampere_dock_staged_v1": "Tampere Dock",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_test_samples(splits: dict) -> list[Sample]:
    for role in ("train", "validation", "test", "unseen"):
        if not splits.get(role):
            raise ValueError(f"Recorded {role} split is empty")
    role_images = {
        role: {Path(item["image"]).resolve() for item in rows}
        for role, rows in splits.items()
    }
    role_masks = {
        role: {Path(item["mask"]).resolve() for item in rows}
        for role, rows in splits.items()
    }
    for first in role_images:
        for second in role_images:
            if first >= second:
                continue
            if role_images[first] & role_images[second] or role_masks[first] & role_masks[second]:
                raise ValueError(f"Recorded {first} and {second} splits overlap")
    samples = [
        Sample(Path(item["image"]).resolve(), Path(item["mask"]).resolve())
        for item in splits["test"]
    ]
    validate_dataset(samples)
    return samples


def evaluate_one(directory: Path, label: str) -> dict:
    from ewasr_mlx import create_head_finetuning_model

    state_path = directory / "stage_state.json"
    split_path = directory / "resolved_splits.json"
    config_path = directory / "resolved_config.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if state["stages"]["6"]["status"] != "complete":
        raise ValueError(f"Staged selection is incomplete: {directory}")
    samples = recorded_test_samples(splits)
    args = SimpleNamespace(
        batch_size=2,
        width=int(config["preprocessing"]["input_width"]),
        height=int(config["preprocessing"]["input_height"]),
    )
    selections = {
        "pretrained": Path(config["model"]["checkpoint"]).resolve(),
        "baseline": Path(state["stages"]["1"]["selected"]["checkpoint"]).resolve(),
        "staged_final": Path(state["stages"]["6"]["selected"]["checkpoint"]).resolve(),
    }
    result = {
        "dataset": label,
        "experiment_dir": str(directory),
        "test_count": len(samples),
        "test_split_sha256": digest(split_path),
        "stage_state_sha256": digest(state_path),
        "reference": "SAM3 pseudo-label masks at 128x96 head resolution",
        "checkpoints": {},
    }
    for name, checkpoint in selections.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = create_head_finetuning_model(checkpoint)
        metrics = evaluate(model, samples, args)
        result["checkpoints"][name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "selected_epoch": None if name == "pretrained" else
                state["stages"]["1" if name == "baseline" else "6"]["selected"]["best_epoch"],
            "validation_water_iou": state["pretrained_validation"]["binary_water_iou"]
                if name == "pretrained" else
                state["stages"]["1" if name == "baseline" else "6"]["selected"]["best_validation"]["binary_water_iou"],
            "test": metrics,
        }
        del model
        gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "Single recorded test split per dataset; no test-based model selection",
        "datasets": [],
    }
    rows = []
    for name, label in DATASETS.items():
        item = evaluate_one(HERE / "outputs" / name, label)
        results["datasets"].append(item)
        for model_name, record in item["checkpoints"].items():
            metrics = record["test"]
            rows.append({
                "dataset": label,
                "test_images": item["test_count"],
                "model": model_name,
                "selected_epoch": record["selected_epoch"],
                "validation_water_iou": record["validation_water_iou"],
                "test_water_iou": metrics["binary_water_iou"],
                "test_loss": metrics["loss"],
                "test_three_class_mean_iou": metrics["mean_iou"],
                "test_water_as_obstacle_rate": metrics["water_as_obstacle_rate"],
            })
        print(f"{label}: evaluated {item['test_count']} reserved test images", flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
