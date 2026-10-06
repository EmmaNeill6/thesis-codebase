"""Separate channel-only experiment using the dock experiment's random split.

Reuse its seeded random assignment and overlap audit, and the shared head-only
training/LR-comparison implementation. Earlier channel and dock outputs are
not used as initial weights or overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from run_experiment import load_config, run, serializable, split_manifest, validate_configuration
from run_tampere_dock import SPLIT_COUNTS, audit_split_independence, random_image_splits
from train_head import SUPPORTED_IMAGE_EXTENSIONS, find_samples, validate_dataset


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_CONFIG = HERE.parent / "DataSource/head_only_tampere_channel_random_v1.yaml"


def prepare_splits(config):
    validate_configuration(config)
    if config["split_policy"]["method"] != "random_images":
        raise ValueError("This channel experiment requires the dock run's random image split")
    images_dir = (PROJECT / "data/Tampere-WaterSeg/images/channel").resolve()
    masks_dir = (HERE.parent / "SAM3_Pseudo_Labels_Generator/output/Tampere_Channel_5ppt/masks").resolve()
    for role, count in SPLIT_COUNTS.items():
        spec = config["data"][role]
        if (Path(spec["images_dir"]).expanduser().resolve() != images_dir or
                Path(spec["masks_dir"]).expanduser().resolve() != masks_dir):
            raise ValueError("Use only Tampere channel images and Tampere_Channel_5ppt masks")
        if int(spec["expected_count"]) != count:
            raise ValueError(f"{role} must specify {count} channel pairs")
    samples = find_samples(images_dir, masks_dir)
    names = {path.name for path in images_dir.iterdir()
             if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS}
    if len(names) != 200 or {sample.image_path.name for sample in samples} != names:
        raise ValueError("Exactly 200 fully paired channel images are required")
    validate_dataset(samples)
    splits = random_image_splits(samples, int(config["split_policy"]["seed"]))
    audit_split_independence(splits)
    return splits


def resolved_config(config, splits):
    """Persist actual membership for existing evaluation/visualization scripts."""
    resolved = copy.deepcopy(config)
    for role, samples in splits.items():
        resolved["data"][role]["sources"] = [
            {"source_id": sample.image_path.stem,
             "images_dir": str(sample.image_path.parent),
             "masks_dir": str(sample.mask_path.parent), "expected_count": 1,
             "selection": {"kind": "inclusive_range",
                           "index_pattern": r"^channel_image_(\d+)$",
                           "start": int(sample.image_path.stem.rsplit("_", 1)[1]),
                           "end": int(sample.image_path.stem.rsplit("_", 1)[1])}}
            for sample in samples
        ]
    return resolved


def verify_completed_training(config, output):
    results = json.loads((output / "experiment_results.json").read_text())
    checks = []
    with np.load(config["model"]["checkpoint"]) as pretrained:
        frozen = [key for key in pretrained.files if not key.startswith("decoder.seg_head.")]
        for trial in results["trials"]:
            folder = Path(trial["best_checkpoint"]).parent
            history = [json.loads(line) for line in (folder / "training_history.jsonl").read_text().splitlines()]
            assert [row["epoch"] for row in history] == list(range(1, config["training"]["epochs"] + 1))
            best = max(history, key=lambda row: row["validation"]["mean_iou"])
            assert trial["best_epoch"] == best["epoch"]
            assert trial["best_validation"] == best["validation"]
            run_config = json.loads((folder / "run_config.json").read_text())
            assert run_config["trainable_parameter_count"] == 67075
            assert len(run_config["trainable_parameter_names"]) == 6
            for filename in ("best_head_only.npz", "latest_head_only.npz"):
                with np.load(folder / filename) as trained:
                    assert set(pretrained.files) == set(trained.files)
                    changed = [key for key in pretrained.files
                               if not np.array_equal(pretrained[key], trained[key])]
                    assert changed and all(key.startswith("decoder.seg_head.") for key in changed)
                    assert all(key in changed for key in run_config["trainable_parameter_names"])
                    checks.append({"checkpoint": str((folder / filename).resolve()),
                                   "unchanged_non_head_tensor_count": len(frozen),
                                   "changed_tensor_names": changed})
    selected_trial = max(results["trials"], key=lambda trial: trial["best_validation"]["mean_iou"])
    assert results["selected"]["learning_rate"] == selected_trial["learning_rate"]
    (output / "frozen_parameter_verification.json").write_text(
        json.dumps({"all_checks_passed": True, "checkpoints_checked": checks}, indent=2) + "\n",
        encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    splits = prepare_splits(config)
    for role, samples in splits.items():
        print(f"{role}: {len(samples)} matched channel pairs", flush=True)
    print("Verified: no shared image/mask paths or identical image files across splits.", flush=True)
    if args.check_data:
        return
    output = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output_dir; earlier files already exist at {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_audit.json").write_text(
        json.dumps(audit_split_independence(splits), indent=2) + "\n", encoding="utf-8")
    (output / "resolved_splits.json").write_text(
        json.dumps({role: split_manifest(samples) for role, samples in splits.items()}, indent=2),
        encoding="utf-8")
    (output / "resolved_experiment.yaml").write_text(
        yaml.safe_dump(serializable(resolved_config(config, splits)), sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    run(config, splits)
    verify_completed_training(config, output)
    provenance = (
        "\nCHANNEL-ONLY RANDOM SPLIT PROVENANCE\n"
        "200 channel pairs; seed 4022; fixed 120 train / 40 validation / 40 test.\n"
        "Only channel images and Tampere_Channel_5ppt masks are used.\n"
        "File-level separation verified; nearby frames may still correlate.\n"
        "Input resolution 512x384; reported evaluation resolution 128x96.\n"
        "Six checkpoints verified: only segmentation-head tensors and head\n"
        "BatchNorm running state changed; all other tensors stayed identical.\n"
    )
    summary = output / "experiment_summary.txt"
    summary.write_text(summary.read_text() + provenance, encoding="utf-8")
    print("All completed histories, checkpoint selection and frozen tensors verified.", flush=True)


if __name__ == "__main__":
    main()
