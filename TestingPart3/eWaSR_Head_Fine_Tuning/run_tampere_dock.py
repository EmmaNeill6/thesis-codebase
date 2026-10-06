"""Run the separate dock experiment with fixed, disjoint dataset splits.

Random images use a seeded 60/20/20 assignment. Optional recording CSVs supply
image_name,sequence_id,split. No trip boundaries are inferred from filenames.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
from pathlib import Path

from run_experiment import (
    ensure_disjoint, load_config, load_split, run, serializable,
    split_manifest, validate_configuration,
)
from train_head import SUPPORTED_IMAGE_EXTENSIONS, find_samples, validate_dataset


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent / "DataSource/head_only_tampere_dock_v1.yaml"
SPLIT_COUNTS = {"train": 120, "validation": 40, "test": 40}


def random_image_splits(samples, seed):
    """Randomize membership once; keep it identical across all LR trials."""
    ordered = sorted(samples, key=lambda sample: str(sample.image_path))
    if len(ordered) != sum(SPLIT_COUNTS.values()):
        raise ValueError("Random dock split requires exactly 200 samples")
    random.Random(seed).shuffle(ordered)
    splits = {}
    start = 0
    for role, count in SPLIT_COUNTS.items():
        splits[role] = sorted(ordered[start:start + count],
                              key=lambda sample: str(sample.image_path))
        start += count
    ensure_disjoint(splits)
    return splits


def audit_split_independence(splits):
    """Reject reused image/mask paths and identical image bytes across roles.

    This establishes file-level separation, not statistical independence of
    nearby frames or recordings with missing trip metadata.
    """
    ensure_disjoint(splits)
    mask_roles, image_hash_roles, fingerprints = {}, {}, []
    for role, samples in splits.items():
        for sample in samples:
            mask_path = sample.mask_path.resolve()
            if mask_path in mask_roles and mask_roles[mask_path] != role:
                raise ValueError(f"Mask reused across splits: {mask_path}")
            mask_roles[mask_path] = role
            digest = hashlib.sha256(sample.image_path.read_bytes()).hexdigest()
            if digest in image_hash_roles and image_hash_roles[digest] != role:
                raise ValueError(f"Identical image bytes cross split boundaries: {sample.image_path}")
            image_hash_roles[digest] = role
            fingerprints.append({"image_name": sample.image_path.name,
                                 "split": role, "image_sha256": digest})
    return {
        "image_path_overlap": 0, "mask_path_overlap": 0,
        "identical_image_bytes_across_splits": 0,
        "independence_scope": "disjoint files and no identical image bytes across splits",
        "recording_independence_verified": False,
        "limitation": "Trip IDs unavailable; nearby frames may share scene content across splits.",
        "image_fingerprints": fingerprints,
    }


def read_sequence_assignments(path: Path, image_names: set[str]) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Recording manifest not found: {path}. Supply a CSV with columns "
            "image_name,sequence_id,split. Local dock metadata has no trip IDs; "
            "chronological frame blocks are not verified separate recordings."
        )
    assignments = {}
    sequence_roles = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not {"image_name", "sequence_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires image_name,sequence_id,split columns")
        for row in reader:
            name, sequence, role = (row[key].strip() for key in
                                    ("image_name", "sequence_id", "split"))
            if name not in image_names or name in assignments:
                raise ValueError(f"Unknown or duplicate manifest image: {name}")
            if not sequence or role not in SPLIT_COUNTS:
                raise ValueError(f"Invalid sequence or split for {name}")
            if sequence in sequence_roles and sequence_roles[sequence] != role:
                raise ValueError(f"Recording {sequence} crosses split boundaries")
            sequence_roles[sequence] = role
            assignments[name] = role
    missing = image_names - assignments.keys()
    if missing:
        raise ValueError(f"Recording manifest omits {len(missing)} dock images")
    for role, count in SPLIT_COUNTS.items():
        actual = sum(value == role for value in assignments.values())
        if actual != count:
            raise ValueError(f"{role}: {actual} images, expected {count}. Whole recordings "
                             "cannot be cut to force 60/20/20; revise the requested "
                             "ratio or provide compatible recording groups.")
    return assignments


def prepare_splits(config):
    validate_configuration(config)
    policy = config["split_policy"]
    first = config["data"]["train"]
    images_dir = Path(first["images_dir"]).expanduser().resolve()
    masks_dir = Path(first["masks_dir"]).expanduser().resolve()
    expected_images = (HERE.parents[1] / "data/Tampere-WaterSeg/images/dock").resolve()
    if images_dir != expected_images:
        raise ValueError("This separate experiment accepts Tampere dock images only")
    expected_masks = (HERE.parent / "SAM3_Pseudo_Labels_Generator/output/Tampere-Dock_5ppt/masks").resolve()
    if masks_dir != expected_masks:
        raise ValueError("Dock training must use the confirmed Tampere-Dock_5ppt masks only")
    for role in SPLIT_COUNTS:
        spec = config["data"][role]
        if (Path(spec["images_dir"]).resolve() != images_dir or
                Path(spec["masks_dir"]).resolve() != masks_dir):
            raise ValueError("All dock splits must share the same image/mask source")
    all_samples = find_samples(images_dir, masks_dir)
    image_names = {p.name for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS}
    if len(image_names) != 200 or {s.image_path.name for s in all_samples} != image_names:
        raise ValueError("The dock experiment requires exactly 200 fully paired images")
    validate_dataset(all_samples)
    if policy["method"] == "random_images":
        splits = random_image_splits(all_samples, int(policy["seed"]))
    elif policy["method"] == "recording_sequence":
        manifest = Path(policy["sequence_manifest"]).expanduser()
        if not manifest.is_absolute():
            manifest = config["_config_path"].parent / manifest
        roles = read_sequence_assignments(manifest, image_names)
        splits = {role: [s for s in all_samples if roles[s.image_path.name] == role]
                  for role in SPLIT_COUNTS}
    elif policy["method"] == "chronological_blocks":
        if policy.get("chronological_fallback_authorized") is not True:
            raise ValueError("Chronological fallback requires explicit authorization")
        splits = {role: load_split(config, role) for role in SPLIT_COUNTS}
    else:
        raise ValueError("split_policy.method must be random_images, recording_sequence or chronological_blocks")
    audit_split_independence(splits)
    for role, count in SPLIT_COUNTS.items():
        if len(splits[role]) != count:
            raise ValueError(f"{role} must contain {count} pairs")
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    splits = prepare_splits(config)
    for role, samples in splits.items():
        print(f"{role}: {len(samples)} matched dock pairs", flush=True)
    print("Verified: no shared image/mask paths or identical image files across splits.", flush=True)
    if args.check_data:
        print("Dock split, complete pairing, dimensions and label IDs passed validation.")
        return
    if config["experiment"].get("label_source_confirmed") is not True:
        raise ValueError("Resolve the channel-versus-dock pseudo-label source before training")
    output = Path(config["experiment"]["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Experiment output already contains files: {output}. "
                              "Choose a new output_dir for a new attempt.")
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_split_independence(splits)
    if config["split_policy"]["method"] == "recording_sequence":
        audit["recording_independence_verified"] = True
        audit["limitation"] = "Recording independence follows supplied sequence metadata."
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = copy.deepcopy(config)
    if config["split_policy"]["method"] == "recording_sequence":
        manifest = Path(config["split_policy"]["sequence_manifest"]).expanduser()
        if not manifest.is_absolute():
            manifest = config["_config_path"].parent / manifest
        snapshot = output / "recording_sequences.csv"
        snapshot.write_bytes(manifest.read_bytes())
        resolved["split_policy"]["sequence_manifest"] = str(snapshot)
    # Preserve actual membership for existing test-visualization and evaluation tools.
    for role, samples in splits.items():
        resolved["data"][role]["sources"] = [
            {"source_id": sample.image_path.stem,
             "images_dir": str(sample.image_path.parent),
             "masks_dir": str(sample.mask_path.parent),
             "expected_count": 1,
             "selection": {"kind": "inclusive_range", "index_pattern": r"^dock_image_(\d+)$",
                           "start": int(sample.image_path.stem.rsplit("_", 1)[1]),
                           "end": int(sample.image_path.stem.rsplit("_", 1)[1])}}
            for sample in samples
        ]
    import yaml
    (output / "resolved_experiment.yaml").write_text(
        yaml.safe_dump(serializable(resolved), sort_keys=False), encoding="utf-8")
    (output / "resolved_splits.json").write_text(
        json.dumps({k: split_manifest(v) for k, v in splits.items()}, indent=2),
        encoding="utf-8")
    run(config, splits)


if __name__ == "__main__":
    main()
