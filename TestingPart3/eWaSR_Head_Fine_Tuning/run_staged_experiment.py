"""Resumable, explicitly gated head-only experiment stages 1–6.

Preview with --step N; execute that one stage with --step N --continue.
Steps 4–6 require an explicit human choice before the following stage opens.
The 5% unseen split is indexed for membership checks and never evaluated here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_experiment import (
    load_config, load_split, require_mapping, select_samples, split_manifest,
    validate_configuration,
)
from train_head import Sample, evaluate, find_samples, train


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent / "DataSource/staged_experiment_template.yaml"
ROLES = ("train", "validation", "test", "unseen")
FRACTIONS = {"train": .60, "validation": .20, "test": .15, "unseen": .05}
STAGE_NAMES = {
    1: "Baseline: AdamW, LR 1e-3, decay 1e-4, batch 2, 10 epochs, weights 1/1/1",
    2: "Learning rates: 3e-4, 1e-3, 3e-3",
    3: "Weight decay: 0, 1e-5, 1e-4, 1e-3",
    4: "Duration: 10 versus 20 epochs, optionally 30",
    5: "Batch size: 2, 4, 8 if memory permits",
    6: "Class weights: 1/1/1, 2/1/1, 3/1/1, 3/2/1, 1/2/1",
}


def resolve_path(value, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def normalize_paths(config):
    config = copy.deepcopy(config)
    base = config["_config_path"].parent
    config["experiment"]["output_dir"] = str(resolve_path(config["experiment"]["output_dir"], base))
    config["model"]["checkpoint"] = str(resolve_path(config["model"]["checkpoint"], base))
    for role in ROLES:
        split = config["data"][role]
        for source in split.get("sources", [split]):
            for field in ("images_dir", "masks_dir"):
                source[field] = str(resolve_path(source[field], base))
    return config


def expected_counts(total: int) -> dict[str, int]:
    """Largest-remainder rounding keeps the exact 60/20/15/5 target."""
    raw = {role: total * share for role, share in FRACTIONS.items()}
    result = {role: math.floor(value) for role, value in raw.items()}
    for role in sorted(ROLES, key=lambda role: (-(raw[role] - result[role]), ROLES.index(role)))[:total - sum(result.values())]:
        result[role] += 1
    return result


def validate_recipe(config):
    validate_configuration(config)
    if "REPLACE_" in json.dumps({key: value for key, value in config.items() if key != "_config_path"}):
        raise ValueError("Fill every REPLACE_ field in the dataset YAML before checking or running it")
    policy = require_mapping(config.get("split_policy"), "split_policy")
    if policy.get("fractions") != FRACTIONS:
        raise ValueError("split_policy.fractions must be train=.60, validation=.20, test=.15, unseen=.05")
    tolerance = float(policy.get("max_fraction_deviation", 0))
    if not 0 <= tolerance < 1:
        raise ValueError("split_policy.max_fraction_deviation must be between 0 and 1")
    if config["model"].get("architecture") != "ewasr_resnet18":
        raise ValueError("This runner requires the pretrained ResNet-18 eWaSR")
    preprocessing = config["preprocessing"]
    if (preprocessing.get("image_normalization") != "imagenet" or
        preprocessing.get("image_mean") != [.485, .456, .406] or
        preprocessing.get("image_std") != [.229, .224, .225]):
        raise ValueError("Image normalization must match the pretrained ImageNet preprocessing")
    training = config["training"]
    required = {"optimizer": "adamw", "initial_learning_rate": .001,
                "weight_decay": .0001, "batch_size": 2, "epochs": 10,
                "loss": "cross_entropy", "class_weights": None,
                "checkpoint_selection_metric": "validation_binary_water_iou"}
    for key, value in required.items():
        if training.get(key) != value:
            raise ValueError(f"Baseline training.{key} must equal {value!r}")
    if [float(x) for x in training["learning_rates"]] != [.0003, .001, .003]:
        raise ValueError("Learning rates must be 3e-4, 1e-3, 3e-3")
    if not Path(config["model"]["checkpoint"]).is_file():
        raise FileNotFoundError("Pretrained non-IMU eWaSR checkpoint is missing")


def load_unseen(config) -> list[Sample]:
    """Index unseen filenames without opening its image or reference pixels."""
    split = require_mapping(config["data"]["unseen"], "data.unseen")
    selected = []
    for index, source in enumerate(split.get("sources", [split])):
        source = require_mapping(source, f"data.unseen.sources[{index}]")
        available = find_samples(Path(source["images_dir"]), Path(source["masks_dir"]),
                                 ignore_masks_without_images=bool(source.get("shared_masks_dir", False)))
        subset = select_samples(available, require_mapping(source["selection"], "unseen.selection"), "unseen")
        if "expected_count" in source and len(subset) != int(source["expected_count"]):
            raise ValueError("Unseen source count differs from YAML")
        selected.extend(subset)
    if "expected_count" in split and len(selected) != int(split["expected_count"]):
        raise ValueError("Unseen split count differs from YAML")
    return selected


def load_dataset(config) -> dict[str, list[Sample]]:
    splits = {role: load_split(config, role) for role in ROLES if role != "unseen"}
    splits["unseen"] = load_unseen(config)
    owners = {}
    mask_owners = {}
    group_owners = {}
    source_specs = {}
    image_groups = {}
    for role in ROLES:
        split = config["data"][role]
        for source in split.get("sources", [split]):
            selection = source["selection"]
            sequence_pattern = (re.compile(str(selection["pattern"]))
                                if selection.get("kind") == "sequence_ids" else None)
            fixed_group = source.get("sequence_id", split.get("sequence_id"))
            if sequence_pattern is None and (not isinstance(fixed_group, str) or not fixed_group.strip()):
                raise ValueError(f"Every {role} source needs a real sequence_id")
            spec = (Path(source["images_dir"]).resolve(), Path(source["masks_dir"]).resolve(),
                    bool(source.get("shared_masks_dir", False)))
            source_specs[spec] = source
            subset = select_samples(find_samples(
                spec[0], spec[1], ignore_masks_without_images=spec[2]),
                selection, role)
            for sample in subset:
                group = (sequence_pattern.fullmatch(sample.image_path.stem).group(1)
                         if sequence_pattern is not None else fixed_group)
                if group in group_owners and group_owners[group] != role:
                    raise ValueError(f"Sequence {group!r} crosses split boundaries")
                group_owners[group] = role
                old = image_groups.setdefault(sample.image_path.resolve(), group)
                if old != group:
                    raise ValueError(f"Conflicting sequence IDs for {sample.image_path}")
        for sample in splits[role]:
            image, mask = sample.image_path.resolve(), sample.mask_path.resolve()
            if image in owners or mask in mask_owners:
                raise ValueError(f"Repeated image or mask across dataset roles: {image}")
            owners[image], mask_owners[mask] = role, role
    if any(not splits[role] for role in ROLES):
        raise ValueError("All four data roles must have at least one pair")
    actual = {role: len(splits[role]) for role in ROLES}
    tolerance = float(config["split_policy"].get("max_fraction_deviation", 0))
    if tolerance == 0:
        if actual != expected_counts(len(owners)):
            raise ValueError(f"Split counts must use 60/20/15/5: expected {expected_counts(len(owners))}")
    elif any(abs(actual[role] / len(owners) - FRACTIONS[role]) > tolerance for role in ROLES):
        raise ValueError(f"Split counts deviate from 60/20/15/5 beyond the configured tolerance: {actual}")
    # Every labelled image in the declared source universe must have a role.
    available = set()
    for (images, masks, shared), _ in source_specs.items():
        available.update(sample.image_path.resolve() for sample in find_samples(
            images, masks, ignore_masks_without_images=shared))
    if available != set(owners):
        raise ValueError(f"YAML leaves {len(available - owners.keys())} paired images unassigned")
    for parent in {path.parent for path in image_groups}:
        ordered = [image_groups[path] for path in sorted(image_groups) if path.parent == parent]
        runs = [group for index, group in enumerate(ordered)
                if index == 0 or group != ordered[index - 1]]
        if len(runs) != len(set(runs)):
            raise ValueError(f"Sequence groups interleave in {parent}; consecutive frames must stay together")
    # Exact duplicate bytes are checked among development splits. Unseen pixels
    # are deliberately not read, hashed, evaluated, or used for stage selection.
    hashes = {}
    for role in ("train", "validation", "test"):
        for sample in splits[role]:
            digest = hashlib.sha256(sample.image_path.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != role:
                raise ValueError("Identical image bytes cross development split boundaries")
            hashes[digest] = role
    return splits


def config_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    """Represent absent class metrics as null in strict, resumable JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_state(path: Path, state: dict):
    target = path.with_suffix(".tmp")
    target.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target.replace(path)


def load_state(output: Path, fingerprint: str, splits):
    path = output / "stage_state.json"
    if path.is_file():
        state = json.loads(path.read_text())
        if state["config_sha256"] != fingerprint:
            raise ValueError("YAML changed after stage 1; use a new output_dir")
        recorded = json.loads((output / "resolved_splits.json").read_text())
        if recorded != {role: split_manifest(samples) for role, samples in splits.items()}:
            raise ValueError("Dataset membership changed after stage 1")
        return state
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory already contains unrelated files: {output}")
    return {"schema_version": 1, "config_sha256": fingerprint,
            "checkpoint_metric": "validation_binary_water_iou", "stages": {}}


def initialize(output: Path, state: dict, config, splits):
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value
                    for key, value in config.items()}, indent=2) + "\n", encoding="utf-8")
    (output / "resolved_splits.json").write_text(
        json.dumps({role: split_manifest(samples) for role, samples in splits.items()}, indent=2) + "\n",
        encoding="utf-8")
    save_state(output / "stage_state.json", state)


def trial_args(config, params: dict, output: Path):
    data = config["data"]
    first = lambda role: data[role].get("sources", [data[role]])[0]
    preprocessing = config["preprocessing"]
    return SimpleNamespace(
        train_images=Path(first("train")["images_dir"]),
        train_masks=Path(first("train")["masks_dir"]),
        val_images=Path(first("validation")["images_dir"]),
        val_masks=Path(first("validation")["masks_dir"]),
        weights=Path(config["model"]["checkpoint"]), output_dir=output,
        epochs=params["epochs"], batch_size=params["batch_size"],
        learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
        class_weights=",".join(str(value) for value in params["class_weights"]),
        width=int(preprocessing["input_width"]), height=int(preprocessing["input_height"]),
        seed=int(config["experiment"]["seed"]), check_data=False,
        selection_metric="binary_water_iou",
    )


def baseline_params():
    return {"learning_rate": .001, "weight_decay": .0001, "epochs": 10,
            "batch_size": 2, "class_weights": [1, 1, 1]}


def best_candidate(candidates: dict, previous: dict | None = None):
    ordered = ([previous] if previous is not None else []) + list(candidates.values())
    return max(ordered, key=lambda item: item["best_validation"]["binary_water_iou"])


def candidate_record(config, state, stage: int, label: str, params: dict, output: Path, splits):
    current = state["stages"].setdefault(str(stage), {"status": "running", "candidates": {}})
    if label in current["candidates"]:
        return current["candidates"][label]
    for previous in state["stages"].values():
        for existing in previous.get("candidates", {}).values():
            if existing["parameters"] == params:
                reused = {**existing, "reused_from": existing["checkpoint"]}
                current["candidates"][label] = reused
                save_state(output / "stage_state.json", state)
                return reused
    directory = output / f"step_{stage:02d}" / re.sub(r"[^A-Za-z0-9._-]", "_", label)
    result = train(trial_args(config, params, directory), splits["train"], splits["validation"])
    # A finished candidate must have a complete history and still-frozen body.
    history = [json.loads(line) for line in Path(result["history"]).read_text().splitlines()]
    if [row["epoch"] for row in history] != list(range(1, params["epochs"] + 1)):
        raise RuntimeError("Incomplete epoch history")
    if result["best_epoch"] != max(history, key=lambda row: row["validation"]["binary_water_iou"])["epoch"]:
        raise RuntimeError("Checkpoint did not select peak validation binary water IoU")
    with np.load(config["model"]["checkpoint"]) as original, np.load(result["best_checkpoint"]) as trained:
        if set(original.files) != set(trained.files) or any(
            not np.array_equal(original[key], trained[key])
            for key in original.files if not key.startswith("decoder.seg_head.")
        ):
            raise RuntimeError("A frozen eWaSR tensor changed")
    scores = [row["validation"]["binary_water_iou"] for row in history]
    record = {"parameters": params, "best_epoch": result["best_epoch"],
              "checkpoint": str(Path(result["best_checkpoint"]).resolve()),
              "history": str(Path(result["history"]).resolve()),
              "best_validation": json_safe(result["best_record"]["validation"]),
              "last_three_validation_water_iou_std": float(np.std(scores[-3:])),
              "trainable_parameter_count": result["trainable_parameter_count"]}
    current["candidates"][label] = record
    save_state(output / "stage_state.json", state)
    return record


def stage_previous(state, stage: int):
    if stage == 1:
        return None
    previous = state["stages"].get(str(stage - 1), {})
    if previous.get("status") != "complete":
        raise ValueError(f"Complete step {stage - 1} and its selection before step {stage}")
    return previous["selected"]


def stage_report(output: Path, stage: int, record: dict):
    lines = [f"STEP {stage}: {STAGE_NAMES[stage]}",
             "Checkpoint/parameter selection: validation binary water IoU.",
             "Test (15%) and unseen (5%) were not evaluated.", ""]
    for label, item in record.get("candidates", {}).items():
        metrics = item["best_validation"]
        lines += [f"{label}: best epoch {item['best_epoch']}; water IoU {metrics['binary_water_iou']:.6f}; "
                  f"mean three-class IoU {metrics['mean_iou']:.6f}; recent stability SD "
                  f"{item['last_three_validation_water_iou_std']:.6f}; checkpoint {item['checkpoint']}"]
        if stage == 6:
            lines += [f"  obstacle missed {metrics['obstacle_miss_rate']}; "
                      f"water mislabelled obstacle {metrics['water_as_obstacle_rate']}; "
                      f"obstacle precision {metrics['obstacle_precision']}"]
    lines += ["", f"Status: {record['status']}"]
    if record.get("selected"):
        selected = record["selected"]
        lines += [f"Selected: {selected['parameters']}; epoch {selected['best_epoch']}; "
                  f"validation water IoU {selected['best_validation']['binary_water_iou']:.6f}"]
    (output / f"step_{stage:02d}_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_stage(config, splits, state, output: Path, stage: int, *, extend30=False, skip_batch8=False):
    previous = stage_previous(state, stage)
    current = state["stages"].setdefault(str(stage), {"status": "running", "candidates": {}})
    if current["status"] == "complete":
        raise ValueError(f"Step {stage} is already complete")
    base = baseline_params() if stage == 1 else copy.deepcopy(previous["parameters"])
    if stage == 1:
        if "pretrained_validation" not in state:
            from ewasr_mlx import create_head_finetuning_model
            model = create_head_finetuning_model(Path(config["model"]["checkpoint"]))
            state["pretrained_validation"] = json_safe(evaluate(
                model, splits["validation"], trial_args(config, base, output / "pretrained")))
            save_state(output / "stage_state.json", state)
        plans = [("baseline", base)]
    elif stage == 2:
        plans = [(f"lr_{rate:g}", {**base, "learning_rate": rate}) for rate in (.0003, .001, .003)]
    elif stage == 3:
        plans = [(f"decay_{decay:g}", {**base, "weight_decay": decay})
                 for decay in (0, .00001, .0001, .001)]
    elif stage == 4:
        if extend30 and "epochs_20" not in current["candidates"]:
            raise ValueError("Train the 20-epoch candidate before requesting 30")
        plans = [("epochs_30" if extend30 else "epochs_20",
                  {**base, "epochs": 30 if extend30 else 20})]
    elif stage == 5:
        plans = [(f"batch_{size}", {**base, "batch_size": size})
                 for size in (2, 4, 8) if size != 8 or not skip_batch8]
        if skip_batch8:
            current["skipped_batch_8"] = "user requested omission because memory does not permit batch 8"
    else:
        plans = [("weights_" + "_".join(str(x) for x in weights),
                  {**base, "class_weights": list(weights)})
                 for weights in ((1, 1, 1), (2, 1, 1), (3, 1, 1), (3, 2, 1), (1, 2, 1))]
    for label, params in plans:
        candidate_record(config, state, stage, label, params, output, splits)
    if stage <= 3:
        current["selected"] = best_candidate(current["candidates"], previous if stage in (2, 3) else None)
        current["status"] = "complete"
    else:
        current["status"] = "pending_choice"
    save_state(output / "stage_state.json", state)
    stage_report(output, stage, current)


def choose_stage(state, output: Path, stage: int, choice: str):
    if stage not in (4, 5, 6):
        raise ValueError("Manual choices apply only to steps 4–6")
    stage_previous(state, stage)
    current = state["stages"].get(str(stage), {})
    if current.get("status") != "pending_choice":
        raise ValueError(f"Run step {stage} candidates before choosing")
    if stage == 4 and choice == "use10":
        selected = state["stages"]["3"]["selected"]
    else:
        label = ({"use20": "epochs_20", "use30": "epochs_30"}.get(choice)
                 if stage == 4 else f"batch_{choice}" if stage == 5 else
                 "weights_" + choice.replace(",", "_") if stage == 6 else None)
        if label not in current["candidates"]:
            raise ValueError(f"Choice {choice!r} has no completed candidate")
        selected = current["candidates"][label]
    current["selected"] = selected
    current["status"] = "complete"
    current["human_choice"] = choice
    save_state(output / "stage_state.json", state)
    stage_report(output, stage, current)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--step", type=int, choices=range(1, 7))
    parser.add_argument("--continue", dest="proceed", action="store_true",
                        help="Explicitly execute the ONE requested stage")
    parser.add_argument("--choose", help="Finish step 4 (use10/use20/use30), 5 (2/4/8), or 6 (1,1,1 etc.)")
    parser.add_argument("--extend30", action="store_true", help="Run the optional 30-epoch step-4 trial")
    parser.add_argument("--skip-batch8", action="store_true", help="Omit batch 8 when memory does not permit it")
    return parser.parse_args()


def main():
    args = parse_args()
    config = normalize_paths(load_config(args.config))
    validate_recipe(config)
    output = Path(config["experiment"]["output_dir"])
    if args.status:
        path = output / "stage_state.json"
        if path.is_file():
            state = json.loads(path.read_text())
            for stage in range(1, 7):
                print(f"Step {stage}: {state['stages'].get(str(stage), {}).get('status', 'not started')}")
        else:
            print("No stage has started.")
        return
    splits = load_dataset(config)
    print("Validated train/validation/test/unseen pairs:", {role: len(splits[role]) for role in ROLES})
    print("The 5% unseen split was indexed by filename only; its pixels were not read.")
    if args.check_data:
        return
    if args.step is None:
        raise ValueError("Specify --step N, --check-data, or --status")
    fingerprint = config_fingerprint(config["_config_path"])
    state = load_state(output, fingerprint, splits)
    print(f"Step {args.step}: {STAGE_NAMES[args.step]}")
    if args.choose is not None:
        if args.proceed or args.extend30 or args.skip_batch8:
            raise ValueError("--choose must be issued separately from execution options")
        choose_stage(state, output, args.step, args.choose)
        print(f"Recorded step {args.step} choice: {args.choose}")
        return
    if not args.proceed:
        stage_previous(state, args.step)
        print(f"Preview only. Run again with --step {args.step} --continue to execute this stage.")
        return
    if args.extend30 and args.step != 4 or args.skip_batch8 and args.step != 5:
        raise ValueError("--extend30 is only for step 4; --skip-batch8 only for step 5")
    stage_previous(state, args.step)
    if not (output / "stage_state.json").is_file():
        initialize(output, state, config, splits)
    execute_stage(config, splits, state, output, args.step,
                  extend30=args.extend30, skip_batch8=args.skip_batch8)
    print((output / f"step_{args.step:02d}_report.txt").read_text())
    if state["stages"][str(args.step)]["status"] == "complete" and args.step < 6:
        print(f"Next: --step {args.step + 1} previews the next stage; it does not start it.")
    elif state["stages"][str(args.step)]["status"] == "pending_choice":
        print("Review the report, then use --step N --choose VALUE before advancing.")


if __name__ == "__main__":
    main()
