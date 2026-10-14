"""Compare nested training budgets while preserving a selected split's evaluation sets.

Percentages refer to the selected run's training pool, not the full inventory.
Three selection seeds vary temporal sampling phases within the existing training
sub-trips; the training seed stays fixed. The 100% endpoint is a single run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

import split_sweep as sweep

HERE = Path(__file__).resolve().parent
RESULT_FOLDER = "head_only_lr1e-3_wd1e-3_20ep"
PERCENTAGES = (25, 50, 75, 100)
ROLES = ("train", "validation", "test", "held_out")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def training_subtrips(rows: list[dict], steps: dict[str, int]) -> dict[str, list[dict]]:
    """Use the original manifest groups; never rebuild or bridge their boundaries."""
    groups = defaultdict(list)
    for row in rows:
        if not row.get("subtrip_id"):
            raise ValueError("Training rows require subtrip_id from a sub-trip split")
        groups[row["subtrip_id"]].append(row)
    for name, group in groups.items():
        group.sort(key=lambda row: int(row["frame_number"]))
        trip = group[0]["trip_id"]
        if (any(row["trip_id"] != trip for row in group) or
                any(int(right["frame_number"]) - int(left["frame_number"]) != steps[trip]
                    for left, right in zip(group, group[1:]))):
            raise ValueError(f"Training sub-trip is not contiguous: {name}")
    return dict(sorted(groups.items()))


def phase_order(size: int, phase: float) -> list[int]:
    """Progressively fill equal temporal intervals using a shifted dyadic grid.

    This is nested, approximately uniform temporal sampling, not a strict
    every-k stride. Filtering repeated bins yields each frame exactly once.
    """
    bits = max(1, (size - 1).bit_length())
    denominator = 1 << bits
    order, seen = [], set()
    for rank in range(denominator):
        reversed_bits = int(f"{rank:0{bits}b}"[::-1], 2)
        index = min(size - 1, math.floor(((phase + reversed_bits / denominator) % 1) * size))
        if index not in seen:
            seen.add(index)
            order.append(index)
    if len(order) != size:
        raise AssertionError("Temporal ordering did not cover every frame")
    return order


def uniform_selection_order(rows: list[dict], steps: dict[str, int], seed: int) -> list[dict]:
    groups = training_subtrips(rows, steps)
    rng = random.Random(seed)
    ranked = []
    for name, group in groups.items():
        phase, tie_break = rng.random(), rng.random()
        for local_rank, index in enumerate(phase_order(len(group), phase)):
            # Interleave group orders by their fractional sampling density.
            # All budgets are prefixes of this one ordering, so quotas cannot
            # decrease at a larger budget (unlike separate rounded allocations).
            ranked.append(((local_rank + .5) / len(group), tie_break, name,
                           {**group[index], "temporal_phase": phase}))
    ranked.sort(key=lambda item: item[:3])
    return [row for _, _, _, row in ranked]


def nested_selections(rows: list[dict], steps: dict[str, int], seed: int,
                      percentages=PERCENTAGES) -> tuple[dict, list[dict]]:
    ordered = uniform_selection_order(rows, steps, seed)
    result = {}
    for percentage in percentages:
        target = math.floor(len(rows) * percentage / 100 + .5)
        selected = ordered[:target]
        if not selected:
            raise ValueError("Training pool too small for the requested budgets")
        result[percentage] = {row["image_id"] for row in selected}
    return result, ordered


def assert_baseline_result(config: dict, rows: list[dict], result_dir: Path) -> None:
    """Only reuse a completed result with the exact recipe and membership."""
    expected = {"epochs": 20, "batch_size": 2, "learning_rate": .001,
                "weight_decay": .001, "class_weights": "1,1,1",
                "width": 512, "height": 384,
                "seed": int(config["experiment"]["seed"]),
                "selection_metric": "binary_water_iou"}
    run_config = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8"))
    if any(run_config.get(key) != value for key, value in expected.items()):
        raise ValueError("Baseline result does not match the fixed size-sweep recipe")
    if Path(run_config["weights"]).resolve() != Path(config["model"]["checkpoint"]).resolve():
        raise ValueError("Baseline starting weights differ")
    resolved = {"unseen" if role == "held_out" else role:
                [{"image": row["image_path"], "mask": row["mask_path"]}
                 for row in rows if row["role"] == role] for role in ROLES}
    if json.loads((result_dir / "resolved_splits.json").read_text(encoding="utf-8")) != resolved:
        raise ValueError("Baseline result membership differs from the selected manifest")
    history = (result_dir / "training_history.jsonl").read_text(encoding="utf-8").splitlines()
    if len(history) != 20 or not (result_dir / "best_head_only.npz").is_file():
        raise ValueError("Baseline training is incomplete")


def generate(spec_path: Path, base_config: Path, output_dir: Path, *,
             percentages=PERCENTAGES, selection_seeds=None, reuse_baseline=False,
             study_config: Path | None = None) -> None:
    spec = sweep.load_spec(spec_path)
    percentages = tuple(percentages)
    if (not percentages or percentages != tuple(sorted(set(percentages))) or
            percentages[-1] != 100 or any(not isinstance(p, int) or not 0 < p <= 100 for p in percentages)):
        raise ValueError("percentages must be increasing unique integers, ending at 100")
    selection_seeds = list(selection_seeds if selection_seeds is not None else spec["split_seeds"])
    if len(selection_seeds) != 3 or len(set(selection_seeds)) != 3 or any(not isinstance(s, int) for s in selection_seeds):
        raise ValueError("Use three distinct integer selection seeds")
    frames = sweep.discover_frames(spec)
    base_config = base_config.resolve()
    base_study = base_config.parents[2]
    sweep.verify_generated_run(base_study, base_config.parent.name, base_config)
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if config.get("split_policy", {}).get("sorting_version") != 2:
        raise ValueError("Choose an existing sub-trip split (sorting_version: 2)")
    if Path(config["model"]["checkpoint"]).resolve() != Path(spec["model_checkpoint"]).resolve():
        raise ValueError("Selected split uses a different starting checkpoint from the source spec")
    rows = read_csv(Path(config["data"]["manifest_csv"]))
    frame_paths = {frame.image_id: (str(frame.image_path), str(frame.mask_path)) for frame in frames}
    if {row["image_id"]: (row["image_path"], row["mask_path"]) for row in rows} != frame_paths:
        raise ValueError("Selected split must cover exactly the supplied data source")
    if config["training"]["checkpoint_selection_metric"] != "validation_binary_water_iou":
        raise ValueError("Size sweep requires selection by validation water IoU")
    baseline_result = base_config.parent / RESULT_FOLDER
    reuse = reuse_baseline and (baseline_result / "evaluation.txt").is_file()
    if reuse_baseline and not reuse:
        raise FileNotFoundError("Requested reuse but the selected baseline is not complete")
    if reuse:
        assert_baseline_result(config, rows, baseline_result)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Study directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir()
    sweep.write_csv(output_dir / "base_split_manifest.csv", sweep.MANIFEST_FIELDS, rows)
    (output_dir / "base_config.yaml").write_text(base_config.read_text(encoding="utf-8"), encoding="utf-8")
    train_rows = [row for row in rows if row["role"] == "train"]
    targets = [math.floor(len(train_rows) * p / 100 + .5) for p in percentages]
    if min(targets) < 1 or len(set(targets)) != len(targets):
        raise ValueError("Training pool is too small for distinct requested budgets")
    steps = {source["trip_id"]: int(source.get("frame_step", 1)) for source in spec["sources"]}
    groups = training_subtrips(train_rows, steps)
    study = {"schema_version": 1, "dataset": spec["dataset"],
             "source_spec": str(spec["_path"]), "source_spec_sha256": sweep.sha256_file(spec["_path"]),
             "base_config": str(base_config), "base_config_sha256": sweep.sha256_file(base_config),
             "base_manifest_sha256": sweep.sha256_file(output_dir / "base_split_manifest.csv"),
             "starting_checkpoint": spec["model_checkpoint"],
             "starting_checkpoint_sha256": sweep.sha256_file(Path(spec["model_checkpoint"])),
             "inventory_count": len(rows), "base_counts": dict(Counter(row["role"] for row in rows)),
             "training_seed": int(config["experiment"]["seed"]), "selection_seeds": selection_seeds,
             "percentages": list(percentages), "frame_steps": steps,
             "training_subtrip_count": len(groups), "baseline_reused": bool(reuse),
             "sampling": "Nested approximately uniform temporal sampling within original training sub-trips. Shifted dyadic phases and proportional density ordering; no image-content ranking or strict every-k stride.",
             "budget": "Fixed 20 epochs; larger subsets receive more optimizer updates.",
             "repeat_interpretation": "Subset-selection variation with one fixed training seed; 100% is a single endpoint.",
             "runs": [], "run_audit_sha256": {}}
    if study_config is not None:
        study["study_config"] = str(study_config.resolve())
        study["study_config_sha256"] = sweep.sha256_file(study_config)
    seen = {percentage: set() for percentage in percentages[:-1]}
    (output_dir / "selection_orders").mkdir()
    for iteration, seed in enumerate(selection_seeds, start=1):
        selections, ordered = nested_selections(train_rows, steps, seed, percentages)
        order_path = output_dir / "selection_orders" / f"selection_{iteration}.csv"
        sweep.write_csv(order_path, ("rank", "image_id", "subtrip_id", "temporal_phase"),
                        ({"rank": rank, **{key: row[key] for key in ("image_id", "subtrip_id", "temporal_phase")}}
                         for rank, row in enumerate(ordered, start=1)))
        for percentage, selected in selections.items():
            if percentage == 100 and iteration != 1:
                continue
            if percentage < 100:
                frozen = frozenset(selected)
                if frozen in seen[percentage]:
                    raise ValueError(f"Duplicate subset for {percentage}%; change selection seeds")
                seen[percentage].add(frozen)
            run_id = f"{spec['dataset']}_size_{percentage:03d}_{iteration}"
            run_dir = output_dir / "runs" / run_id
            run_dir.mkdir()
            active_rows = [row for row in rows if row["role"] != "train" or row["image_id"] in selected]
            sweep.write_csv(run_dir / "split_manifest.csv", sweep.MANIFEST_FIELDS, active_rows)
            membership = [{**row,
                           "role": "unused_train" if row["role"] == "train" and row["image_id"] not in selected else row["role"]}
                          for row in rows]
            sweep.write_csv(run_dir / "full_membership.csv", sweep.MANIFEST_FIELDS, membership)
            generated_config = copy.deepcopy(config)
            generated_config["experiment"] = {"name": run_id, "output_dir": str(run_dir),
                                               "seed": study["training_seed"]}
            counts = dict(Counter(row["role"] for row in active_rows))
            generated_config["data"]["manifest_csv"] = str(run_dir / "split_manifest.csv")
            generated_config["data"]["expected_counts"] = counts
            generated_config["data_size_policy"] = {"percentage": percentage,
                                                     "selection_seed": seed if percentage < 100 else None,
                                                     "sampling_method": "nested_uniform_temporal",
                                                     "base_config": str(base_config)}
            config_path = run_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(generated_config, sort_keys=False), encoding="utf-8")
            result_dir = baseline_result if percentage == 100 and reuse else run_dir / RESULT_FOLDER
            audit = {"run_id": run_id, "percentage": percentage,
                     "actual_training_percentage": len(selected) / len(train_rows) * 100,
                     "iteration": iteration, "selection_seed": seed if percentage < 100 else None,
                     "training_seed": study["training_seed"], "counts": counts,
                     "unused_train": len(train_rows) - len(selected),
                     "training_group_counts": dict(Counter(row["subtrip_id"] for row in active_rows if row["role"] == "train")),
                     "selection_order_path": str(order_path), "selection_order_sha256": sweep.sha256_file(order_path),
                     "epochs": 20, "batch_size": 2,
                     "optimizer_updates": 20 * math.ceil(len(selected) / 2),
                     "train_ids_sha256": sweep.sha256_ids(selected),
                     "config_sha256": sweep.sha256_file(config_path),
                     "manifest_sha256": sweep.sha256_file(run_dir / "split_manifest.csv"),
                     "full_membership_sha256": sweep.sha256_file(run_dir / "full_membership.csv"),
                     "result_dir": str(result_dir), "reused_baseline": percentage == 100 and reuse}
            if audit["reused_baseline"]:
                audit["reused_evaluation_sha256"] = sweep.sha256_file(result_dir / "evaluation.txt")
                audit["reused_run_config_sha256"] = sweep.sha256_file(result_dir / "run_config.json")
                audit["reused_checkpoint_sha256"] = sweep.sha256_file(result_dir / "best_head_only.npz")
            write_json(run_dir / "size_audit.json", audit)
            study["runs"].append(run_id)
            study["run_audit_sha256"][run_id] = sweep.sha256_file(run_dir / "size_audit.json")
    write_json(output_dir / "study.json", study)
    verify(output_dir)
    summarize(output_dir)
    print(f"Generated {len(study['runs'])} unique runs; 100% training pool = {len(train_rows)} images.")


def generate_from_config(path: Path) -> None:
    path = path.resolve()
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    if recipe.get("schema_version") != 1 or recipe.get("kind") != "training_data_size_sweep":
        raise ValueError("Expected schema_version: 1 and kind: training_data_size_sweep")
    if recipe.get("sampling_method") != "nested_uniform_temporal":
        raise ValueError("Only nested_uniform_temporal sampling is supported")
    generate(sweep.resolve_path(recipe["source_spec"], path.parent),
             sweep.resolve_path(recipe["base_config"], path.parent),
             sweep.resolve_path(recipe["output_dir"], path.parent),
             percentages=recipe.get("percentages", PERCENTAGES),
             selection_seeds=recipe.get("selection_seeds"),
             reuse_baseline=recipe.get("reuse_baseline_result", False), study_config=path)


def verify(study_dir: Path) -> None:
    study_dir = study_dir.resolve()
    study = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    if study.get("study_config") and sweep.sha256_file(Path(study["study_config"])) != study["study_config_sha256"]:
        raise ValueError("Sweep recipe changed after generation")
    for file, digest in ((study["source_spec"], study["source_spec_sha256"]),
                         (study["base_config"], study["base_config_sha256"]),
                         (study["starting_checkpoint"], study["starting_checkpoint_sha256"]),
                         (study_dir / "base_split_manifest.csv", study["base_manifest_sha256"])):
        if sweep.sha256_file(Path(file)) != digest:
            raise ValueError(f"Study input changed: {file}")
    base = read_csv(study_dir / "base_split_manifest.csv")
    parent = {row["image_id"]: row for row in base}
    base_train_count = sum(row["role"] == "train" for row in base)
    base_groups = training_subtrips([row for row in base if row["role"] == "train"], study["frame_steps"])
    nested: dict[int, dict[int, set[str]]] = {}
    for name in study["runs"]:
        directory = study_dir / "runs" / name
        if sweep.sha256_file(directory / "size_audit.json") != study["run_audit_sha256"][name]:
            raise ValueError(f"Run audit changed: {name}")
        audit = json.loads((directory / "size_audit.json").read_text(encoding="utf-8"))
        for file, key in (("config.yaml", "config_sha256"), ("split_manifest.csv", "manifest_sha256"),
                          ("full_membership.csv", "full_membership_sha256")):
            if sweep.sha256_file(directory / file) != audit[key]:
                raise ValueError(f"Generated file changed: {directory / file}")
        rows = read_csv(directory / "split_manifest.csv")
        ids = [row["image_id"] for row in rows]
        if len(ids) != len(set(ids)) or any(row != parent.get(row["image_id"]) for row in rows):
            raise ValueError(f"Duplicate or changed membership: {name}")
        if dict(Counter(row["role"] for row in rows)) != audit["counts"]:
            raise ValueError(f"Incorrect counts: {name}")
        for role in ROLES[1:]:
            if [row for row in rows if row["role"] == role] != [row for row in base if row["role"] == role]:
                raise ValueError(f"Evaluation set changed: {role} in {name}")
        train_ids = {row["image_id"] for row in rows if row["role"] == "train"}
        expected_count = math.floor(base_train_count * audit["percentage"] / 100 + .5)
        if (len(train_ids) != expected_count or sweep.sha256_ids(train_ids) != audit["train_ids_sha256"] or
                audit["optimizer_updates"] != 20 * math.ceil(expected_count / 2)):
            raise ValueError(f"Incorrect training budget: {name}")
        order_path = Path(audit["selection_order_path"])
        if sweep.sha256_file(order_path) != audit["selection_order_sha256"]:
            raise ValueError(f"Selection order changed: {name}")
        ordered = read_csv(order_path)
        if (len(ordered) != base_train_count or {row["image_id"] for row in ordered} !=
                {row["image_id"] for row in base if row["role"] == "train"} or
                {row["image_id"] for row in ordered[:expected_count]} != train_ids):
            raise ValueError(f"Training subset does not match its selection order: {name}")
        membership = read_csv(directory / "full_membership.csv")
        expected_membership = [{**row, "role": "unused_train" if row["role"] == "train" and
                                row["image_id"] not in train_ids else row["role"]} for row in base]
        if membership != expected_membership:
            raise ValueError(f"Unused training membership changed: {name}")
        actual_groups = dict(Counter(row["subtrip_id"] for row in rows if row["role"] == "train"))
        if actual_groups != audit["training_group_counts"] or not set(actual_groups).issubset(base_groups):
            raise ValueError(f"Training sub-trip assignment changed: {name}")
        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        if (Path(config["data"]["manifest_csv"]) != directory / "split_manifest.csv" or
                config["data"]["expected_counts"] != audit["counts"] or
                config["experiment"]["seed"] != study["training_seed"]):
            raise ValueError(f"Config does not match experiment: {name}")
        nested.setdefault(audit["iteration"], {})[audit["percentage"]] = train_ids
        if audit["reused_baseline"]:
            result = Path(audit["result_dir"])
            if (sweep.sha256_file(result / "evaluation.txt") != audit["reused_evaluation_sha256"] or
                    sweep.sha256_file(result / "run_config.json") != audit["reused_run_config_sha256"] or
                    sweep.sha256_file(result / "best_head_only.npz") != audit["reused_checkpoint_sha256"]):
                raise ValueError("Reused baseline result changed")
            config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
            assert_baseline_result(config, rows, result)
    endpoint = nested[1][100]
    for selections in nested.values():
        chain = [ids for _, ids in sorted(selections.items())] + [endpoint]
        if any(not left.issubset(right) for left, right in zip(chain, chain[1:])):
            raise ValueError("Training subsets are not nested")
    print("Verified exact nested training counts and fixed validation/test/held-out sets.", flush=True)


def check(study_dir: Path, python: str, run_id: str | None = None) -> None:
    verify(study_dir)
    study = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    if run_id is not None and run_id not in study["runs"]:
        raise ValueError(f"Unknown run ID: {run_id}")
    for name in study["runs"]:
        if run_id is None or run_id == name:
            print(f"Checking data: {name}", flush=True)
            subprocess.run([python, str(HERE / "train_head.py"), "--config",
                            str(study_dir / "runs" / name / "config.yaml"), "--check-data"], check=True)
    write_json(study_dir / "data_check.json", {"status": "passed", "run_ids":
               study["runs"] if run_id is None else [run_id]})


def read_result(path: Path) -> dict:
    result = sweep.read_evaluation(path)
    result = {key: value if value != "unavailable" else "" for key, value in result.items()}
    section = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Training elapsed: "):
            result["training_seconds"] = line.partition(": ")[2].split()[0]
        if line == "Best validation epoch:":
            section = "validation"
        elif line == "Selected checkpoint on test split:":
            section = "test"
        elif line == "Selected checkpoint on common test core:":
            section = "common_test"
        elif section and line.startswith("  ") and ": " in line:
            key, value = line.strip().split(": ", 1)
            if key in {"pixel_accuracy", "obstacle_precision", "obstacle_recall", "water_as_obstacle_rate"}:
                result[f"{section}_{key}"] = value if value != "unavailable" else ""
    return result


def summarize(study_dir: Path) -> None:
    study = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    rows = []
    for name in study["runs"]:
        audit = json.loads((study_dir / "runs" / name / "size_audit.json").read_text(encoding="utf-8"))
        evaluation = Path(audit["result_dir"]) / "evaluation.txt"
        row = {"run_id": name, "percentage": audit["percentage"],
               "actual_training_percentage": audit["actual_training_percentage"],
               "iteration": audit["iteration"], "selection_seed": audit["selection_seed"],
               "training_seed": audit["training_seed"], **audit["counts"],
               "unused_train": audit["unused_train"], "reused_baseline": audit["reused_baseline"],
               "optimizer_updates": audit["optimizer_updates"],
               "status": "complete" if evaluation.is_file() else "generated"}
        if evaluation.is_file():
            row.update(read_result(evaluation))
        rows.append(row)
    fields = ("run_id", "percentage", "actual_training_percentage", "iteration", "selection_seed",
              "training_seed", "train", "validation", "test", "held_out", "unused_train",
              "reused_baseline", "status", "best_epoch", "training_seconds", "elapsed_seconds", "optimizer_updates",
              "validation_binary_water_iou", "validation_mean_iou", "test_binary_water_iou",
              "test_mean_iou", "common_test_binary_water_iou", "common_test_mean_iou",
              "common_test_obstacle_iou", "common_test_obstacle_miss_rate",
              "test_pixel_accuracy", "test_obstacle_precision", "test_obstacle_recall", "test_water_as_obstacle_rate")
    sweep.write_csv(study_dir / "summary.csv", fields,
                    ({field: row.get(field, "") for field in fields} for row in rows))
    aggregate = []
    for percentage in study["percentages"]:
        members = [row for row in rows if row["percentage"] == percentage]
        completed = [row for row in members if row["status"] == "complete"]
        record = {"percentage": percentage, "train": members[0]["train"],
                  "optimizer_updates": members[0]["optimizer_updates"],
                  "completed": len(completed), "planned_unique_runs": len(members)}
        for metric in ("test_binary_water_iou", "test_mean_iou", "common_test_binary_water_iou",
                       "validation_binary_water_iou", "training_seconds", "elapsed_seconds"):
            values = [float(row[metric]) for row in completed if row.get(metric, "") != ""]
            if values:
                record.update({metric + "_mean": statistics.mean(values),
                               metric + "_sd": statistics.stdev(values) if len(values) > 1 else "",
                               metric + "_min": min(values), metric + "_max": max(values)})
        aggregate.append(record)
    aggregate_fields = list(dict.fromkeys(key for row in aggregate for key in row))
    sweep.write_csv(study_dir / "aggregate.csv", aggregate_fields,
                    ({field: row.get(field, "") for field in aggregate_fields} for row in aggregate))
    lines = [f"# {study['dataset']} training data-size evaluation", "",
             f"Base split: `{Path(study['base_config']).parent.name}`.",
             f"Source inventory: {study['inventory_count']} frames. Fixed base counts: {study['base_counts']}.",
             "", "Percentages refer to the base training pool. The original sub-trip assignments are preserved.",
             "Frames are selected within each training sub-trip using seeded temporal phases on a",
             "progressively refined, equally spaced dyadic grid. Group orders are interleaved by fractional",
             "sampling density to allocate images approximately proportionally while meeting exact counts.",
             "All sizes are prefixes of one selection order per seed, giving nested subsets.",
             "This is approximately uniform temporal sampling, not a strict every-k stride or content-based selection.",
             "Excluded training images stay unused; they never enter validation or test.",
             "", "Training uses the existing head-only recipe: AdamW, LR 0.001, decay 0.001,",
             "20 epochs, batch 2, class weights 1/1/1, starting from the same pretrained weights.",
             "Checkpoint selection uses validation binary water IoU. The training seed is fixed;",
             "the three smaller-size repetitions measure subset-selection variation.",
             "The 100% endpoint is one run; its spread is unavailable.",
             "An existing result was reused." if study["baseline_reused"] else
             "The 100% endpoint will be trained afresh alongside the other runs for a comparable timing measurement.",
             "Training seconds include validation passes and checkpoint saving within training; total elapsed",
             "also includes initial loading and final evaluation. Neither is inference latency or SAM3 labelling cost.",
             "", "| Training pool | Images | Complete / planned | Test water IoU, mean ± sample SD | Test range |",
             "|---|---:|---:|---:|---:|"]
    for row in aggregate:
        value = row.get("test_binary_water_iou_mean")
        sd = row.get("test_binary_water_iou_sd")
        score = "pending" if value is None else f"{value:.6f}" + (f" ± {sd:.6f}" if sd != "" else " (single run)")
        interval = "pending" if value is None else f"{row['test_binary_water_iou_min']:.6f}–{row['test_binary_water_iou_max']:.6f}"
        lines.append(f"| {row['percentage']}% | {row['train']} | {row['completed']} / {row['planned_unique_runs']} | {score} | {interval} |")
    lines += ["", "| Training pool | Optimizer updates | Mean training seconds |",
              "|---|---:|---:|"]
    for row in aggregate:
        seconds = row.get("training_seconds_mean")
        lines.append(f"| {row['percentage']}% | {row['optimizer_updates']} | " +
                     (f"{seconds:.2f}" if seconds is not None else "pending") + " |")
    lines += ["", "Interpretation limits: larger subsets receive more optimizer updates under the fixed",
              "epoch budget. Scores measure agreement with SAM3 pseudo-labels at the model's output",
              "resolution. This is one recording with correlated frames, so the experiment does not",
              "establish a universal minimum data size or generalisation to independent trips.",
              "The held-out reserve remains unevaluated. The test set has already been used in the split sweep.", ""]
    (study_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated {study_dir / 'report.md'}", flush=True)


def train(study_dir: Path, python: str, run_id: str | None) -> None:
    verify(study_dir)
    study = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    if run_id is not None and run_id not in study["runs"]:
        raise ValueError(f"Unknown run ID: {run_id}")
    for name in study["runs"]:
        if run_id is not None and name != run_id:
            continue
        run_dir = study_dir / "runs" / name
        audit = json.loads((run_dir / "size_audit.json").read_text(encoding="utf-8"))
        result_dir = Path(audit["result_dir"])
        if (result_dir / "evaluation.txt").is_file():
            assert_baseline_result(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")),
                                   read_csv(run_dir / "split_manifest.csv"), result_dir)
            print(f"Already complete: {name}", flush=True)
            continue
        if result_dir.exists() and any(result_dir.iterdir()):
            raise FileExistsError(f"Partial result requires inspection: {result_dir}")
        print(f"Training {name}: {audit['counts']['train']} images", flush=True)
        subprocess.run([python, str(HERE / "train_head.py"), "--config",
                        str(run_dir / "config.yaml"), "--check-data"], check=True)
        with (run_dir / "training.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([python, "-u", str(HERE / "train_head.py"),
                                     "--config", str(run_dir / "config.yaml")], stdout=log, stderr=subprocess.STDOUT)
        summarize(study_dir)
        if result.returncode:
            raise RuntimeError(f"Training failed; inspect {run_dir / 'training.log'}")
    if all((Path(json.loads((study_dir / "runs" / name / "size_audit.json").read_text())["result_dir"])
            / "evaluation.txt").is_file() for name in study["runs"]):
        summarize(study_dir)
        subprocess.run([python, str(HERE / "plot_data_size_sweep.py"), str(study_dir)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--study-config", type=Path)
    generate_parser.add_argument("--spec", type=Path)
    generate_parser.add_argument("--base-config", type=Path)
    generate_parser.add_argument("--output-dir", type=Path)
    generate_parser.add_argument("--reuse-baseline", action="store_true")
    for command in ("check", "train", "summary", "plot"):
        subparser = commands.add_parser(command)
        subparser.add_argument("study_dir", type=Path)
        if command in ("check", "train", "plot"):
            subparser.add_argument("--python", default=sys.executable)
        if command in ("check", "train"):
            subparser.add_argument("--run-id")
    args = parser.parse_args()
    if args.command == "generate":
        if args.study_config is not None:
            if any((args.spec, args.base_config, args.output_dir, args.reuse_baseline)):
                parser.error("Use --study-config or direct generation arguments, not both")
            generate_from_config(args.study_config)
        else:
            if not all((args.spec, args.base_config, args.output_dir)):
                parser.error("Provide --study-config, or --spec, --base-config and --output-dir")
            generate(args.spec, args.base_config, args.output_dir, reuse_baseline=args.reuse_baseline)
    elif args.command == "check":
        check(args.study_dir.resolve(), args.python, args.run_id)
    elif args.command == "train":
        train(args.study_dir.resolve(), args.python, args.run_id)
    elif args.command == "summary":
        verify(args.study_dir.resolve())
        summarize(args.study_dir.resolve())
    else:
        verify(args.study_dir.resolve())
        summarize(args.study_dir.resolve())
        subprocess.run([args.python, str(HERE / "plot_data_size_sweep.py"), str(args.study_dir.resolve())], check=True)


if __name__ == "__main__":
    main()
