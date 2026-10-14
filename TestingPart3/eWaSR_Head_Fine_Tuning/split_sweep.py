"""Generate and run a reproducible head-only dataset split sweep.

The generated manifests are the source of truth for exact image membership.
Generation and --check-data never train or evaluate the held-out images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
INVENTORY_FIELDS = ("image_id", "image_path", "mask_path", "trip_id", "frame_number",
                    "image_sha256", "mask_sha256")
MANIFEST_FIELDS = ("image_id", "image_path", "mask_path", "trip_id", "frame_number",
                   "subtrip_id", "role")
RATIOS = {
    1: (60 / 95, 20 / 95, 15 / 95),  # exact form of the initial 60/20/15/5 split
    2: (.60, .20, .20),
    3: (.70, .15, .15),
    4: (.75, .10, .15),
    5: (.80, .10, .10),
}
METHODS = {1: "consecutive", 2: "subtrips", 3: "random_images"}


@dataclass(frozen=True)
class Frame:
    image_id: str
    image_path: Path
    mask_path: Path
    trip_id: str
    frame_number: int
    image_sha256: str
    mask_sha256: str


@dataclass(frozen=True)
class Subtrip:
    group_id: str
    frames: tuple[Frame, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def configured_ids(spec: dict, name: str) -> list[str] | None:
    inline = spec.get(f"{name}_image_ids")
    file_name = spec.get(f"{name}_ids_file")
    if inline is not None and file_name is not None:
        raise ValueError(f"Choose either {name}_image_ids or {name}_ids_file")
    if file_name is not None:
        path = resolve_path(file_name, spec["_path"].parent)
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    elif inline is not None:
        values = list(inline)
    else:
        return None
    if any(not isinstance(value, str) or not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{name} IDs must be unique nonempty strings")
    return values


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_spec(path: Path) -> dict:
    path = path.expanduser().resolve()
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("Sweep spec requires schema_version: 1")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(spec.get("dataset", ""))):
        raise ValueError("dataset must use letters, digits, and underscores")
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Sweep spec requires a nonempty sources list")
    trip_ids = [source.get("trip_id") for source in sources]
    if any(not isinstance(item, str) or not item for item in trip_ids) or len(set(trip_ids)) != len(trip_ids):
        raise ValueError("Each source needs a unique nonempty trip_id")
    seeds = spec.get("split_seeds", [4022, 4023, 4024])
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("split_seeds must contain three distinct integers")
    spec["split_seeds"] = seeds
    spec["training_seed"] = int(spec.get("training_seed", 4022))
    spec["subtrip_size"] = int(spec.get("subtrip_size", 10))
    if spec["subtrip_size"] < 2:
        raise ValueError("subtrip_size must be at least 2 frames")
    checkpoint = resolve_path(spec["model_checkpoint"], path.parent)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    spec["model_checkpoint"] = str(checkpoint)
    spec["_path"] = path
    return spec


def discover_frames(spec: dict) -> list[Frame]:
    frames = []
    seen_ids, seen_images, seen_masks = set(), set(), set()
    for source in spec["sources"]:
        trip = source["trip_id"]
        images_dir = resolve_path(source["images_dir"], spec["_path"].parent)
        masks_dir = resolve_path(source["masks_dir"], spec["_path"].parent)
        if not images_dir.is_dir() or not masks_dir.is_dir():
            raise NotADirectoryError(f"Missing image or mask directory for {trip}")
        pattern = re.compile(source.get("frame_pattern", r"^.*?(?P<frame>\d+)$"))
        if "frame" not in pattern.groupindex:
            raise ValueError(f"{trip} frame_pattern needs a named (?P<frame>...) capture")
        first = source.get("first_frame")
        last = source.get("last_frame")
        step = int(source.get("frame_step", 1))
        if step < 1:
            raise ValueError("frame_step must be positive")
        if step > 1 and first is None:
            raise ValueError(f"{trip} frame_step needs first_frame as its anchor")
        if first is not None and last is not None and int(first) > int(last):
            raise ValueError(f"{trip} first_frame must not exceed last_frame")
        images = defaultdict(list)
        for path in images_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images[path.stem].append(path.resolve())
        selected_numbers = []
        for mask in sorted(masks_dir.rglob("*.png")):
            match = pattern.fullmatch(mask.stem)
            if match is None:
                continue
            number = int(match.group("frame"))
            if first is not None and number < int(first):
                continue
            if last is not None and number > int(last):
                continue
            if first is not None and (number - int(first)) % step != 0:
                continue
            matched_images = images.get(mask.stem, [])
            if len(matched_images) != 1:
                raise ValueError(f"Expected one image for {mask}; found {len(matched_images)}")
            image = matched_images[0]
            mask = mask.resolve()
            image_id = f"{trip}/{mask.stem}"
            if image_id in seen_ids or image in seen_images or mask in seen_masks:
                raise ValueError(f"Repeated image, mask, or image ID: {image_id}")
            seen_ids.add(image_id)
            seen_images.add(image)
            seen_masks.add(mask)
            selected_numbers.append(number)
            hash_content = bool(spec.get("hash_content", False))
            frames.append(Frame(image_id, image, mask, trip, number,
                                sha256_file(image) if hash_content else "",
                                sha256_file(mask) if hash_content else ""))
        if not selected_numbers:
            raise ValueError(f"No paired frames selected for {trip}")
        if len(selected_numbers) != len(set(selected_numbers)):
            raise ValueError(f"Repeated frame number in {trip}")
        if source.get("require_complete_range", False):
            if first is None or last is None:
                raise ValueError("require_complete_range needs first_frame and last_frame")
            expected = list(range(int(first), int(last) + 1, step))
            if sorted(selected_numbers) != expected:
                raise ValueError(f"{trip} does not have every requested frame and mask")
    return sorted(frames, key=lambda frame: (frame.trip_id, frame.frame_number))


def largest_remainder(total: int, shares: tuple[float, ...], names: tuple) -> dict:
    if len(shares) != len(names) or not math.isclose(sum(shares), 1, abs_tol=1e-9):
        raise ValueError("Shares must match names and sum to one")
    raw = [total * share for share in shares]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(len(names)), key=lambda i: (-(raw[i] - counts[i]), i))[:total - sum(counts)]:
        counts[index] += 1
    return dict(zip(names, counts))


def proportional_counts(frames: list[Frame], target: int) -> dict[str, int]:
    by_trip = defaultdict(list)
    for frame in frames:
        by_trip[frame.trip_id].append(frame)
    trips = sorted(by_trip)
    return largest_remainder(target, tuple(len(by_trip[trip]) / len(frames) for trip in trips), tuple(trips))


def held_out_ids(frames: list[Frame], spec: dict) -> set[str]:
    explicit = configured_ids(spec, "held_out")
    count = int(spec.get("heldout_count", len(explicit) if explicit is not None
                         else math.floor(len(frames) * .05 + .5)))
    if not 0 < count < len(frames):
        raise ValueError("Held-out count must be between zero and inventory size")
    if explicit is not None:
        selected = set(explicit)
        if len(selected) != count or not selected.issubset({frame.image_id for frame in frames}):
            raise ValueError("held_out_image_ids must contain exactly heldout_count valid IDs")
        return selected
    quotas = proportional_counts(frames, count)
    by_trip = defaultdict(list)
    for frame in frames:
        by_trip[frame.trip_id].append(frame)
    return {frame.image_id for trip, members in by_trip.items()
            for frame in members[-quotas[trip]:] if quotas[trip]}


def ordered_test_candidates(frames: list[Frame]) -> list[Frame]:
    """Interleave late frames from each trip; every test size is a prefix."""
    by_trip = defaultdict(list)
    for frame in frames:
        by_trip[frame.trip_id].append(frame)
    ranked = []
    for trip in sorted(by_trip):
        descending = sorted(by_trip[trip], key=lambda frame: frame.frame_number, reverse=True)
        for index, frame in enumerate(descending):
            ranked.append(((index + .5) / len(descending), trip, frame))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [frame for _, _, frame in ranked]


def consecutive_validation(pool: list[Frame], target: int) -> set[str]:
    quotas = proportional_counts(pool, target)
    by_trip = defaultdict(list)
    for frame in pool:
        by_trip[frame.trip_id].append(frame)
    return {frame.image_id for trip, members in by_trip.items()
            for frame in members[-quotas[trip]:] if quotas[trip]}


def group_choice(groups: list[Subtrip], target: int, seed: int | None = None) -> set[str] | None:
    ordered = list(groups)
    if seed is not None:
        random.Random(seed).shuffle(ordered)
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered:
        size = len(group.frames)
        for count, picked in sorted(list(reachable.items()), reverse=True):
            new_count = count + size
            if new_count <= target and new_count not in reachable:
                reachable[new_count] = picked + (group.group_id,)
    return set(reachable[target]) if target in reachable else None


def group_assignment_count(groups: list[Subtrip], target: int) -> int:
    ways = [0] * (target + 1)
    ways[0] = 1
    for group in groups:
        size = len(group.frames)
        for count in range(target, size - 1, -1):
            ways[count] += ways[count - size]
    return ways[target]


def make_subtrips(pool: list[Frame], size: int, target: int,
                  frame_steps: dict[str, int]) -> list[Subtrip]:
    by_trip = defaultdict(list)
    for frame in pool:
        by_trip[frame.trip_id].append(frame)

    def build_group(trip: str, members: list[Frame]) -> Subtrip:
        return Subtrip(f"{trip}:{members[0].frame_number}-{members[-1].frame_number}", tuple(members))

    groups = []
    for trip, members in sorted(by_trip.items()):
        contiguous_runs = []
        current = []
        for frame in members:
            if current and frame.frame_number - current[-1].frame_number != frame_steps[trip]:
                contiguous_runs.append(current)
                current = []
            current.append(frame)
        if current:
            contiguous_runs.append(current)
        for run in contiguous_runs:
            chunks = [run[index:index + size] for index in range(0, len(run), size)]
            if len(chunks) > 1 and len(chunks[-1]) < max(2, size // 2):
                chunks[-2].extend(chunks.pop())
            groups.extend(build_group(trip, chunk) for chunk in chunks)
    if group_assignment_count(groups, target) >= 3:
        return groups
    # Move a boundary first, keeping all windows reasonably sized.
    options = []
    for index in range(len(groups) - 1):
        left, right = groups[index:index + 2]
        if left.frames[0].trip_id != right.frames[0].trip_id:
            continue
        if right.frames[0].frame_number - left.frames[-1].frame_number != frame_steps[left.frames[0].trip_id]:
            continue
        combined = list(left.frames + right.frames)
        for cut in range(2, len(combined) - 1):
            if cut == len(left.frames):
                continue
            replacements = [build_group(combined[0].trip_id, combined[:cut]),
                            build_group(combined[0].trip_id, combined[cut:])]
            if max(len(part.frames) for part in replacements) > math.ceil(size * 1.5):
                continue
            candidate = groups[:index] + replacements + groups[index + 2:]
            ways = group_assignment_count(candidate, target)
            if ways >= 3:
                options.append((min(len(part.frames) for part in replacements),
                                ways, -index, -abs(cut - len(left.frames)), candidate))
    if options:
        return max(options, key=lambda option: option[:4])[4]
    # If boundary movement cannot do it, split one window into two fixed
    # contiguous sub-trips. Never create a one-frame sub-trip.
    for index, group in enumerate(groups):
        members = list(group.frames)
        for cut in range(2, len(members) - 1):
            replacements = [build_group(members[0].trip_id, members[:cut]),
                            build_group(members[0].trip_id, members[cut:])]
            candidate = groups[:index] + replacements + groups[index + 1:]
            ways = group_assignment_count(candidate, target)
            if ways >= 3:
                options.append((min(cut, len(members) - cut), ways, -index, -cut, candidate))
    if not options:
        raise ValueError(f"Cannot create three exact {target}-image validation mixes by sub-trip")
    return max(options, key=lambda option: option[:4])[4]


def temporal_gap_counts(frames: list[Frame], roles: dict[str, str],
                        frame_steps: dict[str, int]) -> dict[str, int]:
    by_role_trip = defaultdict(list)
    for frame in frames:
        by_role_trip[(roles[frame.image_id], frame.trip_id)].append(frame.frame_number)
    result = {role: 0 for role in ("train", "validation", "test", "held_out")}
    for (role, trip), numbers in by_role_trip.items():
        ordered = sorted(numbers)
        result[role] += sum(right - left != frame_steps[trip]
                            for left, right in zip(ordered, ordered[1:]))
    return result


def image_content_overlap_count(frames: list[Frame], roles: dict[str, str]) -> int:
    owners = defaultdict(set)
    for frame in frames:
        if frame.image_sha256:
            owners[frame.image_sha256].add(roles[frame.image_id])
    return sum(len(assigned_roles) > 1 for assigned_roles in owners.values())


def shuffled_validation(pool: list[Frame], target: int, method: int,
                        seed: int, groups: list[Subtrip] | None,
                        seen: set[frozenset[str]]) -> tuple[set[str], int]:
    for attempt in range(100):
        effective_seed = seed + 9973 * attempt
        if method == 2:
            assert groups is not None
            picked = group_choice(groups, target, effective_seed)
            assert picked is not None
            selected = {frame.image_id for group in groups if group.group_id in picked
                        for frame in group.frames}
        else:
            selected = {frame.image_id for frame in random.Random(effective_seed).sample(pool, target)}
        frozen = frozenset(selected)
        if frozen not in seen:
            seen.add(frozen)
            return selected, effective_seed
    raise ValueError(f"Could not create three distinct mixes: method={method}, "
                     f"target={target}, groups={[len(group.frames) for group in groups or []]}, "
                     f"existing={len(seen)}")


def generate(spec_path: Path, output_dir: Path, *, refresh: bool = False) -> None:
    spec = load_spec(spec_path)
    frames = discover_frames(spec)
    held = held_out_ids(frames, spec)
    development = [frame for frame in frames if frame.image_id not in held]
    counts = {version: largest_remainder(len(development), shares,
              ("train", "validation", "test")) for version, shares in RATIOS.items()}
    core_count = min(entry["test"] for entry in counts.values())
    test_order = ordered_test_candidates(development)
    common_ids = {frame.image_id for frame in test_order[:core_count]}
    by_id = {frame.image_id: frame for frame in frames}
    frame_steps = {source["trip_id"]: int(source.get("frame_step", 1))
                   for source in spec["sources"]}
    pinned_test = configured_ids(spec, "test_order")
    if pinned_test is not None:
        if (len(pinned_test) < max(entry["test"] for entry in counts.values()) or
            not set(pinned_test).issubset({frame.image_id for frame in development})):
            raise ValueError("test_order IDs must include every possible test image and exclude held-out IDs")
        pinned_set = set(pinned_test)
        test_order = [by_id[image_id] for image_id in pinned_test] + [
            frame for frame in test_order if frame.image_id not in pinned_set]
        common_ids = {frame.image_id for frame in test_order[:core_count]}

    # The consecutive method requires each train and validation assignment to
    # form an unbroken block within each recording. A scattered pinned reserve
    # or test selection can make that impossible even when counts still match.
    for version, expected in counts.items():
        test_ids = {frame.image_id for frame in test_order[:expected["test"]]}
        pool = [frame for frame in development if frame.image_id not in test_ids]
        val_ids = consecutive_validation(pool, expected["validation"])
        roles = {frame.image_id: ("held_out" if frame.image_id in held else
                                  "test" if frame.image_id in test_ids else
                                  "validation" if frame.image_id in val_ids else "train")
                 for frame in frames}
        gaps = temporal_gap_counts(frames, roles, frame_steps)
        if gaps["train"] or gaps["validation"]:
            raise ValueError(f"Ratio version {version} cannot make consecutive train/validation "
                             "blocks with the chosen held-out/test IDs; adjust the fixed selections")

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        existing = output_dir / "study.json"
        if not refresh or not existing.is_file():
            raise FileExistsError(f"Study directory is not empty: {output_dir}")
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous.get("spec_path") != str(spec["_path"]) or previous.get("dataset") != spec["dataset"]:
            raise ValueError("Refusing to refresh a study created from a different spec")
        if any((output_dir / "runs" / run / "head_only_lr1e-3_wd1e-3_20ep").exists()
               for run in previous["runs"]):
            raise ValueError("Refusing to refresh a study after training has started")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(exist_ok=refresh)
    inventory_rows = [{"image_id": frame.image_id, "image_path": str(frame.image_path),
                       "mask_path": str(frame.mask_path), "trip_id": frame.trip_id,
                       "frame_number": frame.frame_number, "image_sha256": frame.image_sha256,
                       "mask_sha256": frame.mask_sha256} for frame in frames]
    write_csv(output_dir / "source_inventory.csv", INVENTORY_FIELDS, inventory_rows)
    test_rank = {frame.image_id: index + 1 for index, frame in enumerate(test_order)}
    write_csv(output_dir / "fixed_partitions.csv", ("image_id", "partition", "test_rank"),
              ({"image_id": frame.image_id,
                "partition": "held_out" if frame.image_id in held else "development",
                "test_rank": "" if frame.image_id in held else test_rank[frame.image_id]}
               for frame in frames))
    inventory_hash = sha256_file(output_dir / "source_inventory.csv")
    fixed_hash = sha256_file(output_dir / "fixed_partitions.csv")
    study = {"dataset": spec["dataset"], "spec_path": str(spec["_path"]),
             "spec_sha256": sha256_file(spec["_path"]), "inventory_count": len(frames),
             "model_checkpoint_path": spec["model_checkpoint"],
             "model_checkpoint_sha256": sha256_file(Path(spec["model_checkpoint"])),
             "development_count": len(development), "held_out_count": len(held),
             "held_out_fraction_actual": len(held) / len(frames),
             "common_test_count": core_count, "held_out_ids_sha256": sha256_ids(held),
             "common_test_ids_sha256": sha256_ids(common_ids),
             "inventory_csv_sha256": inventory_hash, "fixed_partitions_csv_sha256": fixed_hash,
             "content_hashes_recorded": bool(spec.get("hash_content", False)),
             "training_recipe": {"optimizer": "AdamW", "learning_rate": .001,
                                 "weight_decay": .001, "epochs": 20, "batch_size": 2,
                                 "class_weights": [1, 1, 1]},
             "runs": []}
    summary_rows = []
    for version in RATIOS:
        expected = counts[version]
        test_ids = {frame.image_id for frame in test_order[:expected["test"]]}
        pool = [frame for frame in development if frame.image_id not in test_ids]
        if len(pool) != expected["train"] + expected["validation"]:
            raise AssertionError("Incorrect train/validation pool")
        subtrips = make_subtrips(pool, spec["subtrip_size"],
                                expected["validation"], frame_steps)
        group_by_id = {frame.image_id: group.group_id for group in subtrips
                       for frame in group.frames}
        for method in METHODS:
            seen: set[frozenset[str]] = set()
            iterations = (1,) if method == 1 else (1, 2, 3)
            for iteration in iterations:
                if method == 1:
                    val_ids = consecutive_validation(pool, expected["validation"])
                    split_seed = None
                else:
                    val_ids, split_seed = shuffled_validation(
                        pool, expected["validation"], method,
                        spec["split_seeds"][iteration - 1],
                        subtrips if method == 2 else None, seen)
                train_ids = {frame.image_id for frame in pool} - val_ids
                assert len(train_ids) == expected["train"]
                roles = {image_id: "train" for image_id in train_ids}
                roles.update({image_id: "validation" for image_id in val_ids})
                roles.update({image_id: "test" for image_id in test_ids})
                roles.update({image_id: "held_out" for image_id in held})
                if set(roles) != set(by_id):
                    raise AssertionError("An inventory image was left unassigned")
                run_id = f"{spec['dataset']}_{version}_{method}_{iteration}"
                run_dir = output_dir / "runs" / run_id
                run_dir.mkdir(exist_ok=refresh)
                manifest_path = run_dir / "split_manifest.csv"
                write_csv(manifest_path, MANIFEST_FIELDS,
                          ({"image_id": frame.image_id, "image_path": str(frame.image_path),
                            "mask_path": str(frame.mask_path), "trip_id": frame.trip_id,
                            "frame_number": frame.frame_number,
                            "subtrip_id": group_by_id.get(frame.image_id, "") if method == 2 else "",
                            "role": roles[frame.image_id]} for frame in frames))
                actual = {role: sum(value == role for value in roles.values())
                          for role in ("train", "validation", "test", "held_out")}
                if actual != {**expected, "held_out": len(held)}:
                    raise AssertionError(f"Incorrect split counts for {run_id}: {actual}")
                audit = {"run_id": run_id, "ratio_version": version,
                         "development_ratio": dict(zip(("train", "validation", "test"), RATIOS[version])),
                         "sorting_version": method, "sorting_method": METHODS[method],
                         "iteration": iteration, "split_seed": split_seed,
                         "training_seed": spec["training_seed"], "counts": actual,
                         "unique_image_ids": len(roles), "path_overlap_count": 0,
                         "image_content_overlap_count": (
                             image_content_overlap_count(frames, roles)
                             if spec.get("hash_content", False) else None),
                         "inventory_csv_sha256": inventory_hash,
                         "fixed_partitions_csv_sha256": fixed_hash,
                         "held_out_ids_sha256": sha256_ids(held),
                         "test_ids_sha256": sha256_ids(test_ids),
                         "common_test_ids_sha256": sha256_ids(common_ids),
                         "split_manifest_csv_sha256": sha256_file(manifest_path),
                         "subtrip_sizes": sorted({len(group.frames) for group in subtrips}) if method == 2 else [],
                         "temporal_gap_counts": temporal_gap_counts(frames, roles, frame_steps),
                         "warning": "All windows come from the declared recordings; sub-trips are not independent trips."}
                run_yaml = {
                    "schema_version": 1, "dataset": spec["dataset"],
                    "experiment": {"name": run_id, "output_dir": str(run_dir),
                                   "seed": spec["training_seed"]},
                    "labels": {"obstacle": 0, "water": 1, "sky": 2},
                    "data": {"manifest_csv": str(manifest_path),
                             "expected_counts": actual,
                             "common_test_image_paths": [str(frame.image_path)
                                                         for frame in test_order[:core_count]]},
                    "split_policy": {"ratio_version": version, "sorting_version": method,
                                     "iteration": iteration, "split_seed": split_seed},
                    "model": {"architecture": "ewasr_resnet18",
                              "checkpoint": spec["model_checkpoint"], "imu": False,
                              "trainable_module": "decoder.seg_head"},
                    "preprocessing": {"input_width": 512, "input_height": 384,
                                      "mask_width": 128, "mask_height": 96,
                                      "mask_interpolation": "nearest",
                                      "image_normalization": "imagenet",
                                      "image_mean": [.485, .456, .406],
                                      "image_std": [.229, .224, .225]},
                    "training": {"optimizer": "adamw", "learning_rate": .001,
                                 "weight_decay": .001, "epochs": 20, "batch_size": 2,
                                 "class_weights": [1, 1, 1],
                                 "checkpoint_selection_metric": "validation_binary_water_iou"},
                }
                config_path = run_dir / "config.yaml"
                config_path.write_text(yaml.safe_dump(run_yaml, sort_keys=False), encoding="utf-8")
                audit["config_yaml_sha256"] = sha256_file(config_path)
                (run_dir / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
                study["runs"].append(run_id)
                summary_rows.append({"run_id": run_id, "ratio_version": version,
                                     "sorting_version": method, "iteration": iteration,
                                     "train": actual["train"], "validation": actual["validation"],
                                     "test": actual["test"], "held_out": actual["held_out"],
                                     "status": "generated"})
    (output_dir / "study.json").write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "summary.csv",
              ("run_id", "ratio_version", "sorting_version", "iteration", "train",
               "validation", "test", "held_out", "status"), summary_rows)
    print(f"Generated {len(study['runs'])} runs from {len(frames)} paired images in {output_dir}")
    print(f"Held out: {len(held)}; common test core: {core_count}; no training started")


def selected_runs(study_dir: Path, run_id: str | None) -> list[tuple[str, Path]]:
    study = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    ids = study["runs"]
    if run_id is not None:
        if run_id not in ids:
            raise ValueError(f"Unknown run ID: {run_id}")
        ids = [run_id]
    return [(item, study_dir / "runs" / item / "config.yaml") for item in ids]


def verify_generated_run(study_dir: Path, name: str, config_path: Path) -> None:
    study = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    if sha256_file(Path(study["spec_path"])) != study["spec_sha256"]:
        raise ValueError("Sweep spec changed after generation")
    if sha256_file(Path(study["model_checkpoint_path"])) != study["model_checkpoint_sha256"]:
        raise ValueError("Model starting checkpoint changed after generation")
    inventory_path = study_dir / "source_inventory.csv"
    fixed_path = study_dir / "fixed_partitions.csv"
    if (sha256_file(inventory_path) != study["inventory_csv_sha256"] or
        sha256_file(fixed_path) != study["fixed_partitions_csv_sha256"]):
        raise ValueError("Source inventory or fixed partitions changed after generation")
    audit = json.loads((config_path.parent / "split_audit.json").read_text(encoding="utf-8"))
    if sha256_file(config_path) != audit["config_yaml_sha256"]:
        raise ValueError(f"Generated run config changed after generation: {name}")
    manifest_path = config_path.parent / "split_manifest.csv"
    if sha256_file(manifest_path) != audit["split_manifest_csv_sha256"]:
        raise ValueError(f"Split manifest changed after generation: {name}")
    with manifest_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    ids = [row["image_id"] for row in rows]
    with inventory_path.open(newline="", encoding="utf-8") as source:
        inventory = list(csv.DictReader(source))
    if len(ids) != len(set(ids)) or set(ids) != {row["image_id"] for row in inventory}:
        raise ValueError(f"Split manifest does not cover the source inventory: {name}")
    with fixed_path.open(newline="", encoding="utf-8") as source:
        fixed = list(csv.DictReader(source))
    held = {row["image_id"] for row in rows if row["role"] == "held_out"}
    test = {row["image_id"] for row in rows if row["role"] == "test"}
    expected_test = {row["image_id"] for row in fixed if row["test_rank"] and
                     int(row["test_rank"]) <= audit["counts"]["test"]}
    if (sha256_ids(held) != study["held_out_ids_sha256"] or
        sha256_ids(test) != audit["test_ids_sha256"] or test != expected_test):
        raise ValueError(f"Fixed held-out or nested test assignment changed: {name}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    common_paths = set(config["data"]["common_test_image_paths"])
    common = {row["image_id"] for row in rows if row["role"] == "test" and
              row["image_path"] in common_paths}
    if sha256_ids(common) != study["common_test_ids_sha256"]:
        raise ValueError(f"Common test core changed: {name}")


def check(study_dir: Path, run_id: str | None, python: str) -> None:
    for name, config in selected_runs(study_dir, run_id):
        print(f"Checking {name}", flush=True)
        verify_generated_run(study_dir, name, config)
        subprocess.run([python, str(HERE / "train_head.py"), "--config", str(config), "--check-data"], check=True)


def train(study_dir: Path, run_id: str | None, all_runs: bool, python: str) -> None:
    if (run_id is None) == (not all_runs):
        raise ValueError("Choose exactly one of --run-id or --all for training")
    for name, config in selected_runs(study_dir, run_id):
        verify_generated_run(study_dir, name, config)
        output = config.parent / "head_only_lr1e-3_wd1e-3_20ep"
        if (output / "evaluation.txt").is_file():
            print(f"Already complete: {name}", flush=True)
            continue
        if output.exists() and {path.name for path in output.iterdir()} - {"resolved_splits.json"}:
            raise FileExistsError(f"Partial output for {name}; inspect it before retrying: {output}")
        print(f"Training {name}", flush=True)
        subprocess.run([python, str(HERE / "train_head.py"), "--config", str(config), "--check-data"], check=True)
        log_path = config.parent / "training.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [python, "-u", str(HERE / "train_head.py"), "--config", str(config)],
                stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Training {name} failed; inspect {log_path}")
        summarize(study_dir)


def read_evaluation(path: Path) -> dict:
    result = {}
    section = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "Best validation epoch:":
            section = "validation"
        elif line == "Selected checkpoint on test split:":
            section = "test"
        elif line == "Selected checkpoint on common test core:":
            section = "common_test"
        elif line.startswith("Best epoch: "):
            result["best_epoch"] = line.partition(": ")[2]
        elif line.startswith("Total elapsed (data loading, training and evaluation): "):
            result["elapsed_seconds"] = line.partition(": ")[2].split()[0]
        elif section and line.startswith("  ") and ": " in line:
            name, value = line.strip().split(": ", 1)
            if name in {"binary_water_iou", "mean_iou", "obstacle_iou", "obstacle_miss_rate"}:
                result[f"{section}_{name}"] = value
    return result


def summarize(study_dir: Path) -> None:
    rows = []
    for name, config in selected_runs(study_dir, None):
        audit = json.loads((config.parent / "split_audit.json").read_text(encoding="utf-8"))
        evaluation = config.parent / "head_only_lr1e-3_wd1e-3_20ep" / "evaluation.txt"
        row = {"run_id": name, "ratio_version": audit["ratio_version"],
               "sorting_version": audit["sorting_version"], "iteration": audit["iteration"],
               **audit["counts"], "status": "complete" if evaluation.is_file() else "generated"}
        if evaluation.is_file():
            row.update(read_evaluation(evaluation))
        rows.append(row)
    fields = ("run_id", "ratio_version", "sorting_version", "iteration", "train",
              "validation", "test", "held_out", "status", "best_epoch", "elapsed_seconds",
              "validation_binary_water_iou", "validation_mean_iou",
              "test_binary_water_iou", "test_mean_iou", "common_test_binary_water_iou",
              "common_test_mean_iou", "common_test_obstacle_iou",
              "common_test_obstacle_miss_rate")
    write_csv(study_dir / "summary.csv", fields,
              ({field: row.get(field, "") for field in fields} for row in rows))
    print(f"Updated {study_dir / 'summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate", help="Create manifests and configs; do not train")
    generate_parser.add_argument("--spec", required=True, type=Path)
    generate_parser.add_argument("--output-dir", required=True, type=Path)
    generate_parser.add_argument("--refresh", action="store_true",
                                 help="Replace generated manifests only before any training starts")
    for command in ("check", "train"):
        subparser = commands.add_parser(command)
        subparser.add_argument("study_dir", type=Path)
        subparser.add_argument("--run-id")
        subparser.add_argument("--python", default=sys.executable)
        if command == "train":
            subparser.add_argument("--all", action="store_true")
    summary_parser = commands.add_parser("summary")
    summary_parser.add_argument("study_dir", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.spec, args.output_dir, refresh=args.refresh)
    elif args.command == "check":
        check(args.study_dir.resolve(), args.run_id, args.python)
    elif args.command == "train":
        train(args.study_dir.resolve(), args.run_id, args.all, args.python)
    else:
        summarize(args.study_dir.resolve())


if __name__ == "__main__":
    main()
