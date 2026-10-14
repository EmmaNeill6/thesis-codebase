import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


SCRIPT_DIR = Path(__file__).parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


split_sweep = load_module("split_sweep_under_test", "split_sweep.py")
train_head = load_module("train_head_for_sweep_test", "train_head.py")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


class SplitSweepTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        sources = []
        for trip in ("canal", "pier"):
            images = self.root / trip / "images"
            masks = self.root / trip / "masks"
            images.mkdir(parents=True)
            masks.mkdir()
            for index in range(1, 51):
                name = f"frame_{index:03d}"
                Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(images / f"{name}.png")
                Image.fromarray(np.ones((8, 12), dtype=np.uint8)).save(masks / f"{name}.png")
            sources.append({"trip_id": trip, "images_dir": str(images), "masks_dir": str(masks),
                            "frame_pattern": r"^frame_(?P<frame>\d+)$",
                            "first_frame": 1, "last_frame": 50, "require_complete_range": True})
        checkpoint = self.root / "checkpoint.npz"
        checkpoint.write_bytes(b"dry-run only")
        self.spec = self.root / "spec.yaml"
        self.spec.write_text(yaml.safe_dump({"schema_version": 1, "dataset": "small_trip",
                                             "model_checkpoint": str(checkpoint),
                                             "subtrip_size": 10,
                                             "split_seeds": [4022, 4023, 4024],
                                             "sources": sources}), encoding="utf-8")
        self.study = self.root / "study"

    def tearDown(self):
        self.temporary.cleanup()

    def test_generate_exact_reproducible_nested_splits_and_manifest_loader(self):
        split_sweep.generate(self.spec, self.study)
        study = json.loads((self.study / "study.json").read_text())
        self.assertEqual(len(study["runs"]), 35)
        self.assertEqual(study["inventory_count"], 100)
        self.assertEqual(study["held_out_count"], 5)
        self.assertEqual(len(read_rows(self.study / "source_inventory.csv")), 100)
        all_test_sets = {}
        held_sets = set()
        validation_sets = {}
        for run_id in study["runs"]:
            run_dir = self.study / "runs" / run_id
            rows = read_rows(run_dir / "split_manifest.csv")
            audit = json.loads((run_dir / "split_audit.json").read_text())
            split_sweep.verify_generated_run(self.study, run_id, run_dir / "config.yaml")
            self.assertEqual(len(rows), 100)
            self.assertEqual(len({row["image_id"] for row in rows}), 100)
            counts = {role: sum(row["role"] == role for row in rows)
                      for role in ("train", "validation", "test", "held_out")}
            self.assertEqual(counts, audit["counts"])
            self.assertEqual(audit["path_overlap_count"], 0)
            self.assertIsNone(audit["image_content_overlap_count"])
            if audit["sorting_version"] == 1:
                self.assertEqual(audit["temporal_gap_counts"]["train"], 0)
                self.assertEqual(audit["temporal_gap_counts"]["validation"], 0)
            held_sets.add(frozenset(row["image_id"] for row in rows if row["role"] == "held_out"))
            test_set = frozenset(row["image_id"] for row in rows if row["role"] == "test")
            all_test_sets.setdefault(audit["ratio_version"], set()).add(test_set)
            if audit["sorting_version"] in (2, 3):
                key = (audit["ratio_version"], audit["sorting_version"])
                validation_sets.setdefault(key, set()).add(frozenset(
                    row["image_id"] for row in rows if row["role"] == "validation"))
            if audit["sorting_version"] == 2:
                self.assertGreaterEqual(min(audit["subtrip_sizes"]), 2)
                group_roles = {}
                for row in rows:
                    if row["subtrip_id"]:
                        group_roles.setdefault(row["subtrip_id"], set()).add(row["role"])
                self.assertTrue(group_roles)
                self.assertTrue(all(len(roles) == 1 for roles in group_roles.values()))
        self.assertEqual(len(held_sets), 1)
        self.assertTrue(all(len(sets) == 1 for sets in all_test_sets.values()))
        self.assertTrue(all(len(sets) == 3 for sets in validation_sets.values()))
        nested = [next(iter(sets)) for _, sets in sorted(all_test_sets.items(),
                  key=lambda item: len(next(iter(item[1]))))]
        self.assertTrue(all(left.issubset(right) for left, right in zip(nested, nested[1:])))

        run_id = "small_trip_3_2_1"
        config, splits = train_head.load_configured_splits(self.study / "runs" / run_id / "config.yaml")
        self.assertEqual(sum(map(len, splits.values())), 100)
        self.assertEqual(len(splits["unseen"]), 5)
        self.assertTrue(all(sample.image_id for sample in splits["test"]))
        self.assertEqual(len(config["data"]["common_test_image_paths"]), study["common_test_count"])

    def test_rejects_missing_frame_in_required_consecutive_range(self):
        (self.root / "canal" / "masks" / "frame_025.png").unlink()
        with self.assertRaisesRegex(ValueError, "does not have every requested frame"):
            split_sweep.generate(self.spec, self.study)

    def test_explicit_fixed_heldout_and_test_order_are_preserved(self):
        spec = yaml.safe_load(self.spec.read_text(encoding="utf-8"))
        held = [f"canal/frame_{number:03d}" for number in range(46, 51)]
        test_order = [f"pier/frame_{number:03d}" for number in range(50, 31, -1)]
        spec["held_out_image_ids"] = held
        spec["test_order_image_ids"] = test_order
        self.spec.write_text(yaml.safe_dump(spec), encoding="utf-8")
        split_sweep.generate(self.spec, self.study)
        rows = read_rows(self.study / "runs" / "small_trip_2_1_1" / "split_manifest.csv")
        self.assertEqual({row["image_id"] for row in rows if row["role"] == "held_out"}, set(held))
        self.assertEqual({row["image_id"] for row in rows if row["role"] == "test"}, set(test_order))
        grouped = {}
        for row in read_rows(self.study / "runs" / "small_trip_2_2_1" / "split_manifest.csv"):
            if row["subtrip_id"]:
                grouped.setdefault(row["subtrip_id"], []).append(int(row["frame_number"]))
        self.assertTrue(all(all(right - left == 1 for left, right in
                                zip(sorted(numbers), sorted(numbers)[1:]))
                            for numbers in grouped.values()))

    def test_rejects_unanchored_frame_step(self):
        spec = yaml.safe_load(self.spec.read_text(encoding="utf-8"))
        spec["sources"][0].pop("first_frame")
        spec["sources"][0]["frame_step"] = 2
        self.spec.write_text(yaml.safe_dump(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frame_step needs first_frame"):
            split_sweep.generate(self.spec, self.study)

    def test_rejects_scattered_fixed_test_for_consecutive_method(self):
        spec = yaml.safe_load(self.spec.read_text(encoding="utf-8"))
        spec["held_out_image_ids"] = [f"canal/frame_{number:03d}"
                                      for number in range(46, 51)]
        spec["test_order_image_ids"] = [f"pier/frame_{number:03d}"
                                         for number in range(31, 50)]
        self.spec.write_text(yaml.safe_dump(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cannot make consecutive train/validation"):
            split_sweep.generate(self.spec, self.study)
        self.assertFalse(self.study.exists())

    def test_optional_content_hash_audit_counts_cross_role_duplicates(self):
        spec = yaml.safe_load(self.spec.read_text(encoding="utf-8"))
        spec["hash_content"] = True
        self.spec.write_text(yaml.safe_dump(spec), encoding="utf-8")
        split_sweep.generate(self.spec, self.study)
        audit = json.loads((self.study / "runs" / "small_trip_1_1_1" /
                            "split_audit.json").read_text())
        self.assertEqual(audit["image_content_overlap_count"], 1)

    def test_preflight_rejects_changed_run_config_and_spec(self):
        split_sweep.generate(self.spec, self.study)
        run_id = "small_trip_1_1_1"
        config = self.study / "runs" / run_id / "config.yaml"
        original = config.read_text()
        config.write_text(original + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "run config changed"):
            split_sweep.verify_generated_run(self.study, run_id, config)
        config.write_text(original)
        self.spec.write_text(self.spec.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "Sweep spec changed"):
            split_sweep.verify_generated_run(self.study, run_id, config)


if __name__ == "__main__":
    unittest.main()
