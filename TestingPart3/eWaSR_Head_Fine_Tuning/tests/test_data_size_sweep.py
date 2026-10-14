"""Check experiment isolation before spending GPU time on a size sweep."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
import data_size_sweep as size_sweep
import split_sweep
import train_head


class DataSizeSweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        images, masks = self.root / "images", self.root / "masks"
        images.mkdir()
        masks.mkdir()
        for index in range(100):
            Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(images / f"frame_{index:03}.png")
            Image.fromarray(np.ones((8, 12), dtype=np.uint8)).save(masks / f"frame_{index:03}.png")
        checkpoint = self.root / "weights.npz"
        checkpoint.write_bytes(b"no GPU required")
        self.spec = self.root / "source.yaml"
        self.spec.write_text(yaml.safe_dump({"schema_version": 1, "dataset": "fixture",
            "model_checkpoint": str(checkpoint), "subtrip_size": 10,
            "split_seeds": [4022, 4023, 4024], "sources": [{"trip_id": "trip",
                "images_dir": str(images), "masks_dir": str(masks),
                "frame_pattern": r"^frame_(?P<frame>\d+)$", "first_frame": 0,
                "last_frame": 99, "frame_step": 1, "require_complete_range": True}]}))
        self.parent_study = self.root / "splits"
        split_sweep.generate(self.spec, self.parent_study)
        self.base = self.parent_study / "runs" / "fixture_3_2_1" / "config.yaml"
        self.study = self.root / "sizes"

    def tearDown(self):
        self.temp.cleanup()

    def test_training_changes_but_all_evaluation_images_are_fixed_and_unused_is_excluded(self):
        size_sweep.generate(self.spec, self.base, self.study)
        study = json.loads((self.study / "study.json").read_text())
        self.assertEqual(len(study["runs"]), 10)
        parent = size_sweep.read_csv(self.study / "base_split_manifest.csv")
        expected_train = sum(row["role"] == "train" for row in parent)
        selections = {}
        for name in study["runs"]:
            directory = self.study / "runs" / name
            audit = json.loads((directory / "size_audit.json").read_text())
            config, loaded = train_head.load_configured_splits(directory / "config.yaml")
            self.assertEqual(len(loaded["train"]), int(expected_train * audit["percentage"] / 100 + .5))
            self.assertEqual(audit["optimizer_updates"], 20 * ((len(loaded["train"]) + 1) // 2))
            active = size_sweep.read_csv(directory / "split_manifest.csv")
            membership = size_sweep.read_csv(directory / "full_membership.csv")
            self.assertEqual(len(membership), 100)
            unused = {row["image_id"] for row in membership if row["role"] == "unused_train"}
            self.assertTrue(unused.isdisjoint({row["image_id"] for row in active}))
            for role in ("validation", "test", "held_out"):
                self.assertEqual([row for row in active if row["role"] == role],
                                 [row for row in parent if row["role"] == role])
            # Even the smallest subset should cover all available training
            # groups in this fixture, rather than selecting only a few windows.
            self.assertEqual({row["subtrip_id"] for row in active if row["role"] == "train"},
                             {row["subtrip_id"] for row in parent if row["role"] == "train"})
            selections.setdefault(audit["iteration"], {})[audit["percentage"]] = {
                row["image_id"] for row in active if row["role"] == "train"}
            self.assertEqual(config["experiment"]["seed"], 4022)
        full = selections[1][100]
        for sizes in selections.values():
            self.assertTrue(sizes[25] < sizes[50] < sizes[75] < full)
        for percentage in (25, 50, 75):
            self.assertEqual(len({frozenset(sizes[percentage]) for sizes in selections.values()}), 3)
        aggregate = size_sweep.read_csv(self.study / "aggregate.csv")
        self.assertEqual([int(row["planned_unique_runs"]) for row in aggregate], [3, 3, 3, 1])
        self.assertTrue(all(row["completed"] == "0" for row in aggregate))

    def test_temporal_prefixes_are_uniform_and_nested_on_equal_windows(self):
        for phase in (0, .11, .37, .79):
            order = size_sweep.phase_order(16, phase)
            self.assertEqual(set(order), set(range(16)))
            for count in (4, 8):
                selected = sorted(order[:count])
                gaps = [b - a for a, b in zip(selected, selected[1:] + [selected[0] + 16])]
                self.assertEqual(set(gaps), {16 // count})

    def test_result_summary_keeps_training_and_total_time_separate(self):
        report = self.root / "evaluation.txt"
        report.write_text("Training elapsed: 123.45 seconds\n"
            "Total elapsed (data loading, training and evaluation): 130.50 seconds\n"
            "Best epoch: 7\nSelected checkpoint on test split:\n"
            "  binary_water_iou: 0.970000\n  mean_iou: 0.960000\n"
            "  obstacle_recall: 0.980000\n  water_as_obstacle_rate: 0.004000\n")
        metrics = size_sweep.read_result(report)
        self.assertEqual(float(metrics["training_seconds"]), 123.45)
        self.assertEqual(float(metrics["elapsed_seconds"]), 130.50)
        self.assertEqual(float(metrics["test_binary_water_iou"]), .97)
        self.assertEqual(float(metrics["test_obstacle_recall"]), .98)

    def test_recipe_generation_and_integrity_check(self):
        recipe = self.root / "recipe.yaml"
        recipe.write_text(yaml.safe_dump({"schema_version": 1, "kind": "training_data_size_sweep",
            "source_spec": self.spec.name, "base_config": str(self.base.relative_to(self.root)),
            "output_dir": self.study.name, "sampling_method": "nested_uniform_temporal",
            "percentages": [25, 50, 75, 100], "selection_seeds": [4022, 4023, 4024],
            "reuse_baseline_result": False}))
        size_sweep.generate_from_config(recipe)
        size_sweep.verify(self.study)
        study = json.loads((self.study / "study.json").read_text())
        self.assertFalse(study["baseline_reused"])
        manifest = self.study / "runs" / study["runs"][0] / "split_manifest.csv"
        manifest.write_text(manifest.read_text().replace(",train", ",test", 1))
        with self.assertRaisesRegex(ValueError, "Generated file changed"):
            size_sweep.verify(self.study)

    def test_summary_estimates_subset_spread_but_not_single_endpoint_spread(self):
        size_sweep.generate(self.spec, self.base, self.study)
        study = json.loads((self.study / "study.json").read_text())
        for name in study["runs"]:
            directory = self.study / "runs" / name
            audit = json.loads((directory / "size_audit.json").read_text())
            result = Path(audit["result_dir"])
            result.mkdir()
            score = .9 + .001 * audit["iteration"]
            (result / "evaluation.txt").write_text(
                f"Training elapsed: {100 + audit['iteration']} seconds\n"
                "Total elapsed (data loading, training and evaluation): 120 seconds\n"
                "Best epoch: 7\nBest validation epoch:\n"
                "  binary_water_iou: 0.950000\n"
                f"Selected checkpoint on test split:\n  binary_water_iou: {score:.6f}\n"
                "  mean_iou: 0.940000\nSelected checkpoint on common test core:\n"
                f"  binary_water_iou: {score:.6f}\n")
        size_sweep.summarize(self.study)
        aggregate = size_sweep.read_csv(self.study / "aggregate.csv")
        self.assertAlmostEqual(float(aggregate[0]["test_binary_water_iou_mean"]), .902)
        self.assertAlmostEqual(float(aggregate[0]["test_binary_water_iou_sd"]), .001)
        self.assertEqual(aggregate[-1]["completed"], "1")
        self.assertEqual(aggregate[-1]["test_binary_water_iou_sd"], "")
        self.assertEqual(float(aggregate[0]["training_seconds_mean"]), 102)

    def test_baseline_reuse_rejects_changed_training_recipe_and_membership(self):
        config = yaml.safe_load(self.base.read_text())
        rows = size_sweep.read_csv(Path(config["data"]["manifest_csv"]))
        result = self.root / "baseline"
        result.mkdir()
        actual = {"epochs": 20, "batch_size": 2, "learning_rate": .001,
                  "weight_decay": .001, "class_weights": "1,1,1", "width": 512,
                  "height": 384, "seed": 4022, "selection_metric": "binary_water_iou",
                  "weights": config["model"]["checkpoint"]}
        size_sweep.write_json(result / "run_config.json", actual)
        resolved = {"unseen" if role == "held_out" else role:
                    [{"image": row["image_path"], "mask": row["mask_path"]}
                     for row in rows if row["role"] == role] for role in size_sweep.ROLES}
        size_sweep.write_json(result / "resolved_splits.json", resolved)
        (result / "training_history.jsonl").write_text("{}\n" * 20)
        (result / "best_head_only.npz").write_bytes(b"fixture")
        size_sweep.assert_baseline_result(config, rows, result)
        actual["seed"] = 999
        size_sweep.write_json(result / "run_config.json", actual)
        with self.assertRaisesRegex(ValueError, "recipe"):
            size_sweep.assert_baseline_result(config, rows, result)
        actual["seed"] = 4022
        size_sweep.write_json(result / "run_config.json", actual)
        resolved["test"] = resolved["test"][1:]
        size_sweep.write_json(result / "resolved_splits.json", resolved)
        with self.assertRaisesRegex(ValueError, "membership"):
            size_sweep.assert_baseline_result(config, rows, result)


if __name__ == "__main__":
    unittest.main()
