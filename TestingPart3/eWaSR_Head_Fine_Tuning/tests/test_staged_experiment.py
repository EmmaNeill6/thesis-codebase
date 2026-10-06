import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_staged_experiment as staged
from run_experiment import select_samples
from train_head import Sample, validation_selection_score


class StagedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images, self.masks = self.root / "images", self.root / "masks"
        self.images.mkdir()
        self.masks.mkdir()
        for index in range(1, 21):
            image = np.full((8, 16, 3), index, dtype=np.uint8)
            Image.fromarray(image).save(
                self.images / f"frame_{index:03d}.jpg")
            mask = np.zeros((8, 16), dtype=np.uint8)
            mask[:, 2:10] = 1
            mask[:, 10:] = 2
            if index == 20:
                mask.fill(255)  # Must stay unread by the staged data check.
            Image.fromarray(mask, mode="L").save(self.masks / f"frame_{index:03d}.png")
        self.checkpoint = self.root / "pretrained.npz"
        self.checkpoint.write_bytes(b"data-only test")
        self.config = yaml.safe_load(staged.DEFAULT_CONFIG.read_text())
        self.config["_config_path"] = self.root / "experiment.yaml"
        self.config["experiment"] = {"name": "tiny", "output_dir": str(self.root / "output"), "seed": 4022}
        self.config["model"]["checkpoint"] = str(self.checkpoint)
        for role, start, end in (("train", 1, 12), ("validation", 13, 16),
                                 ("test", 17, 19), ("unseen", 20, 20)):
            split = self.config["data"][role]
            split["expected_count"] = end - start + 1
            split["sources"] = [{"sequence_id": f"trip_{role}",
                                  "images_dir": str(self.images),
                                  "masks_dir": str(self.masks),
                                  "expected_count": end - start + 1,
                                  "selection": {"kind": "inclusive_range",
                                                "index_pattern": r"^frame_(\d+)$",
                                                "start": start, "end": end}}]
        to_save = copy.deepcopy(self.config)
        del to_save["_config_path"]
        self.config["_config_path"].write_text(yaml.safe_dump(to_save))

    def tearDown(self):
        self.temp.cleanup()

    def test_rounded_ratios_and_unseen_pixels_untouched(self):
        staged.validate_recipe(self.config)
        splits = staged.load_dataset(self.config)
        self.assertEqual({role: len(value) for role, value in splits.items()},
                         {"train": 12, "validation": 4, "test": 3, "unseen": 1})
        self.assertEqual(staged.expected_counts(292),
                         {"train": 175, "validation": 58, "test": 44, "unseen": 15})
        self.assertEqual(self.masks.joinpath("frame_020.png").read_bytes()[:8],
                         b"\x89PNG\r\n\x1a\n")

    def test_sequence_group_cannot_cross_roles(self):
        self.config["data"]["unseen"]["sources"][0]["sequence_id"] = "trip_train"
        with self.assertRaisesRegex(ValueError, "crosses split boundaries"):
            staged.load_dataset(self.config)

    def test_interleaved_consecutive_frames_are_rejected(self):
        train = self.config["data"]["train"]
        odd = copy.deepcopy(train["sources"][0])
        odd["sequence_id"] = "odd_trip"
        odd["selection"] = {"kind": "every_nth", "index_pattern": r"^frame_(\d+)$",
                            "start": 1, "end": 19, "step": 2}
        odd["expected_count"] = 10
        early_even = copy.deepcopy(odd)
        early_even["sequence_id"] = "early_even_trip"
        early_even["selection"].update(start=2, end=4)
        early_even["expected_count"] = 2
        train["sources"] = [odd, early_even]
        for role, start, end, count in (("validation", 6, 12, 4),
                                        ("test", 14, 18, 3), ("unseen", 20, 20, 1)):
            source = self.config["data"][role]["sources"][0]
            source["selection"] = {"kind": "every_nth", "index_pattern": r"^frame_(\d+)$",
                                   "start": start, "end": end, "step": 2}
            source["expected_count"] = count
        with self.assertRaisesRegex(ValueError, "interleave"):
            staged.load_dataset(self.config)

    def test_explicit_gate_previews_without_writing(self):
        with patch.object(sys, "argv", ["run_staged_experiment.py", "--config",
                                        str(self.config["_config_path"]), "--step", "1"]):
            staged.main()
        self.assertFalse((self.root / "output").exists())
        with self.assertRaisesRegex(ValueError, "Complete step 1"):
            staged.stage_previous({"stages": {}}, 2)

    def test_checkpoint_metric_is_binary_water_iou(self):
        record = {"validation": {"mean_iou": .12, "binary_water_iou": .84}}
        self.assertEqual(validation_selection_score(record, "binary_water_iou"), .84)
        with self.assertRaises(ValueError):
            validation_selection_score(record, "test_iou")

    def test_duration_choice_blocks_next_step(self):
        previous = {"parameters": staged.baseline_params(), "best_epoch": 10,
                    "best_validation": {"binary_water_iou": .8}}
        twenty = {"parameters": {**staged.baseline_params(), "epochs": 20},
                  "best_epoch": 16, "best_validation": {"binary_water_iou": .85, "mean_iou": .8},
                  "last_three_validation_water_iou_std": .01, "checkpoint": "best.npz"}
        state = {"stages": {str(n): {"status": "complete", "selected": previous}
                            for n in (1, 2, 3)}}
        state["stages"]["4"] = {"status": "pending_choice", "candidates": {"epochs_20": twenty}}
        with self.assertRaisesRegex(ValueError, "Complete step 4"):
            staged.stage_previous(state, 5)
        staged.choose_stage(state, self.root, 4, "use20")
        self.assertEqual(staged.stage_previous(state, 5)["parameters"]["epochs"], 20)
        self.assertTrue((self.root / "step_04_report.txt").is_file())

    def test_fifth_class_weight_trial_is_planned(self):
        selected = {"parameters": staged.baseline_params()}
        state = {"stages": {str(n): {"status": "complete", "selected": selected}
                            for n in range(1, 6)}}
        seen = []

        def fake_candidate(config, state, stage, label, params, output, splits):
            seen.append(tuple(params["class_weights"]))
            state["stages"][str(stage)]["candidates"][label] = {
                "parameters": params, "best_epoch": 1,
                "best_validation": {"binary_water_iou": .5, "mean_iou": .5,
                                    "obstacle_miss_rate": .1,
                                    "water_as_obstacle_rate": .2,
                                    "obstacle_precision": .7},
                "last_three_validation_water_iou_std": 0,
                "checkpoint": "fake.npz"}

        with patch.object(staged, "candidate_record", side_effect=fake_candidate):
            staged.execute_stage(self.config, {}, state, self.root, 6)
        self.assertEqual(seen, [(1, 1, 1), (2, 1, 1), (3, 1, 1),
                                (3, 2, 1), (1, 2, 1)])
        self.assertEqual(state["stages"]["6"]["status"], "pending_choice")

    def test_sequence_id_selection_keeps_whole_recording(self):
        samples = [Sample(Path(f"davimar_seq_{seq}_{frame:05d}.jpg"), Path("mask.png"))
                   for seq, frame in (("01", 10), ("01", 20), ("02", 30))]
        selection = {"kind": "sequence_ids", "pattern": r"^(davimar_seq_\d+)_\d+$",
                     "ids": ["davimar_seq_01"]}
        self.assertEqual(len(select_samples(samples, selection, "train")), 2)
        selection["ids"] = ["davimar_seq_03"]
        with self.assertRaisesRegex(ValueError, "missing sequence IDs"):
            select_samples(samples, selection, "train")


if __name__ == "__main__":
    unittest.main()
