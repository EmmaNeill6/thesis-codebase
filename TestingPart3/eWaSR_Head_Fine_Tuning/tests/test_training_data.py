import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).parents[1] / "train_head.py"
SPEC = importlib.util.spec_from_file_location("train_head", MODULE_PATH)
train_head = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = train_head
SPEC.loader.exec_module(train_head)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.masks = self.root / "masks"
        self.images.mkdir()
        self.masks.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def write_pair(self, name="frame"):
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        image[..., 2] = 255
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[:, 32:96] = 1
        mask[:, 96:] = 2
        Image.fromarray(image).save(self.images / f"{name}.jpg")
        Image.fromarray(mask).save(self.masks / f"{name}.png")

    def test_mask_subset_is_authoritative(self):
        self.write_pair()
        Image.fromarray(np.zeros((64, 128, 3), dtype=np.uint8)).save(
            self.images / "unlabelled.jpg"
        )
        samples = train_head.find_samples(self.images, self.masks)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].image_path.name, "frame.jpg")

    def test_load_sample_matches_evasr_head_resolution(self):
        self.write_pair()
        sample = train_head.find_samples(self.images, self.masks)[0]
        image, mask = train_head.load_sample(sample, width=512, height=384)
        self.assertEqual(image.shape, (384, 512, 3))
        self.assertEqual(mask.shape, (96, 128))
        self.assertEqual(set(np.unique(mask)), {0, 1, 2})

    def test_invalid_mask_values_are_rejected(self):
        self.write_pair()
        Image.fromarray(np.full((64, 128), 255, dtype=np.uint8)).save(
            self.masks / "frame.png"
        )
        with self.assertRaises(ValueError):
            train_head.validate_mask(self.masks / "frame.png")

    def test_confusion_metrics(self):
        metrics = train_head.confusion_metrics(np.diag([10, 20, 30]))
        self.assertEqual(metrics["pixel_accuracy"], 1.0)
        self.assertEqual(metrics["mean_iou"], 1.0)

    def test_yaml_range_selection_uses_frame_numbers(self):
        samples = [train_head.Sample(self.images / f"left_frame_{number:04d}.png",
                                     self.masks / f"left_frame_{number:04d}.png")
                   for number in (75, 80, 81, 85, 90)]
        chosen = train_head.select_configured_samples(samples, {
            "kind": "every_nth", "index_pattern": r"^left_frame_(\d+)$",
            "start": 75, "end": 85, "step": 5,
        }, "train")
        self.assertEqual([sample.image_path.stem for sample in chosen],
                         ["left_frame_0075", "left_frame_0080", "left_frame_0085"])

    def test_evaluation_report_records_timing_and_reserved_split(self):
        metrics = train_head.confusion_metrics(np.diag([10, 20, 30]))
        metrics["loss"] = 0.1
        checkpoint = self.root / "best_head_only.npz"
        result = {"best_epoch": 3, "best_checkpoint": checkpoint,
                  "best_record": {"validation": metrics},
                  "training_elapsed_seconds": 12.5}
        args = SimpleNamespace(learning_rate=.001, weight_decay=.001,
                               epochs=20, batch_size=2, class_weights="1,1,1")
        path = train_head.write_evaluation_text(
            self.root, result, args=args, config_path=self.root / "data.yaml",
            counts={"train": 360, "validation": 120, "test": 90, "unseen": 30},
            test_metrics=metrics, started_at=datetime.now(timezone.utc),
            total_elapsed_seconds=14.0, selection_metric="binary_water_iou")
        report = path.read_text()
        self.assertIn("Total elapsed", report)
        self.assertIn("Selected checkpoint on test split", report)
        self.assertIn("Unseen split: reserved", report)


if __name__ == "__main__":
    unittest.main()
