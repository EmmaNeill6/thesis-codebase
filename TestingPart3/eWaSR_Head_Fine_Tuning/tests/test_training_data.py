import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
