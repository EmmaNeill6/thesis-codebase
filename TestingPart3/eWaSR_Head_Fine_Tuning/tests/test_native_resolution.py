"""Checks for native preprocessing and exact recorded test membership."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "evaluate_native_resolution", Path(__file__).parents[1] / "evaluate_native_resolution.py"
)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class NativeResolutionTests(unittest.TestCase):
    def test_native_pixels_are_preserved_and_padding_is_normalized_zero(self):
        pixels = np.zeros((65, 130, 3), dtype=np.uint8)
        pixels[..., 0] = 255
        array, size = native.prepare_native(Image.fromarray(pixels))
        self.assertEqual(size, (65, 130))
        self.assertEqual(array.shape, (1, 128, 192, 3))
        np.testing.assert_allclose(array[0, 0, 0], (np.array([1, 0, 0]) - native.MEAN) / native.STD)
        np.testing.assert_array_equal(array[0, 65:, :, :], 0)
        np.testing.assert_array_equal(array[0, :, 130:, :], 0)

    def test_aligned_images_do_not_gain_padding(self):
        array, size = native.prepare_native(Image.new("RGB", (192, 128)))
        self.assertEqual(array.shape, (1, 128, 192, 3))
        self.assertEqual(size, (128, 192))

    def test_intermediate_input_size_and_original_image_are_separate(self):
        image = Image.new("RGB", (1920, 1080))
        array, size = native.prepare_input(image, (768, 576))
        self.assertEqual(image.size, (1920, 1080))
        self.assertEqual(array.shape, (1, 576, 768, 3))
        self.assertEqual(size, (576, 768))

    def test_scores_are_interpolated_before_labels_are_selected(self):
        scores = np.zeros((1, 2, 3), dtype=np.float32)
        scores[0, 0, 1] = 100
        scores[0, 1, 0] = 1
        resized = native.resize_class_scores(scores, (8, 1))
        self.assertEqual(resized.shape, (1, 8, 3))
        labels = np.argmax(resized, axis=-1)
        nearest_labels = np.repeat(np.argmax(scores, axis=-1), 4, axis=1)
        self.assertEqual(labels[0, 4], 1)
        self.assertEqual(nearest_labels[0, 4], 0)

    def write_pair(self, root, name):
        image, mask = root / f"{name}.jpg", root / f"{name}.png"
        Image.new("RGB", (128, 64)).save(image)
        Image.fromarray(np.ones((64, 128), dtype=np.uint8)).save(mask)
        return {"image": str(image), "mask": str(mask)}

    def test_recorded_membership_is_used_without_regenerating_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test = self.write_pair(root, "test")
            train = self.write_pair(root, "train")
            self.write_pair(root, "unselected")
            (root / "resolved_splits.json").write_text(json.dumps({"test": [test], "train": [train], "validation": []}))
            self.assertEqual(native.load_recorded_test(root), [(Path(test["image"]).resolve(), Path(test["mask"]).resolve())])

    def test_overlap_and_resized_references_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test = self.write_pair(root, "test")
            manifest = {"test": [test], "train": [test], "validation": []}
            (root / "resolved_splits.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "overlap"):
                native.load_recorded_test(root)
            manifest["train"] = []
            (root / "resolved_splits.json").write_text(json.dumps(manifest))
            Image.fromarray(np.ones((16, 32), dtype=np.uint8)).save(test["mask"])
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                native.load_recorded_test(root)

    def test_validation_membership_is_used_and_test_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test = self.write_pair(root, "test")
            validation = self.write_pair(root, "validation")
            manifest = {"test": [test], "train": [], "validation": [validation]}
            (root / "resolved_splits.json").write_text(json.dumps(manifest))
            pairs = native.load_recorded_split(root, "validation")
            self.assertEqual(pairs, [(Path(validation["image"]).resolve(), Path(validation["mask"]).resolve())])


if __name__ == "__main__":
    unittest.main()
