"""Unit tests for SAM 3 result merging and output rendering."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).with_name("predict_sam3_mlx.py")
SPEC = importlib.util.spec_from_file_location("predict_sam3_mlx", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SemanticMaskTests(unittest.TestCase):
    def test_score_report_distinguishes_instances_from_fallback_obstacle(self):
        masks = np.zeros((2, 2, 3), dtype=bool)
        masks[0, 0, :2] = True
        masks[1, 1, 2] = True
        fake_result = types.SimpleNamespace(
            labels=["navigable water", "sky"],
            masks=masks,
            scores=np.array([0.82, 0.93]),
        )
        generate = types.ModuleType("mlx_vlm.models.sam3.generate")
        generate.predict_multi = lambda *args, **kwargs: fake_result
        predictor = types.SimpleNamespace()

        with patch.dict(sys.modules, {"mlx_vlm.models.sam3.generate": generate}):
            class_mask, report = MODULE.predict_image_with_scores(
                predictor, Image.new("RGB", (3, 2)), 0.5, 0
            )

        np.testing.assert_array_equal(class_mask, [[1, 1, 0], [0, 0, 2]])
        self.assertEqual(report["instance_scores_by_class"]["navigable water"], [0.82])
        self.assertEqual(report["instance_scores_by_class"]["sky"], [0.93])
        self.assertEqual(report["instance_scores_by_class"]["obstacle"], [])
        self.assertEqual(report["uncovered_pixels_assigned_obstacle"], 3)

    def test_reflection_padding_scales_with_each_image_dimension(self):
        pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        image = Image.fromarray(pixels, mode="RGB")

        padded_image, pad_y, pad_x = MODULE.add_reflection_padding(image, 25.0)
        padded = np.asarray(padded_image)

        self.assertEqual((pad_y, pad_x), (1, 2))
        self.assertEqual(padded.shape, (6, 10, 3))
        np.testing.assert_array_equal(padded[pad_y : pad_y + 4, pad_x : pad_x + 6], pixels)
        np.testing.assert_array_equal(
            padded,
            np.pad(pixels, ((1, 1), (2, 2), (0, 0)), mode="reflect"),
        )

    def test_crop_padding_restores_original_dimensions(self):
        padded_mask = np.arange(6 * 10, dtype=np.uint8).reshape(6, 10)
        cropped = MODULE.crop_padding(
            padded_mask,
            original_shape=(4, 6),
            pad_y=1,
            pad_x=2,
        )
        np.testing.assert_array_equal(cropped, padded_mask[1:5, 2:8])

    def test_overlap_uses_highest_confidence_and_uncovered_is_obstacle(self):
        masks = np.zeros((3, 2, 3), dtype=np.uint8)
        masks[0, 0, :2] = 1  # navigable water, score 0.7
        masks[1, 0, 1:] = 1  # sky, score 0.9
        masks[2, 1, 0] = 1  # explicit obstacle

        result = MODULE.semantic_mask_from_instances(
            labels=["navigable water", "sky", "obstacle"],
            masks=masks,
            scores=np.array([0.7, 0.9, 0.8]),
            image_shape=(2, 3),
        )

        expected = np.array([[1, 2, 2], [0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_empty_prediction_is_all_obstacle(self):
        result = MODULE.semantic_mask_from_instances(
            labels=[],
            masks=np.zeros((0, 1, 1), dtype=np.uint8),
            scores=np.zeros((0,), dtype=np.float32),
            image_shape=(2, 4),
        )
        np.testing.assert_array_equal(result, np.zeros((2, 4), dtype=np.uint8))

    def test_saved_mask_retains_class_ids(self):
        image = Image.new("RGB", (2, 2), (100, 100, 100))
        mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            mask_path = root / "masks/test.png"
            overlay_path = root / "overlays/test.png"
            MODULE.save_outputs(image, mask, mask_path, overlay_path)

            with Image.open(mask_path) as saved:
                saved_mask = np.asarray(saved)
            np.testing.assert_array_equal(saved_mask, mask)
            with Image.open(overlay_path) as saved_overlay:
                self.assertEqual(saved_overlay.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
