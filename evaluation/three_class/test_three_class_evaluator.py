"""Known-confusion and saved-mask integration checks for three-class evaluation."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.three_class.class_metrics import evaluate_pair, metrics_from_confusion
from evaluation.three_class.three_class_evaluator import main, pair_predictions, parse_args


def save_mask(path, values, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(values, dtype=np.uint8))
    if mode == "P":
        # Deliberately non-grayscale palette: IDs must survive unchanged.
        image = image.convert("P")
        image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (768 - 9))
    image.save(path)


class ClassMetricTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prediction = self.root / "prediction.png"
        self.reference = self.root / "reference.png"

    def test_known_confusion_scores(self):
        metrics = metrics_from_confusion(np.array([[3, 1, 0], [1, 2, 1], [0, 1, 3]]))
        self.assertAlmostEqual(metrics["pixel_accuracy"], 8 / 12)
        self.assertAlmostEqual(metrics["obstacle_iou"], 3 / 5)
        self.assertAlmostEqual(metrics["water_iou"], 2 / 6)
        self.assertAlmostEqual(metrics["sky_iou"], 3 / 5)
        self.assertAlmostEqual(metrics["mean_iou"], (3 / 5 + 2 / 6 + 3 / 5) / 3)
        self.assertAlmostEqual(metrics["water_dice"], 1 / 2)
        self.assertAlmostEqual(metrics["mean_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["mean_recall"], 2 / 3)
        self.assertEqual(metrics["water_true_negative"], 6)

    def test_confusion_orientation_ignore_and_palette_indices(self):
        save_mask(self.reference, [[0, 1, 2], [4, 1, 2]], mode="P")
        save_mask(self.prediction, [[1, 1, 0], [2, 2, 2]], mode="P")
        matrix, valid, shape = evaluate_pair(self.prediction, self.reference, ignore_value=4)
        np.testing.assert_array_equal(matrix, [[0, 1, 0], [0, 1, 1], [1, 0, 1]])
        self.assertEqual(valid, 5)
        self.assertEqual(shape, (2, 3))

    def test_absent_classes_excluded_but_false_positive_class_penalized(self):
        metrics = metrics_from_confusion(np.diag([0, 5, 0]))
        self.assertEqual(metrics["mean_iou"], 1)
        self.assertTrue(math.isnan(metrics["obstacle_iou"]))
        matrix = np.array([[0, 0, 0], [1, 4, 0], [0, 0, 0]])
        metrics = metrics_from_confusion(matrix)
        self.assertEqual(metrics["obstacle_iou"], 0)
        self.assertAlmostEqual(metrics["mean_iou"], 0.4)

    def test_ground_truth_mapping(self):
        save_mask(self.reference, [[10, 20, 30, 255]])
        save_mask(self.prediction, [[0, 1, 2, 1]])
        matrix, valid, _ = evaluate_pair(
            self.prediction, self.reference,
            ground_truth_class_values=(10, 20, 30), ignore_value=255,
        )
        np.testing.assert_array_equal(matrix, np.eye(3, dtype=int))
        self.assertEqual(valid, 3)

    def test_rejects_invalid_prediction_ground_truth_and_empty_reference(self):
        for predicted, actual, ignore, message in (
            ([[0, 255]], [[0, 1]], None, "Prediction must contain"),
            ([[0, 1]], [[0, 3]], None, "Unsupported ground-truth"),
            ([[0, 1]], [[4, 4]], 4, "no valid pixels"),
            ([[0, 1]], [[0]], None, "size mismatch"),
        ):
            with self.subTest(message=message):
                save_mask(self.prediction, predicted)
                save_mask(self.reference, actual)
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_pair(self.prediction, self.reference, ignore_value=ignore)

    def test_rejects_rgb_visualization(self):
        Image.new("RGB", (2, 2), (41, 167, 224)).save(self.prediction)
        save_mask(self.reference, [[1, 1], [1, 1]])
        with self.assertRaisesRegex(ValueError, "single-channel"):
            evaluate_pair(self.prediction, self.reference)


class EvaluatorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.predictions = self.root / "predictions"
        self.references = self.root / "references"
        self.results = self.root / "results"
        save_mask(self.references / "nested/a.png", [[0, 0, 1], [1, 2, 2]])
        save_mask(self.predictions / "nested/a.png", [[0, 0, 1], [1, 2, 2]])
        save_mask(self.references / "b.png", [[1, 1], [1, 1]])
        save_mask(self.predictions / "b.png", [[0, 0], [0, 0]])
        self.args = [
            "--ground-truth-dir", str(self.references),
            "--prediction-dir", str(self.predictions), "--model-name", "test_model",
            "--run-name", "test_run", "--results-dir", str(self.results),
        ]

    def test_reports_global_and_per_image_macro_without_mixing(self):
        save_mask(self.predictions / "extra.png", [[0]])
        output = main(self.args)
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(report["evaluation_type"], "three_class")
        self.assertEqual(report["number_of_images"], 2)
        self.assertEqual(report["valid_pixel_count"], 10)
        self.assertEqual(report["confusion_matrix"], [[2, 0, 0], [4, 2, 0], [0, 0, 2]])
        self.assertAlmostEqual(report["global_metrics"]["mean_iou"], 5 / 9)
        self.assertAlmostEqual(report["macro_metrics"]["mean_iou"], 0.5)
        self.assertEqual(report["pairing"]["predictions_without_ground_truth"], 1)
        with (output / "per_image.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(int(row["valid_pixel_count"]) for row in rows), 10)
        self.assertTrue((output / "confusion_matrix.csv").is_file())
        self.assertIn("extra.png", (output / "mask_pairing_report.txt").read_text())

    def test_selection_deduplicates_and_json_undefined_is_null(self):
        selection = self.root / "selection.txt"
        selection.write_text("# subset\nb.png\nb.png\n")
        output = main(self.args + ["--selection-file", str(selection)])
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(report["number_of_images"], 1)
        self.assertIsNone(report["global_metrics"]["sky_iou"])
        self.assertEqual(report["global_metrics"]["water_iou"], 0)
        def reject_nonfinite(value):
            raise AssertionError(f"Non-standard JSON number: {value}")
        json.loads((output / "summary.json").read_text(), parse_constant=reject_nonfinite)
        selection.write_text("unknown.png\n")
        with self.assertRaisesRegex(ValueError, "Unknown evaluation keys"):
            main(self.args + ["--selection-file", str(selection)])

    def test_missing_or_invalid_predictions_withhold_metric_files(self):
        (self.predictions / "b.png").unlink()
        with self.assertRaisesRegex(ValueError, "predictions are missing"):
            main(self.args)
        self.assertEqual(len(list(self.results.rglob("mask_pairing_report.txt"))), 1)
        self.assertFalse(list(self.results.rglob("per_image.csv")))
        save_mask(self.predictions / "b.png", [[0, 255], [0, 0]])
        with self.assertRaisesRegex(ValueError, "Prediction must contain"):
            main(self.args)
        self.assertFalse(list(self.results.rglob("summary.json")))

    def test_mastr_aliases_and_duplicate_detection(self):
        requests = {"001m.png": (Path("001m.png"), Path("001.jpg"))}
        aliases = {"001.png": "001m.png"}
        predictions, extras = pair_predictions({"001.png": Path("001.png")}, requests, aliases)
        self.assertEqual(predictions, {"001m.png": Path("001.png")})
        self.assertEqual(extras, [])
        with self.assertRaisesRegex(ValueError, "Multiple predicted masks"):
            pair_predictions({"001.png": Path("a"), "001m.png": Path("b")}, requests, aliases)

    def test_mastr_default_ignore_value(self):
        args = parse_args(["--dataset", "mastr1325", "--prediction-dir", ".", "--model-name", "test"])
        self.assertEqual(args.ignore_value, 4)
        self.assertEqual(args.ground_truth_class_values, (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
