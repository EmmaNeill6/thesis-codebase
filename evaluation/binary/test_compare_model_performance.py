"""Tests for the paired model-performance report."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.binary.compare_model_performance import (
    QUALITY_METRICS,
    build_comparison,
    load_evaluation,
    load_timing,
    main,
)
from timing_comparison import append_memory_usage, write_timing_comparison


def write_evaluation(path: Path, model: str, scores: dict[str, float]) -> None:
    path.mkdir(parents=True)
    fields = [
        "evaluation_key", "dataset_key", "model_name", "run_name",
        "image_width_px", "image_height_px", "valid_pixel_count", *QUALITY_METRICS,
    ]
    with (path / "per_image.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for key, water_iou in scores.items():
            row = {
                "evaluation_key": key,
                "dataset_key": "mastr1325",
                "model_name": model,
                "run_name": "test_run",
                "image_width_px": 512,
                "image_height_px": 384,
                "valid_pixel_count": 186536,
                **{metric: 0.5 for metric in QUALITY_METRICS},
                "water_iou": water_iou,
                "dice": water_iou,
            }
            writer.writerow(row)


class CompareModelPerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ewasr_run = self.root / "ewasr_run"
        self.sam3_run = self.root / "sam3_run"
        self.ewasr_eval = self.root / "ewasr_eval"
        self.sam3_eval = self.root / "sam3_eval"
        write_timing_comparison(
            self.ewasr_run,
            [
                {"evaluation_key": "a.png", "inference_ms": 10, "final_mask_latency_ms": 15},
                {"evaluation_key": "b.png", "inference_ms": 20, "final_mask_latency_ms": 30},
            ],
        )
        write_timing_comparison(
            self.sam3_run,
            [
                {"evaluation_key": "a.png", "inference_ms": 100, "final_mask_latency_ms": 120},
                {"evaluation_key": "b.png", "inference_ms": 200, "final_mask_latency_ms": 230},
            ],
        )
        write_evaluation(self.ewasr_eval, "ewasr", {"a.png": 0.9, "b.png": 0.8})
        write_evaluation(self.sam3_eval, "sam3", {"a.png": 0.7, "b.png": 0.6})
        for evaluation_dir, run_dir in (
            (self.ewasr_eval, self.ewasr_run),
            (self.sam3_eval, self.sam3_run),
        ):
            (evaluation_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "number_of_images": 2,
                        "configuration": {
                            "dataset": "mastr1325",
                            "prediction_dir": str((run_dir / "predicted_masks").resolve()),
                        },
                    }
                )
            )

    def test_cli_writes_paired_reports(self) -> None:
        append_memory_usage(
            self.ewasr_run, model="eWaSR", dataset="mastr1325",
            attempted_images=2, successful_images=2,
        )
        append_memory_usage(
            self.sam3_run, model="SAM 3", dataset="mastr1325",
            attempted_images=2, successful_images=2,
        )
        output = self.root / "comparison"
        argv = [
            "compare_model_performance.py",
            "--ewasr-run-dir", str(self.ewasr_run),
            "--sam3-run-dir", str(self.sam3_run),
            "--ewasr-evaluation", str(self.ewasr_eval),
            "--sam3-evaluation", str(self.sam3_eval),
            "--output-dir", str(output),
        ]
        with patch("sys.argv", argv):
            main()
        summary = json.loads((output / "comparison_summary.json").read_text())
        self.assertEqual(summary["paired_images"], 2)
        self.assertAlmostEqual(summary["models"]["ewasr"]["macro_quality_on_paired_images"]["water_iou"], 0.85)
        self.assertAlmostEqual(summary["models"]["sam3"]["final_mask_latency_ms"]["median"], 175)
        self.assertAlmostEqual(summary["paired_sam3_minus_ewasr"]["water_iou"]["mean"], -0.2)
        self.assertEqual(summary["memory"]["sam3"]["status"], "single_invocation_same_count")
        with (output / "paired_per_image.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([row["evaluation_key"] for row in rows], ["a.png", "b.png"])
        self.assertTrue((output / "comparison_summary.md").is_file())

    def test_mismatched_timing_sets_require_explicit_intersection(self) -> None:
        ewasr = load_timing(self.ewasr_run, "eWaSR")
        sam3 = load_timing(self.sam3_run, "SAM 3")
        sam3.pop("b.png")
        e_eval = load_evaluation(self.ewasr_eval, "eWaSR")
        s_eval = load_evaluation(self.sam3_eval, "SAM 3")
        with self.assertRaisesRegex(ValueError, "Timing image sets differ"):
            build_comparison(ewasr, sam3, e_eval, s_eval)
        rows, summary = build_comparison(
            ewasr, sam3, e_eval, s_eval, allow_intersection=True
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["pairing"]["ewasr_only_excluded_count"], 1)

    def test_rejects_nonmatching_timing_protocol(self) -> None:
        path = self.sam3_run / "comparable_timing_summary.json"
        summary = json.loads(path.read_text())
        summary["timing_protocol_version"] = 1
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "timing protocol"):
            load_timing(self.sam3_run, "SAM 3")

    def test_rejects_evaluation_of_other_prediction_run(self) -> None:
        path = self.sam3_eval / "summary.json"
        summary = json.loads(path.read_text())
        summary["configuration"]["prediction_dir"] = str(self.root / "other_masks")
        path.write_text(json.dumps(summary))
        argv = [
            "compare_model_performance.py",
            "--ewasr-run-dir", str(self.ewasr_run),
            "--sam3-run-dir", str(self.sam3_run),
            "--ewasr-evaluation", str(self.ewasr_eval),
            "--sam3-evaluation", str(self.sam3_eval),
            "--output-dir", str(self.root / "comparison"),
        ]
        with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "Re-evaluate"):
            main()


if __name__ == "__main__":
    unittest.main()
