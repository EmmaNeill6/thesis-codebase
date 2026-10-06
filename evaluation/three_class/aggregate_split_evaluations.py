"""Pool native-resolution three-class evaluator results across disjoint splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from class_metrics import CONFUSION_NAMES, json_safe, macro_average, metrics_from_confusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", required=True, metavar="NAME=PER_IMAGE_CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-description", required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    counts: dict[str, int] = {}
    sources: set[str] = set()
    common: tuple[str, str] | None = None
    for entry in args.split:
        name, separator, path_text = entry.partition("=")
        if not separator or not name or not path_text or name in counts:
            raise ValueError(f"Expected a unique NAME=PER_IMAGE_CSV: {entry}")
        path = Path(path_text)
        with path.open(newline="", encoding="utf-8") as source:
            split_rows = list(csv.DictReader(source))
        if not split_rows:
            raise ValueError(f"No rows in {path}")
        for row in split_rows:
            identity = (row["model_name"], row["image_resolution"])
            if common is None:
                common = identity
            elif identity != common:
                raise ValueError(f"Mixed models or resolutions in {path}: {identity}")
            if row["prediction_path"] in sources:
                raise ValueError(f"Duplicate prediction: {row['prediction_path']}")
            sources.add(row["prediction_path"])
            rows.append({"split": name, **row})
        counts[name] = len(split_rows)

    confusion = np.array(
        [sum(int(row[field]) for row in rows) for field in CONFUSION_NAMES],
        dtype=np.int64,
    ).reshape(3, 3)
    if int(confusion.sum()) != sum(int(row["valid_pixel_count"]) for row in rows):
        raise ValueError("Confusion counts disagree with valid pixel counts")
    macro = macro_average(rows)
    global_metrics = metrics_from_confusion(confusion)
    report = {
        "evaluation_type": "three_class_pooled_split_results",
        "reference_description": args.reference_description,
        "splits": counts,
        "number_of_images": len(rows),
        "valid_pixel_count": int(confusion.sum()),
        "class_ids": {"obstacle": 0, "water": 1, "sky": 2},
        "confusion_matrix": confusion.tolist(),
        "macro_metrics": macro,
        "global_metrics": global_metrics,
        "note": "Rows are reference classes and columns are predictions; split CSVs retain source mask paths.",
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "Pooled three-class split evaluation",
        f"Reference: {args.reference_description}",
        f"Images: {len(rows)} ({', '.join(f'{name}={count}' for name, count in counts.items())})",
        f"Valid pixels: {int(confusion.sum())}",
        "Macro metrics (per-image average):",
        *(f"  {name}: {value:.6f}" for name, value in macro.items()),
        "Global metrics (all pixels):",
        *(f"  {name}: {value:.6f}" for name, value in global_metrics.items() if isinstance(value, float)),
        "Confusion matrix (reference rows, prediction columns; obstacle/water/sky):",
        *("  " + " ".join(str(value) for value in row) for row in confusion.tolist()),
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Pooled {len(rows)} images: global mean IoU {global_metrics['mean_iou']:.6f}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
