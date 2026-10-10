"""Plot staged and V2 validation comparisons for four datasets.

Each figure uses the recorded validation water IoU. It is a model-selection
illustration, not a held-out test evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
DATASETS = {
    "catarob_staged_v1": ("CataRob", "Catarob_v2/head_only_lr1e-3_wd1e-3_20ep"),
    "lars_channel_staged_v1": ("Lars channel", "lars_channel_v2"),
    "pohang_canal_pier_staged_v1": ("Pohang canal and pier", "Pohang_canal_pier_v2"),
    "tampere_dock_staged_v1": ("Tampere dock", "Tampere_dock_v2"),
}
COLORS = {
    "Pretrained": "#68747e", "Baseline": "#2166ac",
    "Staged best": "#d95f0e", "Staged settings": "#d95f0e",
    "Chosen V2": "#1b9e77",
}


def read_history(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not rows or [row["epoch"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"Incomplete epoch history: {path}")
    return rows


def fixed20_record(dataset: str, variant: str, staged_splits: dict) -> dict:
    base = HERE / "outputs" / "fixed20_comparison" / dataset
    candidates = [base / variant]
    if variant == "baseline" and dataset == "pohang_canal_pier":
        candidates.append(base / "baseline_retry")
    for run_dir in candidates:
        metadata_path = run_dir / "comparison_run.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        split_path = run_dir / "resolved_splits.json"
        if json.loads(split_path.read_text(encoding="utf-8")) != staged_splits:
            raise ValueError(f"Fixed-20 split differs from staged split: {run_dir}")
        if (metadata["dataset"] != dataset or metadata["variant"] != variant or
            metadata["parameters"]["epochs"] != 20 or
            metadata["selection_metric"] != "validation_binary_water_iou"):
            raise ValueError(f"Invalid fixed-20 run metadata: {run_dir}")
        return {
            "best_epoch": metadata["best_epoch"],
            "best_validation": metadata["best_validation"],
            "checkpoint": str(run_dir / "best_head_only.npz"),
            "history": str(run_dir / "training_history.jsonl"),
        }
    raise FileNotFoundError(f"No completed fixed-20 {variant} run for {dataset}")


def load_dataset(directory: Path, v2_directory: Path, *, fixed20: bool = False) -> dict:
    state = json.loads((directory / "stage_state.json").read_text(encoding="utf-8"))
    if any(state["stages"][str(i)]["status"] != "complete" for i in range(1, 7)):
        raise ValueError(f"Stages are incomplete: {directory}")
    staged_splits = json.loads((directory / "resolved_splits.json").read_text(encoding="utf-8"))
    v2_splits = json.loads((v2_directory / "resolved_splits.json").read_text(encoding="utf-8"))
    if staged_splits != v2_splits:
        raise ValueError(f"Staged and V2 splits differ: {directory}, {v2_directory}")
    staged_config = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
    v2_config = json.loads((v2_directory / "run_config.json").read_text(encoding="utf-8"))
    if (v2_config["selection_metric"] != "binary_water_iou" or
        v2_config["weights"] != staged_config["model"]["checkpoint"] or
        v2_config["seed"] != staged_config["experiment"]["seed"] or
        (v2_config["width"], v2_config["height"]) !=
        (staged_config["preprocessing"]["input_width"],
         staged_config["preprocessing"]["input_height"])):
        raise ValueError(f"Staged and V2 evaluation settings differ: {directory}, {v2_directory}")
    candidates = [candidate for stage in state["stages"].values()
                  for candidate in stage["candidates"].values()]
    baseline = state["stages"]["1"]["candidates"]["baseline"]
    best = max(candidates, key=lambda item: item["best_validation"]["binary_water_iou"])
    if fixed20:
        dataset = directory.name.removesuffix("_staged_v1")
        baseline = fixed20_record(dataset, "baseline", staged_splits)
        if best["parameters"]["epochs"] != 20:
            best = fixed20_record(dataset, "staged_best", staged_splits)
    baseline_history = read_history(baseline["history"])
    best_history = read_history(best["history"])
    chosen_history = read_history(str(v2_directory / "training_history.jsonl"))
    if fixed20 and any(len(history) != 20 for history in
                       (baseline_history, best_history, chosen_history)):
        raise ValueError(f"A comparison run does not have 20 epochs: {directory}")
    chosen_row = max(chosen_history, key=lambda row: row["validation"]["binary_water_iou"])
    chosen = {
        "best_epoch": chosen_row["epoch"],
        "best_validation": chosen_row["validation"],
        "checkpoint": str((v2_directory / "best_head_only.npz").resolve()),
        "history": str((v2_directory / "training_history.jsonl").resolve()),
    }
    if not Path(chosen["checkpoint"]).is_file():
        raise FileNotFoundError(chosen["checkpoint"])
    for record, history in ((baseline, baseline_history), (best, best_history),
                            (chosen, chosen_history)):
        actual_best = max(history, key=lambda row: row["validation"]["binary_water_iou"])
        if (record["best_epoch"] != actual_best["epoch"] or
            abs(record["best_validation"]["binary_water_iou"] -
                actual_best["validation"]["binary_water_iou"]) > 1e-10):
            raise ValueError(f"Checkpoint and epoch history disagree: {record['history']}")
    return {
        "pretrained": state["pretrained_validation"]["binary_water_iou"],
        "baseline": baseline,
        "chosen": chosen,
        "best": best,
        "baseline_history": baseline_history,
        "best_history": best_history,
        "chosen_history": chosen_history,
    }


def draw(dataset_name: str, results: dict, output: Path, *, fixed20: bool = False) -> None:
    pretrained = results["pretrained"]
    baseline = results["baseline"]["best_validation"]["binary_water_iou"]
    best = results["best"]["best_validation"]["binary_water_iou"]
    chosen = results["chosen"]["best_validation"]["binary_water_iou"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    fig, (summary_ax, curve_ax) = plt.subplots(
        1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )
    fig.suptitle(dataset_name, fontsize=16, fontweight="bold", x=0.055, ha="left")
    subtitle = ("Validation Water IoU  |  equal 20-epoch training budget"
                if fixed20 else
                "Validation Water IoU  |  staged best and V2 are separate experiments")
    fig.text(0.055, 0.91, subtitle,
             color="#4b5563", fontsize=9.5)

    stage_label = "Staged settings" if fixed20 else "Staged best"
    points = [("Pretrained", pretrained, 3), ("Baseline", baseline, 2),
              (stage_label, best, 1), ("Chosen V2", chosen, 0)]
    for label, value, y in points:
        summary_ax.hlines(y, 0.8, value, color=COLORS[label], linewidth=2.3, alpha=0.82)
        summary_ax.scatter([value], [y], s=105, color=COLORS[label], zorder=3)
        summary_ax.text(value + 0.004, y, f"{value:.4f}", va="center", fontsize=9,
                        color="#273442")
    summary_ax.set_yticks([3, 2, 1, 0], [label for label, _, _ in points])
    summary_ax.set_xlim(0.8, 1.015)
    summary_ax.set_ylim(-0.9, 3.5)
    summary_ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    summary_ax.set_xlabel("Validation Water IoU (zoomed axis)")
    summary_ax.set_title("A  Model comparison", loc="left", fontweight="bold", fontsize=11)
    summary_ax.grid(axis="x", alpha=0.2)
    summary_ax.set_axisbelow(True)
    delta_pp = 100 * (chosen - best)
    summary_ax.text(0.8, -0.62, f"V2 − staged: {delta_pp:+.3f} percentage points",
                    fontsize=9, color=COLORS["Chosen V2"])

    for label, record, history in (
        ("Baseline", results["baseline"], results["baseline_history"]),
        (stage_label, results["best"], results["best_history"]),
        ("Chosen V2", results["chosen"], results["chosen_history"]),
    ):
        epochs = [row["epoch"] for row in history]
        values = [row["validation"]["binary_water_iou"] for row in history]
        curve_ax.plot(epochs, values, color=COLORS[label], marker="o", markersize=3.5,
                      linewidth=2.2, linestyle="--" if label == "Chosen V2" else "-",
                      label=f"{label} (epoch {record['best_epoch']})")
        epoch = record["best_epoch"]
        curve_ax.scatter([epoch], [values[epoch - 1]], s=95, marker="*",
                         color=COLORS[label], edgecolors="white", linewidths=0.6, zorder=4)
    all_values = [row["validation"]["binary_water_iou"]
                  for history in (results["baseline_history"], results["best_history"],
                                  results["chosen_history"])
                  for row in history]
    curve_ax.set_ylim(max(0, min(all_values) - 0.007), min(1.0, max(all_values) + 0.008))
    curve_ax.set_xlim(0.6, max(len(results["baseline_history"]), len(results["best_history"]),
                                len(results["chosen_history"])) + 0.7)
    curve_ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    curve_ax.set_xlabel("Training epoch")
    curve_ax.set_ylabel("Validation Water IoU (zoomed axis)")
    curve_ax.set_title("B  Validation learning curves", loc="left", fontweight="bold",
                       fontsize=11)
    curve_ax.grid(alpha=0.2)
    curve_ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    note = ("Stars mark validation-selected checkpoints. Staged settings were selected "
            "previously; each displayed training run has 20 epochs."
            if fixed20 else
            "Stars mark validation-selected checkpoints. V2 uses the same recipe across datasets: "
            "LR 0.001, decay 0.001, 20-epoch budget, batch 2, weights 1/1/1.")
    fig.text(0.055, 0.062, note, fontsize=8.2, color="#4b5563")
    fig.text(0.055, 0.034, "Validation results only; no held-out test scores are shown.",
             fontsize=8.2, color="#4b5563")
    fig.tight_layout(rect=(0.02, 0.11, 0.99, 0.87), w_pad=3.0)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "validation_model_examples")
    parser.add_argument("--fixed20", action="store_true", help="Use equal-budget 20-epoch runs")
    parser.add_argument("--dataset", action="append", choices=DATASETS,
                        help="Select one or more staged datasets; default is all four")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory_name in (args.dataset or DATASETS):
        title, v2_name = DATASETS[directory_name]
        results = load_dataset(HERE / "outputs" / directory_name,
                               HERE / "outputs" / v2_name, fixed20=args.fixed20)
        output = args.output_dir / directory_name.removesuffix("_staged_v1")
        draw(title, results, output, fixed20=args.fixed20)
        for label, record in (("pretrained", None), ("baseline", results["baseline"]),
                              ("staged_settings_20ep" if args.fixed20 else "staged_best",
                               results["best"]),
                              ("chosen_v2", results["chosen"])):
            rows.append({
                "dataset": title, "model": label,
                "validation_water_iou": results["pretrained"] if record is None
                else record["best_validation"]["binary_water_iou"],
                "best_epoch": "" if record is None else record["best_epoch"],
                "checkpoint": "pretrained model" if record is None else record["checkpoint"],
            })
        print(f"Wrote {output.with_suffix('.png')} and .pdf")
    with (args.output_dir / "comparison_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
