"""Export the completed data-size experiment as a scientific figure."""
import argparse
import csv
import json
from pathlib import Path

try:
    import matplotlib
except ImportError:
    raise SystemExit("Plotting requires Matplotlib: install it in the Python environment used for this command.")
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(study_dir: Path) -> Path:
    with (study_dir / "summary.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    with (study_dir / "aggregate.csv").open(newline="", encoding="utf-8") as source:
        aggregate = list(csv.DictReader(source))
    if any(row["status"] != "complete" for row in rows):
        raise ValueError("Finish every run before creating the final figure")
    study = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    endpoint = next(row for row in rows if int(row["percentage"]) == 100)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), constrained_layout=True)
    for axis, metric, title in zip(axes,
            ("test_binary_water_iou", "validation_binary_water_iou"),
            (f"Fixed test set ({study['base_counts']['test']} images)",
             f"Fixed validation set ({study['base_counts']['validation']} images)")):
        all_values = []
        for iteration in (1, 2, 3):
            members = sorted([row for row in rows if int(row["iteration"]) == iteration and
                              int(row["percentage"]) < 100] + [endpoint],
                             key=lambda row: int(row["percentage"]))
            values = [float(row[metric]) * 100 for row in members]
            all_values.extend(values)
            axis.plot([int(row["percentage"]) for row in members], values,
                      color="#94a3b8", linewidth=1.1, alpha=.7, marker=".",
                      label="Subset selections" if iteration == 1 else None)
        small = [row for row in aggregate if int(row["percentage"]) < 100]
        axis.errorbar([int(row["percentage"]) for row in small],
                      [float(row[metric + "_mean"]) * 100 for row in small],
                      yerr=[float(row[metric + "_sd"]) * 100 for row in small],
                      color="#0f4c81", marker="o", capsize=5, linewidth=2,
                      label="Mean ± sample SD (3 selections)")
        axis.scatter([100], [float(endpoint[metric]) * 100], color="#d97706", s=60,
                     zorder=5, marker="D", label="100% endpoint (single run)")
        low, high = min(all_values), max(all_values)
        pad = max(.12, (high - low) * .18)
        axis.set_ylim(max(0, low - pad), min(100, high + pad))
        axis.set_xticks([int(row["percentage"]) for row in aggregate],
                        [f"{row['percentage']}%\n({row['train']} images)" for row in aggregate])
        axis.set_xlabel("Fraction of the fixed training pool")
        axis.set_ylabel("Water IoU (%) against pseudo-labels")
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", color="#e2e8f0")
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="best", fontsize=8)
    figure.suptitle(f"{study['dataset']}: effect of training data size\n20 epochs, fixed training seed, nested subsets", fontsize=12)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(study_dir / f"data_size_curve.{extension}", dpi=200)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for iteration in (1, 2, 3):
        members = sorted([row for row in rows if int(row["iteration"]) == iteration and
                          int(row["percentage"]) < 100] + [endpoint], key=lambda row: int(row["train"]))
        axis.plot([int(row["train"]) for row in members], [float(row["training_seconds"]) for row in members],
                  color="#94a3b8", alpha=.7, linewidth=1.1, marker=".",
                  label="Subset selections" if iteration == 1 else None)
    small = [row for row in aggregate if int(row["percentage"]) < 100]
    axis.errorbar([int(row["train"]) for row in small],
                  [float(row["training_seconds_mean"]) for row in small],
                  yerr=[float(row["training_seconds_sd"]) for row in small],
                  color="#0f4c81", marker="o", capsize=5, linewidth=2,
                  label="Mean ± sample SD (3 selections)")
    axis.scatter([int(endpoint["train"])], [float(endpoint["training_seconds"])],
                 color="#d97706", marker="D", s=60, zorder=5, label="100% endpoint (single run)")
    axis.set_xticks([int(row["train"]) for row in aggregate])
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Training images")
    axis.set_ylabel("Training time including validation and saving (seconds)")
    axis.set_title("Head-only adaptation cost · fixed 20 epochs", fontsize=12)
    axis.grid(axis="y", color="#e2e8f0")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=8)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(study_dir / f"training_cost_curve.{extension}", dpi=200)
    plt.close(figure)
    return study_dir / "data_size_curve.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    args = parser.parse_args()
    print(plot(args.study_dir.resolve()))
