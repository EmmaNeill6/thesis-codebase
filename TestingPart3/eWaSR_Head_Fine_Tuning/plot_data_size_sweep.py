"""Export the completed data-size experiment as a scientific figure."""
import argparse
import csv
import json
import math
from pathlib import Path

try:
    import matplotlib
except ImportError:
    raise SystemExit("Plotting requires Matplotlib: install it in the Python environment used for this command.")
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_study(study_dir: Path) -> tuple[list[dict], list[dict], dict]:
    with (study_dir / "summary.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    with (study_dir / "aggregate.csv").open(newline="", encoding="utf-8") as source:
        aggregate = list(csv.DictReader(source))
    if not rows or any(row["status"] != "complete" for row in rows):
        raise ValueError("Finish every run before creating the final figure")
    study = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    return rows, aggregate, study


def completed_studies(root: Path) -> list[Path]:
    studies = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "study.json").is_file():
            continue
        if not all((path / name).is_file() for name in ("summary.csv", "aggregate.csv")):
            continue
        with (path / "summary.csv").open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        if rows and all(row["status"] == "complete" for row in rows):
            studies.append(path)
    return studies


def shared_limits(study_dirs: list[Path]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Include individual runs and every mean ± SD bar in the common axes."""
    performance, seconds = [], []
    for path in study_dirs:
        rows, aggregate, _ = load_study(path)
        for row in rows:
            performance.extend(float(row[key]) * 100 for key in
                               ("test_binary_water_iou", "validation_binary_water_iou"))
            seconds.append(float(row["training_seconds"]))
        for row in aggregate:
            for metric in ("test_binary_water_iou", "validation_binary_water_iou"):
                mean = float(row[metric + "_mean"]) * 100
                sd = float(row.get(metric + "_sd") or 0) * 100
                performance.extend((mean - sd, mean + sd))
            mean = float(row["training_seconds_mean"])
            sd = float(row.get("training_seconds_sd") or 0)
            seconds.append(mean + sd)
    if not performance:
        raise ValueError("No completed studies available to determine shared axes")
    performance_limits = (max(0, math.floor(min(performance) - .25)),
                          min(100, math.ceil(max(performance) + .25)))
    cost_limits = (0, max(100, math.ceil(max(seconds) * 1.05 / 100) * 100))
    return performance_limits, cost_limits


def plot(study_dir: Path, performance_limits=None, cost_limits=None) -> Path:
    rows, aggregate, study = load_study(study_dir)
    if performance_limits is None or cost_limits is None:
        common_performance, common_cost = shared_limits(completed_studies(study_dir.parent))
        performance_limits = performance_limits or common_performance
        cost_limits = cost_limits or common_cost
    dataset = {"catarob_all_5pct_sweep": "CataRob", "pohang_canal_sweep": "Pohang Canal",
               "tampere_channel_sweep": "Tampere Channel", "tampere_dock_sweep": "Tampere Dock"}.get(
                   study["dataset"], study["dataset"].replace("_", " "))
    endpoint = next(row for row in rows if int(row["percentage"]) == 100)
    figure, axis = plt.subplots(figsize=(8, 5.7), constrained_layout=True)
    for metric, role, label, color, marker, linestyle in (
            ("test_binary_water_iou", "test", "Test", "#0f4c81", "o", "-"),
            ("validation_binary_water_iou", "validation", "Validation", "#c65d12", "s", "--")):
        for iteration in (1, 2, 3):
            members = sorted([row for row in rows if int(row["iteration"]) == iteration and
                              int(row["percentage"]) < 100] + [endpoint],
                             key=lambda row: int(row["percentage"]))
            values = [float(row[metric]) * 100 for row in members]
            axis.plot([int(row["percentage"]) for row in members], values,
                      color=color, linewidth=.9, alpha=.22, linestyle=linestyle)
        small = [row for row in aggregate if int(row["percentage"]) < 100]
        axis.plot([int(row["percentage"]) for row in aggregate],
                  [float(row[metric + "_mean"]) * 100 for row in aggregate],
                  color=color, linewidth=2, linestyle=linestyle)
        axis.errorbar([int(row["percentage"]) for row in small],
                      [float(row[metric + "_mean"]) * 100 for row in small],
                      yerr=[float(row[metric + "_sd"]) * 100 for row in small],
                      color=color, marker=marker, capsize=5, linewidth=1.5, linestyle="none",
                      label=f"{label} ({study['base_counts'][role]} images)")
        axis.scatter([100], [float(endpoint[metric]) * 100], color=color, s=65,
                     zorder=5, marker="D")
    axis.set_ylim(*performance_limits)
    axis.set_yticks(range(int(performance_limits[0]), int(performance_limits[1]) + 1))
    axis.set_xticks([int(row["percentage"]) for row in aggregate],
                    [f"{row['percentage']}%\n({row['train']} images)" for row in aggregate])
    axis.set_xlabel("Fraction of the fixed training pool")
    axis.set_ylabel("Water IoU (%) against pseudo-labels")
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=9)
    axis.grid(axis="y", color="#e2e8f0")
    axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{dataset}: effect of training data size\n20 epochs, fixed training seed, nested subsets", fontsize=12)
    figure.supxlabel("Thin lines: individual selections. Bars: sample SD. Diamonds at 100%: one run.", fontsize=8)
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
    axis.set_ylim(*cost_limits)
    axis.set_xlabel("Training images")
    axis.set_ylabel("Training time including validation and saving (seconds)")
    axis.set_title(f"{dataset}: head-only adaptation cost\nFixed 20 epochs", fontsize=12)
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
    parser.add_argument("--all", action="store_true",
                        help="Plot all completed studies under study_dir with shared vertical axes")
    args = parser.parse_args()
    if args.all:
        studies = completed_studies(args.study_dir.resolve())
        performance_limits, cost_limits = shared_limits(studies)
        print(f"Shared Water IoU axis: {performance_limits}; training seconds axis: {cost_limits}")
        for study_dir in studies:
            print(plot(study_dir, performance_limits, cost_limits))
    else:
        print(plot(args.study_dir.resolve()))
