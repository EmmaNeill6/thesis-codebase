"""Rerun staged baseline or best settings with an equal 20-epoch budget.

Uses the same dataset split, pretrained checkpoint, seed, and validation
Water IoU checkpoint rule as the saved staged experiment. Test and unseen
splits are not evaluated. Results go to new output directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_staged_experiment import (
    HERE,
    json_safe,
    trial_args,
    validate_recipe,
)
from train_head import Sample, train


DATASETS = {
    "catarob": "catarob_staged_v1",
    "lars_channel": "lars_channel_staged_v1",
    "pohang_canal_pier": "pohang_canal_pier_staged_v1",
    "tampere_dock": "tampere_dock_staged_v1",
}
TARGET_EPOCHS = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--variant", choices=("baseline", "staged_best"), required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="Use a fresh output directory for a retry")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the planned run only")
    args = parser.parse_args()

    staged_dir = HERE / "outputs" / DATASETS[args.dataset]
    state = json.loads((staged_dir / "stage_state.json").read_text(encoding="utf-8"))
    if any(state["stages"][str(i)]["status"] != "complete" for i in range(1, 7)):
        raise ValueError(f"Staged experiment is incomplete: {staged_dir}")
    config = json.loads((staged_dir / "resolved_config.json").read_text(encoding="utf-8"))
    validate_recipe(config)
    config_path = Path(config["_config_path"])
    recorded_splits = json.loads((staged_dir / "resolved_splits.json").read_text(
        encoding="utf-8"))
    if set(recorded_splits) != {"train", "validation", "test", "unseen"}:
        raise ValueError(f"Incomplete recorded split manifest: {staged_dir}")
    splits = {
        role: [Sample(Path(item["image"]), Path(item["mask"])) for item in entries]
        for role, entries in recorded_splits.items()
    }
    for role, samples in splits.items():
        if not samples or any(not sample.image_path.is_file() or
                              not sample.mask_path.is_file() for sample in samples):
            raise FileNotFoundError(f"A recorded {role} image or mask is missing: {staged_dir}")

    if args.variant == "baseline":
        source = state["stages"]["1"]["candidates"]["baseline"]
    else:
        source = max(
            (candidate for stage in state["stages"].values()
             for candidate in stage["candidates"].values()),
            key=lambda candidate: candidate["best_validation"]["binary_water_iou"],
        )
    if source["parameters"]["epochs"] == TARGET_EPOCHS:
        print("This configuration already has a 20-epoch run:")
        print(source["history"])
        return

    parameters = {**source["parameters"], "epochs": TARGET_EPOCHS}
    output = (args.output_dir or
              HERE / "outputs" / "fixed20_comparison" / args.dataset / args.variant)
    output = output.expanduser().resolve()
    print(f"Dataset: {args.dataset}")
    print(f"Variant: {args.variant}")
    print(f"Parameters: {parameters}")
    print(f"Split counts: { {role: len(samples) for role, samples in splits.items()} }")
    print(f"Starting checkpoint: {config['model']['checkpoint']}")
    print(f"Output: {output}")
    if args.dry_run:
        print("Dry run complete; training has not started.")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")

    result = train(trial_args(config, parameters, output), splits["train"], splits["validation"])
    history = [json.loads(line) for line in Path(result["history"]).read_text(
        encoding="utf-8").splitlines()]
    if [row["epoch"] for row in history] != list(range(1, TARGET_EPOCHS + 1)):
        raise RuntimeError("Training finished without a complete 20-epoch history")
    best = max(history, key=lambda row: row["validation"]["binary_water_iou"])
    if best["epoch"] != result["best_epoch"]:
        raise RuntimeError("Best checkpoint epoch disagrees with validation history")

    (output / "resolved_splits.json").write_text(
        json.dumps(recorded_splits, indent=2) + "\n", encoding="utf-8")
    (output / "comparison_run.json").write_text(json.dumps({
        "dataset": args.dataset,
        "variant": args.variant,
        "source_staged_history": source["history"],
        "source_parameters": source["parameters"],
        "parameters": parameters,
        "config_path": str(config_path),
        "source_resolved_config": str(staged_dir / "resolved_config.json"),
        "source_config_sha256": state["config_sha256"],
        "selection_metric": "validation_binary_water_iou",
        "best_epoch": result["best_epoch"],
        "best_validation": json_safe(result["best_record"]["validation"]),
        "best_checkpoint": str(result["best_checkpoint"]),
        "history": str(result["history"]),
        "test_evaluated": False,
        "unseen_evaluated": False,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Best validation Water IoU: {best['validation']['binary_water_iou']:.6f} "
          f"at epoch {best['epoch']}")


if __name__ == "__main__":
    main()
