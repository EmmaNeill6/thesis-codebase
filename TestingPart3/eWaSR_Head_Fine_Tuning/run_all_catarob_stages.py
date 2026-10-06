"""Run the CataRob staged experiment from one interactive terminal command.

Completed stages are skipped. A stage marked running must finish in its
original terminal before this launcher is used, to avoid concurrent writes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUNNER = HERE / "run_staged_experiment.py"
CONFIG = PROJECT / "TestingPart3/DataSource/catarob_split_60_20_15_5.yaml"
SETTINGS = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
OUTPUT = (CONFIG.parent / SETTINGS["experiment"]["output_dir"]).resolve()
STATE = OUTPUT / "stage_state.json"


def read_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {"stages": {}}


def current(stage: int) -> dict:
    return read_state()["stages"].get(str(stage), {})


def run(stage: int, *options: str) -> None:
    command = [sys.executable, "-u", str(RUNNER), "--config", str(CONFIG), "--step", str(stage), *options]
    print("\nRunning stage", stage, flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def show_report(stage: int) -> None:
    report = OUTPUT / f"step_{stage:02d}_report.txt"
    if report.is_file():
        print("\n" + report.read_text(encoding="utf-8"), flush=True)


def ensure_executed(stage: int, *options: str) -> None:
    status = current(stage).get("status", "not started")
    if status == "complete":
        print(f"Stage {stage} is complete; skipping.", flush=True)
    elif status == "pending_choice":
        print(f"Stage {stage} has finished its candidates; choosing next.", flush=True)
        show_report(stage)
    elif status == "running":
        raise SystemExit(
            f"Stage {stage} is marked running. Let its existing terminal finish "
            "before restarting this launcher; do not start a second run."
        )
    else:
        run(stage, *options, "--continue")


def ask_choice(prompt: str, options: list[str], default: str) -> str:
    while True:
        response = input(f"{prompt} [{'/'.join(options)}] (Enter = {default}): ").strip()
        response = response or default
        if response in options:
            return response
        print("Choose one of:", ", ".join(options), flush=True)


def best_by_water_iou(options: dict[str, dict]) -> str:
    return max(options, key=lambda key: options[key]["best_validation"]["binary_water_iou"])


def choose(stage: int, options: dict[str, dict]) -> None:
    if current(stage).get("status") == "complete":
        return
    show_report(stage)
    default = best_by_water_iou(options)
    selection = ask_choice(f"Stage {stage} selection", list(options), default)
    run(stage, "--choose", selection)


def main() -> None:
    print("CataRob stages 1–6. Parameter choices use validation water IoU as the default.", flush=True)
    for stage in (1, 2, 3):
        ensure_executed(stage)

    ensure_executed(4)
    if current(4).get("status") != "complete":
        if "epochs_30" not in current(4)["candidates"]:
            answer = input("Also train the optional 30-epoch candidate? [y/N]: ").strip().lower()
            if answer in ("y", "yes"):
                run(4, "--extend30", "--continue")
        stage_four = current(4)["candidates"]
        options = {"use10": read_state()["stages"]["3"]["selected"],
                   "use20": stage_four["epochs_20"]}
        if "epochs_30" in stage_four:
            options["use30"] = stage_four["epochs_30"]
        choose(4, options)

    if current(5).get("status", "not started") == "not started":
        answer = input("Skip batch size 8 if GPU memory is limited? [y/N]: ").strip().lower()
        extra = ("--skip-batch8",) if answer in ("y", "yes") else ()
    else:
        extra = ()
    ensure_executed(5, *extra)
    if current(5).get("status") != "complete":
        options = {label.removeprefix("batch_"): record
                   for label, record in current(5)["candidates"].items()}
        choose(5, options)

    ensure_executed(6)
    if current(6).get("status") != "complete":
        options = {label.removeprefix("weights_").replace("_", ","): record
                   for label, record in current(6)["candidates"].items()}
        choose(6, options)

    selected = current(6)["selected"]
    print("\nAll six stages complete.", flush=True)
    print("Selected parameters:", selected["parameters"], flush=True)
    print("Best validation water IoU:", selected["best_validation"]["binary_water_iou"], flush=True)
    print("Checkpoint:", selected["checkpoint"], flush=True)


if __name__ == "__main__":
    main()
