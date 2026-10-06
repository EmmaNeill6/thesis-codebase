"""Prepare the requested CataRob frame selection and index its SAM3 outputs."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_DIR / "data/Catarob/left_frames"
OUTPUT_DIR = Path(__file__).resolve().parent / "output/CataRob_It_1_5ppt"

# Each tuple is (split, first frame, stated range end, number to select).
BLOCKS = (
    ("train", 75, 525, 90),
    ("train", 1600, 2050, 90),
    ("train", 2390, 2840, 90),
    ("train", 2845, 3295, 90),
    ("val", 825, 1125, 60),
    ("val", 2070, 2370, 60),
    ("test", 1200, 1426, 45),
    ("test", 1700, 1926, 45),
    ("held_out", 3385, 3536, 30),
)


def selection() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for split, start, end, count in BLOCKS:
        last = start + 5 * (count - 1)
        if last > end:
            raise ValueError(f"{split} range {start}-{end} cannot hold {count} fifth frames")
        for frame in range(start, last + 1, 5):
            filename = f"left_frame_{frame:04d}.png"
            if not (IMAGES_DIR / filename).is_file():
                raise FileNotFoundError(IMAGES_DIR / filename)
            rows.append(
                {
                    "split": split,
                    "range_start": start,
                    "range_end": end,
                    "frame_number": frame,
                    "image": f"data/Catarob/left_frames/{filename}",
                    "mask": f"masks/{filename}",
                    "split_mask": f"by_split/masks/{split}/{filename}",
                }
            )
    counts = Counter(row["split"] for row in rows)
    assert counts == {"train": 360, "val": 120, "test": 90, "held_out": 30}
    return rows


def prepare(output_dir: Path, rows: list[dict[str, str | int]]) -> None:
    selected = output_dir / "selected_unique_images"
    selected.mkdir(parents=True, exist_ok=True)
    for frame in sorted({int(row["frame_number"]) for row in rows}):
        filename = f"left_frame_{frame:04d}.png"
        link = selected / filename
        target = IMAGES_DIR / filename
        if link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f"Unexpected existing link: {link}")
        elif link.exists():
            raise FileExistsError(f"Expected a symlink at {link}")
        else:
            link.symlink_to(target)

    manifest = output_dir / "selection_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    train_frames = {int(row["frame_number"]) for row in rows if row["split"] == "train"}
    test_frames = {int(row["frame_number"]) for row in rows if row["split"] == "test"}
    overlap = sorted(train_frames & test_frames)
    (output_dir / "selection_notes.txt").write_text(
        "CataRob SAM3 selection, 5% reflection padding per side.\n"
        "Every block selects the stated count starting at its first frame and stepping by 5.\n"
        "Requested entries: train 360, val 120, test 90, held_out 30; total 600.\n"
        f"Distinct source images: {len({row['frame_number'] for row in rows})}.\n"
        f"Train/test overlap: {len(overlap)} frames, {overlap[0]}-{overlap[-1]} every 5th frame.\n"
        "Those test entries are not an independent holdout; revise the ranges before evaluation.\n"
        "Masks use 0=obstacle, 1=water, 2=sky.\n"
        "Inference runs once per distinct image; by_split links duplicate overlapping masks.\n"
    )
    print(f"Prepared {len(rows)} entries and {len(list(selected.iterdir()))} distinct images")
    print(f"Train/test overlap: {len(overlap)} frames")
    print(f"Manifest: {manifest}")


def finalize(output_dir: Path, rows: list[dict[str, str | int]]) -> None:
    for row in rows:
        filename = Path(str(row["mask"])).name
        split = str(row["split"])
        for kind in ("masks", "overlays"):
            source = output_dir / kind / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = output_dir / "by_split" / kind / split / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not os.path.samefile(source, destination):
                    raise FileExistsError(destination)
            else:
                os.link(source, destination)
    print("Indexed 600 mask and overlay entries under by_split/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    rows = selection()
    if args.action == "prepare":
        prepare(args.output_dir, rows)
    else:
        finalize(args.output_dir, rows)


if __name__ == "__main__":
    main()
