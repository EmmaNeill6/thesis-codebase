"""Build a complete every-fifth-frame CataRob SAM3 label set.

Reuse the existing 5%-padded mask/overlay pairs as hard links and infer only
frames absent from all four source folders. Safe to rerun after interruption.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from PIL import Image

from predict_sam3_mlx import create_predictor, predict_image, save_outputs


ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data/Catarob/left_frames"
OUTPUT = Path(__file__).resolve().parent / "output/CataRob_All_5pct"
SOURCES = (
    "CataRob_It_1_5ppt",
    "CataRob_Chronological_5ppt",
    "CataRob_V5_V6_5ppt",
    "CataRob_Channel_5ppt",
)
PADDING_PERCENT = 5.0
CONFIDENCE_THRESHOLD = 0.5


def main() -> None:
    source_root = OUTPUT.parent
    images = sorted(IMAGES.glob("left_frame_*.png"))
    frame_numbers = [int(image.stem.rsplit("_", 1)[1]) for image in images]
    expected = list(range(0, max(frame_numbers) + 1, 5))
    image_by_frame = dict(zip(frame_numbers, images))
    missing_images = [frame for frame in expected if frame not in image_by_frame]
    if missing_images:
        raise FileNotFoundError(f"Missing source frames: {missing_images}")

    (OUTPUT / "masks").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "overlays").mkdir(parents=True, exist_ok=True)

    origins: dict[int, str] = {}
    pending: list[int] = []
    for frame in expected:
        filename = image_by_frame[frame].name
        pair = [OUTPUT / kind / filename for kind in ("masks", "overlays")]
        existing = next(
            (
                name
                for name in SOURCES
                if all((source_root / name / kind / filename).is_file()
                       for kind in ("masks", "overlays"))
            ),
            None,
        )
        if existing is not None:
            for kind, destination in zip(("masks", "overlays"), pair):
                source = (source_root / existing / kind / filename).resolve()
                if destination.exists():
                    if not os.path.samefile(source, destination):
                        raise FileExistsError(f"Unexpected existing output: {destination}")
                else:
                    os.link(source, destination)
            origins[frame] = existing
        elif all(path.is_file() for path in pair):
            origins[frame] = "generated_5pct"
        elif any(path.exists() for path in pair):
            raise FileExistsError(f"Incomplete output pair for frame {frame}")
        else:
            pending.append(frame)
            origins[frame] = "generated_5pct"

    print(f"Expected frames: {len(expected)}; existing pairs: {len(expected) - len(pending)}; "
          f"frames to infer: {len(pending)}", flush=True)
    if pending:
        predictor = create_predictor("facebook/sam3", CONFIDENCE_THRESHOLD)
        for index, frame in enumerate(pending, 1):
            with Image.open(image_by_frame[frame]) as source:
                image = source.convert("RGB")
            mask = predict_image(predictor, image, CONFIDENCE_THRESHOLD, PADDING_PERCENT)
            filename = image_by_frame[frame].name
            save_outputs(image, mask, OUTPUT / "masks" / filename,
                         OUTPUT / "overlays" / filename)
            print(f"[{index}/{len(pending)}] {filename}", flush=True)

    manifest = OUTPUT / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame_number", "image", "mask", "overlay", "source", "padding_percent"))
        for frame in expected:
            filename = image_by_frame[frame].name
            writer.writerow((frame, f"data/Catarob/left_frames/{filename}",
                             f"masks/{filename}", f"overlays/{filename}",
                             origins[frame], PADDING_PERCENT))
    print(f"Complete: {len(expected)} mask/overlay pairs in {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
