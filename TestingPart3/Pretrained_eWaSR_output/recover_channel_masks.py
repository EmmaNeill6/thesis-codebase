"""Recover exact class masks from this project's lossless eWaSR overlays.

Only writes a mask when every overlay pixel has exactly one matching class
under the blend used by predict_pretrained_ewasr.py.
"""

from pathlib import Path

import numpy as np
from PIL import Image

# Keep in sync with predict_pretrained_ewasr.py. Importing that module would
# load MLX and require GPU access for this entirely CPU-based recovery.
CLASS_COLOURS = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], dtype=np.uint8)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/lars_v1-2"
OUTPUT = ROOT / "TestingPart3/Pretrained_eWaSR_output/LaSR_eWaSR"


def main() -> None:
    for split in ("test", "train", "val"):
        images = sorted((DATA / split / "images/Channel").glob("*.jpg"))
        for image_path in images:
            relative = Path(split) / "images/Channel" / f"{image_path.stem}.png"
            overlay_path = OUTPUT / "overlays" / relative
            mask_path = OUTPUT / "masks" / relative
            with Image.open(image_path) as source:
                original = np.asarray(source.convert("RGB"), dtype=np.float32)
            with Image.open(overlay_path) as source:
                overlay = np.asarray(source.convert("RGB"), dtype=np.uint8)
            if original.shape != overlay.shape:
                raise ValueError(f"Image and overlay sizes differ: {image_path}")
            matches = np.zeros(original.shape[:2], dtype=np.uint8)
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            for class_id, colour in enumerate(CLASS_COLOURS.astype(np.float32)):
                expected = np.clip(original * 0.7 + colour * 0.3, 0, 255).astype(np.uint8)
                matched = np.all(expected == overlay, axis=2)
                matches += matched
                mask[matched] = class_id
            if not np.all(matches == 1):
                raise ValueError(
                    f"Overlay does not uniquely encode every pixel: {overlay_path} "
                    f"({np.count_nonzero(matches != 1)} ambiguous pixels)"
                )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(mask_path)
        print(f"{split}: recovered {len(images)} masks", flush=True)


if __name__ == "__main__":
    main()
