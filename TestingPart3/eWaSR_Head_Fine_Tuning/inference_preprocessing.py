"""Shared preprocessing for fixed-size pipeline inference; no experiment runner."""
import numpy as np
from PIL import Image

MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

def prepare_input(image: Image.Image, input_size=None) -> tuple[np.ndarray, tuple[int, int]]:
    """Optionally resize the RGB input; references always stay full-resolution."""
    if input_size is not None:
        image = image.resize(tuple(input_size), Image.Resampling.BILINEAR)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = array.shape[:2]
    array = (array - MEAN) / STD
    array = np.pad(array, ((0, (-height) % 64), (0, (-width) % 64), (0, 0)))
    return array[None, ...], (height, width)

def resize_class_scores(scores: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize continuous scores rather than interpolating discrete class IDs."""
    if scores.ndim != 3 or scores.shape[-1] != 3:
        raise ValueError("Expected HxWx3 class scores")
    if (scores.shape[1], scores.shape[0]) == target_size:
        return scores
    return np.stack([
        np.asarray(Image.fromarray(scores[..., index].astype(np.float32)).resize(
            target_size, Image.Resampling.BILINEAR
        )) for index in range(3)
    ], axis=-1)
