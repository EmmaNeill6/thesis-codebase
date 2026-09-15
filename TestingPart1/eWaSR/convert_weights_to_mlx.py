"""One-time conversion of the original PyTorch eWaSR weights to MLX."""

from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Pretrained/ewasr_resnet18.pth"
DESTINATION = ROOT / "Pretrained/ewasr_resnet18_mlx.npz"


def main():
    state = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]

    converted = {}
    for name, tensor in state.items():
        if name.endswith("num_batches_tracked"):
            continue
        value = tensor.detach().numpy()
        if value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        converted[name] = value

    np.savez(DESTINATION, **converted)
    print(f"Converted {len(converted)} arrays to {DESTINATION}")


if __name__ == "__main__":
    main()
