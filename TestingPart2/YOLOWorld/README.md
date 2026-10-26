# YOLO-World object-detection test on Apple Silicon

This experiment runs the Ultralytics YOLO-World V2 small checkpoint through
PyTorch's MPS backend. MPS uses the Apple GPU; it is the Apple equivalent of a
PyTorch CUDA device, not a CUDA implementation.

YOLO-World returns bounding boxes. The outputs from this folder must not be
passed to the project's binary-water mask evaluator.

## Environment

Create an isolated Python 3.11 environment from the project root:

```bash
python3.11 -m venv .venv-yoloworld-mps
source .venv-yoloworld-mps/bin/activate
python -m pip install --upgrade pip
python -m pip install -r TestingPart2/YOLOWorld/requirements-mps.txt
```

Confirm that PyTorch can see the Apple GPU:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## One-image smoke test

The first run downloads `weights/yolov8s-worldv2.pt`. From the project root:

```bash
.venv-yoloworld-mps/bin/python \
  TestingPart2/YOLOWorld/predict_yoloworld.py \
  --source data/MaSTR1325/images/0071.jpg \
  --prompts water person boat crane dock \
  --device mps
```

The runner writes an annotated JPEG and a JSON record below
`TestingPart2/YOLOWorld/output_mps/smoke_test/`. The JSON preserves prompts,
thresholds, package versions, timings and every detected box.

The default confidence threshold is deliberately low (`0.15`) for exploration.
It should be frozen before a quantitative experiment.

## Verified environment

The smoke test was verified on macOS 26.5.2, Apple Silicon, Python 3.11.16,
PyTorch 2.14.0 and Ultralytics 8.4.143. Both
`torch.backends.mps.is_built()` and `torch.backends.mps.is_available()` returned
`True` when executed with host GPU access.
