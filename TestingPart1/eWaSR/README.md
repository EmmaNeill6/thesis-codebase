# eWaSR MLX inference

Inference-only MLX port of **eWaSR**, the embedded-compute-ready maritime
obstacle segmentation network by Matija Teršek, Lojze Žust, and Matej Kristan.
The original paper and model are described in [Sensors 2023](https://www.mdpi.com/1424-8220/23/12/5386).

This copy runs the ResNet-18, non-IMU model natively on Apple silicon through
MLX. The prediction path does not import PyTorch or PyTorch Lightning.

## Setup

Use Python 3.11 on an Apple silicon Mac and install the inference dependencies:

```bash
python -m pip install -r requirements.txt
```

The converted checkpoint must be available at:

```text
Pretrained/ewasr_resnet18_mlx.npz
```

`convert_weights_to_mlx.py` can recreate that file from the original
`Pretrained/ewasr_resnet18.pth`. Conversion requires PyTorch, but inference does
not.

## Run inference

Change the `DATASET` value near the top of `predict_binary_water.py` to one of
`"tampere"`, `"waternet"`, `"mastr1325"`, or `"pohang"`, then run:

```bash
python predict_binary_water.py
```

The script saves grayscale masks (`0 = non-water`, `255 = water`), two-colour
overlays, and per-image Metal execution times below `../output_eWaSR`.
Dataset paths and output-name rules are defined in the project-root
`../../dataset_specs.py`.

For a direct timing comparison with SAM 3, the script also writes
`comparable_timings.csv` and `comparable_timing_summary.json`. Inference time
starts after image preprocessing and ends after the model output has been
materialized as a NumPy array. Final-mask latency starts when the source image
is opened and ends when the 0/255 water mask is ready in memory. Neither metric
includes output-file writing; one warm-up image is excluded. The summary gives
mean, median, p95 and p99 for both timings. Its FPS value is based only on mean
inference time, not whole-run throughput.

Each invocation also appends a record to `memory_usage.jsonl` containing its
peak process resident memory (MiB) and number of successfully processed images.
This peak is measured since process launch, so it includes model loading,
warm-up, inference and output writing. It is not a separate GPU-memory figure.

Inputs are padded internally to 64-pixel boundaries for integer pyramid-pooling
kernels, then predictions are cropped back to the original image dimensions.

## Validation

The MLX port was compared with the original PyTorch/MPS model using the same
checkpoint and preprocessing. On the checked 2048×1080 Pohang frame, the binary
water masks agreed on 99.99995% of pixels (one differing pixel).

## Citation

```bib
@article{tersek2023ewasr,
  author = {Ter\v{s}ek, Matija and \v{Z}ust, Lojze and Kristan, Matej},
  title = {eWaSR -- An Embedded-Compute-Ready Maritime Obstacle Detection Network},
  journal = {Sensors},
  year = {2023},
  volume = {23},
  number = {12},
  pages = {5386},
  doi = {10.3390/s23125386},
}
```

## License

The upstream eWaSR project and pretrained weights are licensed under Apache-2.0.
See `LICENSE`.
