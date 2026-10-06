# Binary water-mask evaluation

This evaluator scores water versus non-water. Predictions are single-channel
0/255 PNG masks, with pixels >=128 treated as water. Dataset-specific ground-truth
encodings and ignore regions come from `dataset_specs.py`.

From `thesis-codebase`:

```sh
.venv-sam3-mlx/bin/python evaluation/binary/water_mask_evaluator.py \
  --dataset tampere \
  --prediction-dir /absolute/path/to/predicted_masks \
  --model-name ewasr \
  --run-name main
```

The labelled datasets are `tampere`, `waternet` and `mastr1325`. Predictions must
use the relative output names defined in `dataset_specs.py`. `--selection-file`
accepts relative evaluation keys, one per line; blank lines and `#` comments are
ignored. The resolution audit at `dataset_metadata/dataset_image_resolutions.csv`
is required.

Each run produces `per_image.csv`, `summary.json`, `summary.txt`, and
`mask_pairing_report.txt` under:

```text
evaluation/binary/results/<model>/<dataset>/<run>/<timestamp>/
```

Metrics remain water IoU, Dice, precision, recall, specificity, balanced accuracy,
pixel accuracy, signed water-area error and symmetric water-boundary error.
Summaries contain per-image macro averages and metrics from summed pixel counts.

The binary timing/quality comparison is now located at
`evaluation/binary/compare_model_performance.py`. Its arguments are unchanged;
its default output is under `evaluation/binary/results/comparison/`.

Module invocation is also supported:

```sh
.venv-sam3-mlx/bin/python -m evaluation.binary.water_mask_evaluator --help
```
