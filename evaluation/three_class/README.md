# Three-class saved-mask evaluation

The class convention matches the eWaSR fine-tuning workflow:

| Class ID | Class |
|---|---|
| 0 | obstacle |
| 1 | water |
| 2 | sky |

Use raw single-channel or indexed-palette PNG masks. Palette indices are preserved.
RGB colour masks, overlays, invalid labels and mismatched mask sizes are rejected.
Scoring uses the saved masks at their native resolution, without automatic resizing.

## Directory-based reference masks

From `thesis-codebase`:

```sh
.venv-sam3-mlx/bin/python evaluation/three_class/three_class_evaluator.py \
  --dataset custom \
  --dataset-name "My three-class test set" \
  --prediction-dir /absolute/path/to/prediction/masks \
  --ground-truth-dir /absolute/path/to/reference/masks \
  --model-name ewasr \
  --run-name pretrained
```

Masks pair by exact relative PNG path. For example, `dock/image_001.png` in the
reference folder pairs with `dock/image_001.png` in the prediction folder.
Point at the mask directory itself, not a run folder containing masks and overlays.
Every reference is required unless a selection file specifies a subset.

For a ground-truth encoding that uses different values, supply them in
**obstacle, water, sky** order, for example:

```sh
--ground-truth-class-values 10 20 30 --ignore-value 255
```

Predictions always use 0, 1, 2. Ignore pixels are excluded only according to the
reference mask; an ignore value is never a fourth evaluated class. Custom data
have no ignore value by default. Record reference provenance with `--dataset-name`;
SAM3 pseudo-labels should be identified as pseudo-label references.

## MaSTR1325 preset

```sh
.venv-sam3-mlx/bin/python evaluation/three_class/three_class_evaluator.py \
  --dataset mastr1325 \
  --prediction-dir /absolute/path/to/three_class/masks \
  --model-name ewasr \
  --run-name pretrained
```

This loads local images and reference masks through `dataset_specs.py`, uses class
IDs 0/1/2 and ignores reference value 4. Both the dataset mask name (`0001m.png`)
and the image-stem prediction name (`0001.png`) are accepted. Supplying both for
one image is an error. MaSTR1325 image dimensions are checked against mask dimensions.

`--selection-file` accepts canonical reference keys, one per line. For MaSTR1325
these are mask names such as `0001m.png`. Blank lines, duplicate keys and `#`
comments are handled; unknown keys are rejected.

Tampere and WaterNet ground truths are binary and cannot independently score
obstacle and sky. Use their three-class pseudo-labels through custom mode if that
is the intended reference comparison.

## Metrics and outputs

Each timestamped run writes to
`evaluation/three_class/results/<model>/<dataset>/<run>/<timestamp>/`, or to
`--results-dir` when supplied:

- `per_image.csv`: dimensions, source paths, valid/ignored pixel counts, all nine
  confusion counts, per-class TP/FP/FN/TN, IoU, Dice/F1, precision and recall,
  and pixel accuracy and class means.
- `summary.json` and `summary.txt`: configuration, reference provenance, pairing
  counts, per-image macro averages and metrics from the aggregate pixel confusion.
- `confusion_matrix.csv`: aggregate 3×3 confusion matrix. Rows are ground truth;
  columns are predictions; both use obstacle, water, sky order.
- `mask_pairing_report.txt`: missing predictions and predictions without references.

The metric definitions are:

- Class IoU = TP / (TP + FP + FN).
- Class Dice/F1 = 2TP / (2TP + FP + FN).
- Class precision = TP / (TP + FP); class recall = TP / (TP + FN).
- Pixel accuracy = correctly classified valid pixels / all valid pixels.
- Mean IoU averages the class IoUs with a nonzero union.

Undefined ratios are `nan` in CSV, `null` in JSON, and excluded from the relevant
average. A class absent from both reference and prediction is excluded from mean
IoU; a class predicted incorrectly still contributes an IoU of zero. Global
metrics use the sum of all image confusion matrices; macro metrics give each image
equal weight. These quantities can differ.

Missing selected predictions, invalid masks and all-ignored references stop the
run before metric files are written. Extra predictions are reported and excluded.
No inference loss is reported because only saved class masks are available.

Module invocation is also supported:

```sh
.venv-sam3-mlx/bin/python -m evaluation.three_class.three_class_evaluator --help
```
