# SAM 3 pseudo-label generator with MLX

`predict_sam3_mlx.py` is an inference-only pseudo-label generator. It loads
SAM 3 through MLX-VLM, reads an image or directory, and writes:

- a single-channel PNG mask containing the raw class IDs `0 = obstacle`,
  `1 = water`, and `2 = sky`;
- an RGB overlay using the repository's eWaSR/MaSTr colours.

Add `--save-scores` to write `scores/<input-name>.json`, print accepted SAM 3
instance scores, and rebuild `instance_scores.csv` for all score reports in the
output directory. The CSV has one row per detected instance, with its image,
class and score. A class with no accepted instance has one row with an empty
score and `instance_count = 0`. Each score belongs to one detected instance,
not every pixel in that region. Pixels not covered by any accepted instance
are assigned obstacle without a SAM 3 score; the reports include their count.
Existing PNG masks contain only class IDs, so their scores cannot be recovered
without running inference again.

There is no timing, benchmarking, manifest, or evaluation code in this runner.

SAM 3 is prompted for `obstacle`, `navigable water`, and `sky` in one
multi-prompt call. The `navigable water` result is stored as class `1`.
When instances overlap, the instance with the highest confidence wins. Pixels
not covered by an accepted SAM 3 instance default to obstacle, so the output is
a dense three-class mask like eWaSR's output.

## Reflection padding

By default, the runner adds a reflected border equal to 5% of each image
dimension before inference. The new pixels mirror content next to the original
edge, which moves the real image boundary into the interior of SAM 3's input
and gives the model context there. After inference, the border is cropped off;
the saved mask and overlay therefore retain exactly the original resolution.

For a 1920x1080 image, 5% means 96 pixels on the left and right and 54 pixels
on the top and bottom. The padded model input is 2112x1188, while the saved
output remains 1920x1080. Different resolutions receive proportionally sized
borders.

Use `--padding-percent 0` to disable padding, or provide another percentage
from 0 to 50. Larger values give SAM 3 more artificial boundary context but
also make the original scene occupy less of the model input after resizing.

## Environment

The existing TestingPart2 MLX environment can be reused:

```bash
source .venv-sam3-mlx/bin/activate
python -m pip install -r TestingPart3/SAM3_Pseudo_Labels_Generator/requirements-mlx.txt
```

SAM 3 weights are gated. Authenticate once before the first download:

```bash
hf auth login
```

## Tampere-WaterSeg open subset

From the `thesis-codebase` directory, run one image first:

```bash
.venv-sam3-mlx/bin/python TestingPart3/SAM3_Pseudo_Labels_Generator/predict_sam3_mlx.py \
  --input data/Tampere-WaterSeg \
  --subset open \
  --max-images 1 \
  --padding-percent 5 \
  --output-dir TestingPart3/SAM3_Pseudo_Labels_Generator/output/confidence_inspection \
  --save-scores
```

Then process all 200 open-subset images by omitting `--max-images`:

```bash
.venv-sam3-mlx/bin/python TestingPart3/SAM3_Pseudo_Labels_Generator/predict_sam3_mlx.py \
  --input data/Tampere-WaterSeg \
  --subset open \
  --output-dir TestingPart3/SAM3_Pseudo_Labels_Generator/output/confidence_inspection \
  --save-scores
```

To rebuild the CSV from existing score JSON files without loading SAM 3:

```bash
.venv-sam3-mlx/bin/python TestingPart3/SAM3_Pseudo_Labels_Generator/predict_sam3_mlx.py \
  --output-dir TestingPart3/SAM3_Pseudo_Labels_Generator/output/confidence_inspection \
  --scores-csv-only
```

The defaults already point to this dataset and subset, so this also works:

```bash
.venv-sam3-mlx/bin/python TestingPart3/SAM3_Pseudo_Labels_Generator/predict_sam3_mlx.py --max-images 1
```

Outputs are written beneath
`TestingPart3/SAM3_Pseudo_Labels_Generator/output/Tampere-WaterSeg/open`:

```text
masks/<input-name>.png
overlays/<input-name>.png
scores/<input-name>.json  # only with --save-scores
instance_scores.csv         # with --save-scores or --scores-csv-only
```

Pass `--output-dir PATH` to change the destination. Existing complete mask and
overlay pairs are skipped unless `--overwrite` is supplied. With
`--save-scores`, an image without a score JSON is processed again.

## CataRob every-fifth-frame scores

`output/CataRob_All_5pct/manifest.csv` lists exactly the 708 source frames
used for that pseudo-label set: frame 0000, 0005, ..., 3535. Use it as the
selection source, rather than pointing at the complete source image directory.
From the `thesis-codebase` directory:

```bash
.venv-sam3-mlx/bin/python TestingPart3/SAM3_Pseudo_Labels_Generator/predict_sam3_mlx.py \
  --selection-manifest TestingPart3/SAM3_Pseudo_Labels_Generator/output/CataRob_All_5pct/manifest.csv \
  --output-dir TestingPart3/SAM3_Pseudo_Labels_Generator/output_confidence/CataRob_All_5pct \
  --padding-percent 5 \
  --confidence-threshold 0.5 \
  --scores-only
```

This reruns SAM 3 on the selected source images and writes score JSON plus
`instance_scores.csv` in the separate confidence directory. It does not
rewrite the existing pseudo-label masks or overlays. Add `--max-images 5` for
a short trial, then remove it and rerun the same command to continue; saved
score reports are skipped. Use `--scores-csv-only` with the same `--output-dir`
to rebuild a CSV from completed reports after an interrupted run.
