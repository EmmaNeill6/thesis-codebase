# Restartable SAM 3 maritime inference with MLX

`predict_sam3.py` processes dataset images on Apple Silicon using MLX-VLM's
SAM 3 implementation. It loads dataset paths and evaluator-compatible output
names from the project-root `dataset_specs.py`.

The experiment YAML files contain only the settings that differ between SAM 3
runs: dataset, output location, model ID, confidence threshold and prompts.

## Environment

Use the isolated Apple Silicon environment:

```bash
source .venv-sam3-mlx/bin/activate
```

Before the first download, authenticate for access to Meta's gated SAM 3
weights:

```bash
hf auth login
```

Install or refresh the runner dependencies with:

```bash
python -m pip install -r TestingPart2/SAM3/requirements-mlx.txt
```

## Configurations

The included configurations use the same water prompt and threshold:

- `configs/tampere_waterseg.yaml`
- `configs/waternet.yaml`
- `configs/mastr1325.yaml`

The current final-mask rule is:

```text
union(water instances)
```

Each prompt still receives its own saved union mask. To change prompts,
threshold or other scientific settings, choose a new `run_name` so results
from different experiments cannot be mixed.

## Validate without loading the model

From the project root:

```bash
.venv-sam3-mlx/bin/python TestingPart2/SAM3/predict_sam3.py \
  --config TestingPart2/SAM3/configs/tampere_waterseg.yaml \
  --max-new-images 5 \
  --dry-run
```

The dry run prints the dataset size, completed count, pending count and the
images that would be processed.

## Process five unfinished Tampere images

```bash
.venv-sam3-mlx/bin/python TestingPart2/SAM3/predict_sam3.py \
  --config TestingPart2/SAM3/configs/tampere_waterseg.yaml \
  --max-new-images 5
```

Run the same command again to skip completed images and process the next five.
Omit `--max-new-images` when the runner should continue through every remaining
image.

## Output structure

The shared dataset specification chooses the dataset-specific part of the
output path. One run contains:

```text
<output_root>/binary_water/<dataset>/<run_name>/
  experiment_config.yaml
  run_manifest.json
  run_attempts.jsonl
  predictions.jsonl
  inference_times.txt
  comparable_timings.csv
  comparable_timing_summary.json
  memory_usage.jsonl
  original_inputs/<dataset-relative image>
  prompt_masks/<prompt>/<dataset-relative mask>.png
  predicted_masks/<dataset-relative mask>.png
  prediction_overlays/<dataset-relative mask>.png
  image_records/<dataset-relative image>.json
```

The quantitative masks are single-channel PNG files with `0 = non-water` and
`255 = water`. The overlay uses the same blue-water/yellow-non-water colours as
the eWaSR prediction script.

The versioned `comparable_timings.csv` and `comparable_timing_summary.json`
use the same timing definitions as the eWaSR runner. Inference time starts
after model-specific image preprocessing and ends after returned model outputs
have been materialized as NumPy arrays. Final-mask latency starts when the
source image is opened and ends when the final 0/255 water mask is ready in
memory. Both exclude output-file writing and one unrecorded warm-up inference.
The summary reports mean, median, p95 and p99 for each timing, plus FPS derived
from mean inference time; this is not whole-run throughput.

`inference_times.txt` remains for compatibility. When resuming an older run,
only newly processed records with timing protocol version 2 enter the
comparable report; older records remain in the legacy inference file and are
not silently mixed into the new timing summary. Each image JSON also records
the broader `end_to_end_processing` time, prompt roles, instance scores,
boxes, areas, success or failure, and every output location.

Each predictor invocation appends one `memory_usage.jsonl` record with its
peak process resident memory (MiB) and counts for images attempted and
successfully processed in that invocation. The peak includes model loading,
warm-up, inference and output writing since process launch. It is not a
separate GPU-memory figure. For resumed runs, compare individual invocation
records rather than treating the last peak as a whole-experiment aggregate.

## Restart safety

An image is complete only when its successful JSON record and all declared
outputs exist. Failed or interrupted images remain unfinished and are retried
on the next invocation. Files are written in a temporary staging directory and
the completion record is written last.

The copied YAML and its SHA-256 digest identify the experiment. If the YAML is
edited, the existing run refuses to resume; update `run_name` for the changed
experiment.
