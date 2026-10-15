# eWaSR head-only fine-tuning with MLX

## Training data-size evaluation

`data_size_sweep.py` generates, checks, runs, and reports training-data budgets
while preserving a chosen sub-trip split. The CataRob study recipe is
`../DataSource/data_size_catarob_80_10_10_subtrips.yaml`. It freezes
`catarob_all_5pct_sweep_5_2_1` (80/10/10 after the separate reserve), using
iteration 1 by convention rather than ranking test results. Validation, test,
common test core, and held-out membership remain identical in every run.

From the `thesis-codebase` directory:

```bash
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/data_size_sweep.py generate \
  --study-config TestingPart3/DataSource/data_size_catarob_80_10_10_subtrips.yaml
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/data_size_sweep.py check \
  TestingPart3/eWaSR_Head_Fine_Tuning/outputs/data_size_sweeps/catarob_80_10_10_subtrips
.venv-sam3-mlx/bin/python -u TestingPart3/eWaSR_Head_Fine_Tuning/data_size_sweep.py train \
  TestingPart3/eWaSR_Head_Fine_Tuning/outputs/data_size_sweeps/catarob_80_10_10_subtrips
```

Generation writes files only; it never starts training. Refuse an existing
nonempty study directory rather than overwrite experiments. `check` verifies
provenance, counts, nesting, and fixed memberships, then invokes the unchanged
`train_head.py --check-data` on every generated YAML (no Metal GPU needed).

Budgets refer to the 539-image training pool: 25% = 135, 50% = 270,
75% = 404, and 100% = 539. The fixed validation/test/reserve counts are
67/67/35. Percentages can be configured in the recipe as increasing integers
ending at 100; its source YAML supplies image pairing and frame spacing.

Three selection seeds use shifted temporal phases within the **original**
training sub-trips. Each sub-trip's frame order progressively refines an equally
spaced dyadic grid. Group orders are interleaved by fractional sampling density,
allocating frames approximately proportionally and meeting exact total counts.
Every budget is a prefix of one ordering per repetition. This gives nested,
approximately uniform temporal coverage; it is not a strict every-k stride or
content-based adaptive sampling. `selection_orders/` records ranks and phases.
Unselected training images are omitted from the active manifest and recorded
as `unused_train` in the full inventory. Sampling never moves a frame across
the original sub-trip's training/validation/test assignment.

The existing recipe uses AdamW, LR 0.001, decay 0.001, 20 epochs, batch 2, and
class weights 1/1/1. Checkpoint selection uses validation binary water IoU.
There are ten unique runs: three per smaller size and one 100% endpoint.
The default recipe trains the 100% endpoint afresh alongside the smaller runs
for comparable timings. Optional `reuse_baseline_result: true` requires a
completed result with matching membership, recipe, and hashes; its timing may
have been measured under different machine load. A fixed epoch count
means larger sizes receive more optimizer updates; this is a practical
training-budget experiment, not a comparison at equal compute.

`train` updates `summary.csv`, `aggregate.csv`, and `report.md` after every run.
It skips completed runs and refuses to overwrite partial outputs. `summary`
rebuilds the reports, and `train --run-id RUN_ID` runs one generated subset.
The reported sample SD measures subset-selection variation, with one fixed
training seed. The single 100% endpoint has no estimated SD. Results measure
agreement with pseudo-labels from one recording; the held-out reserve remains
unevaluated. The test set was already used in the split sweep.

Training seconds include validation passes and checkpoint saving within the
training loop. Total elapsed also includes initial loading and final test
evaluation. Planned optimizer updates are 1,360/2,700/4,040/5,400 under
20 epochs and batch 2. These measure adaptation cost, not inference latency or
SAM3 pseudo-labelling time.

After the entire study finishes, `train` invokes the Matplotlib plotter to
export performance and training-cost curves as PNG, PDF, and SVG. Install
Matplotlib in the environment used for training/plotting. Each performance graph
overlays test and validation curves. Completed studies in the same parent folder
share a Water IoU axis and, separately, a training-time axis. The limits include
all individual runs and mean ± sample SD bars. IoU values remain percentages;
this changes the display scale rather than rescaling measured scores.
Rebuild every completed dataset's plots together with:

```bash
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/plot_data_size_sweep.py \
  TestingPart3/eWaSR_Head_Fine_Tuning/outputs/data_size_sweeps --all
```

Rebuild one study with:

```bash
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/data_size_sweep.py plot \
  TestingPart3/eWaSR_Head_Fine_Tuning/outputs/data_size_sweeps/catarob_80_10_10_subtrips
```

Direct generation arguments (`--spec`, `--base-config`, `--output-dir`) are
also supported for another chosen sub-trip run; do not mix them with
`--study-config`. Each generated YAML references its membership CSV and can
be passed directly to the original trainer.

## Reproducible split sweep

`split_sweep.py` generates the five train/validation/test ratios and three
train/validation sorting methods discussed in the methodology. It removes one
fixed held-out reserve first, then applies each ratio to the remaining images.
The initial ratio is the exact `60/95 : 20/95 : 15/95` form of the original
`60/20/15/5` split. The three methods are consecutive blocks, whole contiguous
sub-trip windows, and random images. Consecutive blocks have one iteration;
the other methods each have three distinct split iterations, for 35 runs total.
The training seed and the fixed head-only AdamW recipe stay the same in all runs.

The example spec `../DataSource/split_sweep_pohang_100_smoke.yaml` selects 50
consecutively numbered image/mask pairs from Pohang Canal and 50 from Pier.
From the `thesis-codebase` directory:

```bash
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/split_sweep.py generate \
  --spec TestingPart3/DataSource/split_sweep_pohang_100_smoke.yaml \
  --output-dir TestingPart3/eWaSR_Head_Fine_Tuning/outputs/split_sweeps/pohang_100_smoke
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/split_sweep.py check \
  TestingPart3/eWaSR_Head_Fine_Tuning/outputs/split_sweeps/pohang_100_smoke \
  --run-id pohang_100_smoke_1_1_1
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/split_sweep.py train \
  TestingPart3/eWaSR_Head_Fine_Tuning/outputs/split_sweeps/pohang_100_smoke \
  --run-id pohang_100_smoke_1_1_1
```

`generate` only writes manifests, configs, audits, and an initial summary. `check`
uses `train_head.py --check-data` and does not train. `train` runs one named job;
pass `--all` instead of `--run-id` for the full sweep. It skips completed jobs
and stops at a partial output so that a failed run is inspected before retrying.
Run `split_sweep.py summary STUDY_DIR` to rebuild `summary.csv` from completed
evaluation reports. `--refresh` may regenerate a study only before training
starts. The generated run name is `dataset_splitversion_sortversion_iteration`.

Each study has a `source_inventory.csv`, `fixed_partitions.csv`, `study.json`,
and `summary.csv`. Each run has a `split_manifest.csv` listing every image and
mask, its recording and frame, its sub-trip when applicable, and its exact role.
It also has `split_audit.json` and a generated `config.yaml`. The trainer writes
`resolved_splits.json` from the CSV it actually loaded. All test sets are nested:
the smallest test set is a common core in every run, while larger test sets add
fixed images. The selected checkpoint is evaluated on both its full test set
and the common core; the fixed held-out set is never evaluated by the sweep.
The `summary.csv` records comparable common-core metrics after runs complete.
The trainer also writes `test_per_image_metrics.csv`, keyed by `image_id`, with
each test image's IoU and a flag for common-core membership. For a run completed
before that CSV existed, use `train_head.py --config RUN_CONFIG --per-image-only`
to calculate it from the saved best checkpoint without retraining.
Each audit records split counts, membership hashes, path overlap, and temporal
gaps within each role and recording. Before checking or training, the runner
also verifies that the source spec, starting checkpoint, generated config, and
split manifests still match their saved hashes. `hash_content: true`
additionally records SHA256 hashes and reports exact image bytes found in more
than one role; this is a diagnostic count and does not stop training. With the
default `false`, content overlap is unmeasured. The consecutive method
requires an unbroken train and validation block per recording, so generation
rejects pinned held-out or test choices that break those blocks.

The source spec can list multiple recordings, each with its image and mask
directories, `trip_id`, `frame_pattern` with a named `frame` capture, and an
optional first/last frame range. `require_complete_range: true` rejects missing
numbers or pairs. The default held-out assignment takes the final 5% of images,
apportioned across recordings; the test order then takes late development
frames from each recording. To preserve an existing reserve or test selection,
provide `held_out_image_ids` or `held_out_ids_file`, and `test_order_image_ids`
or `test_order_ids_file`. The test order list must cover at least the largest
requested test set. When using `frame_step` greater than one, also set
`first_frame` to anchor the sequence. The default avoids reading all image
bytes just to plan the sweep. A dataset with adjacent frames from one recording
can still have temporal similarity across split boundaries; its sub-trips are
temporal windows, not independent trips.

## Single CataRob run from the split YAML

`train_head.py` accepts a `TestingPart3/DataSource` YAML through `--config`.
It uses the YAML's exact train, validation, test, and unseen membership and its
pretrained checkpoint. Relative paths are resolved from the YAML file. The
single-run training recipe is fixed at AdamW, learning rate 0.001, weight decay
0.001, 20 epochs, batch size 2, and class weights 1/1/1. The YAML's `training`
block is used only for its validation checkpoint-selection metric; its other
training values belong to the staged runner.

From the `thesis-codebase` directory:

```bash
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/train_head.py --config TestingPart3/DataSource/catarob_split_60_20_15_5.yaml --check-data
.venv-sam3-mlx/bin/python -u TestingPart3/eWaSR_Head_Fine_Tuning/train_head.py --config TestingPart3/DataSource/catarob_split_60_20_15_5.yaml
```

The selected checkpoint is evaluated on the 15% test split after validation
selection. The 5% unseen split remains reserved. Output goes under the YAML's
`experiment.output_dir/head_only_lr1e-3_wd1e-3_20ep/`, separate from staged
results. `evaluation.txt` contains validation and test metrics, the selected
checkpoint, per-run training time, and total elapsed time. Each epoch's time is
also saved in `training_history.jsonl`. The test metrics measure agreement with
SAM3 pseudo-labels.

## Staged dataset experiment (steps 1–6)

The concrete LaSR Channel configuration is
`../DataSource/head_only_lars_channel_staged_v1.yaml`. It records the native
LaSR 140/15/137 folder inventory and assigns all 292 paired images by whole
`davimar_seq_*` sequence to 175/58/44/15 staged roles. Some images from the
native LaSR test folder enter the new training and validation roles, so this is
a separate experiment. Check its paths and membership with:

```bash
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/run_staged_experiment.py --config TestingPart3/DataSource/head_only_lars_channel_staged_v1.yaml --check-data
```

Use one resumable runner for the sequence in `thesis-notes/Second Half Progress
and Method/Post Meeting 8/Order of experiment for the datasets.md`. It makes
the handoff between stages explicit without duplicating six scripts. Copy
`../DataSource/staged_experiment_template.yaml` to a dataset-specific YAML and
fill its real image/mask paths, recording or trip IDs, selections, and expected
counts. The runner targets a 60/20/15/5 train/validation/test/unseen split,
keeps each recording together, and rejects missing or overlapping pairs. Set
`split_policy.max_fraction_deviation` explicitly if whole recordings prevent
an exact allocation; the default is exact largest-remainder counts. The
last 5% is reserved for final evaluation; steps 1–6 do not read its pixels or
evaluate either held-out split.

From the repository root, replace `MY_DATASET.yaml` with the completed YAML:

```bash
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/run_staged_experiment.py --config TestingPart3/DataSource/MY_DATASET.yaml --check-data
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/run_staged_experiment.py --config TestingPart3/DataSource/MY_DATASET.yaml --step 1
.venv-sam3-mlx/bin/python -u TestingPart3/eWaSR_Head_Fine_Tuning/run_staged_experiment.py --config TestingPart3/DataSource/MY_DATASET.yaml --step 1 --continue
```

`--step N` previews a stage and does not train. `--step N --continue` runs only
that stage; run stages 2 through 6 the same way. Each candidate starts from the
same pretrained non-IMU checkpoint, and checkpoints and parameter choices use
validation **binary water IoU**. Results, epoch histories, exact membership,
and resumable state go in the YAML's `experiment.output_dir`. Use `--status` to
inspect progress. Do not edit a YAML after its first run; a different split or
recipe needs a new output directory.

Stages 1–3 pick the best validation candidate automatically. Stage 4 runs 20
epochs and pauses for a choice. If its validation curve still improves, run
`--step 4 --extend30 --continue` to try 30 epochs. Then use `--step 4 --choose
use10`, `use20`, or `use30`. Stage 5 compares batch sizes 2, 4, and 8; add
`--skip-batch8` if memory does not permit 8, then choose with `--step 5 --choose
2` (or `4`/`8`). Stage 6 compares class weights `1,1,1`, `2,1,1`, `3,1,1`,
`3,2,1`, and `1,2,1`; inspect its obstacle miss and water-as-obstacle rates,
then record a choice such as `--step 6 --choose 2,1,1`. Use the same `--config`
argument for every command. No next stage starts until its preceding stage and
any required choice are complete.

## Separate channel-only random-split experiment

`../DataSource/head_only_tampere_channel_random_v1.yaml` uses only Tampere
channel images and `Tampere_Channel_5ppt/masks`, with the dock experiment's
training parameters and fixed random 120/40/40 split (seed 4022). From the
repository root:

```bash
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/run_tampere_channel_random.py --check-data
.venv-sam3-mlx/bin/python -u TestingPart3/eWaSR_Head_Fine_Tuning/run_tampere_channel_random.py
```

Outputs go to `outputs/tampere_channel_random_head_only_v1/`. Each trial reloads
the original pretrained checkpoint. This is separate from the earlier channel
experiment with an external CataRob test set. Exact membership and an overlap
audit are saved; all six output checkpoints are checked for frozen-state
changes. See `../DataSource/IMPORTANT_TAMPERE_CHANNEL_RANDOM_TRAINING_NOTES.txt`.

## Separate Tampere dock experiment

`../DataSource/head_only_tampere_dock_v1.yaml` defines an independent dock-only
attempt with 120/40/40 image pairs and three learning rates (`3e-4`, `1e-3`,
`3e-3`). Use its dedicated runner:

```bash
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/run_tampere_dock.py --check-data
.venv-sam3-mlx/bin/python -u TestingPart3/eWaSR_Head_Fine_Tuning/run_tampere_dock.py
```

These commands run from the repository root. The user confirmed the matching
`Tampere-Dock_5ppt/masks` source and requested a seeded random image split. Seed
4022 assigns 120/40/40 images once for all trials. The runner checks complete
coverage, rejects shared image/mask paths and identical image bytes across
roles, and protects previous experiment outputs. This is file-level separation;
unseen-trip independence cannot be established without recording metadata.
Optional recording CSV support is retained. See
`../DataSource/IMPORTANT_TAMPERE_DOCK_TRAINING_NOTES.txt` for the exact method,
current status, settings and interpretation.

This folder contains the first pseudo-label adaptation baseline for eWaSR. It
uses the validated non-IMU MLX architecture from `TestingPart1`, loads the
predefined ResNet-18 eWaSR checkpoint, and trains only the existing three-class
segmentation head.

The class convention is shared by eWaSR, MaSTr and the SAM 3 generator:

```text
0 = obstacle
1 = water
2 = sky
```

There is deliberately no runtime, FPS or memory benchmarking in this test.
The output is limited to model checkpoints, the run configuration, training
loss and validation segmentation metrics.

## What is frozen

`EWaSR.configure_head_only_training()` first freezes the complete network and
then unfreezes only `decoder.seg_head`. Therefore:

- the ResNet-18 encoder is frozen;
- all MetaFormer attention/refinement blocks are frozen;
- all SIM feature-fusion modules are frozen;
- frozen BatchNorm layers remain in evaluation mode;
- the existing segmentation head and its affine BatchNorm parameters train.

The head is retained rather than reinitialised because the pretrained model
already uses the required water/sky/obstacle taxonomy.

## Pretrained checkpoint

The default path reuses the existing validated checkpoint without creating a
second 230 MB copy:

```text
TestingPart1/eWaSR/Pretrained/ewasr_resnet18_mlx.npz
```

An alternative checkpoint can be supplied with `--weights`.

## Environment

Use Python 3.11 on an Apple-silicon Mac. From this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the initial checks, the already configured TestingPart1 environment can
also be used:

```bash
../../TestingPart1/eWaSR/.venv/bin/python train_head.py --help
```

## YAML-defined first experiment

The reproducible first-attempt configuration is stored at:

```text
TestingPart3/DataSource/head_only_finetuning_v1.yaml
```

It defines the chronological 150/50 Tampere train/validation blocks, the 50
every-30th CataRob test frames, all model and preprocessing settings, and the
three learning-rate trials. Validate it from this folder with:

```bash
../../TestingPart1/eWaSR/.venv/bin/python run_experiment.py \
  --config ../DataSource/head_only_finetuning_v1.yaml \
  --check-data
```

Run the complete comparison by removing `--check-data`. The runner evaluates
the pretrained baseline, selects the learning rate using validation mean IoU,
and evaluates only that selected checkpoint on the test split.

The supplied open-subset masks were not paired with channel images. Matching
`channel_image_*` pseudo-labels were found in `Tampere_Channel_5ppt/masks` and
are used by the YAML. The reason is recorded in both the YAML and
`TestingPart3/DataSource/IMPORTANT_TRAINING_NOTES.txt`.

## Validate a pseudo-labelled subset

This performs pairing, dimensions and class-ID checks without loading MLX:

```bash
../../TestingPart1/eWaSR/.venv/bin/python train_head.py \
  --train-images ../../data/MaSTR1325/images \
  --train-masks ../SAM3_Pseudo_Labels_Generator/output/MaSTR1325/masks \
  --check-data
```

The masks determine the subset. Images without a corresponding pseudo-label
are ignored, while every supplied mask must have exactly one matching image.

## Fine-tune the head

Training and validation directories must be selected before the experiment.
They should be separated by recording sequence rather than by randomly mixing
near-duplicate video frames.

```bash
../../TestingPart1/eWaSR/.venv/bin/python train_head.py \
  --train-images /path/to/train/images \
  --train-masks /path/to/train/pseudo_masks \
  --val-images /path/to/validation/images \
  --val-masks /path/to/validation/masks \
  --epochs 10 \
  --batch-size 2
```

Images are resized to the pretrained eWaSR input size of 512 x 384. Masks use
nearest-neighbour resizing and are downsampled to the head's 128 x 96 output
resolution for loss calculation. Inputs follow the same ImageNet
normalisation as the TestingPart1 inference path.

Generated files are written to `outputs/head_only/`:

```text
run_config.json
training_history.jsonl
latest_head_only.npz
best_head_only.npz
```

When validation data are supplied, `training_history.jsonl` records pixel
accuracy, mean IoU and obstacle/water/sky IoU for every epoch. The final thesis
comparison should still use the fixed, held-out ground-truth test set rather
than SAM 3 masks.

The first completed run is summarised in:

```text
outputs/tampere_channel_head_only_v1/experiment_summary.txt
```

It selected learning rate `1e-3` at epoch 10. Validation agreement with SAM3
improved strongly, but performance against the separate CataRob SAM3 masks was
worse than the pretrained baseline. This is evidence of negative transfer or
over-specialisation to the Tampere sequence, not evidence that head-only
fine-tuning improved the target deployment domain.

## Completed port-domain experiment

The separate port-camera configuration is:

```text
TestingPart3/DataSource/head_only_port_v1.yaml
```

It uses 98 MODS pairs for training, 35 LaRS Port pairs for validation (from
the requested `val`, `test`, and `train` source folders), and the 20 labelled
Pohang Port pairs for the fixed test set. These are source-based roles, not a
random split of neighbouring video frames. LaRS masks are stored in one shared
directory, so the runner explicitly intersects that directory with each
requested LaRS image source.

```bash
../../TestingPart1/eWaSR/.venv/bin/python run_experiment.py \
  --config ../DataSource/head_only_port_v1.yaml --check-data

../../TestingPart1/eWaSR/.venv/bin/python run_experiment.py \
  --config ../DataSource/head_only_port_v1.yaml
```

The completed run selected `1e-3` at epoch 8 using LaRS validation mean IoU.
Its held-out Pohang agreement against SAM3 masks increased from 0.935191 to
0.955051 three-class mIoU. Detailed metrics, exact resolved paths, histories,
and the selected checkpoint are in:

```text
outputs/port_head_only_v1/
```

See `TestingPart3/DataSource/IMPORTANT_PORT_TRAINING_NOTES.txt` for the full
method and report-ready interpretation. The SAM3 result is a pseudo-label
evaluation; compare both models with independently annotated masks before
claiming real-world performance improvement.

## Visualise the selected model on the test split

Generate raw class-ID masks, full-resolution prediction overlays, four-panel
comparisons, and a representative contact sheet for all 50 held-out CataRob
frames. Each comparison shows the original image, original pretrained eWaSR,
the selected fine-tuned eWaSR, and the SAM3 pseudo-label reference:

```bash
../../TestingPart1/eWaSR/.venv/bin/python visualize_test_predictions.py \
  --config ../DataSource/head_only_finetuning_v1.yaml
```

The command automatically reads the chosen checkpoint from
`outputs/tampere_channel_head_only_v1/selected_model.json`. Visualisations are
written beneath `outputs/tampere_channel_head_only_v1/test_visualizations/`.
The `pretrained_masks/` and `pretrained_overlays/` directories preserve the
original model's outputs separately. The comparison's SAM3 panel is a
pseudo-label reference and must not be described as independent ground truth.


## Native-resolution evaluation of the selected checkpoint

Run the already selected checkpoints on the exact test membership recorded in
`resolved_splits.json`, preserving the training configuration and selection:

```sh
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/evaluate_native_resolution.py \
  --experiment-dir TestingPart3/eWaSR_Head_Fine_Tuning/outputs/tampere_dock_head_only_v1
```

Run this command from the `thesis-codebase` directory with GPU access. The runner
uses native RGB dimensions, ImageNet normalization and bottom/right zero padding
(after normalization) to multiples of 64. It linearly upsamples the class scores
before argmax and crops padding, matching TestingPart1 inference. For 1920x1080
frames this gives a 1920x1088 input and a 480x272 internal score grid.

Both the original pretrained checkpoint and the fixed selected fine-tuned
checkpoint are evaluated against the same full-resolution SAM3 pseudo-labels.
No learning rate, epoch, membership or training-resolution settings are changed.
These results quantify pseudo-label agreement, not independent human-ground-truth
accuracy. The input scale, proportions and score interpolation differ from the
original 512x384-input / 128x96-mask training evaluation, so the comparison does
not isolate one resolution variable.

New timestamped outputs are placed under
`outputs/tampere_dock_head_only_v1/native_resolution_test/`: full-resolution masks,
overlays, comparison panels, a contact sheet, an inference manifest with file
fingerprints and a combined `summary.txt` / `comparison_summary.json`. Detailed
per-model scores and confusion matrices are saved separately under
`evaluation/three_class/results/`. The original training results are retained.


## Intermediate inference sizes

The same runner also accepts `--input-size WIDTH HEIGHT` and `--split validation`.
These settings change inference only; the chosen weights and training configuration
remain fixed. For example, from `thesis-codebase`:

```sh
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/evaluate_native_resolution.py \
  --split validation --input-size 768 576

.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/evaluate_native_resolution.py \
  --split validation --input-size 1024 768
```

Both sizes retain the training input's 4:3 proportions. They produce 192x144 and
256x192 class-score grids respectively, compared with 128x96 at 512x384. The RGB
input is resized before normalization. Scores are interpolated to input size,
then rescaled to the original image dimensions before argmax; reference masks
are not resized. Saved masks and overlays remain 1920x1080 for dock frames.

Use identical validation membership and full-resolution references to compare
candidate inference sizes. Choose the inference size using validation before
running it on the fixed test split. A 512x384 run with the same score interpolation
provides a fair full-resolution-reference baseline; the old 128x96 scores use a
different evaluation procedure. Run results live under
`outputs/tampere_dock_head_only_v1/<width>x<height>_resolution_<split>/`.

## Compare saved masks with SAM3

`compare_masks_against_sam3.py` compares one or two folders of saved class-ID
prediction PNGs with SAM3 pseudo-label PNGs. From `thesis-codebase`:

```sh
.venv-sam3-mlx/bin/python TestingPart3/eWaSR_Head_Fine_Tuning/compare_masks_against_sam3.py \
  --sam3-masks /path/to/SAM3/masks \
  --prediction /path/to/first/run-or-predicted_masks \
  --prediction /path/to/second/run-or-predicted_masks \
  --images /path/to/original/images \
  --output-dir /path/to/results
```

Omit the second `--prediction` for a single output. If `--images` is omitted,
the script reads original image paths from a nearby `visualization_manifest.json`;
it asks for `--images` when no manifest is available. A run directory is accepted when it contains
`test_visualizations/predicted_masks/`; direct `predicted_masks/` folders also work.
The command writes `failure_masks/`, `failure_overlays/`, `summary.csv`, and
`manifest.json` inside the supplied output folder.
For two predictions, blue means only the first differs from SAM3, purple means
only the second differs, and red means both differ. For one prediction, red
means it differs. Black means agreement. The prediction folders must contain
the same image names, and all masks for an image must have the same dimensions.
This script compares saved masks; it does not run a model checkpoint.
