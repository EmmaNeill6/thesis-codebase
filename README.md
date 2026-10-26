# Maritime water segmentation and eWaSR adaptation

Research code for comparing pretrained eWaSR and SAM3, generating three-class
SAM3 pseudo-labels, and adapting the eWaSR segmentation head on Apple Silicon.
YOLO-World is retained as a qualitative obstacle-detection pilot. This is a
curated source release, not a complete archive of every experiment in the thesis.

## Setup and external assets

Use Python 3.11 on an Apple Silicon Mac with a working Metal GPU for MLX runs.
Keep model environments separate; evaluation and data checks can run without a GPU.
From the repository root:

```bash
python3.11 -m venv .venv-sam3-mlx
source .venv-sam3-mlx/bin/activate
python -m pip install -r TestingPart3/eWaSR_Head_Fine_Tuning/requirements.txt
python -m pip install -r TestingPart2/SAM3/requirements-mlx.txt
python -m pip install -r evaluation/requirements.txt
```

For eWaSR inference alone use `TestingPart1/eWaSR/requirements.txt`. Conversion
also requires PyTorch (`python -m pip install torch`). The optional original
PyTorch architecture uses `TestingPart1/eWaSR/requirements-reference.txt`;
upstream training/export entry points are outside this release. Plotting scripts
additionally require Matplotlib. For the independent MPS detection environment,
follow [YOLO-World setup](TestingPart2/YOLOWorld/README.md).

Datasets, annotations and weights are external and are not downloaded by setup:

- Place Tampere-WaterSeg under `data/Tampere-WaterSeg/`, MaSTR1325 under
  `data/MaSTR1325/`, WaterNet under `data/water_v2/`, and the selected Pohang
  recordings under `data/pohang00/`. See `dataset_specs.py` for baseline folder
  layouts, image selection and human-mask encodings. Adaptation YAMLs also use
  `data/Catarob/left_frames/`, `data/lars_v1-2/` and `data/mods/` where specified.
  Obtain these from their original providers under their terms; no data licence
  is granted by this repository.
- Obtain the original non-IMU ResNet-18 checkpoint from the
  [eWaSR project](https://github.com/tersekmatija/eWaSR), put it at
  `TestingPart1/eWaSR/Pretrained/ewasr_resnet18.pth`, then run
  `python TestingPart1/eWaSR/convert_weights_to_mlx.py`. This produces the
  `ewasr_resnet18_mlx.npz` used by the MLX runners.
- SAM3 uses `facebook/sam3` through MLX-VLM. Arrange model access, authentication
  and any upstream terms separately. The first model load may download weights.
- The YOLO-World pilot uses `yolov8s-worldv2.pt`; its runner can download the
  checkpoint to `TestingPart2/YOLOWorld/weights/`.
- Adaptation needs original-size single-channel pseudo-labels with IDs
  **0 obstacle, 1 water, 2 sky**. Generate them into the folders named in the
  retained YAMLs; overlays are not labels. Bank/adapted checkpoints must be
  trained or supplied separately. No model bank is bundled.

Paths in experiment YAMLs are relative to the YAML file. Preserve the documented
layout or adjust storage paths before starting a new run. Do not edit a config
inside an existing resumable run. Historical source/published hashes are recorded
in [results/provenance.json](results/provenance.json).

## Commands

Run these from the repository root, using the appropriate environment. Supply
external images, labels and weights before invoking training or inference.

```bash
# eWaSR binary baseline: choose DATASET in the script first.
python TestingPart1/eWaSR/predict_binary_water.py

# SAM3 binary water baseline.
python TestingPart2/SAM3/predict_sam3.py \
  --config TestingPart2/SAM3/configs/tampere_waterseg.yaml

# Three-class pseudo-label generation (5% reflection padding).
python TestingPart3/SAM3_Pseudo_Labels_Generator/predict_sam3_mlx.py \
  --input data/Tampere-WaterSeg --subset dock --padding-percent 5 \
  --output-dir TestingPart3/SAM3_Pseudo_Labels_Generator/output/Tampere-Dock_5ppt

# Reported qualitative detector pilot (in its separate MPS environment).
python TestingPart2/YOLOWorld/test_random_datasets.py \
  --config TestingPart2/YOLOWorld/configs/maritime_random5.yaml

# Staged hyperparameter investigation: check, preview, then execute one stage.
python TestingPart3/eWaSR_Head_Fine_Tuning/run_staged_experiment.py \
  --config TestingPart3/DataSource/pohang_canal_split_60_20_15_5.yaml --check-data
python TestingPart3/eWaSR_Head_Fine_Tuning/run_staged_experiment.py \
  --config TestingPart3/DataSource/pohang_canal_split_60_20_15_5.yaml --step 1
# Add --continue to execute; later stages may require an explicit --choose.

# Established partition study: generate first, check, then train.
python TestingPart3/eWaSR_Head_Fine_Tuning/split_sweep.py generate \
  --spec TestingPart3/DataSource/split_sweep_catarob_all_5pct.yaml \
  --output-dir TestingPart3/eWaSR_Head_Fine_Tuning/outputs/split_sweeps/catarob_all_5pct_sweep
python TestingPart3/eWaSR_Head_Fine_Tuning/split_sweep.py check \
  TestingPart3/eWaSR_Head_Fine_Tuning/outputs/split_sweeps/catarob_all_5pct_sweep
# Use train STUDY_DIR --all to execute the checked study.

# Data-size study: requires the generated parent split named in the recipe.
python TestingPart3/eWaSR_Head_Fine_Tuning/data_size_sweep.py generate \
  --study-config TestingPart3/DataSource/data_size_catarob_80_10_10_subtrips.yaml
# check STUDY_DIR and train STUDY_DIR are separate commands.

# Fixed-recipe head-only training from a generated run config.
python TestingPart3/eWaSR_Head_Fine_Tuning/train_head.py --config PATH_TO_RUN_CONFIG --check-data
# Remove --check-data to train using the enforced common recipe.

# Supplementary selective unfreezing, with portable locked split manifests.
python TestingPart3/eWaSR_Head_Fine_Tuning/train_head_sim4.py --check-data
python TestingPart3/eWaSR_Head_Fine_Tuning/train_lars_head_sim4.py --check-data

# Binary and three-class evaluation of saved masks.
python evaluation/binary/water_mask_evaluator.py --help
python evaluation/three_class/three_class_evaluator.py --help
```

For actual three-class generation arguments and selection CSV format, see the
[generator guide](TestingPart3/SAM3_Pseudo_Labels_Generator/README.md).
The [training guide](TestingPart3/eWaSR_Head_Fine_Tuning/README.md) maps the
retained runners and definitions. The [pipeline guide](TestingPart3/Domain_Adaptation_Pipeline/README.md)
describes the core validation-gated controller. Edit its adjacent `config.yaml`
for your data, interpreter and checkpoints, then run:

```bash
python TestingPart3/Domain_Adaptation_Pipeline/run_pipeline.py --run
```

Its committed example is a **two-epoch execution trial**, not the twenty-epoch
recipe used for the established head-only partition/data-size studies. Shared
preprocessing is retained independently of the excluded native-resolution runner.

## Repository structure

| Path | Purpose |
| --- | --- |
| `TestingPart1/eWaSR/` | eWaSR architecture, conversion, MLX baseline and upstream licence |
| `TestingPart2/SAM3/` | Binary SAM3 baseline and dataset configs |
| `TestingPart2/YOLOWorld/` | Seeded qualitative detection pilot |
| `TestingPart3/SAM3_Pseudo_Labels_Generator/` | Three-class labels and reflection padding |
| `TestingPart3/DataSource/` | Retained experimental recipes and exact split rules |
| `TestingPart3/eWaSR_Head_Fine_Tuning/` | Head training, staged/partition/data-size runners, selective unfreezing and tests |
| `TestingPart3/Domain_Adaptation_Pipeline/` | Core controller, helpers and example configuration |
| `evaluation/` | Established binary and three-class saved-mask evaluators and tests |
| `dataset_specs.py`, `ground_truths/scripts/` | Dataset pairing, encodings and inspection utilities |
| `timing_comparison.py` | Shared warm-up-excluded inference/final-mask latency and process-memory reports |
| `results/` | Selected compact summaries, staged evidence, split manifests and provenance |

## Coverage and interpretation

The established split and data-size implementations remain included. Retained
recipes cover CataRob, Pohang Canal and Tampere Channel/Dock; a separate Pohang
Pier staged definition is also included. Generated sweep manifests are recreated
from these recipes, external data and fixed seeds. Full sweep result collections
are not bundled. The newer LaRS-sequence/Pohang-Pier final sweep and Tampere Open
additions are excluded, so this release does not reproduce every reported final
model or table. Historical v1–v8 partition trials and recovery/one-off launchers
are omitted; a `v1` filename is retained where it is an essential established
experiment definition, rather than being excluded solely for its name.

The selective-unfreezing comparison retains its exact LaRS Channel and combined
Pohang Canal/Pier role memberships. It trains `decoder.sim4` at `1e-5` and the
segmentation head at `1e-3`, with matched preprocessing and twenty epochs. Portable
manifest hashes replace the original absolute-path hashes; role membership and
scientific settings are unchanged. Existing frozen-state and membership checks
remain in place.

Pseudo-label agreement is not independent human-reference accuracy. Grouping
nearby frames within a recording does not establish independent-trip testing.
Test scores were inspected in the historical split study; the same test set in
the data-size study is not an untouched new benchmark. YOLO-World has no matched
box ground truth here and supports qualitative findings only.

Boundary-metric evaluator/pipeline edits, confidence filtering experiments,
native-resolution experiment runners/configs/outputs, validation-to-30
continuations, Chapter 6 challenge holdouts and Chapter 8 assessment/comparison
utilities are outside the published scope. The core pipeline remains available,
but the excluded supplementary pipeline assessments are not reproducible from
this release. Padding CSVs preserve selected reported evidence; their edge-band
analysis runner is not included. They do not imply boundary metrics are supported
by the published evaluators.

Raw datasets, annotations, checkpoints, full labels, masks, overlays, bulk outputs,
virtual environments, caches, migration recovery material, thesis sources/figures
and personal/administrative notes are excluded. Previously tracked artifacts are
untracked, not deleted locally. Ordinary Git history is preserved, so artifacts
from older commits remain in that history.

## Focused checks

```bash
python -m unittest discover -s evaluation -v
python -m unittest discover -s TestingPart3/eWaSR_Head_Fine_Tuning/tests -v
python -m unittest discover -s TestingPart3/SAM3_Pseudo_Labels_Generator -p 'test_*.py' -v
```

Most data/metric tests use synthetic inputs. Tests requiring a pretrained
checkpoint or Metal access require those external resources; passing unit tests
does not reproduce model training or the full benchmark.

## Attribution

The eWaSR code derives from [tersekmatija/eWaSR](https://github.com/tersekmatija/eWaSR),
upstream revision `443bd4a35dafcebb76892e2047d19d743ccb8bce`, by Matija Teršek,
Lojze Žust and Matej Kristan. Its [MIT licence](TestingPart1/eWaSR/LICENSE) is
preserved. Local work includes the MLX port/conversion, dataset-aware prediction,
timing, pseudo-label adaptation and experiment tooling. See the
[eWaSR citation](TestingPart1/eWaSR/README.md). SAM3/MLX-VLM and Ultralytics
YOLO-World are external dependencies with their own licences and model terms;
the eWaSR licence does not relicense them or the datasets.
