# Repository Restructure Decision Brief

## Purpose

This document is for the agent responsible for deciding how to restructure the EEE4022S student project's Git repositories.

The project is fundamentally a **benchmarking and comparison project**. eWaSR, SAM3, and YOLO-World are experiments or methods evaluated within one study, rather than unrelated software products. The thesis will become a separate repository.

No restructuring has been performed yet. The findings below come from read-only inspection of the current workspace.

## Decision required

Choose between:

1. One repository containing all benchmark code and experiments.
2. One repository per testing part.
3. One repository per experiment.
4. Separate experiment repositories plus a separately versioned shared benchmark package.

The current recommendation is **one benchmark repository containing all three experiments, with the thesis in its own repository**. This is a recommendation, not a completed decision.

## Current workspace

The current working directory is:

```text
EEE4022S/
├── TestingPart1/
│   ├── eWaSR/
│   ├── output_eWaSR/
│   ├── .venv/
│   └── .venv-python314-backup/
├── TestingPart2/
│   ├── SAM3/
│   └── YOLOWorld/
├── data/
├── weights/
├── evaluation/
├── dataset_metadata/
├── ground_truths/
├── dataset_specs.py
├── dataset_resolution_audit.py
├── Thesis/
├── Final_Report/
└── Archives /
```

Approximate sizes at inspection time:

| Area | Size | Main reason |
|---|---:|---|
| `TestingPart1` | 24 GB | Approximately 20 GB of eWaSR outputs plus environments |
| `TestingPart2` | 4.6 GB | Almost entirely SAM3 outputs |
| `data` | 7.8 GB | Local datasets |
| Root SAM3 environment | 947 MB | Installed dependencies |
| Root YOLO-World environment | 1.2 GB | Installed dependencies |
| `weights` | 338 MB | Model checkpoints |
| `ground_truths` | 270 MB | Mostly generated visual outputs |
| `Thesis` | 277 MB | Thesis material, intended for a separate repository |
| `evaluation` | 3.5 MB | Evaluation code and compact results |
| `dataset_metadata` | 1.4 MB | Generated dataset audit metadata |

## Experiment roles

### eWaSR

eWaSR produces binary water-segmentation masks and overlays. It is the main traditional segmentation model in Testing Part 1.

Its custom runner is:

```text
TestingPart1/eWaSR/predict_binary_water.py
```

It reads the shared dataset specification from the workspace root and writes to:

```text
TestingPart1/output_eWaSR/
```

### SAM3

SAM3 performs text-prompted segmentation through MLX on Apple Silicon. It produces evaluator-compatible binary masks, per-prompt masks, overlays, manifests, timing information, and per-image completion records.

Its runner is:

```text
TestingPart2/SAM3/predict_sam3.py
```

It uses YAML experiment configurations and writes beneath:

```text
TestingPart2/SAM3/output_mlx/
```

### YOLO-World

YOLO-World performs open-vocabulary object detection. Its results are qualitative because the datasets do not provide matching bounding-box ground truth. Its outputs must not be passed to the binary-mask evaluator.

Its main experiment runner is:

```text
TestingPart2/YOLOWorld/test_random_datasets.py
```

It writes beneath:

```text
TestingPart2/YOLOWorld/output_mps/
```

## Shared benchmarking contract

The experiments share more than a common parent directory. They depend on a common scientific contract implemented primarily by `dataset_specs.py`:

- Dataset discovery.
- Image inclusion and exclusion rules.
- Ground-truth mask pairing.
- Ground-truth encodings.
- Stable per-image evaluation keys.
- Prediction output naming.

The evaluator and dataset-resolution audit use the same contract. This is important: allowing experiment-specific copies of this logic to drift could invalidate comparisons even if every individual runner still executes successfully.

The verified dataset inventory is:

| Dataset key | Dataset | Selected images | Paired ground truths |
|---|---|---:|---:|
| `tampere` | Tampere-WaterSeg | 600 | 600 |
| `waternet` | WaterNet | 2,390 | 2,390 |
| `mastr1325` | MaSTR1325 | 1,325 | 1,325 |
| `pohang` | Pohang | 8 | 0 |

The ordered evaluation keys and these counts should be treated as migration invariants.

## Existing Git state

The workspace root, `TestingPart1`, and `TestingPart2` are not currently Git repositories.

`TestingPart1/eWaSR` is an existing nested clone with remote:

```text
https://github.com/tersekmatija/eWaSR.git
```

The inspected upstream commit is:

```text
443bd4a35dafcebb76892e2047d19d743ccb8bce
```

Its working tree contains important local changes:

- Modified `README.md`.
- Modified `datasets/mods.py`.
- Modified `wasr/models.py`.
- Deleted upstream `predict.py`.
- Added `predict_binary_water.py`.
- Added `configs/my_five_images.yaml`.
- Untracked `.DS_Store` files.

The local eWaSR work must be preserved before changing or removing its nested `.git` directory.

## Major restructure risks

### 1. Fixed directory-depth assumptions

The eWaSR, SAM3, and YOLO-World runners derive the project root using expressions such as:

```python
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
```

The ground-truth scripts similarly depend on `parents[2]`.

Renaming or adding nesting levels can make these expressions point to the wrong directory. The likely failures are missing `dataset_specs.py`, failed imports, and missing datasets.

### 2. `dataset_specs.py` is anchored to its own location

The shared module currently defines:

```python
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
```

Moving it directly into `src/maritime_benchmark/` would make it search for `src/maritime_benchmark/data`, not the repository-level `data` directory.

Long-term, dataset discovery should accept an explicit `data_root` rather than infer data location from the module location.

### 3. YAML paths are relative to the YAML file

SAM3 and YOLO-World resolve paths such as these relative to each configuration file:

```yaml
output_root: ../output_mlx
checkpoint: ../weights/yolov8s-worldv2.pt
```

Moving a YAML file without simultaneously moving or updating its target changes runtime behaviour.

### 4. SAM3 resume identity is too sensitive

SAM3 hashes the complete YAML file. Every successful per-image record is associated with this SHA-256 value.

At inspection time:

- MaSTR1325 resumes as 1,325/1,325 complete.
- WaterNet resumes as 2,390/2,390 complete.
- Pohang resumes as 8/8 complete.
- Tampere reports 0/600 complete even though its stored manifest says 600/600 complete.

The Tampere mismatch is caused only by a changed YAML comment. The scientific settings are unchanged. This demonstrates that changing a path or comment during restructuring can invalidate resume recognition.

The preferred long-term fix is to hash a canonical representation of scientific settings, excluding comments and storage locations. Existing hashes need a legacy compatibility or explicit migration strategy.

### 5. Existing records contain absolute paths

At least 4,362 existing JSON, JSONL, and YAML result records contain the current absolute workspace path. These include:

- Input image paths.
- Configuration paths.
- Model checkpoint paths.
- Prediction directories.
- The dataset-specification path.

Many are provenance rather than operational dependencies, so moving outputs may not prevent them from being read. They will nevertheless become stale and misleading after relocation.

Historical manifests should normally remain immutable. A migration record can document the old and new roots. Future manifests should prefer repository-relative paths where appropriate.

### 6. Large and generated content

The repository must not accidentally include:

- Approximately 20 GB of eWaSR outputs.
- Approximately 4.6 GB of SAM3 outputs.
- Local virtual environments.
- Local datasets.
- Caches.
- Downloaded model weights.

The eWaSR checkpoint is approximately 230 MB, exceeding GitHub's normal per-file limit. The YOLO-World checkpoint is approximately 25 MB. Weights should be ignored and downloaded through documented scripts or managed deliberately with Git LFS.

Ignore rules must be installed and verified before the first broad `git add`.

### 7. Virtual environments are not portable

There are separate environments for eWaSR, SAM3, YOLO-World, and other prior work. They should be recreated, not moved or committed. Activation scripts and installed command entry points may embed their original paths.

Separate environments remain appropriate because the dependency requirements differ significantly:

- eWaSR uses older model-stack dependencies, including `timm==0.6.5`.
- SAM3 uses MLX-VLM and MLX.
- YOLO-World pins a newer PyTorch and Ultralytics stack.
- Evaluation requires NumPy, Pillow, and SciPy.

Repository consolidation does not imply environment consolidation.

### 8. Evaluation paths name the old layout

The evaluator explicitly uses `TestingPart1/output_eWaSR` as a default output-root argument when loading datasets. The resolution-audit script contains the same topology assumption. Both will require adjustment if outputs are centralized.

### 9. Existing unrelated defects

Some existing paths are already stale or missing and should not be mistaken for migration regressions:

- `evaluation_images.txt` is referenced but was not present at the workspace root.
- Old eWaSR training configurations refer to dataset directories that are not present inside the eWaSR clone.
- `tools/mods_inference.py` contains `/home/paperspace/eWaSR`.
- The Tampere SAM3 YAML hash already differs from its completed run because of a comment-only edit.

## Repository strategy options

### Option A: One benchmark repository

One repository contains eWaSR, SAM3, YOLO-World, the shared dataset contract, evaluation code, configurations, and small result summaries.

Advantages:

- One commit can update an experiment and its evaluator together.
- The shared dataset contract has one authoritative implementation.
- Comparisons are easier to reproduce.
- Cross-experiment tests can run in one repository.
- Documentation and versioning are centralized.
- Separate virtual environments can still be used.

Disadvantages:

- The initial eWaSR nested-repository migration needs care.
- The repository contains several dependency ecosystems.
- Ignore rules and artifact boundaries must be well maintained.

Assessment: best fit for the current benchmarking purpose.

### Option B: One repository per experiment

Separate repositories would exist for eWaSR, SAM3, and YOLO-World.

Advantages:

- Clear individual ownership and release boundaries.
- Smaller source repositories.
- eWaSR can naturally remain a fork of its upstream project.
- Dependencies are visibly isolated.

Disadvantages:

- `dataset_specs.py` and evaluation logic must be duplicated or moved to a fourth repository.
- Copied dataset rules can silently drift and invalidate comparisons.
- Coordinated benchmark changes require multiple commits or pull requests.
- Reproducing the study requires recording a compatible set of repository commits.
- Cross-experiment validation and documentation become harder.

Assessment: appropriate only if the experiments will have independent maintainers or lifecycles beyond the study.

### Option C: Separate experiments plus a shared benchmark-core repository

This produces four repositories: three experiment repositories and one versioned package for datasets and metrics.

Advantages:

- Strong separation with a single authoritative benchmark contract.
- Each experiment can pin an exact core-package revision.
- Suitable if the projects will be independently maintained or published.

Disadvantages:

- Highest initial setup and release-management burden.
- Compatibility must be managed across four repositories.
- Probably excessive for a single student benchmarking project.

Assessment: architecturally defensible, but operationally expensive for the present scale.

### Option D: One repository per testing part

This would put eWaSR in one repository and SAM3 plus YOLO-World in another.

This division reflects the historical folder names but not a clean technical boundary. Both parts still share datasets and evaluation logic, while SAM3 and eWaSR are more directly comparable as segmentation methods than SAM3 and YOLO-World are as task types.

Assessment: weakest conceptual boundary of the available options.

## Candidate long-term structure

If Option A is selected, a reasonable destination is:

```text
EEE4022S/                                  # Ordinary parent directory
├── maritime-perception-benchmarks/       # Git repository
│   ├── pyproject.toml
│   ├── README.md
│   ├── src/
│   │   └── maritime_benchmark/
│   │       ├── datasets.py
│   │       ├── paths.py
│   │       └── metrics.py
│   ├── experiments/
│   │   ├── ewasr/
│   │   │   ├── runner/
│   │   │   └── vendor/ewasr/
│   │   ├── sam3/
│   │   └── yolo_world/
│   ├── configs/
│   ├── evaluation/
│   ├── tools/
│   ├── tests/
│   ├── data/                             # Contents ignored
│   ├── weights/                          # Contents ignored
│   └── results/
│       ├── raw/                          # Ignored
│       └── summaries/                    # Selected compact artifacts tracked
├── thesis/                               # Separate Git repository
└── archives/                             # Outside active repositories
```

Important design detail: vendored eWaSR code and project-specific eWaSR runner code should be distinguishable. This makes upstream provenance and local modifications easier to understand.

## Suggested artifact policy

### Track in Git

- Source code.
- YAML experiment definitions.
- Dependency specifications and lock information.
- Tests.
- Documentation.
- Dataset download/preparation instructions.
- Checksums and source URLs for model weights.
- Compact final CSV/JSON result summaries selected for the report.
- eWaSR upstream license and attribution.

### Ignore

- Raw datasets.
- Downloaded model checkpoints.
- Generated masks and overlays.
- Complete raw inference outputs.
- Virtual environments.
- Python caches.
- `.DS_Store` files.
- Temporary staging files.

### Consider separately

- Important raw results may be deposited in institutional storage, Zenodo, OSF, or another artifact store.
- Git LFS may be appropriate only if the owner deliberately wants large binary assets versioned and accepts its storage and bandwidth implications.

## Recommended migration sequence if Option A is chosen

### Phase 0: Capture a baseline

- Record the current file inventory.
- Save the eWaSR upstream commit, diff, and untracked source files.
- Record hashes of scientific configurations.
- Record dataset counts and ordered evaluation keys.
- Record representative prediction hashes.
- Record existing evaluation summaries.
- Record SAM3 completion counts.

### Phase 1: Establish the new repository with minimal path changes

Initially place the existing relevant structure under `maritime-perception-benchmarks/` without immediately renaming `TestingPart1` and `TestingPart2`.

This preserves most relative relationships and separates relocation failures from semantic refactoring failures.

Do not move virtual environments. Recreate them after relocation.

### Phase 2: Preserve and vendor eWaSR safely

- Keep the original clone intact until verification is complete.
- Copy the working source without `.git` into the new repository.
- Retain its license and upstream commit reference.
- Verify the copied source against the recorded local diff.
- Ensure the parent repository tracks ordinary files, not an accidental Git link.

### Phase 3: Create a shared installable package

- Move dataset selection and metrics into `src/maritime_benchmark/`.
- Replace `sys.path` modification with package imports.
- Accept data, weight, and result roots explicitly.
- Keep experiment-scientific settings separate from machine-local storage settings.
- Add a schema version to experiment manifests.

### Phase 4: Move to semantic directory names

Only after paths are configurable:

```text
TestingPart1/eWaSR       -> experiments/ewasr
TestingPart2/SAM3        -> experiments/sam3
TestingPart2/YOLOWorld   -> experiments/yolo_world
ground_truths/scripts    -> tools/ground_truths
```

Centralize generated outputs under `results/raw/` if desired.

### Phase 5: Verify and remove the old layout

The old copies should be removed only after the acceptance criteria below pass.

## Acceptance criteria

A successful restructure should satisfy all of the following:

1. Dataset counts remain exactly 600, 2,390, 1,325, and 8.
2. Ordered evaluation keys are identical before and after migration.
3. Ground-truth encodings are unchanged.
4. SAM3 dry runs recognize the intended existing completion records.
5. A representative eWaSR image produces an identical binary-mask hash before and after migration.
6. Existing evaluator inputs reproduce their prior metric summaries within exact or justified numerical tolerance.
7. YOLO-World selects the same images for seed `4022`.
8. The source contains no unintended references to `TestingPart1`, `TestingPart2`, or the old absolute root.
9. A fresh clone can recreate each environment from documented dependency files.
10. `git status --ignored` confirms data, weights, raw outputs, environments, and caches are excluded.
11. No file above the hosting provider's size limit is present in ordinary Git history.
12. eWaSR attribution, license, upstream commit, and local modifications are documented.

## Questions for the deciding agent

1. Will any experiment be maintained or published independently after the thesis?
2. Will different people own different experiments?
3. Must existing SAM3 runs remain resumable, or may completed runs be archived as immutable artifacts?
4. Should raw results live only locally, or be published to a durable external artifact store?
5. Should compact evaluation summaries be tracked in Git?
6. Is vendoring eWaSR acceptable, or is preserving it as an independently maintained fork important?
7. Should dataset and result roots be controlled by CLI arguments, a local untracked configuration file, environment variables, or a combination?
8. Which current utilities are active, and which old eWaSR training/MODS scripts may be archived instead of migrated?
9. Is the missing `evaluation_images.txt` still required?
10. Should the final repository support only Apple Silicon, or also CUDA/Linux reproduction?

## Provisional recommendation

Use two repositories at the top level:

1. `maritime-perception-benchmarks` for shared dataset rules, eWaSR, SAM3, YOLO-World, evaluation, and reproducibility tooling.
2. `thesis` for the written academic artifact.

Keep separate virtual environments per experiment. Keep `data/`, `weights/`, and `results/raw/` physically inside the benchmark repository for convenience but ignore their contents. Track instructions, checksums, configurations, tests, and selected compact result summaries.

Perform relocation and path refactoring as separate phases. Do not combine moving folders, changing configuration paths, migrating eWaSR Git history, and centralizing results into one unverified operation.
