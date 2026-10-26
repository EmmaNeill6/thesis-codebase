# Domain adaptation pipeline

Edit the adjacent [config.yaml](config.yaml) for your dataset, external bank
checkpoints and Python interpreter. From the repository root:

```bash
python TestingPart3/Domain_Adaptation_Pipeline/run_pipeline.py --run
```

The flow is: prepare fixed splits → reuse/generate pseudo-labels → assess pretrained weights → assess bank if needed → fine-tune if needed → evaluate the recorded checkpoint on the reserved test split.

## Images and existing pseudo-labels

Set `dataset.images_dir` to the input images and `sam3.existing_masks_dir` to their existing raw PNG mask folder or the SAM3 output parent containing `masks/`, or `null` when none are available. Each mask must match its image's full filename stem, use original image dimensions, and store class IDs 0 = obstacle, 1 = water, 2 = sky in a single-channel image. RGB overlays are previews and cannot be used as labels.

For each selected image, the pipeline checks the configured mask folder first, then the persistent generated-label cache. It runs SAM3 only for missing masks. Existing masks are referenced directly rather than copied or overwritten. The cache is separated by image folder and SAM3 settings so unchanged inputs can reuse generated labels across run folders. Source image files are expected to remain unchanged; clear the corresponding cache if images are replaced in place.

Each run saves `pseudo_labels.json` with the reused/generated counts and mask-cache location. Changing generation settings does not regenerate masks supplied through `existing_masks_dir`.

## Configuration

Edit the same configuration for each dataset. Training is currently limited to two epochs for small execution trials. The three domain bank checkpoints remain enabled, so training runs only when pretrained and every bank candidate fail the 0.95 acceptance gate.

- `dataset.selection_mode: all` selects every image in the configured folder. Use `frame_range` with `first_frame`, `last_frame`, and `frame_step` for numbered frame selection.
- `splitting.grouping: sequence` groups filenames such as LaRS `davimar_seq_08_00037.jpg` by the prefix before the final `_<frame>`. For individual numbered images such as MaSTR `0001.jpg`, use `grouping: sub_trip` and `group_size: 1`. Larger `group_size` values group consecutive selected images into blocks.
- `evaluation` enables binary and three-class metrics, 512×384 inference, and comparisons against original-resolution SAM3 masks. Continuous scores are interpolated before argmax.
- `model.bank_checkpoints` lists existing full-model checkpoints to assess; an empty list skips the bank.
- `training` configures the head-only AdamW recipe. The trainer chooses its checkpoint using 128×96 head-grid validation water IoU; the pipeline assesses acceptance using full-resolution validation water IoU.
- `runtime.output_dir` is the external output root. `sam3.cache_dir` is the persistent generated-label root. Paths are resolved relative to `config.yaml`.

The SAM3 generator currently uses its existing reflection padding and maritime prompts/class IDs; the confidence threshold, model ID and padding percentage are passed through.

## Outputs

The source folder contains only the controller, helpers, one configuration and this documentation. Each complete run receives a fresh timestamped folder outside it, so previous selection/training state cannot affect a new run.

```text
TestingPart3/
    Domain_Adaptation_Pipeline/
        run_pipeline.py
        pipeline_helpers.py
        config.yaml
        README.md
    Domain_Adaptation_Outputs/
        runs/<dataset>_<UTC timestamp>/
            run_config.json
            splits.json
            pseudo_labels.json
            candidate_scores.csv
            model_choice.json
            results.json
            pretrained/validation/...
            bank/...
            training/...
            fine_tuned/validation/...
            test/
                prediction_manifest.json
                metrics.json
                per_image.csv
                masks/...
                overlays/...
        pseudo_labels/<dataset and settings cache>/...
        legacy/...
```

`run_config.json` records resolved settings used for that run. `results.json` combines the selected checkpoint/source, validation decision, optional training summary, binary and three-class test metrics, and artifact paths. Reports contain pooled/global metrics, per-image macro metrics and confusion matrices. Undefined ratios are saved as JSON `null`.

Existing trial outputs and their old configurations were preserved under `legacy`; they are historical records, not configurations for new runs. Their internal artifact paths were updated after moving them.

## Inspecting a completed run

Open its `results.json`, or use the same controller and configuration with `--final-evaluate --run-dir` followed by the absolute run-folder path. Matching completed evaluation is read from the saved report without rerunning inference. Other individual stage flags also require `--run-dir`; normal usage is `--run`.

Model selection and acceptance use validation data. Final test scores do not change the selected model or acceptance status. Scores measure agreement with SAM3 pseudo-labels. Small trials verify execution; their small validation/test splits do not establish reliable domain performance.

The published core excludes Chapter 8 assessment utilities and uncommitted boundary-metric changes. See the [root coverage limits](../../README.md). Shared preprocessing lives in `../eWaSR_Head_Fine_Tuning/inference_preprocessing.py`; the native-resolution experiment runner is not required.
