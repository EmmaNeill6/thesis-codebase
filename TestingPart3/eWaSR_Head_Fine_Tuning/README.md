# eWaSR head-only fine-tuning with MLX

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
