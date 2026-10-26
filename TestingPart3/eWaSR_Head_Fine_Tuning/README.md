# eWaSR head adaptation

See the [root setup and coverage guide](../../README.md) before running experiments.
The MLX model uses class IDs 0 obstacle, 1 water and 2 sky. The normal input is
512×384 with a 128×96 target grid. Head-only training freezes the encoder,
refinement/fusion modules and their BatchNorm state, updating `decoder.seg_head`.

`run_experiment.py`, `run_tampere_dock.py` and `run_tampere_channel_random.py`
preserve the early learning-rate experiments and seeded dataset preparation;
the corresponding `head_only_*.yaml` definitions remain in `../DataSource/`.
`run_staged_experiment.py` is the established resumable parameter investigation.
It calls the trainer directly to vary hyperparameters, uses validation water IoU,
and does not evaluate test/reserve splits during search. `--step N` previews;
`--step N --continue` executes. Later stages can require `--choose`; use `--help`.
The retained staged definitions are `catarob_split_60_20_15_5.yaml`,
`pohang_canal_split_60_20_15_5.yaml`, `pohang_pier_split_60_20_15_5.yaml`,
`tampere_dock_staged_v1.yaml` and the earlier LaRS definition.

`split_sweep.py` generates/checks/trains/reports the established allocation and
assignment study. `data_size_sweep.py` generates nested training budgets from a
fixed parent split. Each recipe records seeds and exact membership rules; each
generated run saves a CSV manifest. Data-size recipes require the parent split
at the `base_config` location specified in the recipe. Generate that split first.
The standalone `train_head.py --config` entry point enforces AdamW, LR 0.001,
weight decay 0.001, batch two, equal class weights and twenty epochs.

`train_head_sim4.py` and `train_lars_head_sim4.py` implement the reported
supplementary unfreezing comparison using the locked configs alongside them.
Portable exact historical manifests are in `../../results/manifests/`; source
YAML and manifest hashes are verified. The Pohang recorded roles take precedence
over later source-YAML boundary edits. Original external pairs must still exist.
The output directory must be fresh.

`visualize_test_predictions.py` renders recorded test predictions.
`evaluate_staged_selected_test.py`, `export_validation_per_image_metrics.py` and
`run_fixed20_comparison.py` consume completed run state, configs, histories and
external checkpoints produced by earlier runs; compact evidence in `results/`
is not a resumable replacement for those full run directories. Plotting scripts
also require Matplotlib. Outputs go under ignored `outputs/`.

The native-resolution runner and dedicated tests/outputs, confidence-specific
runs, final split additions, validation continuations and challenge holdouts
are excluded. The shared `inference_preprocessing.py` supplies only the two
preprocessing helpers required by the core pipeline. Metrics against SAM3
measure teacher agreement, not independent annotation accuracy.
