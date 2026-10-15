"""Run the complete pipeline, inspect a run, or print the adjacent configuration."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from pipeline_helpers import (
    evaluate_masks, final_evaluate, generate_pseudo_labels, load_config, predict_ewasr,
    prepare_dataset, select_model, train_head_candidate, write_prediction_manifest,
)


def print_final_report(report):
    """Print the same final summary for a complete run and a saved-run evaluation."""
    selection, test = report["selection"], report["test"]
    print(f"Checkpoint: {selection['checkpoint_path']}; source: {selection['source']}")
    print(f"Validation accepted: {selection['accepted']}; selection status: {selection['status']}")
    print(f"Test images: {test['evaluated_count']}")
    for mode, key in (("binary", "water_iou"), ("three_class", "mean_iou")):
        score = test[mode]["global_metrics"][key]
        print(f"Global test {mode} {key}: {score:.6f}" if score is not None
              else f"Global test {mode} {key}: undefined")
    print(f"Report: {report['artifacts']['results']}")


def run_full_pipeline(config):
    """Run the existing stages in a fresh external directory using one configuration."""
    run_config = deepcopy(config)
    output_root = Path(run_config["runtime"]["output_dir"]).expanduser().resolve()
    pipeline_dir = Path(__file__).resolve().parent
    if output_root == pipeline_dir or pipeline_dir in output_root.parents:
        raise ValueError("runtime.output_dir must be outside the pipeline source directory")
    dataset_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(run_config["dataset"]["name"])).strip("_").lower()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / "runs" / f"{dataset_slug or 'dataset'}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config["runtime"]["output_dir"] = str(run_dir)
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Run directory: {run_dir}", flush=True)

    print("[1/5] Preparing dataset membership", flush=True)
    manifest = prepare_dataset(run_config)
    counts = manifest["counts"]
    print("Images: " + ", ".join(f"{role}={counts[role]['images']}"
                                 for role in ("train", "validation", "test")), flush=True)
    print("[2/5] Reusing pseudo-labels and generating missing masks", flush=True)
    labels = generate_pseudo_labels(run_config)
    print(f"Labels: {labels['total']}; reused: {labels['reused']}; generated: {labels['generated']}", flush=True)
    print("[3/5] Selecting pretrained or bank weights", flush=True)
    choice = select_model(run_config)
    if choice["requires_training"]:
        print("[4/5] Training and assessing the segmentation head", flush=True)
        training = train_head_candidate(run_config)
        print(training["message"], flush=True)
    else:
        print(f"[4/5] Training skipped; selected {choice['candidate_id']}", flush=True)
    print("[5/5] Evaluating the recorded checkpoint on the final test split", flush=True)
    return final_evaluate(run_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument(
        "--run", action="store_true",
        help="Run the complete pipeline with adjacent config.yaml in a fresh external run directory",
    )
    stages.add_argument(
        "--prepare-data", action="store_true",
        help="Select images and save a fixed grouped train/validation/test manifest",
    )
    stages.add_argument(
        "--generate-labels", action="store_true",
        help="Reuse masks and generate missing SAM3 labels for the existing split manifest",
    )
    stages.add_argument(
        "--predict-pretrained", action="store_true",
        help="Predict validation images with the configured pretrained eWaSR checkpoint",
    )
    stages.add_argument(
        "--evaluate-pretrained", action="store_true",
        help="Evaluate the explicit saved pretrained validation predictions against attached masks",
    )
    stages.add_argument(
        "--select-model", action="store_true",
        help="Assess pretrained weights and, if needed, the bank on all validation images",
    )
    stages.add_argument(
        "--train-head", action="store_true",
        help="Train and assess the best head-only checkpoint after a requires_training handoff",
    )
    stages.add_argument(
        "--final-evaluate", action="store_true",
        help="Evaluate the recorded choice on the complete reserved test split and report results",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        help="Artifact directory for an explicit single-stage command; cannot be combined with --run",
    )
    parser.add_argument("--max-images", type=int, help="Limit --predict-pretrained to the first N validation images")
    args = parser.parse_args()
    stage_requested = any(getattr(args, name) for name in (
        "prepare_data", "generate_labels", "predict_pretrained", "evaluate_pretrained",
        "select_model", "train_head", "final_evaluate",
    ))
    if args.run and args.run_dir is not None:
        parser.error("--run-dir cannot be combined with --run")
    if stage_requested and args.run_dir is None:
        parser.error("Single-stage commands require --run-dir to identify their artifact directory")
    if args.run_dir is not None and not stage_requested:
        parser.error("--run-dir requires a single-stage command")
    if args.max_images is not None and (args.max_images <= 0 or not args.predict_pretrained):
        parser.error("--max-images must be positive and requires --predict-pretrained")
    config_path = Path(__file__).with_name("config.yaml")
    config = load_config(config_path)
    if args.run:
        print_final_report(run_full_pipeline(config))
        return
    if args.run_dir is not None:
        config = deepcopy(config)
        config["runtime"]["output_dir"] = str(args.run_dir.expanduser().resolve())
    if args.final_evaluate:
        print_final_report(final_evaluate(config))
        return
    if args.train_head:
        result = train_head_candidate(config)
        print(result["message"])
        if "best_checkpoint" in result:
            print(f"Checkpoint: {result['best_checkpoint']}; best epoch: {result['best_epoch']}")
            print(f"Validation water IoU: head grid 128×96={result['trainer_head_grid_validation_water_iou']:.6f}; "
                  f"original resolution={result['pipeline_original_resolution_validation_water_iou']:.6f}")
            print(f"Accepted: {result['accepted']}; threshold: {result['acceptance_threshold']}")
            print(f"Summary: {Path(config['runtime']['output_dir']) / 'training' / 'training_summary.json'}")
        else:
            print(f"Selected checkpoint: {result['checkpoint_path']}")
        return
    if args.select_model:
        choice = select_model(config)
        print(f"Acceptance threshold: {choice['acceptance_threshold']}")
        for candidate in choice["assessed_candidates"]:
            print(f"{candidate['candidate_id']}: water IoU={candidate['validation_water_iou']:.6f}; accepted={candidate['accepted']}")
        print(f"Selected checkpoint: {choice['checkpoint_path']}" if not choice["requires_training"]
              else "Status: requires_training (handoff recorded; no training run)")
        output_dir = Path(config["runtime"]["output_dir"])
        print(f"Scores: {output_dir / 'candidate_scores.csv'}\nChoice: {output_dir / 'model_choice.json'}")
        return
    if args.evaluate_pretrained:
        split_path = Path(config["runtime"]["output_dir"]) / "splits.json"
        output_dir = split_path.parent / "pretrained" / "validation"
        prediction_path = output_dir / "prediction_manifest.json"
        for path, prerequisite in ((split_path, "--prepare-data"),
                                   (prediction_path, "--predict-pretrained")):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; run {prerequisite} first")
        validation = json.loads(split_path.read_text(encoding="utf-8"))["splits"]["validation"]
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        records = prediction["predictions"]
        selected_ids = {record["image_id"] for record in records}
        validation_ids = {record["image_id"] for record in validation}
        if (len(selected_ids) != len(records) or len(validation_ids) != len(validation)
                or not selected_ids.issubset(validation_ids)):
            raise ValueError("Prediction IDs must be unique and belong to the validation split")
        partial = len(records) < len(validation)
        if (prediction["split"] != "validation"
                or prediction["available_validation_count"] != len(validation)
                or prediction["processed_count"] != len(records)
                or prediction["partial_run"] is not partial):
            raise ValueError("Prediction manifest counts/partial status do not match validation membership")
        metadata = {key: prediction[key] for key in
                    ("checkpoint_path", "split", "input_width", "input_height",
                     "prediction_policy", "available_validation_count", "partial_run")}
        report = evaluate_masks(records, validation, output_dir, config["evaluation"]["modes"], metadata)
        print(f"Evaluated: {report['evaluated_count']}/{len(validation)}; partial run: {partial}")
        for mode, keys in (("binary", ("water_iou", "mean_iou", "water_precision", "water_recall")),
                           ("three_class", ("mean_iou",))):
            if mode in report:
                scores = report[mode]["global_metrics"]
                print(f"Global {mode}: " + ", ".join(
                    f"{key}={scores[key]:.6f}" if scores[key] is not None else f"{key}=undefined"
                    for key in keys))
        print(f"Metrics: {output_dir / 'metrics.json'}\nPer-image: {output_dir / 'per_image.csv'}")
        return
    if args.predict_pretrained:
        split_path = Path(config["runtime"]["output_dir"]) / "splits.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split manifest not found; run --prepare-data first: {split_path}")
        validation = json.loads(split_path.read_text(encoding="utf-8"))["splits"]["validation"]
        selected = validation[:args.max_images] if args.max_images is not None else validation
        output_dir = split_path.parent / "pretrained" / "validation"
        checkpoint = config["model"]["pretrained_checkpoint"]
        width, height = config["evaluation"]["input_width"], config["evaluation"]["input_height"]
        predictions = predict_ewasr(checkpoint, selected, output_dir, width, height)
        manifest = write_prediction_manifest(checkpoint, predictions, output_dir, width, height, len(validation))
        manifest_path = output_dir / "prediction_manifest.json"
        print(f"Predicted: {len(predictions)}/{len(validation)} validation images; partial run: {manifest['partial_run']}")
        print(f"Masks: {output_dir / 'masks'}\nOverlays: {output_dir / 'overlays'}")
        print(f"Manifest: {manifest_path}")
        return
    if args.generate_labels:
        summary = generate_pseudo_labels(config)
        print(f"Selected: {summary['total']}; reused: {summary['reused']}; generated: {summary['generated']}")
        print(f"Manifest: {Path(config['runtime']['output_dir']) / 'splits.json'}")
        return
    if args.prepare_data:
        manifest = prepare_dataset(config)
        counts = manifest["counts"]
        print(f"Selected: {counts['total']['images']} images, {counts['total']['groups']} groups")
        for role in ("train", "validation", "test"):
            print(f"{role}: {counts[role]['images']} images, {counts[role]['groups']} groups")
        print(f"Manifest: {Path(config['runtime']['output_dir']) / 'splits.json'}")
        return
    dataset, model, sam3 = config["dataset"], config["model"], config["sam3"]
    evaluation, splitting = config["evaluation"], config["splitting"]
    training, runtime = config["training"], config["runtime"]

    print(f"Configuration only: {config_path.resolve()}")
    if dataset.get("selection_mode", "frame_range") == "all":
        print(f"Dataset: {dataset['name']}; all images in the configured folder")
    else:
        print(
            f"Dataset: {dataset['name']}; frames {dataset['first_frame']}–"
            f"{dataset['last_frame']} inclusive, step {dataset['frame_step']}"
        )
    print(f"Images: {dataset['images_dir']}")
    print(f"Pretrained checkpoint: {model['pretrained_checkpoint']}")
    print(f"Bank checkpoints: {model['bank_checkpoints']}")
    print(
        f"SAM3: {sam3['model_id']}; confidence {sam3['confidence_threshold']}; "
        f"{sam3['padding_mode']} padding {sam3['padding_percent']}% per side; "
        f"prompts/IDs {sam3['prompts']}"
    )
    print(f"Generator: {sam3['generator_script']}")
    print(f"Existing masks: {sam3.get('existing_masks_dir')}")
    print(f"Label cache: {sam3.get('cache_dir')}")
    print(
        f"Evaluation: {', '.join(evaluation['modes'])}; "
        f"input {evaluation['input_width']}×{evaluation['input_height']}; "
        f"reference masks at {evaluation['reference_mask_resolution']} resolution; "
        f"interpolate scores before argmax: "
        f"{evaluation['interpolate_class_scores_before_argmax']}"
    )
    threshold = evaluation['acceptance_threshold']
    print(
        f"Acceptance: {evaluation['acceptance_metric']}; threshold "
        f"{threshold if threshold is not None else 'unset (to be calibrated)'}"
    )
    grouping_description = splitting["grouping"]
    if splitting["grouping"] == "sub_trip":
        grouping_description += f", group size {splitting['group_size']}"
    print(
        f"Splitting (settings only): train/validation/test "
        f"{splitting['train']}/{splitting['validation']}/{splitting['test']}; "
        f"{grouping_description}, seed {splitting['seed']}"
    )
    print(
        f"Training (settings only): {training['optimizer']}; "
        f"lr {training['learning_rate']}; weight decay {training['weight_decay']}; "
        f"{training['epochs']} epochs; batch {training['batch_size']}; "
        f"class weights {training['class_weights']}"
    )
    print(f"Trainer: {training['trainer_script']}")
    print(f"Python: {runtime['python']}")
    print(f"Output root for --run: {runtime['output_dir']}")


if __name__ == "__main__":
    main()
