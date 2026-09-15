"""Restartable multi-image SAM 3 inference on Apple Silicon with MLX.

The dataset inventory and evaluator-compatible output names come from the
project-root ``dataset_specs.py``. Model and prompt choices live in a
small YAML experiment file so a resumed run cannot silently mix settings.

For every successfully processed image this runner saves:

* an unchanged copy of the input image;
* one union mask for each text prompt;
* a final 0/255 binary water mask;
* an eWaSR-style two-colour overlay; and
* a per-image JSON record containing prompts, scores, boxes and timings.

The per-image JSON record is the completion marker. It is written atomically
only after all image outputs have been written. Re-running the same command
skips valid completion records and selects the next unfinished images.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean, median
from time import perf_counter_ns
from typing import Any, Iterable

import numpy as np
from PIL import Image
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DATASET_SPECS_DIR = PROJECT_DIR
DATASET_SPECS_PATH = DATASET_SPECS_DIR / "dataset_specs.py"
DEFAULT_CONFIG = SCRIPT_DIR / "configs/tampere_waterseg.yaml"

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_PROMPT_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
VALID_PROMPT_ROLES = {"include", "exclude", "record_only"}

WATER_COLOUR = np.array([41, 167, 224], dtype=np.float32)
NON_WATER_COLOUR = np.array([247, 195, 37], dtype=np.float32)


if not DATASET_SPECS_PATH.is_file():
    raise FileNotFoundError(f"Shared dataset specification not found: {DATASET_SPECS_PATH}")
if str(DATASET_SPECS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_SPECS_DIR))

from dataset_specs import load_dataset  # noqa: E402
from timing_comparison import (  # noqa: E402
    TIMING_DEFINITIONS,
    TIMING_PROTOCOL_VERSION,
    append_memory_usage,
    write_timing_comparison,
)


@dataclass(frozen=True)
class PromptSpec:
    """One text prompt and its role in the final binary-water mask."""

    key: str
    text: str
    role: str


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated settings for one restartable MLX experiment."""

    dataset_key: str
    output_root: Path
    run_name: str
    backend: str
    model_id: str
    confidence_threshold: float
    prompts: tuple[PromptSpec, ...]


@dataclass(frozen=True)
class PromptPrediction:
    """The accepted SAM 3 instances for one prompt."""

    masks: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray


def now_iso() -> str:
    """Return a local, timezone-aware timestamp."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a reproducibility digest for a configuration file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    """Return an installed package version without failing provenance."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_run_name(value: str) -> str:
    """Restrict run names to one safe directory component."""

    if not SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            "run_name must start with a letter or number and contain only "
            "letters, numbers, periods, underscores or hyphens"
        )
    return value


def resolve_config_path(config_dir: Path, value: Any, field_name: str) -> Path:
    """Resolve one configured path relative to its YAML file."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    return (config_dir / value).resolve()


def parse_prompts(value: Any) -> tuple[PromptSpec, ...]:
    """Validate prompt names, texts and final-mask roles."""

    if not isinstance(value, list) or not value:
        raise ValueError("prompts must be a non-empty YAML list")

    prompts: list[PromptSpec] = []
    seen_keys: set[str] = set()
    seen_texts: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each prompt must be a YAML mapping")
        key = item.get("key")
        text = item.get("text")
        role = item.get("role", "record_only")
        if not isinstance(key, str) or not SAFE_PROMPT_KEY.fullmatch(key):
            raise ValueError(
                f"Prompt key {key!r} must contain only letters, numbers, _ or -"
            )
        if key in seen_keys:
            raise ValueError(f"Duplicate prompt key: {key}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Prompt {key!r} must have non-empty text")
        text = text.strip()
        if text in seen_texts:
            raise ValueError(
                f"Duplicate prompt text {text!r} is ambiguous in MLX multi-prompt output"
            )
        if role not in VALID_PROMPT_ROLES:
            raise ValueError(
                f"Prompt {key!r} has role {role!r}; expected one of "
                f"{sorted(VALID_PROMPT_ROLES)}"
            )
        prompts.append(PromptSpec(key=key, text=text, role=role))
        seen_keys.add(key)
        seen_texts.add(text)

    if not any(prompt.role == "include" for prompt in prompts):
        raise ValueError("At least one prompt must have role: include")
    return tuple(prompts)


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """Load the small, model-specific YAML experiment definition."""

    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        value = yaml.safe_load(config_file)
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")

    dataset_key = value.get("dataset")
    if not isinstance(dataset_key, str) or not dataset_key.strip():
        raise ValueError("dataset must be a non-empty string")

    backend = value.get("backend", "mlx")
    if backend != "mlx":
        raise ValueError("This runner currently supports only backend: mlx")

    run_name = value.get("run_name")
    if not isinstance(run_name, str):
        raise ValueError("run_name must be a string")

    model = value.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("model must be a YAML mapping")
    model_id = model.get("model_id", "facebook/sam3")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model.model_id must be a non-empty string")
    confidence_threshold = float(model.get("confidence_threshold", 0.5))
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("model.confidence_threshold must be between 0 and 1")

    return ExperimentConfig(
        dataset_key=dataset_key.strip().casefold(),
        output_root=resolve_config_path(
            config_path.parent, value.get("output_root"), "output_root"
        ),
        run_name=validate_run_name(run_name),
        backend=backend,
        model_id=model_id.strip(),
        confidence_threshold=confidence_threshold,
        prompts=parse_prompts(value.get("prompts")),
    )


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a text file atomically so interruption cannot truncate it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write indented, standards-compliant JSON atomically."""

    atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_json_line(path: Path, value: dict[str, Any]) -> None:
    """Append and flush one durable attempt record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def safe_output_path(path: Path) -> Path:
    """Validate dataset-provided output paths before writing below the run."""

    if path.is_absolute() or ".." in path.parts or path.suffix.casefold() != ".png":
        raise ValueError(f"Unsafe or non-PNG dataset output path: {path}")
    return path


def record_path(run_dir: Path, output_path: Path) -> Path:
    """Return the authoritative per-image completion-record path."""

    return run_dir / "image_records" / output_path.with_suffix(".json")


def output_locations(request: Any) -> dict[str, Any]:
    """Return all stable paths, relative to the run directory, for one image."""

    output_path = safe_output_path(request.output_path)
    original_path = (
        Path("original_inputs") / output_path.parent / request.image_path.name
    )
    return {
        "original_input": original_path.as_posix(),
        "prompt_masks": {},
        "final_binary_mask": (Path("predicted_masks") / output_path).as_posix(),
        "overlay": (Path("prediction_overlays") / output_path).as_posix(),
    }


def required_output_paths(outputs: dict[str, Any]) -> list[str]:
    """Flatten the output mapping for restart validation."""

    paths = [
        outputs["original_input"],
        outputs["final_binary_mask"],
        outputs["overlay"],
    ]
    paths.extend(outputs["prompt_masks"].values())
    return paths


def read_json(path: Path) -> dict[str, Any] | None:
    """Read one JSON object, treating malformed files as incomplete."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def completed_record(
    request: Any, run_dir: Path, experiment_sha256: str
) -> dict[str, Any] | None:
    """Return a successful record only when every declared output still exists."""

    value = read_json(record_path(run_dir, safe_output_path(request.output_path)))
    if value is None or value.get("status") != "success":
        return None
    if value.get("experiment_sha256") != experiment_sha256:
        return None
    outputs = value.get("outputs")
    if not isinstance(outputs, dict):
        return None
    try:
        paths = required_output_paths(outputs)
    except (KeyError, AttributeError):
        return None
    if not all((run_dir / relative_path).is_file() for relative_path in paths):
        return None
    return value


def create_predictor(config: ExperimentConfig):
    """Load the MLX SAM 3 model once for the complete invocation."""

    from mlx_vlm.models.sam3.generate import Sam3Predictor
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
    from mlx_vlm.utils import get_model_path, load_model

    model_path = Path(get_model_path(config.model_id)).resolve()
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    predictor = Sam3Predictor(
        model,
        processor,
        score_threshold=config.confidence_threshold,
    )
    return predictor, model_path


def predict_prompts(
    predictor: Any,
    image: Image.Image,
    prompts: tuple[PromptSpec, ...],
    confidence_threshold: float,
) -> tuple[dict[str, PromptPrediction], float]:
    """Time model execution through NumPy materialization of its outputs."""

    from mlx_vlm.models.sam3.generate import predict_multi

    height, width = image.height, image.width

    # Match eWaSR's timing boundary: resize and normalize the image before the
    # timer, but keep NumPy-to-MLX conversion, synchronized model execution and
    # model-result postprocessing inside it. predict_multi normally calls this
    # preprocessing method itself, so temporarily serve the prepared result.
    prepared_image = predictor.processor.preprocess_image(image)
    original_preprocess_image = predictor.processor.preprocess_image
    predictor.processor.preprocess_image = lambda _: prepared_image
    try:
        started_ns = perf_counter_ns()
        result = predict_multi(
            predictor,
            image,
            prompts=[prompt.text for prompt in prompts],
            score_threshold=confidence_threshold,
        )
        labels = list(result.labels or [])
        masks = np.asarray(result.masks)
        scores = np.asarray(result.scores, dtype=float).reshape(-1)
        boxes = np.asarray(result.boxes, dtype=float).reshape(-1, 4)
        inference_ms = (perf_counter_ns() - started_ns) / 1_000_000
    finally:
        predictor.processor.preprocess_image = original_preprocess_image

    if labels and (
        masks.ndim != 3
        or masks.shape[1:] != (height, width)
        or not (len(labels) == len(masks) == len(scores) == len(boxes))
    ):
        raise ValueError(
            "MLX SAM 3 returned inconsistent prompt labels, masks, scores or boxes"
        )

    predictions: dict[str, PromptPrediction] = {}
    for prompt in prompts:
        indices = [index for index, label in enumerate(labels) if label == prompt.text]
        if indices:
            prompt_masks = masks[indices].astype(bool, copy=False)
            prompt_scores = scores[indices]
            prompt_boxes = boxes[indices]
        else:
            prompt_masks = np.zeros((0, height, width), dtype=bool)
            prompt_scores = np.zeros((0,), dtype=float)
            prompt_boxes = np.zeros((0, 4), dtype=float)
        predictions[prompt.key] = PromptPrediction(
            masks=prompt_masks,
            scores=prompt_scores,
            boxes=prompt_boxes,
        )
    return predictions, inference_ms


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    """Save a boolean or 0/255 uint8 mask using the evaluator convention."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = (
        mask if mask.dtype == np.uint8 else np.where(mask, 255, 0).astype(np.uint8)
    )
    Image.fromarray(pixels, mode="L").save(path)


def save_ewasr_style_overlay(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    """Save the same blue-water/yellow-non-water overlay used by eWaSR."""

    colours = np.where(mask[..., None], WATER_COLOUR, NON_WATER_COLOUR)
    overlay = (image.astype(np.float32) * 0.7 + colours * 0.3).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(path)


def prompt_summary(
    prompt: PromptSpec, prediction: PromptPrediction
) -> dict[str, Any]:
    """Create JSON-safe scores, boxes and areas for one prompt."""

    instances = []
    for index, (mask, score, box) in enumerate(
        zip(prediction.masks, prediction.scores, prediction.boxes), start=1
    ):
        instances.append(
            {
                "instance_id": index,
                "score": float(score),
                "box_xyxy_pixels": [float(number) for number in box],
                "area_pixels": int(mask.sum()),
            }
        )
    return {
        "key": prompt.key,
        "text": prompt.text,
        "role": prompt.role,
        "instance_count": len(instances),
        "instances": instances,
    }


def commit_staged_files(staging_dir: Path, run_dir: Path) -> None:
    """Publish fully written image files; the JSON completion record follows."""

    for source_path in sorted(path for path in staging_dir.rglob("*") if path.is_file()):
        relative_path = source_path.relative_to(staging_dir)
        destination = run_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source_path, destination)


def process_one_image(
    request: Any,
    run_dir: Path,
    config: ExperimentConfig,
    predictor: Any,
    experiment_sha256: str,
) -> dict[str, Any]:
    """Process and atomically mark one image as successful."""

    processing_started_ns = perf_counter_ns()
    output_path = safe_output_path(request.output_path)
    outputs = output_locations(request)
    outputs["prompt_masks"] = {
        prompt.key: (
            Path("prompt_masks") / prompt.key / output_path
        ).as_posix()
        for prompt in config.prompts
    }

    staging_root = run_dir / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="image-", dir=staging_root))
    inference_ms: float | None = None
    final_mask_latency_ms: float | None = None
    try:
        original_target = staging_dir / outputs["original_input"]
        original_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(request.image_path, original_target)
        latency_started_ns = perf_counter_ns()
        with Image.open(request.image_path) as source_image:
            image = source_image.convert("RGB")
        image_array = np.asarray(image, dtype=np.uint8)
        height, width = image_array.shape[:2]

        predictions, inference_ms = predict_prompts(
            predictor=predictor,
            image=image,
            prompts=config.prompts,
            confidence_threshold=config.confidence_threshold,
        )

        include_union = np.zeros((height, width), dtype=bool)
        exclude_union = np.zeros((height, width), dtype=bool)
        union_masks = {}
        for prompt in config.prompts:
            prediction = predictions[prompt.key]
            union_mask = (
                np.any(prediction.masks, axis=0)
                if len(prediction.masks)
                else np.zeros((height, width), dtype=bool)
            )
            union_masks[prompt.key] = union_mask
            if prompt.role == "include":
                include_union |= union_mask
            elif prompt.role == "exclude":
                exclude_union |= union_mask

        final_water = include_union & ~exclude_union
        final_water_mask = np.where(final_water, 255, 0).astype(np.uint8)
        final_mask_latency_ms = (perf_counter_ns() - latency_started_ns) / 1_000_000

        summaries = []
        for prompt in config.prompts:
            union_mask = union_masks[prompt.key]
            save_binary_mask(
                union_mask,
                staging_dir / outputs["prompt_masks"][prompt.key],
            )
            prediction = predictions[prompt.key]
            summary = prompt_summary(prompt, prediction)
            summary["union_area_pixels"] = int(union_mask.sum())
            summaries.append(summary)

        save_binary_mask(final_water_mask, staging_dir / outputs["final_binary_mask"])
        save_ewasr_style_overlay(
            image_array,
            final_water,
            staging_dir / outputs["overlay"],
        )

        commit_staged_files(staging_dir, run_dir)
        processing_ms = (perf_counter_ns() - processing_started_ns) / 1_000_000
        record = {
            "status": "success",
            "completed_at": now_iso(),
            "experiment_sha256": experiment_sha256,
            "dataset": config.dataset_key,
            "image_filename": request.image_path.name,
            "input_path": str(request.image_path.resolve()),
            "output_path": output_path.as_posix(),
            "width": width,
            "height": height,
            "prompts": summaries,
            "combination_rule": "union(include) AND NOT union(exclude)",
            "candidate_water_area_pixels": int(final_water.sum()),
            "timing_protocol_version": TIMING_PROTOCOL_VERSION,
            "timing_ms": {
                "inference": inference_ms,
                "final_mask_latency": final_mask_latency_ms,
                "end_to_end_processing": processing_ms,
            },
            "outputs": outputs,
        }
        atomic_write_json(record_path(run_dir, output_path), record)
        return record
    except Exception as error:
        processing_ms = (perf_counter_ns() - processing_started_ns) / 1_000_000
        record = {
            "status": "failed",
            "failed_at": now_iso(),
            "experiment_sha256": experiment_sha256,
            "dataset": config.dataset_key,
            "image_filename": request.image_path.name,
            "input_path": str(request.image_path.resolve()),
            "output_path": output_path.as_posix(),
            "prompts": [asdict(prompt) for prompt in config.prompts],
            "timing_protocol_version": TIMING_PROTOCOL_VERSION,
            "timing_ms": {
                "inference": inference_ms,
                "final_mask_latency": final_mask_latency_ms,
                "end_to_end_processing": processing_ms,
            },
            "outputs": outputs,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        atomic_write_json(record_path(run_dir, output_path), record)
        return record
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def successful_records(
    requests: Iterable[Any], run_dir: Path, experiment_sha256: str
) -> list[dict[str, Any]]:
    """Return successful records in stable dataset order."""

    records = []
    for request in requests:
        record = completed_record(request, run_dir, experiment_sha256)
        if record is not None:
            records.append(record)
    return records


def rewrite_summary_files(run_dir: Path, records: list[dict[str, Any]]) -> None:
    """Regenerate the prediction index and eWaSR-compatible timing table."""

    prediction_text = "".join(
        json.dumps(record, allow_nan=False) + "\n" for record in records
    )
    atomic_write_text(run_dir / "predictions.jsonl", prediction_text)

    times = [float(record["timing_ms"]["inference"]) for record in records]
    timing_lines = ["image\tinference_ms"]
    timing_lines.extend(
        f"{record['input_path']}\t{float(record['timing_ms']['inference']):.3f}"
        for record in records
    )
    if times:
        mean_ms = fmean(times)
        timing_lines.extend(
            [
                f"# mean_ms\t{mean_ms:.3f}",
                f"# median_ms\t{median(times):.3f}",
                f"# fps_from_mean\t{1000 / mean_ms:.3f}",
            ]
        )
    atomic_write_text(run_dir / "inference_times.txt", "\n".join(timing_lines) + "\n")
    write_timing_comparison(
        run_dir,
        [
            {
                "evaluation_key": record["output_path"],
                "inference_ms": record["timing_ms"]["inference"],
                "final_mask_latency_ms": record["timing_ms"]["final_mask_latency"],
            }
            for record in records
            if record.get("timing_protocol_version") == TIMING_PROTOCOL_VERSION
        ],
    )


def prepare_run(
    run_dir: Path,
    config_path: Path,
    config: ExperimentConfig,
    dataset_name: str,
    image_count: int,
    experiment_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Create a run or validate that an existing run is safe to resume."""

    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists():
        manifest = read_json(manifest_path)
        if manifest is None:
            raise RuntimeError(
                f"Run directory exists without a valid manifest: {run_dir}"
            )
        previous_hash = manifest.get("configuration", {}).get("sha256")
        if previous_hash != experiment_sha256:
            raise RuntimeError(
                "The experiment YAML changed since this run was created. "
                "Use a new run_name rather than mixing settings."
            )
    else:
        run_dir.mkdir(parents=True)
        shutil.copy2(config_path, run_dir / "experiment_config.yaml")
        manifest = {
            "status": "created",
            "created_at": now_iso(),
            "dataset": {
                "key": config.dataset_key,
                "name": dataset_name,
                "number_of_images": image_count,
                "specification": str(DATASET_SPECS_PATH),
            },
            "configuration": {
                "source_path": str(config_path),
                "sha256": experiment_sha256,
                "run_name": config.run_name,
                "backend": config.backend,
                "model_id": config.model_id,
                "confidence_threshold": config.confidence_threshold,
                "prompts": [asdict(prompt) for prompt in config.prompts],
                "combination_rule": "union(include) AND NOT union(exclude)",
            },
            "outputs": {
                "original_inputs": "original_inputs/<dataset-relative path>",
                "prompt_masks": "prompt_masks/<prompt>/<dataset-relative mask>",
                "predicted_masks": "predicted_masks/<dataset-relative mask>",
                "prediction_overlays": "prediction_overlays/<dataset-relative mask>",
                "image_records": "image_records/<dataset-relative record>.json",
                "attempt_log": "run_attempts.jsonl",
                "inference_times": "inference_times.txt",
                "comparable_timings": "comparable_timings.csv",
                "comparable_timing_summary": "comparable_timing_summary.json",
                "memory_usage": "memory_usage.jsonl",
            },
        }
        atomic_write_json(manifest_path, manifest)
    return manifest_path, manifest


def parse_args() -> argparse.Namespace:
    """Parse operational controls; scientific settings remain in YAML."""

    parser = argparse.ArgumentParser(
        description="Run restartable, text-prompted SAM 3 inference with MLX."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--max-new-images",
        type=int,
        help="Process at most this many unfinished images during this invocation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show pending images without loading SAM 3",
    )
    return parser.parse_args()


def main() -> None:
    """Run or resume one configured SAM 3 experiment."""

    args = parse_args()
    if args.max_new_images is not None and args.max_new_images <= 0:
        raise ValueError("--max-new-images must be positive")

    config_path = args.config.resolve()
    config = load_experiment_config(config_path)
    experiment_sha256 = sha256_file(config_path)
    dataset = load_dataset(config.dataset_key, config.output_root)
    requests = list(dataset.images)
    run_dir = dataset.output_dir / config.run_name

    completed = {
        request.output_path
        for request in requests
        if completed_record(request, run_dir, experiment_sha256) is not None
    }
    pending = [request for request in requests if request.output_path not in completed]
    selected = (
        pending[: args.max_new_images]
        if args.max_new_images is not None
        else pending
    )

    print(f"Dataset: {dataset.name} ({len(requests)} images)")
    print(f"Run: {config.run_name}")
    print(f"Completed: {len(completed)}")
    print(f"Pending: {len(pending)}")
    print(f"Selected now: {len(selected)}")
    print(
        "Prompts: "
        + ", ".join(
            f"{prompt.key}={prompt.text!r} ({prompt.role})"
            for prompt in config.prompts
        )
    )
    for request in selected[:5]:
        print(f"  {request.image_path} -> {request.output_path.as_posix()}")
    if len(selected) > 5:
        print(f"  ... and {len(selected) - 5} more")
    if args.dry_run:
        print("Dry run complete: dataset, config and restart state are valid.")
        return

    manifest_path, manifest = prepare_run(
        run_dir=run_dir,
        config_path=config_path,
        config=config,
        dataset_name=dataset.name,
        image_count=len(requests),
        experiment_sha256=experiment_sha256,
    )

    if not selected:
        records = successful_records(requests, run_dir, experiment_sha256)
        rewrite_summary_files(run_dir, records)
        manifest["status"] = "complete" if len(records) == len(requests) else "partial"
        manifest["last_checked_at"] = now_iso()
        manifest["progress"] = {
            "completed": len(records),
            "total": len(requests),
            "remaining": len(requests) - len(records),
        }
        atomic_write_json(manifest_path, manifest)
        print("No unfinished images were selected; the model was not loaded.")
        print(f"Run directory: {run_dir}")
        return

    print("Loading facebook/sam3 with MLX...")
    predictor, model_path = create_predictor(config)
    manifest["status"] = "running"
    manifest["last_started_at"] = now_iso()
    manifest["model"] = {
        "implementation": "MLX-VLM SAM 3",
        "model_path": str(model_path),
        "mlx_vlm_version": package_version("mlx-vlm"),
        "mlx_version": package_version("mlx"),
    }
    manifest["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "pillow": package_version("Pillow"),
        "pyyaml": package_version("PyYAML"),
    }
    manifest["timing_strategy"] = {
        "warmup_images": 1,
        "protocol_version": TIMING_PROTOCOL_VERSION,
        "definitions": TIMING_DEFINITIONS,
    }
    manifest.setdefault("outputs", {})["memory_usage"] = "memory_usage.jsonl"
    atomic_write_json(manifest_path, manifest)

    print("Warming up MLX graph...")
    with Image.open(selected[0].image_path) as warmup_source:
        warmup_image = warmup_source.convert("RGB")
    predict_prompts(
        predictor=predictor,
        image=warmup_image,
        prompts=config.prompts,
        confidence_threshold=config.confidence_threshold,
    )

    failures = 0
    successes_this_invocation = 0
    attempts_path = run_dir / "run_attempts.jsonl"
    for index, request in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {request.image_path.name}")
        record = process_one_image(
            request=request,
            run_dir=run_dir,
            config=config,
            predictor=predictor,
            experiment_sha256=experiment_sha256,
        )
        append_json_line(attempts_path, record)
        if record["status"] == "success":
            successes_this_invocation += 1
            inference_ms = float(record["timing_ms"]["inference"])
            print(f"  success ({inference_ms:.3f} ms inference)")
        else:
            failures += 1
            print(
                f"  failed: {record['error']['type']}: "
                f"{record['error']['message']}"
            )

        current_records = successful_records(requests, run_dir, experiment_sha256)
        rewrite_summary_files(run_dir, current_records)
        manifest["progress"] = {
            "completed": len(current_records),
            "total": len(requests),
            "remaining": len(requests) - len(current_records),
        }
        atomic_write_json(manifest_path, manifest)

    records = successful_records(requests, run_dir, experiment_sha256)
    manifest["status"] = "complete" if len(records) == len(requests) else "partial"
    manifest["last_finished_at"] = now_iso()
    manifest["progress"] = {
        "completed": len(records),
        "total": len(requests),
        "remaining": len(requests) - len(records),
        "failures_this_invocation": failures,
    }
    atomic_write_json(manifest_path, manifest)
    memory = append_memory_usage(
        run_dir,
        model="SAM 3",
        dataset=config.dataset_key,
        attempted_images=len(selected),
        successful_images=successes_this_invocation,
    )

    print(f"Completed records: {len(records)}/{len(requests)}")
    print(f"Run directory: {run_dir}")
    print(f"Evaluation masks: {run_dir / 'predicted_masks'}")
    print(f"Inference times: {run_dir / 'inference_times.txt'}")
    print(f"Comparable timing report: {run_dir / 'comparable_timing_summary.json'}")
    print(f"Peak process RSS: {memory['peak_process_rss_mib']:.1f} MiB")
    print(f"Memory report: {run_dir / 'memory_usage.jsonl'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
