from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from .checkpoint import load_official_pair, load_tokenizer
from .config import TTFSConfig
from .conversion import convert_ann_to_ttfs
from .data import load_wikitext2_corpora
from .evaluation import evaluate_cache_equivalence, evaluate_model_pair
from .generation import greedy_generate
from .workflow import calibrate_ann


@dataclass
class ModelMVPConfig:
    id: str
    revision: str
    dtype: str
    device: str
    local_files_only: bool


@dataclass
class DatasetMVPConfig:
    id: str
    config: str
    revision: str
    calibration_split: str
    evaluation_split: str
    calibration_max_tokens: int
    evaluation_max_tokens: int | None
    context_length: int
    batch_size: int
    seed: int
    local_files_only: bool


@dataclass
class CalibrationMVPConfig:
    margin: float
    decode_batches: int


@dataclass
class TTFSMVPConfig:
    finite_precision: int
    finite_allow_clipping: bool
    spike_lm_head: bool


@dataclass
class GenerationMVPConfig:
    prompt: str
    max_new_tokens: int


@dataclass
class AcceptanceMVPConfig:
    official_local_max_abs: float
    continuous_ppl_relative: float
    continuous_top1: float
    cache_max_abs: float
    ttfs_cache_extra_abs: float


@dataclass
class MVPConfig:
    model: ModelMVPConfig
    dataset: DatasetMVPConfig
    calibration: CalibrationMVPConfig
    ttfs: TTFSMVPConfig
    generation: GenerationMVPConfig
    acceptance: AcceptanceMVPConfig
    output_dir: str

    def validate(self) -> None:
        if self.dataset.calibration_split == self.dataset.evaluation_split:
            raise ValueError("Calibration and evaluation splits must differ")
        if self.dataset.context_length < 2 or self.dataset.batch_size <= 0:
            raise ValueError("context_length must be >=2 and batch_size must be positive")
        if self.dataset.calibration_max_tokens < self.dataset.context_length:
            raise ValueError("calibration_max_tokens must cover at least one context")
        if self.dataset.evaluation_max_tokens is not None and self.dataset.evaluation_max_tokens < 2:
            raise ValueError("evaluation_max_tokens must be >=2 or null")
        if self.calibration.margin < 0 or self.calibration.decode_batches < 0:
            raise ValueError("calibration margin and decode_batches must be non-negative")
        if self.ttfs.finite_precision <= 0:
            raise ValueError("finite_precision must be positive")
        if self.generation.max_new_tokens < 8:
            raise ValueError("MVP generation must request at least 8 new tokens")
        if (
            self.acceptance.official_local_max_abs < 0
            or self.acceptance.continuous_ppl_relative < 0
            or self.acceptance.cache_max_abs < 0
            or self.acceptance.ttfs_cache_extra_abs < 0
        ):
            raise ValueError("MVP error tolerances must be non-negative")
        if not 0 <= self.acceptance.continuous_top1 <= 1:
            raise ValueError("continuous_top1 must be between zero and one")


def load_mvp_config(path: str | Path) -> tuple[MVPConfig, str]:
    source = Path(path)
    raw = source.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError("MVP config must be a YAML mapping")
    try:
        config = MVPConfig(
            model=ModelMVPConfig(**payload["model"]),
            dataset=DatasetMVPConfig(**payload["dataset"]),
            calibration=CalibrationMVPConfig(**payload["calibration"]),
            ttfs=TTFSMVPConfig(**payload["ttfs"]),
            generation=GenerationMVPConfig(**payload["generation"]),
            acceptance=AcceptanceMVPConfig(**payload["acceptance"]),
            output_dir=payload["output_dir"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid MVP config structure: {exc}") from exc
    config.validate()
    return config, hashlib.sha256(raw).hexdigest()


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def log(self, stage: str, message: str, **fields: Any) -> None:
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "message": message,
            **fields,
        }
        line = json.dumps(event, sort_keys=True, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(f"[{stage}] {message}", flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": []}


def _environment() -> dict[str, Any]:
    import datasets
    import huggingface_hub
    import safetensors
    import transformers

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "gpu": gpu,
    }


def _telemetry(model) -> dict[str, Any]:
    items = {}
    total_elements = 0
    clipped_elements = 0
    for name, stats in sorted(model.codec.telemetry.items()):
        total_elements += stats.elements
        clipped = stats.below + stats.above
        clipped_elements += clipped
        if clipped:
            items[name] = asdict(stats)
    return {
        "encoded_elements": total_elements,
        "clipped_elements": clipped_elements,
        "violating_range_count": len(items),
        "violating_ranges": items,
    }


def _progress(logger: RunLogger, stage: str):
    last = {"batch": 0}

    def report(batch: int, total: int) -> None:
        if batch == total or batch - last["batch"] >= max(1, total // 10):
            last["batch"] = batch
            logger.log(stage, f"evaluated batch {batch}/{total}", batch=batch, total_batches=total)

    return report


def _relative_change(reference: float, candidate: float) -> float:
    return (candidate - reference) / reference


def _acceptance(
    metrics: dict[str, Any], generation: dict[str, Any], thresholds: AcceptanceMVPConfig
) -> dict[str, bool]:
    official = metrics["official_vs_local"]
    continuous = metrics["local_vs_continuous"]
    finite = metrics["local_vs_finite"]
    continuous_telemetry = metrics["continuous_telemetry"]
    ann_cache_error = generation["ann_cache"]["max_abs"]
    ttfs_cache_error = generation["ttfs_cache"]["max_abs"]
    return {
        "official_local_max_abs": (
            official["comparison"]["max_abs"] <= thresholds.official_local_max_abs
        ),
        "continuous_finite_values": math.isfinite(continuous["candidate"]["perplexity"]),
        "continuous_no_clipping": continuous_telemetry["clipped_elements"] == 0,
        "continuous_ppl": (
            abs(continuous["ppl_relative_change"]) <= thresholds.continuous_ppl_relative
        ),
        "continuous_top1": (
            continuous["comparison"]["top1_agreement"] >= thresholds.continuous_top1
        ),
        "ann_cache": ann_cache_error <= thresholds.cache_max_abs,
        "ttfs_cache": (
            ttfs_cache_error <= thresholds.cache_max_abs
            and ttfs_cache_error <= ann_cache_error + thresholds.ttfs_cache_extra_abs
        ),
        "generation_length": generation["new_tokens"] >= 8,
        "finite_completed": math.isfinite(finite["candidate"]["perplexity"]),
    }


def run_mvp(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    device: str | None = None,
    local_files_only: bool | None = None,
) -> dict[str, Any]:
    config, config_sha256 = load_mvp_config(config_path)
    if output_dir is not None:
        config.output_dir = str(output_dir)
    if device is not None:
        config.model.device = device
    if local_files_only is not None:
        config.model.local_files_only = local_files_only
        config.dataset.local_files_only = local_files_only
    config.validate()

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output / "run.log")
    started = time.time()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_sha256,
        "config": asdict(config),
        "git": _git_state(),
        "environment": _environment(),
        "command": sys.argv,
    }
    _write_json(output / "run_manifest.json", manifest)
    metrics: dict[str, Any] = {}

    try:
        torch.manual_seed(config.dataset.seed)
        logger.log("model", "loading official reference, local ANN, and tokenizer")
        pair = load_official_pair(
            config.model.id,
            config.model.revision,
            config.model.device,
            config.model.dtype,
            config.model.local_files_only,
        )
        tokenizer = load_tokenizer(
            config.model.id,
            config.model.revision,
            config.model.local_files_only,
        )

        logger.log("data", "loading and packing WikiText-2 splits")
        calibration_corpus, evaluation_corpus = load_wikitext2_corpora(
            tokenizer=tokenizer,
            calibration_split=config.dataset.calibration_split,
            evaluation_split=config.dataset.evaluation_split,
            calibration_max_tokens=config.dataset.calibration_max_tokens,
            evaluation_max_tokens=config.dataset.evaluation_max_tokens,
            context_length=config.dataset.context_length,
            dataset_id=config.dataset.id,
            dataset_config=config.dataset.config,
            revision=config.dataset.revision,
            local_files_only=config.dataset.local_files_only,
        )
        manifest["data"] = {
            "calibration": calibration_corpus.metadata,
            "evaluation": evaluation_corpus.metadata,
        }
        _write_json(output / "run_manifest.json", manifest)
        logger.log(
            "data",
            "corpora ready",
            calibration_tokens=calibration_corpus.metadata["token_count"],
            evaluation_tokens=evaluation_corpus.metadata["token_count"],
        )

        logger.log("evaluate-ann", "comparing official Hugging Face model and local ANN")
        metrics["official_vs_local"] = evaluate_model_pair(
            pair.reference,
            pair.ann,
            evaluation_corpus,
            config.dataset.batch_size,
            config.model.device,
            _progress(logger, "evaluate-ann"),
        )
        _write_json(output / "metrics.json", metrics)
        del pair.reference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        range_metadata = {
            "model_id": config.model.id,
            "revision": config.model.revision,
            "tokenizer_name": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
            "tokenizer_vocab_size": len(tokenizer),
            "dataset": calibration_corpus.metadata,
            "evaluation_dataset": evaluation_corpus.metadata,
            "margin": config.calibration.margin,
            "decode_batches": config.calibration.decode_batches,
            "config_sha256": config_sha256,
            "git_commit": manifest["git"]["commit"],
        }
        logger.log("calibrate", "collecting activation ranges from train split")
        registry = calibrate_ann(
            pair.ann,
            calibration_corpus.batches(config.dataset.batch_size, config.model.device),
            margin=config.calibration.margin,
            include_decode=config.calibration.decode_batches > 0,
            decode_batches=config.calibration.decode_batches,
            metadata=range_metadata,
        )
        registry.save(output / "ranges.json")
        logger.log("calibrate", "range artifact saved", ranges=len(registry))

        continuous = convert_ann_to_ttfs(
            pair.ann,
            registry,
            TTFSConfig(precision=None, strict_ranges=True, spike_lm_head=config.ttfs.spike_lm_head),
        ).eval()
        logger.log("evaluate-continuous", "evaluating strict Continuous TTFS")
        metrics["local_vs_continuous"] = evaluate_model_pair(
            pair.ann,
            continuous,
            evaluation_corpus,
            config.dataset.batch_size,
            config.model.device,
            _progress(logger, "evaluate-continuous"),
        )
        metrics["local_vs_continuous"]["ppl_relative_change"] = _relative_change(
            metrics["local_vs_continuous"]["reference"]["perplexity"],
            metrics["local_vs_continuous"]["candidate"]["perplexity"],
        )

        finite = convert_ann_to_ttfs(
            pair.ann,
            registry,
            TTFSConfig(
                precision=config.ttfs.finite_precision,
                strict_ranges=not config.ttfs.finite_allow_clipping,
                spike_lm_head=config.ttfs.spike_lm_head,
            ),
        ).eval()
        logger.log("evaluate-finite", f"evaluating P={config.ttfs.finite_precision} smoke path")
        metrics["local_vs_finite"] = evaluate_model_pair(
            pair.ann,
            finite,
            evaluation_corpus,
            config.dataset.batch_size,
            config.model.device,
            _progress(logger, "evaluate-finite"),
        )
        metrics["local_vs_finite"]["ppl_relative_change"] = _relative_change(
            metrics["local_vs_finite"]["reference"]["perplexity"],
            metrics["local_vs_finite"]["candidate"]["perplexity"],
        )
        metrics["continuous_telemetry"] = _telemetry(continuous)
        metrics["finite_telemetry"] = _telemetry(finite)
        _write_json(output / "metrics.json", metrics)

        encoded = tokenizer(config.generation.prompt, return_tensors="pt")
        encoded = {
            name: value.to(config.model.device)
            for name, value in encoded.items()
            if name in {"input_ids", "attention_mask"}
        }
        logger.log("generation", "checking cache equivalence and greedy generation")
        generation = {
            "prompt": config.generation.prompt,
            "ann_cache": evaluate_cache_equivalence(
                pair.ann, encoded["input_ids"], encoded.get("attention_mask")
            ),
            "ttfs_cache": evaluate_cache_equivalence(
                continuous, encoded["input_ids"], encoded.get("attention_mask")
            ),
        }
        ann_tokens = greedy_generate(
            pair.ann,
            encoded["input_ids"],
            encoded.get("attention_mask"),
            config.generation.max_new_tokens,
            eos_token_id=None,
        )
        ttfs_tokens = greedy_generate(
            continuous,
            encoded["input_ids"],
            encoded.get("attention_mask"),
            config.generation.max_new_tokens,
            eos_token_id=None,
        )
        generation.update(
            {
                "new_tokens": ann_tokens.shape[1] - encoded["input_ids"].shape[1],
                "token_agreement": float(
                    (
                        ann_tokens[:, encoded["input_ids"].shape[1] :]
                        == ttfs_tokens[:, encoded["input_ids"].shape[1] :]
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "ann_token_ids": ann_tokens[0].tolist(),
                "ttfs_token_ids": ttfs_tokens[0].tolist(),
                "ann_text": tokenizer.decode(ann_tokens[0], skip_special_tokens=True),
                "ttfs_text": tokenizer.decode(ttfs_tokens[0], skip_special_tokens=True),
            }
        )
        metrics["continuous_telemetry"] = _telemetry(continuous)
        metrics["generation"] = generation
        acceptance = _acceptance(metrics, generation, config.acceptance)
        metrics["acceptance"] = acceptance
        metrics["passed"] = all(acceptance.values())
        _write_json(output / "metrics.json", metrics)

        manifest.update(
            {
                "status": "passed" if metrics["passed"] else "failed_acceptance",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.time() - started,
                "artifacts": {
                    "ranges": "ranges.json",
                    "metrics": "metrics.json",
                    "log": "run.log",
                },
            }
        )
        _write_json(output / "run_manifest.json", manifest)
        logger.log("complete", f"MVP finished with status={manifest['status']}")
        return {"manifest": manifest, "metrics": metrics, "output_dir": str(output.resolve())}
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed_exception",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.time() - started,
                "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            }
        )
        _write_json(output / "run_manifest.json", manifest)
        logger.log("failed", f"{type(exc).__name__}: {exc}")
        raise
