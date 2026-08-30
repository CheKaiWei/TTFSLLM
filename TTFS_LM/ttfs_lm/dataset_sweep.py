from __future__ import annotations

import gc
import hashlib
import json
import math
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
from .data import CorpusSource, PackedCorpus, load_corpus
from .evaluation import evaluate_cache_equivalence, evaluate_model_pair
from .generation import greedy_generate
from .modeling_ttfs import RangeViolationError
from .mvp import (
    AcceptanceMVPConfig,
    CalibrationMVPConfig,
    GenerationMVPConfig,
    ModelMVPConfig,
    RunLogger,
    TTFSMVPConfig,
    _environment,
    _git_state,
    _progress,
    _relative_change,
    _telemetry,
    _write_json,
)
from .workflow import calibrate_ann


@dataclass
class SweepRuntimeConfig:
    context_length: int
    batch_size: int
    seed: int
    local_files_only: bool


@dataclass
class DatasetSweepConfig:
    model: ModelMVPConfig
    runtime: SweepRuntimeConfig
    calibration_dataset: CorpusSource
    evaluation_datasets: list[CorpusSource]
    calibration: CalibrationMVPConfig
    ttfs: TTFSMVPConfig
    generation: GenerationMVPConfig
    acceptance: AcceptanceMVPConfig
    output_dir: str

    def validate(self) -> None:
        if self.runtime.context_length < 2 or self.runtime.batch_size <= 0:
            raise ValueError("context_length must be >=2 and batch_size must be positive")
        self.calibration_dataset.validate()
        if self.calibration_dataset.max_tokens is None:
            raise ValueError("Sweep calibration dataset requires a finite max_tokens")
        if self.calibration_dataset.max_tokens < self.runtime.context_length:
            raise ValueError("Calibration max_tokens must cover at least one context")
        if not self.evaluation_datasets:
            raise ValueError("At least one evaluation dataset is required")
        names = [source.name for source in self.evaluation_datasets]
        if len(names) != len(set(names)):
            raise ValueError("Evaluation dataset names must be unique")
        for source in self.evaluation_datasets:
            source.validate()
        if self.calibration.margin < 0 or self.calibration.decode_batches < 0:
            raise ValueError("Calibration margin and decode_batches must be non-negative")
        if self.ttfs.finite_precision <= 0:
            raise ValueError("finite_precision must be positive")
        if self.generation.max_new_tokens < 8:
            raise ValueError("Sweep generation must request at least eight new tokens")
        if (
            self.acceptance.official_local_max_abs < 0
            or self.acceptance.continuous_ppl_relative < 0
            or self.acceptance.cache_max_abs < 0
            or self.acceptance.ttfs_cache_extra_abs < 0
        ):
            raise ValueError("Sweep error tolerances must be non-negative")
        if not 0 <= self.acceptance.continuous_top1 <= 1:
            raise ValueError("continuous_top1 must be between zero and one")


def load_dataset_sweep_config(path: str | Path) -> tuple[DatasetSweepConfig, str]:
    source = Path(path)
    raw = source.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError("Dataset sweep config must be a YAML mapping")
    try:
        config = DatasetSweepConfig(
            model=ModelMVPConfig(**payload["model"]),
            runtime=SweepRuntimeConfig(**payload["runtime"]),
            calibration_dataset=CorpusSource(**payload["calibration_dataset"]),
            evaluation_datasets=[CorpusSource(**item) for item in payload["evaluation_datasets"]],
            calibration=CalibrationMVPConfig(**payload["calibration"]),
            ttfs=TTFSMVPConfig(**payload["ttfs"]),
            generation=GenerationMVPConfig(**payload["generation"]),
            acceptance=AcceptanceMVPConfig(**payload["acceptance"]),
            output_dir=payload["output_dir"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid dataset sweep config structure: {exc}") from exc
    config.validate()
    return config, hashlib.sha256(raw).hexdigest()


def _pair_metrics_with_relative_change(metrics: dict[str, Any]) -> dict[str, Any]:
    metrics["ppl_relative_change"] = _relative_change(
        metrics["reference"]["perplexity"], metrics["candidate"]["perplexity"]
    )
    return metrics


def _continuous_acceptance(metrics: dict[str, Any], telemetry: dict[str, Any], config: DatasetSweepConfig):
    return {
        "finite_values": math.isfinite(metrics["candidate"]["perplexity"]),
        "no_clipping": telemetry["clipped_elements"] == 0,
        "ppl": abs(metrics["ppl_relative_change"]) <= config.acceptance.continuous_ppl_relative,
        "top1": metrics["comparison"]["top1_agreement"] >= config.acceptance.continuous_top1,
    }


def _evaluate_dataset(
    source: CorpusSource,
    corpus: PackedCorpus,
    ann,
    continuous,
    continuous_diagnostic,
    finite,
    config: DatasetSweepConfig,
    logger: RunLogger,
) -> dict[str, Any]:
    result: dict[str, Any] = {"source": asdict(source), "data": corpus.metadata}
    continuous.codec.reset_telemetry()
    try:
        strict_metrics = _pair_metrics_with_relative_change(
            evaluate_model_pair(
                ann,
                continuous,
                corpus,
                config.runtime.batch_size,
                config.model.device,
                _progress(logger, f"{source.name}-continuous"),
            )
        )
        strict_telemetry = _telemetry(continuous)
        result["continuous"] = {"status": "completed", **strict_metrics}
        result["continuous_telemetry"] = strict_telemetry
        result["continuous_acceptance"] = _continuous_acceptance(
            strict_metrics, strict_telemetry, config
        )
    except RangeViolationError as exc:
        result["continuous"] = {
            "status": "range_failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        result["continuous_telemetry"] = _telemetry(continuous)
        continuous_diagnostic.codec.reset_telemetry()
        logger.log(
            f"{source.name}-continuous-diagnostic",
            "strict range failed; rerunning with explicit clipping for diagnostics",
        )
        diagnostic_metrics = _pair_metrics_with_relative_change(
            evaluate_model_pair(
                ann,
                continuous_diagnostic,
                corpus,
                config.runtime.batch_size,
                config.model.device,
                _progress(logger, f"{source.name}-continuous-diagnostic"),
            )
        )
        result["continuous_clipped_diagnostic"] = {
            "status": "completed",
            **diagnostic_metrics,
        }
        result["continuous_clipped_telemetry"] = _telemetry(continuous_diagnostic)
        result["continuous_acceptance"] = {
            "finite_values": math.isfinite(diagnostic_metrics["candidate"]["perplexity"]),
            "no_clipping": False,
            "ppl": False,
            "top1": False,
        }

    finite.codec.reset_telemetry()
    try:
        finite_metrics = _pair_metrics_with_relative_change(
            evaluate_model_pair(
                ann,
                finite,
                corpus,
                config.runtime.batch_size,
                config.model.device,
                _progress(logger, f"{source.name}-finite"),
            )
        )
        result["finite"] = {"status": "completed", **finite_metrics}
        result["finite_telemetry"] = _telemetry(finite)
    except Exception as exc:
        result["finite"] = {
            "status": "failed",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    return result


def _generation_metrics(ann, continuous, tokenizer, config: DatasetSweepConfig) -> dict[str, Any]:
    encoded = tokenizer(config.generation.prompt, return_tensors="pt")
    encoded = {
        name: value.to(config.model.device)
        for name, value in encoded.items()
        if name in {"input_ids", "attention_mask"}
    }
    ann_cache = evaluate_cache_equivalence(
        ann, encoded["input_ids"], encoded.get("attention_mask")
    )
    continuous.codec.reset_telemetry()
    ttfs_cache = evaluate_cache_equivalence(
        continuous, encoded["input_ids"], encoded.get("attention_mask")
    )
    ann_tokens = greedy_generate(
        ann,
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
    prompt_length = encoded["input_ids"].shape[1]
    return {
        "prompt": config.generation.prompt,
        "new_tokens": ann_tokens.shape[1] - prompt_length,
        "token_agreement": float(
            (ann_tokens[:, prompt_length:] == ttfs_tokens[:, prompt_length:]).float().mean().item()
        ),
        "ann_cache": ann_cache,
        "ttfs_cache": ttfs_cache,
        "ann_token_ids": ann_tokens[0].tolist(),
        "ttfs_token_ids": ttfs_tokens[0].tolist(),
        "ann_text": tokenizer.decode(ann_tokens[0], skip_special_tokens=True),
        "ttfs_text": tokenizer.decode(ttfs_tokens[0], skip_special_tokens=True),
        "continuous_telemetry": _telemetry(continuous),
    }


def _summarize(metrics: dict[str, Any], config: DatasetSweepConfig) -> dict[str, Any]:
    datasets = {}
    execution_completed = True
    strict_passed = 0
    diagnostic_quality_passed = 0
    for name, result in metrics["datasets"].items():
        continuous = result.get("continuous", {})
        finite = result.get("finite", {})
        acceptance = result.get("continuous_acceptance", {})
        strict_ok = continuous.get("status") == "completed" and all(acceptance.values())
        strict_passed += int(strict_ok)
        diagnostic = result.get("continuous_clipped_diagnostic", {})
        diagnostic_telemetry = result.get("continuous_clipped_telemetry", {})
        diagnostic_completed = diagnostic.get("status") == "completed"
        diagnostic_ok = (
            diagnostic_completed
            and math.isfinite(diagnostic["candidate"]["perplexity"])
            and abs(diagnostic["ppl_relative_change"])
            <= config.acceptance.continuous_ppl_relative
            and diagnostic["comparison"]["top1_agreement"]
            >= config.acceptance.continuous_top1
        )
        diagnostic_quality_passed += int(diagnostic_ok)
        execution_completed &= (
            continuous.get("status") == "completed" or diagnostic_completed
        ) and finite.get("status") == "completed"
        finite_telemetry = result.get("finite_telemetry", {})
        diagnostic_encoded = diagnostic_telemetry.get("encoded_elements", 0)
        diagnostic_clipped = diagnostic_telemetry.get("clipped_elements", 0)
        finite_encoded = finite_telemetry.get("encoded_elements", 0)
        finite_clipped = finite_telemetry.get("clipped_elements", 0)
        datasets[name] = {
            "strict_continuous_passed": strict_ok,
            "continuous_status": continuous.get("status"),
            "strict_error": continuous.get("error", {}).get("message"),
            "continuous_ppl": continuous.get("candidate", {}).get("perplexity"),
            "continuous_top1": continuous.get("comparison", {}).get("top1_agreement"),
            "continuous_clipped_elements": result.get("continuous_telemetry", {}).get(
                "clipped_elements"
            ),
            "diagnostic_quality_passed": diagnostic_ok,
            "diagnostic_ann_ppl": diagnostic.get("reference", {}).get("perplexity"),
            "diagnostic_continuous_ppl": diagnostic.get("candidate", {}).get("perplexity"),
            "diagnostic_ppl_relative_change": diagnostic.get("ppl_relative_change"),
            "diagnostic_top1": diagnostic.get("comparison", {}).get("top1_agreement"),
            "diagnostic_max_abs": diagnostic.get("comparison", {}).get("max_abs"),
            "diagnostic_clipped_elements": diagnostic_clipped,
            "diagnostic_encoded_elements": diagnostic_encoded,
            "diagnostic_clipped_ratio": (
                diagnostic_clipped / diagnostic_encoded if diagnostic_encoded else None
            ),
            "diagnostic_violating_range_count": diagnostic_telemetry.get(
                "violating_range_count"
            ),
            "finite_status": finite.get("status"),
            "finite_ppl": finite.get("candidate", {}).get("perplexity"),
            "finite_top1": finite.get("comparison", {}).get("top1_agreement"),
            "finite_clipped_elements": finite_clipped,
            "finite_encoded_elements": finite_encoded,
            "finite_clipped_ratio": finite_clipped / finite_encoded if finite_encoded else None,
            "finite_violating_range_count": finite_telemetry.get("violating_range_count"),
            "target_tokens": result.get("data", {}).get("target_token_count"),
        }
    return {
        "execution_completed": execution_completed,
        "dataset_count": len(datasets),
        "strict_continuous_passed": strict_passed,
        "strict_continuous_failed": len(datasets) - strict_passed,
        "clipped_diagnostic_quality_passed": diagnostic_quality_passed,
        "finite_precision": config.ttfs.finite_precision,
        "datasets": datasets,
    }


def run_dataset_sweep(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    device: str | None = None,
    local_files_only: bool | None = None,
) -> dict[str, Any]:
    config, config_sha256 = load_dataset_sweep_config(config_path)
    if output_dir is not None:
        config.output_dir = str(output_dir)
    if device is not None:
        config.model.device = device
    if local_files_only is not None:
        config.model.local_files_only = local_files_only
        config.runtime.local_files_only = local_files_only
    config.validate()

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output / "run.log")
    started = time.time()
    manifest: dict[str, Any] = {
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
    _write_json(output / "sweep_manifest.json", manifest)
    metrics: dict[str, Any] = {"datasets": {}}

    try:
        torch.manual_seed(config.runtime.seed)
        logger.log("tokenizer", "loading pinned tokenizer")
        tokenizer = load_tokenizer(
            config.model.id,
            config.model.revision,
            config.model.local_files_only,
        )
        source_cache = output / "sources"
        logger.log("data-calibration", f"loading {config.calibration_dataset.name}")
        calibration_corpus = load_corpus(
            config.calibration_dataset,
            tokenizer,
            config.runtime.context_length,
            config.runtime.local_files_only,
            source_cache,
        )
        corpora: dict[str, PackedCorpus] = {}
        for source in config.evaluation_datasets:
            logger.log(f"data-{source.name}", "loading and packing evaluation corpus")
            corpora[source.name] = load_corpus(
                source,
                tokenizer,
                config.runtime.context_length,
                config.runtime.local_files_only,
                source_cache,
            )
            logger.log(
                f"data-{source.name}",
                "corpus ready",
                tokens=corpora[source.name].metadata["token_count"],
            )
        manifest["data"] = {
            "calibration": calibration_corpus.metadata,
            "evaluations": {name: corpus.metadata for name, corpus in corpora.items()},
        }
        _write_json(output / "sweep_manifest.json", manifest)

        logger.log("model", "loading official reference and local ANN")
        pair = load_official_pair(
            config.model.id,
            config.model.revision,
            config.model.device,
            config.model.dtype,
            config.model.local_files_only,
        )
        logger.log("evaluate-ann", "checking official and local ANN on calibration corpus")
        metrics["official_vs_local"] = evaluate_model_pair(
            pair.reference,
            pair.ann,
            calibration_corpus,
            config.runtime.batch_size,
            config.model.device,
            _progress(logger, "evaluate-ann"),
        )
        del pair.reference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        range_metadata = {
            "model_id": config.model.id,
            "revision": config.model.revision,
            "tokenizer_name": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
            "tokenizer_vocab_size": len(tokenizer),
            "calibration_dataset": calibration_corpus.metadata,
            "evaluation_datasets": {name: corpus.metadata for name, corpus in corpora.items()},
            "margin": config.calibration.margin,
            "decode_batches": config.calibration.decode_batches,
            "config_sha256": config_sha256,
            "git_commit": manifest["git"]["commit"],
        }
        logger.log("calibrate", "collecting shared ranges from calibration corpus")
        registry = calibrate_ann(
            pair.ann,
            calibration_corpus.batches(config.runtime.batch_size, config.model.device),
            margin=config.calibration.margin,
            include_decode=config.calibration.decode_batches > 0,
            decode_batches=config.calibration.decode_batches,
            metadata=range_metadata,
        )
        registry.save(output / "ranges.json")

        continuous = convert_ann_to_ttfs(
            pair.ann,
            registry,
            TTFSConfig(None, True, config.ttfs.spike_lm_head),
        ).eval()
        continuous_diagnostic = convert_ann_to_ttfs(
            pair.ann,
            registry,
            TTFSConfig(None, False, config.ttfs.spike_lm_head),
        ).eval()
        finite = convert_ann_to_ttfs(
            pair.ann,
            registry,
            TTFSConfig(
                config.ttfs.finite_precision,
                not config.ttfs.finite_allow_clipping,
                config.ttfs.spike_lm_head,
            ),
        ).eval()

        for source in config.evaluation_datasets:
            logger.log(source.name, "starting ANN-to-TTFS dataset evaluation")
            metrics["datasets"][source.name] = _evaluate_dataset(
                source,
                corpora[source.name],
                pair.ann,
                continuous,
                continuous_diagnostic,
                finite,
                config,
                logger,
            )
            _write_json(output / "metrics.json", metrics)

        logger.log("generation", "checking shared cache and generation path")
        metrics["generation"] = _generation_metrics(
            pair.ann, continuous, tokenizer, config
        )
        metrics["summary"] = _summarize(metrics, config)
        _write_json(output / "metrics.json", metrics)
        _write_json(output / "summary.json", metrics["summary"])

        manifest.update(
            {
                "status": (
                    "completed" if metrics["summary"]["execution_completed"] else "completed_with_failures"
                ),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.time() - started,
                "artifacts": {
                    "ranges": "ranges.json",
                    "metrics": "metrics.json",
                    "summary": "summary.json",
                    "log": "run.log",
                },
            }
        )
        _write_json(output / "sweep_manifest.json", manifest)
        logger.log("complete", f"dataset sweep finished with status={manifest['status']}")
        return {"manifest": manifest, "metrics": metrics, "output_dir": str(output.resolve())}
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed_exception",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.time() - started,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        _write_json(output / "sweep_manifest.json", manifest)
        logger.log("failed", f"{type(exc).__name__}: {exc}")
        raise


def format_sweep_summary(result: dict[str, Any]) -> str:
    payload = {
        "output_dir": result["output_dir"],
        "status": result["manifest"]["status"],
        **result["metrics"].get("summary", {}),
    }
    return json.dumps(payload, indent=2, sort_keys=True)
