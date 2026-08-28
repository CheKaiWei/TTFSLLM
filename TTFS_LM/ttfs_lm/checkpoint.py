from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import DEFAULT_MODEL_ID, DEFAULT_REVISION, QwenConfig
from .modeling_ann import QwenForCausalLM


@dataclass
class OfficialModelPair:
    ann: QwenForCausalLM
    reference: torch.nn.Module


def _checkpoint_source(model_id: str, revision: str, local_files_only: bool) -> tuple[str, str | None]:
    if not local_files_only:
        return model_id, revision
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(model_id, revision=revision, local_files_only=True)
    return snapshot, None


def _parse_dtype(dtype: str | torch.dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    choices = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return choices[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(choices)}") from exc


def load_official_pair(
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = "float32",
    local_files_only: bool = False,
) -> OfficialModelPair:
    """Load the official Hugging Face model and an exactly keyed local ANN copy."""

    from transformers import AutoModelForCausalLM

    source, source_revision = _checkpoint_source(model_id, revision, local_files_only)
    reference = AutoModelForCausalLM.from_pretrained(
        source,
        revision=source_revision,
        dtype=_parse_dtype(dtype),
        attn_implementation="eager",
        local_files_only=local_files_only,
    )
    config = QwenConfig.from_hf_config(reference.config)
    if config.num_hidden_layers != 24 and model_id == DEFAULT_MODEL_ID:
        raise ValueError(f"Official baseline must contain 24 layers, got {config.num_hidden_layers}")

    reference_dtype = next(reference.parameters()).dtype
    ann = QwenForCausalLM(config).to(dtype=reference_dtype)
    incompatible = ann.load_state_dict(reference.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Official state_dict mismatch: {incompatible}")
    ann.tie_weights()
    ann.ttfs_provenance = {"model_id": model_id, "revision": revision}

    target = torch.device(device)
    reference = reference.to(target).eval()
    ann = ann.to(target).eval()
    return OfficialModelPair(ann=ann, reference=reference)


def load_official_ann(
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = "float32",
    local_files_only: bool = False,
) -> QwenForCausalLM:
    pair = load_official_pair(model_id, revision, device, dtype, local_files_only)
    ann = pair.ann
    del pair.reference
    return ann


def load_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    local_files_only: bool = False,
):
    from transformers import AutoTokenizer

    source, source_revision = _checkpoint_source(model_id, revision, local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=source_revision,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
