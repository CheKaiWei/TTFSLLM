from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .calibration import MinMaxCalibrator, RangeRegistry
from .modeling_ann import QwenForCausalLM


DEFAULT_CALIBRATION_TEXTS = (
    "The purpose of this experiment is to compare an artificial neural network with a time-to-first-spike model.",
    "A causal language model predicts the next token using only the tokens that came before it.",
    "时间到首脉冲编码使用单个脉冲的发放时刻表示连续激活值。",
    "请用简洁的语言解释为什么因果注意力不能读取未来的词元。",
)


def tokenize_texts(tokenizer, texts: Iterable[str], device: torch.device | str, max_length: int = 128):
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {name: value.to(device) for name, value in encoded.items() if name in {"input_ids", "attention_mask"}}


@torch.no_grad()
def calibrate_ann(
    model: QwenForCausalLM,
    batches: Iterable[dict[str, torch.Tensor]],
    margin: float = 0.05,
    include_decode: bool = True,
    metadata: dict[str, Any] | None = None,
) -> RangeRegistry:
    calibrator = MinMaxCalibrator()
    for batch in batches:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            observer=calibrator,
        )
        if include_decode:
            cache = None
            for index in range(input_ids.shape[1]):
                prefix_mask = attention_mask[:, : index + 1] if attention_mask is not None else None
                output = model(
                    input_ids[:, index : index + 1],
                    attention_mask=prefix_mask,
                    cache=cache,
                    use_cache=True,
                    observer=calibrator,
                )
                cache = output.cache
    return calibrator.finalize(margin=margin, metadata=metadata)


def tensor_comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Cannot compare shapes {tuple(reference.shape)} and {tuple(candidate.shape)}")
    reference_float = reference.float()
    candidate_float = candidate.float()
    if mask is not None:
        mask = torch.broadcast_to(mask.to(device=reference.device, dtype=torch.bool), reference.shape)
        reference_float = reference_float[mask]
        candidate_float = candidate_float[mask]
        if reference_float.numel() == 0:
            raise ValueError("Comparison mask selects no elements")
    difference = (reference_float - candidate_float).abs()
    cosine = torch.nn.functional.cosine_similarity(
        reference_float.reshape(1, -1), candidate_float.reshape(1, -1), dim=-1
    )
    if mask is None:
        top1 = (reference.argmax(dim=-1) == candidate.argmax(dim=-1)).float().mean()
    else:
        token_mask = mask.reshape(*reference.shape[:-1], reference.shape[-1]).any(dim=-1)
        top1 = (reference.argmax(dim=-1)[token_mask] == candidate.argmax(dim=-1)[token_mask]).float().mean()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "cosine": float(cosine.item()),
        "top1_agreement": float(top1.item()),
    }
