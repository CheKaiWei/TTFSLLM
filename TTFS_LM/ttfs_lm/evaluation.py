from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from .data import PackedCorpus
from .workflow import tensor_comparison


def _metric_from_nll(nll_sum: float, target_tokens: int) -> dict[str, float | int]:
    if target_tokens <= 0:
        raise ValueError("No target tokens were evaluated")
    mean_nll = nll_sum / target_tokens
    perplexity = math.exp(mean_nll) if mean_nll < 700 else math.inf
    return {
        "nll": mean_nll,
        "perplexity": perplexity,
        "bits_per_token": mean_nll / math.log(2),
        "target_tokens": target_tokens,
    }


def _batch_nll(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[float, int]:
    shifted_logits = logits[:, :-1].float()
    shifted_targets = input_ids[:, 1:]
    valid = attention_mask.to(torch.bool)
    target_mask = valid[:, :-1] & valid[:, 1:]
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_targets.reshape(-1),
        reduction="none",
    ).view_as(shifted_targets)
    return float(losses[target_mask].sum().item()), int(target_mask.sum().item())


@torch.no_grad()
def evaluate_model_pair(
    reference_model,
    candidate_model,
    corpus: PackedCorpus,
    batch_size: int,
    device: str | torch.device,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    reference_nll = 0.0
    candidate_nll = 0.0
    target_tokens = 0
    comparison_tokens = 0
    comparison_elements = 0
    absolute_error_sum = 0.0
    max_absolute_error = 0.0
    top1_matches = 0
    kl_sum = 0.0
    batches = math.ceil(len(corpus) / batch_size)

    for batch_index, batch in enumerate(corpus.batches(batch_size, device), start=1):
        reference_logits = reference_model(**batch, use_cache=False).logits
        candidate_logits = candidate_model(**batch, use_cache=False).logits
        reference_batch_nll, batch_targets = _batch_nll(
            reference_logits, batch["input_ids"], batch["attention_mask"]
        )
        candidate_batch_nll, candidate_targets = _batch_nll(
            candidate_logits, batch["input_ids"], batch["attention_mask"]
        )
        if candidate_targets != batch_targets:
            raise RuntimeError("Reference and candidate target counts differ")
        reference_nll += reference_batch_nll
        candidate_nll += candidate_batch_nll
        target_tokens += batch_targets

        valid = batch["attention_mask"].to(torch.bool)
        prediction_mask = valid[:, :-1] & valid[:, 1:]
        reference_valid = reference_logits[:, :-1][prediction_mask].float()
        candidate_valid = candidate_logits[:, :-1][prediction_mask].float()
        difference = (reference_valid - candidate_valid).abs()
        absolute_error_sum += float(difference.sum().item())
        max_absolute_error = max(max_absolute_error, float(difference.max().item()))
        comparison_elements += difference.numel()
        top1_matches += int(
            (reference_valid.argmax(dim=-1) == candidate_valid.argmax(dim=-1)).sum().item()
        )
        comparison_tokens += reference_valid.shape[0]

        reference_log_probs = F.log_softmax(reference_valid, dim=-1)
        candidate_log_probs = F.log_softmax(candidate_valid, dim=-1)
        kl_sum += float(
            F.kl_div(candidate_log_probs, reference_log_probs, log_target=True, reduction="sum").item()
        )
        if progress is not None:
            progress(batch_index, batches)

    reference_metrics = _metric_from_nll(reference_nll, target_tokens)
    candidate_metrics = _metric_from_nll(candidate_nll, target_tokens)
    return {
        "reference": reference_metrics,
        "candidate": candidate_metrics,
        "comparison": {
            "max_abs": max_absolute_error,
            "mean_abs": absolute_error_sum / comparison_elements,
            "top1_agreement": top1_matches / comparison_tokens,
            "mean_kl": kl_sum / comparison_tokens,
            "comparison_tokens": comparison_tokens,
        },
    }


@torch.no_grad()
def evaluate_cache_equivalence(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if input_ids.shape[0] != 1:
        raise ValueError("Cache equivalence MVP check expects batch size 1")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    full = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
    cache = None
    pieces = []
    for index in range(input_ids.shape[1]):
        output = model(
            input_ids[:, index : index + 1],
            attention_mask=attention_mask[:, : index + 1],
            cache=cache,
            use_cache=True,
        )
        cache = output.cache
        pieces.append(output.logits)
    cached = torch.cat(pieces, dim=1)
    metrics = tensor_comparison(full, cached, attention_mask[:, :, None])
    metrics["cache_length"] = cache.get_seq_length()
    return metrics
