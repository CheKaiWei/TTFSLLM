from __future__ import annotations

import torch


@torch.no_grad()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int = 20,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    generated = input_ids
    if max_new_tokens == 0:
        return generated
    output = model(input_ids, attention_mask=attention_mask, use_cache=True)
    cache = output.cache
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    for step in range(max_new_tokens):
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        if eos_token_id is not None:
            eos = torch.full_like(next_token, eos_token_id)
            next_token = torch.where(finished[:, None], eos, next_token)
            finished |= next_token[:, 0].eq(eos_token_id)
        generated = torch.cat((generated, next_token), dim=-1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)), dim=-1
        )
        if (eos_token_id is not None and finished.all()) or step + 1 == max_new_tokens:
            break
        output = model(next_token, attention_mask=attention_mask, cache=cache, use_cache=True)
        cache = output.cache
    return generated
