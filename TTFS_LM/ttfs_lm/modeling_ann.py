from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .calibration import Observer, observe
from .config import QwenConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, kv_heads, repeats, seq_len, head_dim)
    return expanded.reshape(batch, kv_heads * repeats, seq_len, head_dim)


def build_valid_attention_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    key_valid: torch.Tensor,
) -> torch.Tensor:
    causal = key_positions[:, None, None, :] <= query_positions[:, None, :, None]
    return causal & key_valid[:, None, None, :]


def masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid_mask = torch.broadcast_to(valid_mask.to(torch.bool), logits.shape)
    work = logits.float().masked_fill(~valid_mask, float("-inf"))
    has_value = valid_mask.any(dim=-1, keepdim=True)
    work = torch.where(has_value, work, torch.zeros_like(work))
    probabilities = torch.softmax(work, dim=-1)
    probabilities = torch.where(valid_mask, probabilities, torch.zeros_like(probabilities))
    return probabilities.to(logits.dtype)


@dataclass
class KVCacheEntry:
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor


class QwenCache:
    def __init__(self, num_layers: int):
        self.entries: list[KVCacheEntry | None] = [None] * num_layers

    def get_seq_length(self) -> int:
        for entry in self.entries:
            if entry is not None:
                return entry.key.shape[-2]
        return 0

    def key_valid(self) -> torch.Tensor | None:
        for entry in self.entries:
            if entry is not None:
                return entry.valid
        return None

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> KVCacheEntry:
        previous = self.entries[layer_idx]
        if previous is None:
            entry = KVCacheEntry(key, value, positions, valid)
        else:
            if previous.key.shape[:2] != key.shape[:2] or previous.key.shape[-1] != key.shape[-1]:
                raise ValueError("KV cache shape changed within one cache lifecycle")
            entry = KVCacheEntry(
                torch.cat((previous.key, key), dim=-2),
                torch.cat((previous.value, value), dim=-2),
                torch.cat((previous.positions, positions), dim=-1),
                torch.cat((previous.valid, valid), dim=-1),
            )
        self.entries[layer_idx] = entry
        return entry

    def reorder(self, beam_indices: torch.Tensor) -> None:
        for index, entry in enumerate(self.entries):
            if entry is not None:
                self.entries[index] = KVCacheEntry(
                    entry.key.index_select(0, beam_indices),
                    entry.value.index_select(0, beam_indices),
                    entry.positions.index_select(0, beam_indices),
                    entry.valid.index_select(0, beam_indices),
                )


@dataclass
class QwenCausalLMOutput:
    logits: torch.Tensor
    cache: QwenCache | None = None
    loss: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        observer: Observer | None = None,
        prefix: str = "norm",
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_float = hidden_states.float()
        square = observe(observer, f"{prefix}.square", hidden_float.pow(2), token_mask)
        variance = observe(observer, f"{prefix}.variance", square.mean(dim=-1, keepdim=True), token_mask)
        inv_rms = observe(
            observer,
            f"{prefix}.inv_rms",
            torch.rsqrt(variance + self.variance_epsilon),
            token_mask,
        )
        normalized = observe(observer, f"{prefix}.normalized", hidden_float * inv_rms, token_mask)
        output = self.weight * normalized.to(input_dtype)
        return observe(observer, f"{prefix}.output", output, token_mask)


class QwenRotaryEmbedding(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        positions = position_ids[:, None, :].float()
        freqs = (inv_freq @ positions).transpose(1, 2)
        embedding = torch.cat((freqs, freqs), dim=-1)
        return embedding.cos().to(x.dtype), embedding.sin().to(x.dtype)


class QwenMLP(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        observer: Observer | None,
        prefix: str,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        gate = observe(observer, f"{prefix}.gate_proj", self.gate_proj(hidden_states), token_mask)
        gate = observe(observer, f"{prefix}.gate_act", F.silu(gate), token_mask)
        up = observe(observer, f"{prefix}.up_proj", self.up_proj(hidden_states), token_mask)
        product = observe(observer, f"{prefix}.product", gate * up, token_mask)
        return observe(observer, f"{prefix}.down_proj", self.down_proj(product), token_mask)


class QwenAttention(nn.Module):
    def __init__(self, config: QwenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_key_value_groups
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        current_valid: torch.Tensor,
        cache: QwenCache | None,
        observer: Observer | None,
        prefix: str,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        token_mask = current_valid[:, :, None]
        query = observe(observer, f"{prefix}.q_proj", self.q_proj(hidden_states), token_mask)
        key = observe(observer, f"{prefix}.k_proj", self.k_proj(hidden_states), token_mask)
        value = observe(observer, f"{prefix}.v_proj", self.v_proj(hidden_states), token_mask)

        query = query.view(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        query = observe(observer, f"{prefix}.q_rope", query, current_valid[:, None, :, None])
        key = observe(observer, f"{prefix}.k_rope", key, current_valid[:, None, :, None])

        if cache is not None:
            entry = cache.update(self.layer_idx, key, value, position_ids, current_valid)
            key, value, key_positions, key_valid = entry.key, entry.value, entry.positions, entry.valid
        else:
            key_positions, key_valid = position_ids, current_valid

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        valid_mask = build_valid_attention_mask(position_ids, key_positions, key_valid)
        active_mask = valid_mask & current_valid[:, None, :, None]
        logits = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        logits = observe(observer, f"{prefix}.logits", logits, active_mask)
        probabilities = masked_softmax(logits, valid_mask)
        probabilities = observe(observer, f"{prefix}.probs", probabilities, active_mask)
        context = torch.matmul(probabilities, value)
        context = observe(observer, f"{prefix}.context", context, current_valid[:, None, :, None])
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return observe(observer, f"{prefix}.o_proj", self.o_proj(context), token_mask)


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = QwenAttention(config, layer_idx)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        current_valid: torch.Tensor,
        cache: QwenCache | None,
        observer: Observer | None,
    ) -> torch.Tensor:
        prefix = f"model.layers.{self.layer_idx}"
        token_mask = current_valid[:, :, None]
        residual = hidden_states
        normalized = self.input_layernorm(
            hidden_states, observer, f"{prefix}.input_layernorm", token_mask
        )
        attention = self.self_attn(
            normalized,
            position_embeddings,
            position_ids,
            current_valid,
            cache,
            observer,
            f"{prefix}.self_attn",
        )
        hidden_states = observe(observer, f"{prefix}.attn_residual", residual + attention, token_mask)

        residual = hidden_states
        normalized = self.post_attention_layernorm(
            hidden_states, observer, f"{prefix}.post_attention_layernorm", token_mask
        )
        mlp_output = self.mlp(normalized, observer, f"{prefix}.mlp", token_mask)
        return observe(observer, f"{prefix}.mlp_residual", residual + mlp_output, token_mask)


class QwenModel(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([QwenDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = QwenRotaryEmbedding(config)

    @staticmethod
    def _prepare_inputs(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        cache: QwenCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = input_ids.shape
        past_len = cache.get_seq_length() if cache is not None else 0
        if attention_mask is None:
            current_valid = torch.ones((batch, seq_len), dtype=torch.bool, device=input_ids.device)
        else:
            if attention_mask.shape[0] != batch:
                raise ValueError("attention_mask batch size does not match input_ids")
            if attention_mask.shape[-1] == seq_len:
                current_valid = attention_mask.to(torch.bool)
            elif attention_mask.shape[-1] == past_len + seq_len:
                previous_valid = cache.key_valid() if cache is not None else None
                if previous_valid is not None and not torch.equal(
                    attention_mask[:, :past_len].to(torch.bool), previous_valid
                ):
                    raise ValueError("attention_mask history does not match KV cache")
                current_valid = attention_mask[:, -seq_len:].to(torch.bool)
            else:
                raise ValueError("attention_mask must cover the current tokens or the complete cached sequence")

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0)
            position_ids = position_ids.expand(batch, -1)
        elif position_ids.shape == (1, seq_len) and batch != 1:
            position_ids = position_ids.expand(batch, -1)
        elif position_ids.shape != (batch, seq_len):
            raise ValueError("position_ids must have shape [batch, current_sequence]")
        return current_valid, position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: QwenCache | None = None,
        use_cache: bool = False,
        observer: Observer | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, QwenCache | None, tuple[torch.Tensor, ...] | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if use_cache and cache is None:
            cache = QwenCache(self.config.num_hidden_layers)
        if not use_cache and cache is not None:
            raise ValueError("A cache was provided while use_cache=False")
        current_valid, position_ids = self._prepare_inputs(input_ids, attention_mask, position_ids, cache)
        token_mask = current_valid[:, :, None]
        hidden_states = observe(observer, "model.embed_tokens", self.embed_tokens(input_ids), token_mask)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        collected = [hidden_states] if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                position_ids,
                current_valid,
                cache,
                observer,
            )
            if collected is not None:
                collected.append(hidden_states)
        hidden_states = self.norm(hidden_states, observer, "model.norm", token_mask)
        if collected is not None:
            collected[-1] = hidden_states
        return hidden_states, cache if use_cache else None, tuple(collected) if collected is not None else None


class QwenForCausalLM(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.model = QwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: QwenCache | None = None,
        use_cache: bool | None = None,
        labels: torch.Tensor | None = None,
        observer: Observer | None = None,
        output_hidden_states: bool = False,
    ) -> QwenCausalLMOutput:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        hidden_states, cache, collected = self.model(
            input_ids,
            attention_mask,
            position_ids,
            cache,
            use_cache,
            observer,
            output_hidden_states,
        )
        token_mask = None
        if attention_mask is not None:
            token_mask = attention_mask[:, -input_ids.shape[1] :, None].to(torch.bool)
        logits = observe(observer, "lm_head", self.lm_head(hidden_states), token_mask)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return QwenCausalLMOutput(logits=logits, cache=cache, loss=loss, hidden_states=collected)
