from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .calibration import RangeRegistry
from .config import TTFSConfig
from .modeling_ann import (
    QwenCausalLMOutput,
    QwenForCausalLM,
    apply_rotary_pos_emb,
    build_valid_attention_mask,
    masked_softmax,
    repeat_kv,
)


class RangeViolationError(ValueError):
    pass


@dataclass(frozen=True)
class SpikeTensor:
    time: torch.Tensor
    range_name: str
    value_dtype: torch.dtype


@dataclass
class RangeTelemetry:
    calls: int = 0
    elements: int = 0
    below: int = 0
    above: int = 0
    max_underflow: float = 0.0
    max_overflow: float = 0.0


class TTFSCodec:
    def __init__(self, registry: RangeRegistry, config: TTFSConfig):
        self.registry = registry
        self.config = config
        self.telemetry: dict[str, RangeTelemetry] = {}

    def reset_telemetry(self) -> None:
        self.telemetry.clear()

    @staticmethod
    def _active_mask(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(value, dtype=torch.bool)
        try:
            return torch.broadcast_to(mask.to(device=value.device, dtype=torch.bool), value.shape)
        except RuntimeError as exc:
            raise ValueError(f"Mask shape {tuple(mask.shape)} cannot broadcast to {tuple(value.shape)}") from exc

    def encode(self, name: str, value: torch.Tensor, mask: torch.Tensor | None = None) -> SpikeTensor:
        spec = self.registry[name]
        time_dtype = torch.float64 if self.config.precision is None else torch.float32
        work_value = value.to(time_dtype)
        active = self._active_mask(work_value, mask)
        if not torch.isfinite(work_value[active]).all():
            raise ValueError(f"Non-finite value passed to TTFS encoder at {name}")

        selected = work_value[active]
        below = selected < spec.lower
        above = selected > spec.upper
        stats = self.telemetry.setdefault(name, RangeTelemetry())
        stats.calls += 1
        stats.elements += selected.numel()
        if selected.numel():
            below_count = int(below.sum().item())
            above_count = int(above.sum().item())
            stats.below += below_count
            stats.above += above_count
            if below_count:
                stats.max_underflow = max(stats.max_underflow, float((spec.lower - selected[below]).max().item()))
            if above_count:
                stats.max_overflow = max(stats.max_overflow, float((selected[above] - spec.upper).max().item()))
            if self.config.strict_ranges and (below_count or above_count):
                raise RangeViolationError(
                    f"{name} exceeded [{spec.lower:.7g}, {spec.upper:.7g}]: "
                    f"below={below_count}, above={above_count}"
                )

        safe_value = torch.where(active, work_value, torch.full_like(work_value, spec.lower))
        time = (safe_value - spec.lower) / (spec.upper - spec.lower)
        if self.config.precision is not None:
            time = torch.round(time * self.config.precision) / self.config.precision
        time = time.clamp(0.0, 1.0)
        return SpikeTensor(time=time, range_name=name, value_dtype=value.dtype)

    def decode(self, spike: SpikeTensor) -> torch.Tensor:
        spec = self.registry[spike.range_name]
        value = spike.time * (spec.upper - spec.lower) + spec.lower
        return value.to(spike.value_dtype)


def _reshape_spike(spike: SpikeTensor, time: torch.Tensor) -> SpikeTensor:
    return SpikeTensor(time=time, range_name=spike.range_name, value_dtype=spike.value_dtype)


def _cat_spikes(left: SpikeTensor, right: SpikeTensor, dim: int) -> SpikeTensor:
    if left.range_name != right.range_name or left.value_dtype != right.value_dtype:
        raise ValueError("Cached TTFS tensors use incompatible RangeSpec identities")
    return _reshape_spike(left, torch.cat((left.time, right.time), dim=dim))


@dataclass
class TTFSKVCacheEntry:
    key: SpikeTensor
    value: SpikeTensor
    positions: torch.Tensor
    valid: torch.Tensor


class TTFSCache:
    def __init__(self, num_layers: int):
        self.entries: list[TTFSKVCacheEntry | None] = [None] * num_layers

    def get_seq_length(self) -> int:
        for entry in self.entries:
            if entry is not None:
                return entry.key.time.shape[-2]
        return 0

    def key_valid(self) -> torch.Tensor | None:
        for entry in self.entries:
            if entry is not None:
                return entry.valid
        return None

    def update(
        self,
        layer_idx: int,
        key: SpikeTensor,
        value: SpikeTensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> TTFSKVCacheEntry:
        previous = self.entries[layer_idx]
        if previous is None:
            entry = TTFSKVCacheEntry(key, value, positions, valid)
        else:
            entry = TTFSKVCacheEntry(
                _cat_spikes(previous.key, key, dim=-2),
                _cat_spikes(previous.value, value, dim=-2),
                torch.cat((previous.positions, positions), dim=-1),
                torch.cat((previous.valid, valid), dim=-1),
            )
        self.entries[layer_idx] = entry
        return entry

    def reorder(self, beam_indices: torch.Tensor) -> None:
        for index, entry in enumerate(self.entries):
            if entry is not None:
                self.entries[index] = TTFSKVCacheEntry(
                    _reshape_spike(entry.key, entry.key.time.index_select(0, beam_indices)),
                    _reshape_spike(entry.value, entry.value.time.index_select(0, beam_indices)),
                    entry.positions.index_select(0, beam_indices),
                    entry.valid.index_select(0, beam_indices),
                )


class TTFSQwenForCausalLM(nn.Module):
    """Decode-compute-encode TTFS reference simulator for a calibrated Qwen ANN."""

    def __init__(self, ann: QwenForCausalLM, registry: RangeRegistry, config: TTFSConfig | None = None):
        super().__init__()
        self.ann = ann
        self.config = config or TTFSConfig()
        missing = sorted(self.required_range_names() - set(registry.names()))
        if missing:
            preview = ", ".join(missing[:8])
            suffix = " ..." if len(missing) > 8 else ""
            raise ValueError(f"Calibration registry is missing {len(missing)} ranges: {preview}{suffix}")
        self.codec = TTFSCodec(registry, self.config)

    @property
    def qwen_config(self):
        return self.ann.config

    def required_range_names(self) -> set[str]:
        names = {"model.embed_tokens", "model.norm.square", "model.norm.variance", "model.norm.inv_rms"}
        names.update({"model.norm.normalized", "model.norm.output"})
        norm_suffixes = {"square", "variance", "inv_rms", "normalized", "output"}
        attention_suffixes = {"q_proj", "k_proj", "v_proj", "q_rope", "k_rope", "logits", "probs", "context", "o_proj"}
        mlp_suffixes = {"gate_proj", "gate_act", "up_proj", "product", "down_proj"}
        for layer_idx in range(self.qwen_config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                names.update(f"{prefix}.{norm_name}.{suffix}" for suffix in norm_suffixes)
            names.update(f"{prefix}.self_attn.{suffix}" for suffix in attention_suffixes)
            names.update(f"{prefix}.mlp.{suffix}" for suffix in mlp_suffixes)
            names.update({f"{prefix}.attn_residual", f"{prefix}.mlp_residual"})
        if self.config.spike_lm_head:
            names.add("lm_head")
        return names

    def _linear(
        self,
        spike: SpikeTensor,
        linear: nn.Linear,
        name: str,
        mask: torch.Tensor | None,
    ) -> SpikeTensor:
        return self.codec.encode(name, linear(self.codec.decode(spike)), mask)

    def _rms_norm(
        self,
        spike: SpikeTensor,
        norm: nn.Module,
        prefix: str,
        token_mask: torch.Tensor,
    ) -> SpikeTensor:
        hidden = self.codec.decode(spike)
        input_dtype = hidden.dtype
        square = self.codec.encode(f"{prefix}.square", hidden.float().pow(2), token_mask)
        variance_value = self.codec.decode(square).mean(dim=-1, keepdim=True)
        variance = self.codec.encode(f"{prefix}.variance", variance_value, token_mask)
        inv_value = torch.rsqrt(self.codec.decode(variance).float() + norm.variance_epsilon)
        inv_rms = self.codec.encode(f"{prefix}.inv_rms", inv_value, token_mask)
        normalized_value = hidden.float() * self.codec.decode(inv_rms).float()
        normalized = self.codec.encode(f"{prefix}.normalized", normalized_value, token_mask)
        output = norm.weight * self.codec.decode(normalized).to(input_dtype)
        return self.codec.encode(f"{prefix}.output", output, token_mask)

    def _attention(
        self,
        hidden: SpikeTensor,
        layer_idx: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        current_valid: torch.Tensor,
        cache: TTFSCache | None,
    ) -> SpikeTensor:
        attention = self.ann.model.layers[layer_idx].self_attn
        prefix = f"model.layers.{layer_idx}.self_attn"
        batch, seq_len = position_ids.shape
        token_mask = current_valid[:, :, None]

        query = self._linear(hidden, attention.q_proj, f"{prefix}.q_proj", token_mask)
        key = self._linear(hidden, attention.k_proj, f"{prefix}.k_proj", token_mask)
        value = self._linear(hidden, attention.v_proj, f"{prefix}.v_proj", token_mask)
        query = _reshape_spike(
            query,
            query.time.view(batch, seq_len, self.qwen_config.num_attention_heads, attention.head_dim).transpose(1, 2),
        )
        key = _reshape_spike(
            key,
            key.time.view(batch, seq_len, self.qwen_config.num_key_value_heads, attention.head_dim).transpose(1, 2),
        )
        value = _reshape_spike(
            value,
            value.time.view(batch, seq_len, self.qwen_config.num_key_value_heads, attention.head_dim).transpose(1, 2),
        )

        query_value, key_value = apply_rotary_pos_emb(
            self.codec.decode(query), self.codec.decode(key), *position_embeddings
        )
        head_mask = current_valid[:, None, :, None]
        query = self.codec.encode(f"{prefix}.q_rope", query_value, head_mask)
        key = self.codec.encode(f"{prefix}.k_rope", key_value, head_mask)

        if cache is not None:
            entry = cache.update(layer_idx, key, value, position_ids, current_valid)
            key, value, key_positions, key_valid = entry.key, entry.value, entry.positions, entry.valid
        else:
            key_positions, key_valid = position_ids, current_valid

        key = _reshape_spike(key, repeat_kv(key.time, attention.num_key_value_groups))
        value = _reshape_spike(value, repeat_kv(value.time, attention.num_key_value_groups))
        valid_mask = build_valid_attention_mask(position_ids, key_positions, key_valid)
        active_mask = valid_mask & current_valid[:, None, :, None]
        logits_value = torch.matmul(self.codec.decode(query), self.codec.decode(key).transpose(2, 3))
        logits_value = logits_value * attention.scaling
        logits = self.codec.encode(f"{prefix}.logits", logits_value, active_mask)
        probabilities_value = masked_softmax(self.codec.decode(logits), valid_mask)
        probabilities = self.codec.encode(f"{prefix}.probs", probabilities_value, active_mask)
        context_value = torch.matmul(self.codec.decode(probabilities), self.codec.decode(value))
        context = self.codec.encode(f"{prefix}.context", context_value, head_mask)
        context = _reshape_spike(
            context,
            context.time.transpose(1, 2).contiguous().view(batch, seq_len, -1),
        )
        return self._linear(context, attention.o_proj, f"{prefix}.o_proj", token_mask)

    def _mlp(
        self,
        hidden: SpikeTensor,
        layer_idx: int,
        token_mask: torch.Tensor,
    ) -> SpikeTensor:
        mlp = self.ann.model.layers[layer_idx].mlp
        prefix = f"model.layers.{layer_idx}.mlp"
        gate = self._linear(hidden, mlp.gate_proj, f"{prefix}.gate_proj", token_mask)
        gate_act = self.codec.encode(f"{prefix}.gate_act", F.silu(self.codec.decode(gate)), token_mask)
        up = self._linear(hidden, mlp.up_proj, f"{prefix}.up_proj", token_mask)
        product_value = self.codec.decode(gate_act) * self.codec.decode(up)
        product = self.codec.encode(f"{prefix}.product", product_value, token_mask)
        return self._linear(product, mlp.down_proj, f"{prefix}.down_proj", token_mask)

    def _add(
        self,
        left: SpikeTensor,
        right: SpikeTensor,
        name: str,
        token_mask: torch.Tensor,
    ) -> SpikeTensor:
        return self.codec.encode(name, self.codec.decode(left) + self.codec.decode(right), token_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: TTFSCache | None = None,
        use_cache: bool | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> QwenCausalLMOutput:
        use_cache = self.qwen_config.use_cache if use_cache is None else use_cache
        if use_cache and cache is None:
            cache = TTFSCache(self.qwen_config.num_hidden_layers)
        if not use_cache and cache is not None:
            raise ValueError("A cache was provided while use_cache=False")
        current_valid, position_ids = self.ann.model._prepare_inputs(
            input_ids, attention_mask, position_ids, cache
        )
        token_mask = current_valid[:, :, None]
        embedding = self.ann.model.embed_tokens(input_ids)
        hidden = self.codec.encode("model.embed_tokens", embedding, token_mask)
        position_embeddings = self.ann.model.rotary_emb(embedding, position_ids)
        collected = [self.codec.decode(hidden)] if output_hidden_states else None

        for layer_idx, layer in enumerate(self.ann.model.layers):
            prefix = f"model.layers.{layer_idx}"
            residual = hidden
            normalized = self._rms_norm(
                hidden, layer.input_layernorm, f"{prefix}.input_layernorm", token_mask
            )
            attention = self._attention(
                normalized, layer_idx, position_embeddings, position_ids, current_valid, cache
            )
            hidden = self._add(residual, attention, f"{prefix}.attn_residual", token_mask)
            residual = hidden
            normalized = self._rms_norm(
                hidden, layer.post_attention_layernorm, f"{prefix}.post_attention_layernorm", token_mask
            )
            mlp = self._mlp(normalized, layer_idx, token_mask)
            hidden = self._add(residual, mlp, f"{prefix}.mlp_residual", token_mask)
            if collected is not None:
                collected.append(self.codec.decode(hidden))

        hidden = self._rms_norm(hidden, self.ann.model.norm, "model.norm", token_mask)
        hidden_value = self.codec.decode(hidden)
        if collected is not None:
            collected[-1] = hidden_value
        logits = self.ann.lm_head(hidden_value)
        if self.config.spike_lm_head:
            logits = self.codec.decode(self.codec.encode("lm_head", logits, token_mask))

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.qwen_config.vocab_size), shift_labels.view(-1))
        return QwenCausalLMOutput(
            logits=logits,
            cache=cache if use_cache else None,
            loss=loss,
            hidden_states=tuple(collected) if collected is not None else None,
        )
