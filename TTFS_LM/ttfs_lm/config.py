from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


@dataclass
class QwenConfig:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    bos_token_id: int | None = 151643
    eos_token_id: int | None = 151643
    pad_token_id: int | None = None
    tie_word_embeddings: bool = True
    use_cache: bool = True

    def __post_init__(self) -> None:
        if self.hidden_act != "silu":
            raise ValueError(f"Only Qwen2 SiLU is supported, got {self.hidden_act!r}")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if min(self.vocab_size, self.intermediate_size, self.num_hidden_layers) <= 0:
            raise ValueError("model dimensions and layer count must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf_config(cls, config: Any) -> "QwenConfig":
        if getattr(config, "model_type", None) != "qwen2":
            raise ValueError(f"Expected a qwen2 checkpoint, got {getattr(config, 'model_type', None)!r}")
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 1_000_000.0))
        return cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=float(rope_theta),
            rms_norm_eps=config.rms_norm_eps,
            attention_dropout=getattr(config, "attention_dropout", 0.0),
            hidden_act=config.hidden_act,
            bos_token_id=getattr(config, "bos_token_id", None),
            eos_token_id=getattr(config, "eos_token_id", None),
            pad_token_id=getattr(config, "pad_token_id", None),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            use_cache=getattr(config, "use_cache", True),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TTFSConfig:
    precision: int | None = None
    strict_ranges: bool = True
    spike_lm_head: bool = False

    def __post_init__(self) -> None:
        if self.precision is not None and self.precision <= 0:
            raise ValueError("precision must be positive or None for continuous time")
