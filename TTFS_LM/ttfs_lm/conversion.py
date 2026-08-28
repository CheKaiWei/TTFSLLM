from __future__ import annotations

from .calibration import RangeRegistry
from .config import TTFSConfig
from .modeling_ann import QwenForCausalLM
from .modeling_ttfs import TTFSQwenForCausalLM


def convert_ann_to_ttfs(
    ann: QwenForCausalLM,
    registry: RangeRegistry,
    config: TTFSConfig | None = None,
) -> TTFSQwenForCausalLM:
    """Convert a calibrated local Qwen ANN into the functional TTFS backend."""

    provenance = getattr(ann, "ttfs_provenance", {})
    for key in ("model_id", "revision"):
        expected = provenance.get(key)
        actual = registry.metadata.get(key)
        if expected is not None and actual is not None and expected != actual:
            raise ValueError(f"Calibration {key}={actual!r} does not match ANN {key}={expected!r}")
    return TTFSQwenForCausalLM(ann, registry, config)
