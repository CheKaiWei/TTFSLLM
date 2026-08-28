from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


class Observer(Protocol):
    def observe(self, name: str, value: torch.Tensor, mask: torch.Tensor | None = None) -> None: ...


@dataclass(frozen=True)
class RangeSpec:
    name: str
    lower: float
    upper: float
    calls: int
    elements: int

    def __post_init__(self) -> None:
        if not self.lower < self.upper:
            raise ValueError(f"Invalid range for {self.name}: [{self.lower}, {self.upper}]")


class RangeRegistry:
    SCHEMA_VERSION = 1

    def __init__(self, specs: dict[str, RangeSpec], metadata: dict[str, Any] | None = None):
        self._specs = dict(specs)
        self.metadata = dict(metadata or {})

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> RangeSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Missing calibration range {name!r}") from exc

    def __len__(self) -> int:
        return len(self._specs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "metadata": self.metadata,
            "ranges": [asdict(self._specs[name]) for name in sorted(self._specs)],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RangeRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported RangeRegistry schema: {payload.get('schema_version')}")
        specs = {item["name"]: RangeSpec(**item) for item in payload["ranges"]}
        return cls(specs, payload.get("metadata"))


@dataclass
class _RunningRange:
    lower: float
    upper: float
    calls: int
    elements: int


class MinMaxCalibrator:
    """Streaming, mask-aware calibration for the P0 correctness baseline."""

    def __init__(self):
        self._ranges: dict[str, _RunningRange] = {}

    @staticmethod
    def _select(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        detached = value.detach()
        if mask is None:
            return detached.reshape(-1)
        mask = mask.detach().to(device=detached.device, dtype=torch.bool)
        try:
            mask = torch.broadcast_to(mask, detached.shape)
        except RuntimeError as exc:
            raise ValueError(f"Mask shape {tuple(mask.shape)} cannot broadcast to {tuple(detached.shape)}") from exc
        return detached[mask]

    def observe(self, name: str, value: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        selected = self._select(value, mask)
        if selected.numel() == 0:
            return
        if not torch.isfinite(selected).all():
            raise ValueError(f"Non-finite activation observed at {name}")
        lower, upper = torch.aminmax(selected.float())
        lo = lower.item()
        hi = upper.item()
        previous = self._ranges.get(name)
        if previous is None:
            self._ranges[name] = _RunningRange(lo, hi, 1, selected.numel())
        else:
            previous.lower = min(previous.lower, lo)
            previous.upper = max(previous.upper, hi)
            previous.calls += 1
            previous.elements += selected.numel()

    def finalize(
        self,
        margin: float = 0.01,
        metadata: dict[str, Any] | None = None,
        fixed_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> RangeRegistry:
        if margin < 0:
            raise ValueError("margin must be non-negative")
        fixed_ranges = fixed_ranges or {}
        specs: dict[str, RangeSpec] = {}
        for name, running in self._ranges.items():
            if name in fixed_ranges:
                lower, upper = fixed_ranges[name]
            elif name.endswith(".probs"):
                lower, upper = 0.0, 1.0
            else:
                width = running.upper - running.lower
                if width == 0:
                    radius = max(abs(running.lower) * 1e-6, 1e-6)
                    lower, upper = running.lower - radius, running.upper + radius
                else:
                    lower = running.lower - margin * width
                    upper = running.upper + margin * width
            if name.endswith((".square", ".variance", ".inv_rms")):
                lower = max(0.0, lower)
            specs[name] = RangeSpec(name, float(lower), float(upper), running.calls, running.elements)
        if not specs:
            raise ValueError("No activations were observed")
        return RangeRegistry(specs, metadata)


def observe(
    observer: Observer | None,
    name: str,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if observer is not None:
        observer.observe(name, value, mask)
    return value
