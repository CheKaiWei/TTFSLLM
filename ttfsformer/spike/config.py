import torch
import torch.nn as nn

import dataclasses


@dataclasses.dataclass
class SpikeConfig:
    """
    All SpikingLayers share the same SpikeConfig.
    """

    precision: int | None = None
    noise: float | None = None

    def emit(self, value: torch.Tensor) -> torch.Tensor:
        """
        Convert the theoretical emitting time to the real emitting time.
        Simulate the precision loss.
        """
        if self.precision is not None:
            return value.mul_(self.precision).round_().div_(self.precision).clip_(0, 1)
        elif self.noise is not None:
            noise = torch.normal(0, self.noise, size=value.shape, device=value.device, dtype=value.dtype)
            return value.add_(noise).clip_(0, 1)
        return value
