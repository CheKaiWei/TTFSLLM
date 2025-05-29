import torch
import torch.nn as nn

from .config import SpikeConfig
from .param import ttfs_to_value, value_to_ttfs, SpikeParamDict
from . import ops, kernel


class SpikingModule(nn.Module):

    def __init__(self, config: SpikeConfig):
        super().__init__()
        self.config = config


class SpikingLayerNorm(SpikingModule):

    def __init__(
        self,
        ann: nn.LayerNorm,
        config: SpikeConfig,
        in_range: tuple[float, float],
        out_range: tuple[float, float],
        param: SpikeParamDict,
    ):
        super().__init__(config)
        assert isinstance(ann, nn.LayerNorm)
        self.normalized_shape = ann.normalized_shape
        self.dim = tuple(range(-len(self.normalized_shape), 0))
        self.eps = ann.eps
        self.weight = ann.weight
        self.bias = ann.bias
        self.in_range = in_range
        self.x_range = param.at("monitor_1")
        self.inv_std_range = param.at("monitor_2")
        self.out_range = out_range

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = ttfs_to_value(t, self.in_range)
        # Layer 1: substract
        mean = x.mean(self.dim, keepdim=True)
        x.sub_(mean)
        t = self.config.emit(value_to_ttfs(x, self.x_range))
        # Layer 2: Invert
        x = ttfs_to_value(t, self.x_range)
        inv_std = (x.var(self.dim, keepdim=True) + self.eps) ** (-1 / 2)
        inv_std = self.config.emit(value_to_ttfs(inv_std, self.inv_std_range))
        # Layer 3/4: Multiply
        # out = x * inv_std
        # out = out * self.weight + self.bias
        out = ops.hadamard(
            t, inv_std, self.x_range, self.inv_std_range, self.out_range, self.config, self.weight, self.bias
        )
        return out


class SpikingSoftmax(SpikingModule):

    def __init__(
        self, ann: nn.Softmax, config: SpikeConfig, in_range: tuple[float, float], mid_range: tuple[float, float]
    ):
        super().__init__(config)
        assert isinstance(ann, nn.Softmax)
        self.dim = ann.dim
        self.in_range = in_range
        self.mid_range = mid_range

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = ttfs_to_value(t, self.in_range)
        x -= torch.max(x, dim=self.dim, keepdim=True).values
        lse = x.exp().sum(dim=self.dim, keepdim=True).log()
        lse = ttfs_to_value(self.config.emit(value_to_ttfs(lse, self.mid_range)), self.mid_range)
        x -= lse
        x.exp_()
        return self.config.emit(x)
