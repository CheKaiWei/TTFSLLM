import torch
import torch.nn as nn

from .config import SpikeConfig
from .module import SpikingModule, SpikingLayerNorm, SpikingSoftmax
from .param import SpikeParamDict


class SpikingWrapped(SpikingModule):

    def __init__(
        self, inner: nn.Module, config: SpikeConfig, in_range: tuple[float, float], out_range: tuple[float, float]
    ):
        super().__init__(config)
        self.inner = inner
        self.in_range = in_range
        self.out_range = out_range

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        in_0, in_1 = self.in_range
        out_0, out_1 = self.out_range
        x = t * (in_1 - in_0) + in_0
        x = self.inner(x)
        x = (x - out_0) / (out_1 - out_0)
        return self.config.emit(x)


def ann_to_snn(
    model: nn.Module,
    config: SpikeConfig,
    in_range: tuple[float, float],
    out_range: tuple[float, float],
    param: SpikeParamDict | None = None,
) -> SpikingModule:

    if isinstance(model, nn.Identity):
        assert in_range == out_range
        return nn.Identity()
    elif isinstance(model, (nn.Conv2d, nn.Linear, nn.ReLU, nn.GELU, nn.SiLU)):
        return SpikingWrapped(model, config, in_range, out_range)
    elif isinstance(model, nn.LayerNorm):
        return SpikingLayerNorm(model, config, in_range, out_range, param)
    elif isinstance(model, nn.Softmax):
        return SpikingSoftmax(model, config, in_range, param.at("monitor_1"))
    else:
        raise NotImplementedError(f"Not implemented for model of type {type(model)}")
