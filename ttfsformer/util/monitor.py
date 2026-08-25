import torch
import torch.nn as nn

from typing import Any

__all__ = ["BaseMonitor", "PercentileMonitor", "RangeMonitor", "MinMaxMonitor"]


class BaseMonitor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def monitor_data(self) -> Any:
        raise NotImplementedError()


class PercentileMonitor(BaseMonitor):

    def __init__(self, percentile: list[float]):
        super().__init__()
        self.num = 0
        self.percentile = torch.tensor(percentile, device="cuda")
        self.value = torch.zeros(len(percentile), device="cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.value += torch.quantile(x, self.percentile)
        self.num += 1
        return x

    def monitor_data(self):
        return (self.value / self.num).tolist()


class RangeMonitor(BaseMonitor):

    def __init__(self):
        super().__init__()
        self.min = None
        self.max = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_min, current_max = torch.aminmax(x)
        current_min = current_min.detach()
        current_max = current_max.detach()
        self.min = current_min if self.min is None else torch.minimum(self.min, current_min)
        self.max = current_max if self.max is None else torch.maximum(self.max, current_max)
        return x

    def monitor_data(self):
        return (self.min.item(), self.max.item()) if self.min is not None else None


class MinMaxMonitor(BaseMonitor):

    def __init__(self):
        super().__init__()
        self.min = None
        self.max = None
        self.num = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate as float64 to match the old Python-float summation while
        # avoiding a device synchronization at every monitored activation.
        current_min, current_max = torch.aminmax(x)
        current_min = current_min.detach().double()
        current_max = current_max.detach().double()
        self.min = current_min if self.min is None else self.min + current_min
        self.max = current_max if self.max is None else self.max + current_max
        self.num += 1
        return x

    def monitor_data(self):
        return ((self.min / self.num).item(), (self.max / self.num).item()) if self.num else None


def get_monitor_data(model: nn.Module) -> dict[str, Any]:
    data = {}
    for name, child in model.named_modules():
        if isinstance(child, BaseMonitor):
            info = child.monitor_data()
            if info is not None:
                data[name] = info
    return data
