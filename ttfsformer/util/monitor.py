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
        self.min = torch.inf
        self.max = -torch.inf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.min = min(self.min, torch.min(x).item())
        self.max = max(self.max, torch.max(x).item())
        return x

    def monitor_data(self):
        return (self.min, self.max)


class MinMaxMonitor(BaseMonitor):

    def __init__(self):
        super().__init__()
        self.min = 0.0
        self.max = 0.0
        self.num = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.min += torch.min(x).item()
        self.max += torch.max(x).item()
        self.num += 1
        return x

    def monitor_data(self):
        return (self.min / self.num, self.max / self.num) if self.num else None


def get_monitor_data(model: nn.Module) -> dict[str, Any]:
    data = {}
    for name, child in model.named_modules():
        if isinstance(child, BaseMonitor):
            info = child.monitor_data()
            if info is not None:
                data[name] = info
    return data
