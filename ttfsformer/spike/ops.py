import torch
import torch.nn as nn

import math

from .param import value_to_ttfs, ttfs_to_value
from .config import SpikeConfig
from .kernel import spiking_matmul_core


def add(
    x: torch.Tensor,
    y: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    out_range: tuple[float, float],
    config: SpikeConfig,
) -> torch.Tensor:
    x = ttfs_to_value(x, x_range)
    y = ttfs_to_value(y, y_range)
    out = value_to_ttfs(x + y, out_range)
    return config.emit(out)


def add_scalar(
    x: torch.Tensor, b: torch.Tensor, x_range: tuple[float, float], out_range: tuple[float, float], config: SpikeConfig
) -> torch.Tensor:
    x = ttfs_to_value(x, x_range)
    out = value_to_ttfs(x + b, out_range)
    return config.emit(out)


def matmul(
    x: torch.Tensor,
    y: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    out_range: tuple[float, float],
    config: SpikeConfig,
    scale: float = 1.0,
) -> torch.Tensor:
    x = ttfs_to_value(x, x_range)
    y = ttfs_to_value(y, y_range)
    max_abs = max(x_range[1], -x_range[0], y_range[1], -y_range[0])
    out = spiking_matmul_core(x, y, max_abs, config.precision or 0)
    out = value_to_ttfs(out * scale, out_range)
    return config.emit(out)


def hadamard(
    x: torch.Tensor,
    y: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    out_range: tuple[float, float],
    config: SpikeConfig,
    scale: float | torch.Tensor = 1.0,
    bias: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    x = ttfs_to_value(x, x_range)
    y = ttfs_to_value(y, y_range)
    mid = torch.log(x.abs() + 1) + torch.log(y.abs() + 1)
    mid_range = (0, math.log(max(x_range[1], -x_range[0]) + 1) + math.log(max(y_range[1], -y_range[0]) + 1))
    mid = config.emit(value_to_ttfs(mid, mid_range))
    mid = ttfs_to_value(mid, mid_range)
    out = mid.exp()
    out -= y.abs()
    out -= x.abs()
    out -= 1
    out *= torch.sign(x) * torch.sign(y)
    out = value_to_ttfs(out * scale + bias, out_range)
    return config.emit(out)


def hadamard_scalar(
    x: torch.Tensor,
    b: torch.Tensor,
    x_range: tuple[float, float],
    out_range: tuple[float, float],
    config: SpikeConfig,
) -> torch.Tensor:
    x = ttfs_to_value(x, x_range)
    out = value_to_ttfs(x * b, out_range)
    return config.emit(out)
