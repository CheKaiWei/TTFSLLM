import torch
import torch.nn as nn

import dataclasses

from typing import Self


def value_to_ttfs(x: torch.Tensor, interval: tuple[float, float]) -> torch.Tensor:
    a, b = interval
    return (x - a) / (b - a)


def ttfs_to_value(t: torch.Tensor, interval: tuple[float, float]) -> torch.Tensor:
    a, b = interval
    return t * (b - a) + a


class SpikeParamDict:
    """
    All SpikeParams among all layers.
    """

    data: dict[str, tuple[float, float]]

    def __init__(self, value: dict[str, tuple[float, float]]):
        self.data = value

    def child(self, key: str) -> Self:
        key += "."
        return SpikeParamDict({k.removeprefix(key): v for k, v in self.data.items() if k.startswith(key)})

    def at(self, key: str) -> tuple[float, float]:
        try:
            return self.data[key]
        except:
            print("self.data =", self.data)
            raise
