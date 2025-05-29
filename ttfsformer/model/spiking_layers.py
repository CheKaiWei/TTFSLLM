import torch
import torch.nn as nn

from . import layers
from ..spike import SpikeConfig, SpikeParamDict, ann_to_snn, SpikingModule, ttfs_to_value, value_to_ttfs, ops


class SpikingPatchEmbed(SpikingModule):

    # meta-data
    img_size: tuple[int, int]
    patch_size: tuple[int, int]
    grid_size: tuple[int, int]
    num_patches: int
    # layers
    proj: nn.Module
    norm: nn.Module

    def __init__(self, ann: layers.PatchEmbed, config: SpikeConfig, param: SpikeParamDict):
        assert isinstance(ann, layers.PatchEmbed)
        super().__init__(config)
        self.param = param
        self.img_size = ann.img_size
        self.patch_size = ann.patch_size
        self.grid_size = ann.grid_size
        self.num_patches = ann.num_patches
        self.proj = ann_to_snn(ann.proj, config, param.at("monitor_0"), param.at("monitor_1"))
        self.norm = ann_to_snn(ann.norm, config, param.at("monitor_1"), param.at("monitor_o"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class SpikingMlp(SpikingModule):
    def __init__(self, ann: layers.Mlp, config: SpikeConfig, param: SpikeParamDict):
        assert isinstance(ann, layers.Mlp)
        super().__init__(config)
        self.param = param
        self.fc1 = ann_to_snn(ann.fc1, config, param.at("monitor_0"), param.at("monitor_1"))
        self.act = ann_to_snn(ann.act, config, param.at("monitor_1"), param.at("monitor_2"))
        self.norm = ann_to_snn(ann.norm, config, param.at("monitor_2"), param.at("monitor_3"), param.child("norm"))
        self.fc2 = ann_to_snn(ann.fc2, config, param.at("monitor_3"), param.at("monitor_o"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class SpikingGluMlp(SpikingModule):

    def __init__(self, ann: layers.GluMlp, config: SpikeConfig, param: SpikeParamDict):
        assert isinstance(ann, layers.GluMlp)
        super().__init__(config)
        self.param = param
        self.chunk_dim = ann.chunk_dim
        self.gate_last = ann.gate_last
        self.fc1 = ann_to_snn(ann.fc1, config, param.at("monitor_0"), param.at("monitor_1"))
        self.act = ann_to_snn(ann.act, config, param.at("monitor_1"), param.at("monitor_2"))
        self.norm = ann_to_snn(ann.norm, config, param.at("monitor_3"), param.at("monitor_4"))
        self.fc2 = ann_to_snn(ann.fc2, config, param.at("monitor_4"), param.at("monitor_o"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        x = ops.hadamard(
            x1 if self.gate_last else self.act(x1),
            self.act(x2) if self.gate_last else x2,
            self.param.at("monitor_1") if self.gate_last else self.param.at("monitor_2"),
            self.param.at("monitor_2") if self.gate_last else self.param.at("monitor_1"),
            self.param.at("monitor_3"),
            self.config,
        )
        x = self.norm(x)
        x = self.fc2(x)
        return x


class SpikingSwiGLU(SpikingModule):

    def __init__(self, ann: layers.SwiGLU, config: SpikeConfig, param: SpikeParamDict):
        assert isinstance(ann, layers.SwiGLU)
        super().__init__(config)
        self.param = param
        self.fc1_g = ann_to_snn(ann.fc1_g, config, param.at("monitor_0"), param.at("monitor_1"))
        self.act = ann_to_snn(ann.act, config, param.at("monitor_1"), param.at("monitor_2"))
        self.fc1_x = ann_to_snn(ann.fc1_x, config, param.at("monitor_0"), param.at("monitor_3"))
        self.norm = ann_to_snn(ann.norm, config, param.at("monitor_4"), param.at("monitor_5"), param.child("norm"))
        self.fc2 = ann_to_snn(ann.fc2, config, param.at("monitor_5"), param.at("monitor_o"))

    def forward(self, x):
        x_gate = self.act(self.fc1_g(x))
        x = self.fc1_x(x)
        x = ops.hadamard(
            x_gate,
            x,
            self.param.at("monitor_2"),
            self.param.at("monitor_3"),
            self.param.at("monitor_4"),
            self.config,
        )
        x = self.norm(x)
        x = self.fc2(x)
        return x


def rot(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def spiking_apply_rot_embed_cat(
    t: torch.Tensor,
    emb: torch.Tensor,
    in_range: tuple[float, float],
    out_range: tuple[float, float],
    config: SpikeConfig,
) -> torch.Tensor:
    sin_emb, cos_emb = emb.tensor_split(2, -1)
    x = ttfs_to_value(t, in_range)
    if sin_emb.ndim == 3:
        out = x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(1).expand_as(x)
    else:
        out = x * cos_emb + rot(x) * sin_emb
    out = value_to_ttfs(out, out_range)
    return config.emit(out)
