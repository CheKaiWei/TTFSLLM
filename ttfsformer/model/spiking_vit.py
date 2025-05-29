import torch
import torch.nn as nn

from . import vit
from ..spike import (
    SpikeConfig,
    SpikeParamDict,
    ann_to_snn,
    SpikingModule,
    value_to_ttfs,
    ttfs_to_value,
    ops,
)
from .spiking_layers import SpikingMlp, SpikingPatchEmbed


class SpikingAttention(SpikingModule):
    def __init__(self, ann: vit.Attention, config: SpikeConfig, param: SpikeParamDict):
        assert isinstance(ann, vit.Attention)
        super().__init__(config)
        self.param = param
        self.num_heads = ann.num_heads
        self.scale = ann.scale
        self.qkv = ann_to_snn(ann.qkv, config, param.at("monitor_0"), param.at("monitor_1"))
        self.softmax = ann_to_snn(ann.softmax, config, param.at("monitor_2"), param.at("monitor_3"), param)
        self.proj = ann_to_snn(ann.proj, config, param.at("monitor_4"), param.at("monitor_o"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = torch.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ops.matmul(
            q,
            k.transpose(-2, -1),
            self.param.at("monitor_1"),
            self.param.at("monitor_1"),
            self.param.at("monitor_2"),
            self.config,
            scale=self.scale,
        )
        attn = self.softmax(attn)
        x = ops.matmul(
            attn,
            v,
            (0, 1),  # self.param.at("monitor_3"),
            self.param.at("monitor_1"),
            self.param.at("monitor_4"),
            self.config,
        )
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x


class SpikingBlock(SpikingModule):
    def __init__(self, ann: vit.Block, config: SpikeConfig, param: SpikeParamDict):
        assert isinstance(ann, vit.Block)
        super().__init__(config)
        self.param = param
        self.norm1 = ann_to_snn(ann.norm1, config, param.at("monitor_0"), param.at("monitor_1"), param.child("norm1"))
        self.attn = SpikingAttention(ann.attn, config, param.child("attn"))
        self.norm2 = ann_to_snn(ann.norm2, config, param.at("monitor_3"), param.at("monitor_4"), param.child("norm2"))
        self.mlp = SpikingMlp(ann.mlp, config, param.child("mlp"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Equivalent to add/mul by scalar before LayerNorm
        tmp = self.attn(self.norm1(x))
        x = ops.add(
            x, tmp, self.param.at("monitor_0"), self.param.at("monitor_2"), self.param.at("monitor_3"), self.config
        )
        tmp = self.mlp(self.norm2(x))
        x = ops.add(
            x, tmp, self.param.at("monitor_3"), self.param.at("monitor_5"), self.param.at("monitor_o"), self.config
        )
        return x


class SpikingVisionTransformer(SpikingModule):

    def __init__(self, ann: vit.VisionTransformer, config: SpikeConfig, param: SpikeParamDict):
        assert isinstance(ann, vit.VisionTransformer)
        super().__init__(config)
        self.param = param

        self.num_classes = ann.num_classes
        self.num_features = ann.num_features
        self.embed_dim = ann.embed_dim
        self.patch_embed = SpikingPatchEmbed(ann.patch_embed, config, param.child("patch_embed"))

        self.cls_token = nn.Parameter(value_to_ttfs(ann.cls_token, param.at("monitor_1")))
        self.pos_embed = nn.Parameter(ann.pos_embed)
        self.blocks = nn.Sequential(
            *[SpikingBlock(block, config, param.child(f"blocks.{i}")) for i, block in enumerate(ann.blocks)]
        )
        self.norm = ann_to_snn(ann.norm, config, param.at("monitor_3"), param.at("monitor_4"), param.child("norm"))
        self.head = ann_to_snn(ann.head, config, param.at("monitor_4"), param.at("monitor_o"))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = ops.add_scalar(x, self.pos_embed, self.param.at("monitor_1"), self.param.at("monitor_2"), self.config)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x
