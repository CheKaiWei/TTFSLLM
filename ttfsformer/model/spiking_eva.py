import math
from typing import Callable, Union

import torch
import torch.nn as nn

from . import eva
from ..spike import (
    SpikeConfig,
    SpikeParamDict,
    ann_to_snn,
    SpikingModule,
    value_to_ttfs,
    ttfs_to_value,
    ops,
)
from .spiking_layers import SpikingMlp, SpikingPatchEmbed, SpikingGluMlp, SpikingSwiGLU, spiking_apply_rot_embed_cat
from .layers import Mlp, GluMlp, SwiGLU


class SpikingEvaAttention(SpikingModule):

    def __init__(self, ann: eva.EvaAttention, config: SpikeConfig, param: SpikeParamDict):
        super().__init__(config)
        self.param = param
        self.dim = ann.dim
        self.num_heads = ann.num_heads
        self.scale = ann.scale
        self.num_prefix_tokens = ann.num_prefix_tokens

        head_dim = self.dim // self.num_heads
        all_head_dim = head_dim * self.num_heads

        qkv = nn.Linear(self.dim, all_head_dim * 3)
        if ann.qkv is not None:
            qkv.weight = ann.qkv.weight
            if ann.q_bias is not None:
                qkv.bias = nn.Parameter(torch.cat((ann.q_bias, ann.k_bias, ann.v_bias)))
            else:
                qkv.bias = None
        else:
            qkv.weight = nn.Parameter(torch.cat((ann.q_proj.weight, ann.k_proj.weight, ann.v_proj.weight), dim=-2))
            zero = torch.zeros(all_head_dim, dtype=qkv.weight.dtype, device=qkv.weight.device)
            qkv.bias = nn.Parameter(
                torch.cat(
                    (
                        ann.q_proj.bias if ann.q_proj.bias is not None else zero,
                        ann.k_proj.bias if ann.k_proj.bias is not None else zero,
                        ann.v_proj.bias if ann.v_proj.bias is not None else zero,
                    ),
                ),
            )

        self.qkv = ann_to_snn(qkv, config, param.at("monitor_0"), param.at("monitor_1"))
        self.softmax = ann_to_snn(
            ann.softmax, config, param.at("monitor_2"), param.at("monitor_3"), param.child("softmax")
        )
        self.norm = ann_to_snn(ann.norm, config, param.at("monitor_4"), param.at("monitor_5"))
        self.proj = ann_to_snn(ann.proj, config, param.at("monitor_5"), param.at("monitor_o"))

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = torch.reshape(qkv, (B, N, 3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat(
                [
                    q[:, :, :npt, :],
                    spiking_apply_rot_embed_cat(
                        q[:, :, npt:, :],
                        rope,
                        self.param.at("monitor_1"),
                        self.param.at("monitor_1"),
                        self.config,
                    ),
                ],
                dim=2,
            ).type_as(v)
            k = torch.cat(
                [
                    k[:, :, :npt, :],
                    spiking_apply_rot_embed_cat(
                        k[:, :, npt:, :],
                        rope,
                        self.param.at("monitor_1"),
                        self.param.at("monitor_1"),
                        self.config,
                    ),
                ],
                dim=2,
            ).type_as(v)

        attn = ops.matmul(
            q,
            k.transpose(-2, -1),
            self.param.at("monitor_1"),
            self.param.at("monitor_1"),
            self.param.at("monitor_2"),
            self.config,
            self.scale,
        )

        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)
            attn = attn.masked_fill(~attn_mask[:, None, None, :], 1)

        attn = self.softmax(attn)

        x = ops.matmul(
            attn,
            v,
            self.param.at("monitor_3"),
            self.param.at("monitor_1"),
            self.param.at("monitor_4"),
            self.config,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        return x


class SpikingEvaBlock(SpikingModule):

    def __init__(self, ann: eva.EvaBlock, config: SpikeConfig, param: SpikeParamDict):
        super().__init__(config)
        self.param = param

        self.norm1 = ann_to_snn(ann.norm1, config, param.at("monitor_0"), param.at("monitor_1"), param.child("norm1"))
        self.attn = SpikingEvaAttention(ann.attn, config, param.child("attn"))
        self.gamma_1 = ann.gamma_1

        self.norm2 = ann_to_snn(ann.norm2, config, param.at("monitor_3"), param.at("monitor_4"), param.child("norm2"))

        if isinstance(ann.mlp, Mlp):
            self.mlp = SpikingMlp(ann.mlp, config, param.child("mlp"))
        elif isinstance(ann.mlp, GluMlp):
            self.mlp = SpikingGluMlp(ann.mlp, config, param.child("mlp"))
        elif isinstance(ann.mlp, SwiGLU):
            self.mlp = SpikingSwiGLU(ann.mlp, config, param.child("mlp"))
        else:
            raise NotImplementedError()

        self.gamma_2 = ann.gamma_2

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.gamma_1 is None:
            x = ops.add(
                x,
                tmp := self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask),
                self.param.at("monitor_0"),
                self.param.at("monitor_2"),
                self.param.at("monitor_3"),
                self.config,
            )
            x = ops.add(
                x,
                self.mlp(self.norm2(x)),
                self.param.at("monitor_3"),
                self.param.at("monitor_5"),
                self.param.at("monitor_o"),
                self.config,
            )
        else:
            x = ops.add(
                x,
                ops.hadamard_scalar(
                    self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask),
                    self.gamma_1,
                    self.param.at("monitor_6"),
                    self.param.at("monitor_2"),
                    self.config,
                ),
                self.param.at("monitor_0"),
                self.param.at("monitor_2"),
                self.param.at("monitor_3"),
                self.config,
            )
            x = ops.add(
                x,
                ops.hadamard_scalar(
                    self.mlp(self.norm2(x)),
                    self.gamma_2,
                    self.param.at("monitor_7"),
                    self.param.at("monitor_5"),
                    self.config,
                ),
                self.param.at("monitor_3"),
                self.param.at("monitor_5"),
                self.param.at("monitor_o"),
                self.config,
            )
        return x


class SpikingEvaBlockPostNorm(SpikingModule):

    def __init__(self, ann: eva.EvaBlockPostNorm, config: SpikeConfig, param: SpikeParamDict):
        super().__init__(config)
        self.param = param
        self.attn = SpikingEvaAttention(ann.attn, config, param.child("attn"))
        self.norm1 = ann_to_snn(ann.norm1, config, param.at("monitor_1"), param.at("monitor_2"))

        if isinstance(ann.mlp, Mlp):
            self.mlp = SpikingMlp(ann.mlp, config, param.at("mlp"))
        elif isinstance(ann.mlp, GluMlp):
            self.mlp = SpikingGluMlp(ann.mlp, config, param.at("mlp"))
        elif isinstance(ann.mlp, SwiGLU):
            self.mlp = SpikingSwiGLU(ann.mlp, config, param.at("mlp"))
        else:
            raise NotImplementedError()

        self.norm2 = ann_to_snn(ann.norm2, config, param.at("monitor_4"), param.at("monitor_5"))

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = ops.add(
            x,
            self.norm1(self.attn(x, rope=rope, attn_mask=attn_mask)),
            self.param.at("monitor_0"),
            self.param.at("monitor_2"),
            self.param.at("monitor_3"),
            self.config,
        )
        x = ops.add(
            x,
            self.norm2(self.mlp(x)),
            self.param.at("monitor_3"),
            self.param.at("monitor_5"),
            self.param.at("monitor_o"),
            self.config,
        )
        return x


class SpikingEva(SpikingModule):

    def __init__(self, ann: eva.Eva, config: SpikeConfig, param: SpikeParamDict):
        super().__init__(config)
        self.param = param
        self.num_classes = ann.num_classes
        self.global_pool = ann.global_pool
        self.num_features = ann.num_features
        self.head_hidden_size = ann.head_hidden_size
        self.embed_dim = ann.embed_dim
        self.num_prefix_tokens = ann.num_prefix_tokens

        self.patch_embed = SpikingPatchEmbed(ann.patch_embed, config, param.child("patch_embed"))
        self.cls_token = ann.cls_token
        self.reg_token = (
            nn.Parameter(value_to_ttfs(ann.reg_token), param.at("monitor_5")) if ann.reg_token is not None else None
        )
        self.pos_embed = ann.pos_embed
        # self.patch_drop = nn.Identity()  # Lazy, dropout disabled when not training
        self.rope = ann.rope
        self.blocks = nn.ModuleList(
            [
                (
                    SpikingEvaBlock(block, config, param.child(f"blocks.{i}"))
                    if isinstance(block, eva.EvaBlock)
                    else SpikingEvaBlockPostNorm(block, config, param.child(f"blocks.{i}"))
                )
                for i, block in enumerate(ann.blocks)
            ]
        )
        self.norm = ann_to_snn(ann.norm, config, param.at("monitor_2"), param.at("monitor_3"))
        self.fc_norm = ann_to_snn(
            ann.fc_norm, config, param.at("monitor_3"), param.at("monitor_4"), param.child("fc_norm")
        )
        self.head = ann_to_snn(ann.head, config, param.at("monitor_4"), param.at("monitor_o"))

    def _pos_embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        pos_embed = self.pos_embed
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        if self.cls_token is not None:
            cls_token = value_to_ttfs(self.cls_token, self.param.at("monitor_1"))
            x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if pos_embed is not None:
            x = ops.add_scalar(
                x,
                pos_embed,
                self.param.at("monitor_1"),
                self.param.at("monitor_5"),
                self.config,
            )

        if self.reg_token is not None:
            to_cat = []
            if self.cls_token is not None:
                cls_token = value_to_ttfs(self.cls_token, self.param.at("monitor_5"))
                to_cat.append(cls_token.expand(x.shape[0], -1, -1))
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
            x = torch.cat(to_cat + [x], dim=1)

        # Note: patch_dropout ignored since we're in eval mode.
        return x, rot_pos_embed

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
