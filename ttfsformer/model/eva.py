import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import Callable, Union

from ..util import BaseMonitor, RangeMonitor, PercentileMonitor, MinMaxMonitor
from .monitor import MonitoredLayerNorm, MonitoredSoftmax

from timm.layers import (
    PatchDropout,
    RotaryEmbeddingCat,
    apply_keep_indices_nlc,
    to_2tuple,
)
from timm.models._features import feature_take_indices
from timm.models.registry import register_model
from .layers import DropPath, PatchEmbed, Mlp, GluMlp, SwiGLU, apply_rot_embed_cat


class EvaAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: int | None = None,
        norm_layer: Callable | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim**-0.5
        self.num_prefix_tokens = num_prefix_tokens

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer("k_bias", torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.monitor_0 = BaseMonitor()
        self.monitor_1 = BaseMonitor()
        self.monitor_2 = BaseMonitor()
        self.monitor_3 = BaseMonitor()
        self.monitor_4 = BaseMonitor()
        self.monitor_5 = BaseMonitor()
        self.monitor_6 = BaseMonitor()
        self.monitor_o = BaseMonitor()

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        # Linear Part 1: Convert input into QKV.
        x = self.monitor_0(x)
        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                qkv = self.qkv(x)
                qkv += qkv_bias
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        q = self.monitor_1(q)
        k = self.monitor_1(k)
        v = self.monitor_1(v)
        # RoPE.
        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)
        # Attention.
        q = self.monitor_1(q)
        k = self.monitor_1(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.monitor_2(attn)

        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)
            attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))

        attn = self.softmax(attn)
        attn = self.monitor_3(attn)

        attn = self.attn_drop(attn)
        x = attn @ v
        x = self.monitor_4(x)
        # Linear Part 2: Output.
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.monitor_5(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.monitor_o(x)
        return x


class EvaBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.0,
        swiglu_mlp: bool = False,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        num_prefix_tokens: int = 1,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn_head_dim: int | None = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.monitor_0 = BaseMonitor()
        self.monitor_1 = BaseMonitor()
        self.monitor_2 = BaseMonitor()
        self.monitor_3 = BaseMonitor()
        self.monitor_4 = BaseMonitor()
        self.monitor_5 = BaseMonitor()
        self.monitor_6 = BaseMonitor()
        self.monitor_7 = BaseMonitor()
        self.monitor_o = BaseMonitor()

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.monitor_0(x)
        if self.gamma_1 is None:
            tmp = self.norm1(x)
            tmp = self.monitor_1(tmp)
            tmp = self.drop_path1(self.attn(tmp, rope=rope, attn_mask=attn_mask))
            tmp = self.monitor_2(tmp)
            x = x + tmp
            x = self.monitor_3(x)
            tmp = self.norm2(x)
            tmp = self.monitor_4(tmp)
            tmp = self.drop_path2(self.mlp(tmp))
            tmp = self.monitor_5(tmp)
            x = x + tmp
            # print("<", x.flatten()[:10])
            # exit(0)
        else:
            tmp = self.norm1(x)
            tmp = self.monitor_1(tmp)
            tmp = self.attn(tmp, rope=rope, attn_mask=attn_mask)
            tmp = self.monitor_6(tmp)
            tmp = self.drop_path1(self.gamma_1 * tmp)
            tmp = self.monitor_2(tmp)
            x = x + tmp
            x = self.monitor_3(x)
            tmp = self.norm2(x)
            tmp = self.monitor_4(tmp)
            tmp = self.mlp(tmp)
            tmp = self.monitor_7(tmp)
            tmp = self.drop_path2(self.gamma_2 * tmp)
            tmp = self.monitor_5(tmp)
            x = x + tmp
        x = self.monitor_o(x)
        return x


class EvaBlockPostNorm(nn.Module):
    """EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.0,
        swiglu_mlp: bool = False,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        num_prefix_tokens: int = 1,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,  # ignore for post-norm
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attn_head_dim: int | None = None,
    ):
        super().__init__()
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.monitor_0 = BaseMonitor()
        self.monitor_1 = BaseMonitor()
        self.monitor_2 = BaseMonitor()
        self.monitor_3 = BaseMonitor()
        self.monitor_4 = BaseMonitor()
        self.monitor_5 = BaseMonitor()
        self.monitor_o = BaseMonitor()

    def forward(
        self, x: torch.Tensor, rope: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.monitor_0(x)
        tmp = self.attn(x, rope=rope, attn_mask=attn_mask)
        tmp = self.monitor_1(tmp)
        tmp = self.drop_path1(self.norm1(tmp))
        tmp = self.monitor_2(tmp)
        x = x + tmp
        x = self.monitor_3(x)
        tmp = self.mlp(x)
        tmp = self.monitor_4(tmp)
        tmp = self.drop_path2(self.norm2(tmp))
        tmp = self.monitor_5(tmp)
        x = x + tmp
        tmp = self.monitor_o(tmp)
        return x


class Eva(nn.Module):
    """Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
    """

    def __init__(
        self,
        img_size: Union[int, tuple[int, int]] = 224,
        patch_size: Union[int, tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "avg",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.0,
        swiglu_mlp: bool = False,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable = nn.LayerNorm,
        init_values: float | None = None,
        class_token: bool = True,
        num_reg_tokens: int = 0,
        use_abs_pos_emb: bool = True,
        use_rot_pos_emb: bool = False,
        use_post_norm: bool = False,
        # dynamic_img_size: bool = False,
        # dynamic_img_pad: bool = False,
        ref_feat_shape: Union[tuple[int, int | None, int]] = None,
        head_init_scale: float = 0.001,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = (1 if class_token else 0) + num_reg_tokens
        # self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_c=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim)) if num_reg_tokens else None

        self.pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=self.patch_embed.grid_size,
                ref_feat_shape=ref_feat_shape,
            )
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qkv_fused=qkv_fused,
                    mlp_ratio=mlp_ratio,
                    swiglu_mlp=swiglu_mlp,
                    scale_mlp=scale_mlp,
                    scale_attn_inner=scale_attn_inner,
                    num_prefix_tokens=self.num_prefix_tokens,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )

        use_fc_norm = self.global_pool == "avg"
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.monitor_0 = BaseMonitor()
        self.monitor_1 = BaseMonitor()
        self.monitor_2 = BaseMonitor()
        self.monitor_3 = BaseMonitor()
        self.monitor_4 = BaseMonitor()
        self.monitor_5 = BaseMonitor()
        self.monitor_o = BaseMonitor()

    def monitor(self):
        for _, mod in self.named_modules():
            for child_name, child_mod in mod.named_children():
                if isinstance(child_mod, nn.LayerNorm):
                    setattr(mod, child_name, MonitoredLayerNorm(child_mod))
                elif isinstance(child_mod, nn.Softmax):
                    setattr(mod, child_name, MonitoredSoftmax(child_mod))
                elif isinstance(child_mod, BaseMonitor):
                    setattr(mod, child_name, RangeMonitor())

    def _pos_embed(self, x) -> tuple[torch.Tensor, torch.Tensor | None]:
        pos_embed = self.pos_embed
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if pos_embed is not None:
            x = x + pos_embed
        x = self.monitor_5(x)

        if self.reg_token is not None:
            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
            x = torch.cat(to_cat + [x], dim=1)

        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
        return x, rot_pos_embed

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.monitor_1(x)
        x, rot_pos_embed = self._pos_embed(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope=rot_pos_embed)
        x = self.monitor_2(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm(x)
        x = self.monitor_4(x)
        x = self.head_drop(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.monitor_0(x)
        x = self.forward_features(x)
        x = self.monitor_3(x)
        x = self.forward_head(x)
        x = self.monitor_o(x)
        return x


@register_model
def eva02_tiny_patch14_336(pretrained=False, **keywords) -> Eva:
    model = Eva(
        img_size=336,
        patch_size=14,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    )
    return model


@register_model
def eva02_small_patch14_336(pretrained=False, **keywords) -> Eva:
    model = Eva(
        img_size=336,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    )
    return model


@register_model
def eva02_base_patch14_448(pretrained=False, **kwargs) -> Eva:
    model = Eva(
        img_size=448,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    return model


@register_model
def eva02_large_patch14_448(pretrained=False, **kwargs) -> Eva:
    model = Eva(
        img_size=448,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    return model


@register_model
def eva_giant_patch14_224(pretrained=False, **kwargs) -> Eva:
    model = Eva(
        img_size=224,
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=6144 / 1408,
        qkv_fused=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    return model
