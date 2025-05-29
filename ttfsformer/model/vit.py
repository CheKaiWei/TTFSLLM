from functools import partial
from timm.models.registry import register_model
import torch
import torch.nn as nn

from .layers import DropPath, PatchEmbed, Mlp
from ..util import RangeMonitor, PercentileMonitor, BaseMonitor, MinMaxMonitor
from .monitor import MonitoredLayerNorm, MonitoredSoftmax


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop_ratio: float = 0.0,
        proj_drop_ratio: float = 0.0,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.monitor_0 = BaseMonitor()
        self.monitor_1 = BaseMonitor()
        self.monitor_2 = BaseMonitor()
        self.monitor_3 = BaseMonitor()
        self.monitor_4 = BaseMonitor()
        self.monitor_o = BaseMonitor()
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]

        x = self.monitor_0(x)
        qkv = self.qkv(x)
        qkv = self.monitor_1(qkv)
        qkv = torch.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads)).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = (qkv[0], qkv[1], qkv[2])

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        attn = q @ k.transpose(-2, -1)
        attn *= self.scale

        attn = self.monitor_2(attn)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = self.monitor_3(attn)

        x = attn @ v
        x = self.monitor_4(x)
        x = x.transpose(1, 2).reshape(B, N, -1)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]

        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.monitor_o(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio,
        )
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop_ratio,
        )
        self.monitor_0 = BaseMonitor()
        self.monitor_1 = BaseMonitor()
        self.monitor_2 = BaseMonitor()
        self.monitor_3 = BaseMonitor()
        self.monitor_4 = BaseMonitor()
        self.monitor_5 = BaseMonitor()
        self.monitor_o = BaseMonitor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.monitor_0(x)
        tmp = self.norm1(x)
        tmp = self.monitor_1(tmp)
        tmp = self.drop_path(self.attn(tmp))
        tmp = self.monitor_2(tmp)
        x = x + tmp
        x = self.monitor_3(x)
        tmp = self.norm2(x)
        tmp = self.monitor_4(tmp)
        tmp = self.drop_path(self.mlp(tmp))
        tmp = self.monitor_5(tmp)
        x = x + tmp
        x = self.monitor_o(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_c=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_ratio=0.0,
        attn_drop_ratio=0.0,
        drop_path_ratio=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    drop_path_ratio=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.monitor_0 = BaseMonitor()
        self.monitor_1 = BaseMonitor()
        self.monitor_2 = BaseMonitor()
        self.monitor_3 = BaseMonitor()
        self.monitor_4 = BaseMonitor()
        self.monitor_o = BaseMonitor()

    def monitor(self):
        for _, mod in self.named_modules():
            for child_name, child_mod in mod.named_children():
                if isinstance(child_mod, nn.LayerNorm):
                    setattr(mod, child_name, MonitoredLayerNorm(child_mod))
                elif isinstance(child_mod, nn.Softmax):
                    setattr(mod, child_name, MonitoredSoftmax(child_mod))
                elif isinstance(child_mod, BaseMonitor):
                    setattr(mod, child_name, MinMaxMonitor())

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.monitor_0(x)
        x = self.patch_embed(x)
        x = self.monitor_1(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.monitor_2(x)
        x = self.blocks(x)
        x = self.monitor_3(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.monitor_4(x)
        x = self.head(x)
        x = self.monitor_o(x)
        return x


@register_model
def vit_tiny_patch16_224(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_tiny_patch16_384(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_tiny_patch16_224_cifar10(pretrained=False, num_classes: int = 10, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_tiny_patch16_224_cifar100(pretrained=False, num_classes: int = 100, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_small_patch32_224(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=32,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_small_patch32_384(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_small_patch16_224(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_small_patch16_384(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_small_patch16_224_cifar10(pretrained=False, num_classes: int = 10, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_small_patch16_224_cifar100(pretrained=False, num_classes: int = 100, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_base_patch32_224(pretrained: bool = False, num_classes: int = 1000, **kwargs) -> VisionTransformer:
    model = VisionTransformer(
        img_size=224,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_base_patch32_384(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_base_patch16_224(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_base_patch16_224_cifar10(pretrained=False, num_classes: int = 10, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_base_patch16_224_cifar100(pretrained=False, num_classes: int = 100, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_base_patch16_384(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_large_patch32_224(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_large_patch32_384(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_large_patch16_224(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_large_patch16_384(pretrained=False, num_classes: int = 1000, **keywords):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_large_patch16_224_cifar10(pretrained=False, num_classes: int = 10, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
    )
    return model


@register_model
def vit_large_patch16_224_cifar100(pretrained=False, num_classes: int = 100, **keywords):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
    )
    return model
