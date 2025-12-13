import timm
import torch
import torch.nn as nn
from timm.layers import PatchEmbed
from timm.layers.pos_embed import resample_abs_pos_embed
from einops import rearrange

from .ffn import Mlp, SwiGLU
from models import get_2d_sincos_pos_embed

class PatchEmbedAdapter(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels: int,
        embed_dim: int,
        use_ffn: bool = False,
        norm_layer: nn.Module = None,
        ffn_layer: Mlp | SwiGLU = Mlp,
        ffn_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.use_ffn = use_ffn

        in_channels = in_channels
        self.embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            strict_img_size=False,
            flatten=False,
        )

        self.num_tokens = self.embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim), requires_grad=False)
        # self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim) * 0.02)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.embed.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        if use_ffn:
            ffn_hidden_dim = int(embed_dim * ffn_ratio)
            self.ffn = ffn_layer(
                in_features=embed_dim,
                hidden_features=ffn_hidden_dim,
                out_features=embed_dim,
                act_layer=act_layer,
                drop=proj_drop,
            )

    def forward(self, x):
        x = self.embed(x)
        B, C, H, W = x.shape
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,  # this unsqueeze might be unnecessary.
            new_size=(H, W),
            # old_size=self.embed.grid_size,
            num_prefix_tokens=0,
        )
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + pos_embed
        x = self.norm(x)
        if self.use_ffn:
            x = self.ffn(x)
        return x


class OutputAdapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_ffn: bool = False,
        norm_layer: nn.Module = None,
        ffn_layer: Mlp | SwiGLU = Mlp,
        ffn_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        self.use_ffn = use_ffn
        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.drop = nn.Dropout(drop)
        if use_ffn:
            ffn_hidden_dim = int(in_features * ffn_ratio)
            self.ffn = ffn_layer(
                in_features=in_features,
                hidden_features=ffn_hidden_dim,
                out_features=out_features,
                act_layer=act_layer,
            )
        else:
            self.ffn = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.drop(x)
        x = self.ffn(x)
        return x