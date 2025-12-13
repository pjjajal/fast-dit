from typing import Tuple, Literal, Optional

import torch
import torch.nn as nn
from .attention import Attention, FlexAttention
from .ffn import Mlp, SwiGLU
from models import modulate


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        ffn_layer: Mlp | SwiGLU = Mlp,
        attn_layer: Attention | FlexAttention = Attention,
    ):
        super().__init__()
        self.ffn_ratio = ffn_ratio
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = attn_layer(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        ffn_hidden_dim = int(dim * ffn_ratio)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = ffn_layer(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, c, freq_cis, block_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), freq_cis, mask=block_mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.ffn(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x
