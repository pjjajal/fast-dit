import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.attention.flex_attention as flex_attn
from timm.layers import get_act_layer, get_norm_layer

from adaperceiver.layers.adapters import OutputAdapter, PatchEmbedAdapter
from adaperceiver.layers.ffn import Mlp
from adaperceiver.layers.fusion import CrossAttentionFusion
from adaperceiver.layers.latents import PatchEmbedOutputLatents
from models import FinalLayer, LabelEmbedder, TimestepEmbedder
from functools import partial

from .adaperceiver import AdaPerceiver, AdaPercevierConfig, get_ffn_layer, AdaPerceiverOutput


@dataclass
class DiTAdaPerceiverConfig:
    img_size: int
    in_channels: int
    patch_size: int
    use_embed_ffn: bool = False
    use_output_ffn: bool = False
    learn_sigma: bool = True
    class_dropout_prob: float = 0.1
    num_classes: int = 1000


class DiTAdaPerceiver(AdaPerceiver):
    def __init__(self, config: AdaPercevierConfig, dit_config: DiTAdaPerceiverConfig):
        super().__init__(config)
        self.dit_config = dit_config
        self.use_embed_ffn = dit_config.use_embed_ffn
        self.use_output_ffn = dit_config.use_output_ffn
        self.img_size = dit_config.img_size
        self.in_channels = dit_config.in_channels
        self.patch_size = dit_config.patch_size

        # DiT Specific
        self.learn_sigma = dit_config.learn_sigma
        self.class_dropout_prob = dit_config.class_dropout_prob
        self.num_classes = dit_config.num_classes
        self.out_channels = (
            self.in_channels * 2 if self.learn_sigma else self.in_channels
        )

        act_layer = get_act_layer(config.act_layer) or nn.GELU
        ffn_layer = get_ffn_layer(config.ffn_layer) or Mlp
        # norm_layer = get_norm_layer(config.norm_layer) or nn.LayerNorm
        norm_layer = partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6)

        self.patch_embed = PatchEmbedAdapter(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            use_ffn=self.use_embed_ffn,
            ffn_layer=ffn_layer,
            ffn_ratio=self.config.ffn_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer if self.use_embed_ffn else None,
        )

        self.output_latents = PatchEmbedOutputLatents(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
        )
        self.output_adapter = OutputAdapter(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            use_ffn=self.use_output_ffn,
            ffn_layer=ffn_layer,
            ffn_ratio=self.config.ffn_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer if self.use_output_ffn else None,
            drop=self.config.head_drop,
        )

        self.read_head = CrossAttentionFusion(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            norm_layer=norm_layer,
        )
        self.write_head = CrossAttentionFusion(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            norm_layer=norm_layer,
        )

        # DiT specific layers
        self.num_patches = self.patch_embed.num_tokens
        self.t_embedder = TimestepEmbedder(self.embed_dim)
        self.y_embedder = LabelEmbedder(
            self.num_classes, self.embed_dim, self.class_dropout_prob
        )
        self.final_layer = FinalLayer(
            self.embed_dim, self.patch_size, self.out_channels
        )

        self.init_weights()

    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_embed.embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.embed.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def read(
        self,
        x: torch.Tensor,
        process_latents: torch.Tensor,
        freq_cis: torch.Tensor,
    ):
        process_latents = process_latents + self.read_head(
            sink=process_latents,
            src=x,
            freq_cis_q=freq_cis,  # apply the RoPE frequencies to the query
            freq_cis_k=None,  # no RoPE on key since they already have positional information
        )
        return process_latents

    def write(
        self,
        process_latents: torch.Tensor,
        output_latents: torch.Tensor,
        freq_cis: torch.Tensor,
    ):
        output_latents = output_latents + self.write_head(
            sink=output_latents,  # NOTE: output_latents is the sink.
            src=process_latents,
            freq_cis_q=None,  # no RoPE on query since they already have positional information
            freq_cis_k=freq_cis,  # NOTE: # apply the RoPE frequencies to the keys
        )
        return output_latents

    def output_head(self, output_latents: torch.Tensor, c: torch.Tensor):
        """
        This is the output head that converts the output latents to the final output.
        It is used in the forward pass to compute the final output.
        """
        return self.final_layer(self.output_adapter(output_latents), c)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_embed.embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward_blocks(
        self,
        process_latents: torch.Tensor,
        c: torch.Tensor,
        freq_cis: torch.Tensor,
        block_mask: flex_attn.BlockMask = None,
    ):
        for i, block in enumerate(self.blocks):
            process_latents = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(block),
                process_latents,
                c,
                freq_cis,
                block_mask,
            )
            # process_latents = block(
            #     x=process_latents,
            #     freq_cis=freq_cis,
            #     mat_dim=mat_dim,
            #     block_mask=block_mask,
            # )
        return process_latents

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        num_tokens=None,
        token_loss=False,
        freq_cis=None,
        **kwargs
    ):
        B = x.shape[0]
        device = x.device

        # Embed input into patches
        patches = self.patch_embed(x)
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y

        # Creates the process and output latents
        N = num_tokens if num_tokens else self.max_latent_tokens  # Number of tokens
        process_latents = self.process_latents((B, N))
        if self.training:
            output_latents = self.output_latents(patches.clone())
        else:
            output_latents = self.output_latents(patches)

        # Compute the RoPE frequencies
        freq_cis = self.compute_freq_cis(N, device, freq_cis)

        # Compute block mask
        block_mask = self.compute_block_mask(
            num_tokens=N,
            device=device,
            mask_type=self.block_mask_type,
            token_grans=self.mask_token_grans,
        )

        # Read inputs into process latents
        process_latents = self.read(patches, process_latents, freq_cis)

        # Forward through the transformer blocks
        process_latents = self.forward_blocks(
            process_latents=process_latents,
            c=c,
            freq_cis=freq_cis,
            block_mask=block_mask,
        )

        # This is the final writeout step.
        final_writeout = self.write(
            process_latents=process_latents,
            output_latents=output_latents,
            freq_cis=freq_cis,
        )
        output = self.output_head(final_writeout, c)

        # Token loss: grab each token granularity and write them out.
        # We require token grans to be specified.
        out_list = []
        token_loss = bool(token_loss)
        num_tokens = bool(num_tokens)
        if token_loss and self.mask_token_grans:
            # We don't compute the output of the last token granularity
            # This is because the readout of the last token granularity is computed earlier.
            for token_gran in self.mask_token_grans[:-1]:
                freq_cis_slice = freq_cis[:token_gran]
                token_writeout = self.write(
                    process_latents=process_latents[:, :token_gran],
                    output_latents=output_latents,
                    freq_cis=freq_cis_slice,
                )
                out_list.append(self.output_head(token_writeout, c))
        # Append the final output to the list
        out_list.append(output)
        out_list = [self.unpatchify(out) for out in out_list]
        return AdaPerceiverOutput(preds=out_list)