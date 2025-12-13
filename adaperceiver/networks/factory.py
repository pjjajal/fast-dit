from omegaconf import DictConfig
from .adaperceiver import AdaPercevierConfig
from .dit_adaperceiver import DiTAdaPerceiverConfig, DiTAdaPerceiver


def create_dit_adaperceiver(cfg: DictConfig, **kwargs):
    encoder_cfg = cfg.encoder_config
    encoder_cfg.ffn_ratio = (
        eval(encoder_cfg.ffn_ratio)
        if isinstance(encoder_cfg.ffn_ratio, str)
        else encoder_cfg.ffn_ratio
    )
    dense_cfg = DiTAdaPerceiverConfig(
        img_size=encoder_cfg.img_size,
        in_channels=encoder_cfg.in_channels,
        patch_size=encoder_cfg.patch_size,
        use_embed_ffn=encoder_cfg.use_embed_ffn,
        use_output_ffn=encoder_cfg.use_output_ffn,
        learn_sigma=encoder_cfg.learn_sigma,
        class_dropout_prob=encoder_cfg.class_dropout_prob,
        num_classes=encoder_cfg.num_classes,
    )
    perceiver_cfg = AdaPercevierConfig(
        embed_dim=encoder_cfg.embed_dim,
        num_heads=encoder_cfg.num_heads,
        depth=encoder_cfg.depth,
        max_latent_tokens=encoder_cfg.max_latent_tokens,
        max_latent_tokens_mult=encoder_cfg.max_latent_tokens_mult,
        rope_theta=encoder_cfg.rope_theta,
        ffn_ratio=encoder_cfg.ffn_ratio,
        qkv_bias=encoder_cfg.qkv_bias,
        proj_bias=encoder_cfg.proj_bias,
        proj_drop=encoder_cfg.proj_drop,
        attn_drop=encoder_cfg.attn_drop,
        act_layer=encoder_cfg.act_layer,
        norm_layer=encoder_cfg.norm_layer,
        ffn_layer=encoder_cfg.ffn_layer,
        attn_layer=encoder_cfg.attn_layer,
        process_token_init=encoder_cfg.process_token_init,
        block_mask=cfg.mask_type,
        mask_token_grans=cfg.token_grans,
    )
    return DiTAdaPerceiver(
        config=perceiver_cfg,
        dit_config=dense_cfg,
    )


