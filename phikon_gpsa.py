"""
Inject GPSA into a pretrained phikon (ViT-B) backbone.

phikon (owkin/phikon) is loaded via HuggingFace `transformers` as a
standard ViTModel. This module replaces the first `local_layers` of its
`ViTAttention` blocks with GPSA, TRANSPLANTING the pretrained
query/key/value/output weights into GPSA's layout so pretraining isn't
thrown away — only the attention *mechanism* changes (content-only ->
gated content+positional), not the learned features it starts from.

Verified against transformers' current ViT implementation, where each
layer's attention is:
    ViTAttention(q_proj, k_proj, v_proj, o_proj)   # all Linear(dim, dim)
and ViTLayer.forward calls:
    hidden_states, _ = self.attention(hidden_states, attention_mask, **kwargs)
so the replacement module must accept (hidden_states, attention_mask, **kwargs)
and return (output, attn_weights_or_None).

Usage:
    from transformers import ViTModel
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    inject_gpsa(model, local_layers=10, locality_strength=1.0)
    # model.layers[:10] now use GPSA (pretrained weights preserved),
    # model.layers[10:] are untouched standard MHSA.
"""

import math
import torch
import torch.nn as nn
from gpsa import GPSA


class GPSAAttentionWrapper(nn.Module):
    """
    Drop-in replacement for HF's ViTAttention. Wraps a GPSA module and
    matches ViTAttention's forward signature/return type so it can be
    swapped into layer.attention without touching ViTLayer at all.
    """

    def __init__(self, gpsa_module):
        super().__init__()
        self.gpsa = gpsa_module

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, N, C = hidden_states.shape
        num_patches = N - 1  # GPSA here is always used with class_token=True
        H = W = int(math.isqrt(num_patches))
        assert H * W == num_patches, (
            f"GPSA wrapper assumes a square patch grid; got {num_patches} patches "
            f"(N={N}). If your input isn't square, pass H, W explicitly instead."
        )
        out = self.gpsa(hidden_states, H, W)
        return out, None  # (output, attn_weights) — weights unused downstream


def _transplant_qkv(gpsa, old_attn):
    """Copy pretrained q_proj/k_proj/v_proj/o_proj weights into GPSA's qk/v/proj."""
    dim = old_attn.q_proj.in_features
    with torch.no_grad():
        gpsa.qk.weight[:dim].copy_(old_attn.q_proj.weight)
        gpsa.qk.weight[dim:].copy_(old_attn.k_proj.weight)
        if old_attn.q_proj.bias is not None:
            gpsa.qk.bias[:dim].copy_(old_attn.q_proj.bias)
            gpsa.qk.bias[dim:].copy_(old_attn.k_proj.bias)

        gpsa.v.weight.copy_(old_attn.v_proj.weight)
        if old_attn.v_proj.bias is not None:
            gpsa.v.bias.copy_(old_attn.v_proj.bias)

        gpsa.proj.weight.copy_(old_attn.o_proj.weight)
        if old_attn.o_proj.bias is not None:
            gpsa.proj.bias.copy_(old_attn.o_proj.bias)


def inject_gpsa(model, local_layers=10, locality_strength=1.0, use_local_init=True):
    """
    Mutates `model` in place: replaces model.layers[:local_layers].attention
    with GPSA, seeded from the pretrained Q/K/V/O weights of each layer.
    model.layers[local_layers:] are left untouched (standard MHSA).

    Args:
        model: a loaded ViTModel (e.g. from transformers.ViTModel.from_pretrained)
        local_layers: number of leading layers to convert to GPSA
        locality_strength: GPSA's init locality strength (higher = more local at init)
        use_local_init: if True, positional attention is soft-conv initialized
            THEN overwritten by pretrained content weights in qk/v/proj — the
            locality bias lives entirely in pos_proj, which is untouched by
            the transplant, so both effects (pretrained features + local bias)
            are present simultaneously.

    Returns:
        model (same object, mutated in place) — returned for convenience/chaining.
    """
    num_heads = model.config.num_attention_heads
    dim = model.config.hidden_size

    for i in range(local_layers):
        layer = model.layers[i]
        old_attn = layer.attention

        gpsa = GPSA(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            locality_strength=locality_strength,
            use_local_init=use_local_init,
            class_token=True,
        )
        _transplant_qkv(gpsa, old_attn)
        layer.attention = GPSAAttentionWrapper(gpsa)

    return model


def get_gating_params(model, local_layers=10):
    """sigmoid(lambda) per head, per converted layer — track pos->content drift."""
    out = {}
    for i in range(local_layers):
        attn = model.layers[i].attention
        if isinstance(attn, GPSAAttentionWrapper):
            out[i] = torch.sigmoid(attn.gpsa.gating_param).detach().cpu()
    return out


if __name__ == "__main__":
    # sanity check using a randomly-initialized model with phikon's config
    # (no network access needed here — same architecture, different weights;
    # the transplant logic is identical when you load actual phikon)
    from transformers import ViTModel, ViTConfig

    cfg = ViTConfig(hidden_size=768, num_attention_heads=12, num_hidden_layers=12,
                     image_size=224, patch_size=16)
    model = ViTModel(cfg, add_pooling_layer=False)

    inject_gpsa(model, local_layers=10, locality_strength=1.0)

    x = torch.randn(2, 3, 224, 224)
    out = model(x).last_hidden_state
    print("output shape:", out.shape)  # (2, 197, 768) = 1 cls + 196 patches
    assert out.shape == (2, 197, 768)

    loss = out.mean()
    loss.backward()
    gates = get_gating_params(model, local_layers=10)
    print("GPSA layers converted:", list(gates.keys()))
    print("layer 0 gate (per head):", gates[0])
    print("layer 0 gate grad is not None:",
          model.layers[0].attention.gpsa.gating_param.grad is not None)
