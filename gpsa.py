"""
Gated Positional Self-Attention (GPSA) — ConViT (d'Ascoli et al., 2021)

GPSA lets each attention head adaptively interpolate between a
content-based attention map (standard QK^T) and a positional attention
map (predicted from relative position, convolution-like at init) via a
learned, per-head gating parameter lambda. This injects a soft locality
inductive bias into a ViT without discarding global attention capacity.

Reference: "ConViT: Improving Vision Transformers with Soft Convolutional
Inductive Biases", d'Ascoli, Touvron, Leavitt, Morcos, Biroli, Sagun, 2021.
https://arxiv.org/abs/2103.10697
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPSA(nn.Module):
    """
    Gated Positional Self-Attention.

    Args:
        dim: token embedding dimension (C)
        num_heads: number of attention heads
        qkv_bias: add bias to q/k/v projections
        attn_drop: dropout on attention weights
        proj_drop: dropout after output projection
        locality_strength: scales the initial convolutional-like locality
            of the positional attention (higher = more local at init)
        use_local_init: initialize positional attention to mimic a
            convolution (as in the paper); if False, positional attention
            starts randomly
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        locality_strength=1.0,
        use_local_init=True,
        class_token=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.locality_strength = locality_strength
        self.use_local_init = use_local_init
        # if True, forward() expects x = [cls_token; patch_tokens], N = 1 + H*W.
        # The cls token has no spatial location, so it gets a relative-position
        # row/col of all zeros -> pos_proj(0,0,0) is a constant score, i.e. the
        # positional attention treats the cls token as equidistant from
        # everything (and everything as equidistant from it).
        self.class_token = class_token

        # content q, k, v — separate q/k so content attention is standard QK^T
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        # positional attention: maps a 3-dim relative-position encoding
        # (delta_x, delta_y, ||delta||) to a per-head scalar score
        self.pos_proj = nn.Linear(3, num_heads)

        # per-head gating parameter lambda, passed through sigmoid.
        # initialized to 1.0 -> sigmoid ~ 0.73, biased toward positional
        # attention early in training, as in the reference implementation
        self.gating_param = nn.Parameter(torch.ones(num_heads))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # cache of relative position indices, keyed by (H, W) grid shape
        self.rel_indices = None
        self.current_grid_size = None

        self.apply(self._init_weights)
        if self.use_local_init:
            self._local_init()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _local_init(self):
        """
        Initialize pos_proj so each head's positional attention starts as
        a Gaussian-ish kernel centered on a distinct nearby offset,
        approximating a depthwise convolution — the "soft conv" init.
        """
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1  # base kernel "radius" in a 3x3-like sense
        kernel_size = int(math.sqrt(self.num_heads))
        center = (kernel_size - 1) / 2 if kernel_size > 1 else 0

        for h in range(self.num_heads):
            if kernel_size > 1:
                h1 = h // kernel_size
                h2 = h % kernel_size
            else:
                h1, h2 = 0, 0
            # this head is "responsible" for the offset (h1 - center, h2 - center)
            self.pos_proj.weight.data[h, 0] = -1 * self.locality_strength * (h1 - center) * locality_distance
            self.pos_proj.weight.data[h, 1] = -1 * self.locality_strength * (h2 - center) * locality_distance
            self.pos_proj.weight.data[h, 2] = self.locality_strength
        self.pos_proj.bias.data.zero_()

    def _get_rel_indices(self, H, W, device):
        """Build (1, N, N, 3) relative-position tensor: (dx, dy, dist^2).
        N = H*W, or 1 + H*W if class_token=True (extra all-zero row/col for cls)."""
        if self.rel_indices is not None and self.current_grid_size == (H, W):
            return self.rel_indices.to(device)

        Np = H * W
        coords = torch.stack(
            torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1
        ).reshape(Np, 2).float()  # (Np, 2) -> (row, col)

        rel = coords[None, :, :] - coords[:, None, :]  # (Np, Np, 2): rel[i, j] = pos_i - pos_j
        dy, dx = rel[..., 0], rel[..., 1]
        dist2 = dx ** 2 + dy ** 2
        rel_indices = torch.stack([dx, dy, dist2], dim=-1)  # (Np, Np, 3)

        if self.class_token:
            N = Np + 1
            padded = torch.zeros(N, N, 3)
            padded[1:, 1:] = rel_indices  # patch-to-patch block unchanged
            # row/col 0 (the cls token) stays all zeros -> pos_proj(0,0,0)
            rel_indices = padded

        rel_indices = rel_indices.unsqueeze(0)  # (1, N, N, 3)
        self.rel_indices = rel_indices
        self.current_grid_size = (H, W)
        return rel_indices.to(device)

    def get_attention(self, x, H, W):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # each (B, heads, N, head_dim)

        # content-based attention
        attn_content = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # positional attention (shared across batch, depends only on grid geometry)
        rel_indices = self._get_rel_indices(H, W, x.device)  # (1, N, N, 3), N = H*W (+1 if class_token)
        attn_pos = self.pos_proj(rel_indices)  # (1, N, N, heads)
        attn_pos = attn_pos.permute(0, 3, 1, 2)  # (1, heads, N, N)

        attn_content = F.softmax(attn_content, dim=-1)
        attn_pos = F.softmax(attn_pos, dim=-1)

        gating = torch.sigmoid(self.gating_param).view(1, -1, 1, 1)  # (1, heads, 1, 1)
        attn = (1.0 - gating) * attn_content + gating * attn_pos
        attn = attn / attn.sum(dim=-1, keepdim=True)  # renormalize after mixing
        attn = self.attn_drop(attn)
        return attn

    def forward(self, x, H, W):
        """
        x: (B, N, C) token embeddings.
           If class_token=False: N = H * W.
           If class_token=True:  x = [cls_token; patch_tokens], N = 1 + H*W,
           with the cls token at index 0.
        H, W: spatial grid dimensions of the patch tokens.
        """
        B, N, C = x.shape
        expected_N = H * W + 1 if self.class_token else H * W
        assert N == expected_N, f"N={N} must equal {expected_N} (class_token={self.class_token})"

        attn = self.get_attention(x, H, W)  # (B, heads, N, N)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, N, hd)

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def get_attention_map(self, x, H, W, return_map=True):
        """Utility for visualizing/debugging locality: returns averaged attn map."""
        attn = self.get_attention(x, H, W)
        if return_map:
            return attn.mean(dim=1)  # average over heads -> (B, N, N)
        return attn


if __name__ == "__main__":
    # quick sanity check
    B, H, W, C, heads = 2, 14, 14, 192, 4
    x = torch.randn(B, H * W, C)
    gpsa = GPSA(dim=C, num_heads=heads, locality_strength=1.0)
    out = gpsa(x, H, W)
    print("output shape:", out.shape)
    assert out.shape == x.shape
    print("gating (sigmoid):", torch.sigmoid(gpsa.gating_param))
