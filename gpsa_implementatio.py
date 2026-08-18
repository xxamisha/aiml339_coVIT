"""
Gated Positional Self-Attention (GPSA), ConViT-style (d'Ascoli et al., 2021).

Core idea: attention = convex combination of content attention and positional
attention, mixed per-head by a learnable gate.

  A_h = (1 - sigmoid(lambda_h)) * softmax(Q_h K_h^T / sqrt(d))     [content]
      +        sigmoid(lambda_h)  * softmax(pos_proj(rel_ij))       [positional]

The positional term depends only on relative position (dx, dy, dx^2+dy^2)
between patches i,j - not on content - so at init it can be shaped into a
Gaussian-like local receptive field per head (locality bias, conv-like).
lambda_h is learned, so each head/layer can shift toward pure content
attention (standard ViT) if that's what training favors.

Assumes a square patch grid (N = num_patches = grid_size^2), consistent with
Phikon-v2 / DINOv2 ViT-L/16 on square input crops.
"""

import math
import torch
import torch.nn as nn


class GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
                 locality_strength=1.0, use_local_init=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # rel position (dx, dy, dx^2+dy^2) -> per-head positional attention score
        self.pos_proj = nn.Linear(3, num_heads)
        # per-head gate; init near 1.0 -> sigmoid(~1) biases toward positional/local
        # attention at the start of training (matches ConViT's "start local, can
        # relax to global if useful" behavior)
        self.gating_param = nn.Parameter(torch.ones(num_heads))

        self.locality_strength = locality_strength
        self.rel_indices = None  # cached, built lazily per sequence length

        self.apply(self._init_weights)
        if use_local_init:
            self._local_init(locality_strength)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _local_init(self, locality_strength):
        """Shape pos_proj so each head's positional attention starts as a
        Gaussian centered on a distinct offset in a sqrt(num_heads) x sqrt(num_heads)
        grid - approximates a set of small conv-like receptive fields, one per head."""
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1
        kernel_size = int(self.num_heads ** 0.5)
        center = (kernel_size - 1) / 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                if position >= self.num_heads:
                    continue
                # weight on dx^2+dy^2 term: negative -> penalizes distance (locality)
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.pos_proj.weight.data *= locality_strength

    def _get_rel_indices(self, num_patches, device):
        grid_size = int(math.isqrt(num_patches))
        assert grid_size * grid_size == num_patches, \
            f"GPSA assumes a square patch grid; got num_patches={num_patches}"
        rel = torch.zeros(1, num_patches, num_patches, 3, device=device)
        ind = torch.arange(grid_size, device=device).view(1, -1) - torch.arange(grid_size, device=device).view(-1, 1)
        indx = ind.repeat(grid_size, grid_size)
        indy = ind.repeat_interleave(grid_size, dim=0).repeat_interleave(grid_size, dim=1)
        rel[:, :, :, 0] = indx
        rel[:, :, :, 1] = indy
        rel[:, :, :, 2] = indx ** 2 + indy ** 2
        return rel

    def get_attention(self, x):
        B, N, C = x.shape
        if self.rel_indices is None or self.rel_indices.shape[1] != N:
            self.rel_indices = self._get_rel_indices(N, x.device)

        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # (B, heads, N, head_dim)

        content_score = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        content_attn = content_score.softmax(dim=-1)

        pos_score = self.pos_proj(self.rel_indices.expand(B, -1, -1, -1))  # (B, N, N, heads)
        pos_score = pos_score.permute(0, 3, 1, 2)  # (B, heads, N, N)
        pos_attn = pos_score.softmax(dim=-1)

        gate = torch.sigmoid(self.gating_param).view(1, -1, 1, 1)
        attn = (1.0 - gate) * content_attn + gate * pos_attn
        attn = attn / attn.sum(dim=-1, keepdim=True)  # renormalize after mixing
        return self.attn_drop(attn)

    def forward(self, x):
        B, N, C = x.shape
        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out