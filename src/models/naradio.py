"""
NACLIP attention module for RADIO's timm ViT backbone.

Drop-in replacement for the last block's attention: K-K similarity
(k_i @ k_j^T) + Gaussian locality bias, matching GaussKernelAttn
from the NARADIO reference implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_bias(
    ph: int, pw: int, sigma: float, num_prefix_tokens: int, device
) -> torch.Tensor:
    """
    Build the [1, total, total] Gaussian locality bias for RADIO's attention.

    Patch-to-patch entries are Gaussian(distance); prefix-token rows/cols are zero.
    """
    window_h, window_w = 2 * ph - 1, 2 * pw - 1
    c = 1.0 / (sigma * math.sqrt(2))
    k1 = torch.linspace(-(ph - 1) * c, (ph - 1) * c, window_h, dtype=torch.float32, device=device)
    k2 = torch.linspace(-(pw - 1) * c, (pw - 1) * c, window_w, dtype=torch.float32, device=device)
    dist_sq = (torch.stack(torch.meshgrid(k1, k2, indexing="ij")) ** 2).sum(0)  # [2ph-1, 2pw-1]
    window = torch.exp(-dist_sq)  # [2ph-1, 2pw-1]

    m = torch.einsum("ij,kl->ijkl", torch.eye(ph, device=device), torch.eye(pw, device=device))
    m = m.permute(0, 3, 1, 2).contiguous()  # [ph, pw, ph, pw]
    out = F.conv2d(
        m.view(-1, ph, pw).unsqueeze(1),          # [ph*pw, 1, ph, pw]
        window.unsqueeze(0).unsqueeze(1),          # [1, 1, 2ph-1, 2pw-1]
        padding=(ph - 1, pw - 1),
    ).squeeze(1)                                   # [ph*pw, ph, pw]
    out = out.view(ph * pw, ph * pw)               # [N, N]

    N = ph * pw
    if num_prefix_tokens > 0:
        pad_rows = torch.zeros(num_prefix_tokens, N, device=device)
        pad_cols = torch.zeros(N + num_prefix_tokens, num_prefix_tokens, device=device)
        out = torch.hstack([pad_cols, torch.vstack([pad_rows, out])])

    return out.unsqueeze(0)  # [1, total, total]


class RadioNaclipAttn(nn.Module):
    """
    NACLIP attention for RADIO's timm ViT backbone.
    """

    def __init__(self, orig_attn: nn.Module, bias: torch.Tensor) -> None:
        super().__init__()
        self.num_heads = orig_attn.num_heads
        self.scale = orig_attn.scale
        self.register_buffer("_bias", bias)  # [1, total, total]

        self.qkv = orig_attn.qkv
        self.q_norm = orig_attn.q_norm
        self.k_norm = orig_attn.k_norm
        self.attn_drop = orig_attn.attn_drop
        self.proj = orig_attn.proj
        self.proj_drop = orig_attn.proj_drop

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        out = self._custom_attn(x.permute(1, 0, 2))
        return out.permute(1, 0, 2)

    def _custom_attn(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // self.num_heads

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        # K-K similarity (NACLIP)
        w = torch.bmm(k, k.transpose(1, 2)) * self.scale
        w = w + self._bias.to(dtype=w.dtype)
        w = F.softmax(w, dim=-1)

        out = torch.bmm(w, v)
        out = out.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
