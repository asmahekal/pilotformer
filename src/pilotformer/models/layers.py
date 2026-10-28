"""Building blocks: Fourier coordinate features, locality-biased attention (Eq. 12), pre-norm blocks (Eqs. 13-14)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """gamma(c) = [c, sin(2^j pi c), cos(2^j pi c)]_{j<J} followed by a two-layer perceptron (Eq. 10)."""

    def __init__(self, n_freq: int = 10, d: int = 128):
        super().__init__()
        self.register_buffer("f", (2.0 ** torch.arange(n_freq)) * math.pi)
        self.proj = nn.Sequential(nn.Linear(4 * n_freq + 2, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        a = c[..., None] * self.f
        return self.proj(torch.cat([c, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], -1))


class SNREmbedding(nn.Module):
    """e(sigma^2) = MLP(log10(sigma^2) / 3)."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, s2: torch.Tensor) -> torch.Tensor:
        return self.net((torch.log10(s2) / 3.0).view(-1, 1).float())


class LocalityBiasedAttention(nn.Module):
    """Multi-head attention with logits q.k / sqrt(d_h) - alpha_h dk^2 - beta_h dl^2 (alpha, beta = softplus, per head).

    Heads are initialized from local to almost global: (0.05, 0.5), (0.01, 0.1), (0.002, 0.02), (1e-5, 1e-4).
    """

    INIT = ((0.05, 0.5), (0.01, 0.1), (0.002, 0.02), (1e-5, 1e-4))

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        init = torch.tensor(self.INIT)[:heads] if heads <= 4 else torch.full((heads, 2), 0.01)
        self.raw = nn.Parameter(torch.log(torch.expm1(init)))   # inverse softplus

    def bandwidths(self) -> torch.Tensor:
        """(heads, 2) learned (alpha_h, beta_h)."""
        return F.softplus(self.raw)

    def forward(self, xq: torch.Tensor, xkv: torch.Tensor, dist2: torch.Tensor, need_weights: bool = False):
        B, Q, _ = xq.shape
        P = xkv.shape[1]
        q = self.q(xq).view(B, Q, self.h, self.dh).transpose(1, 2)
        k = self.k(xkv).view(B, P, self.h, self.dh).transpose(1, 2)
        v = self.v(xkv).view(B, P, self.h, self.dh).transpose(1, 2)
        bias = -(dist2 @ self.bandwidths().T).permute(2, 0, 1)[None].to(q.dtype)
        if need_weights:
            w = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh) + bias, -1)
            out, w = w @ v, w.mean(1)
        else:
            out, w = F.scaled_dot_product_attention(q, k, v, attn_mask=bias), None
        return self.o(out.transpose(1, 2).reshape(B, Q, -1)), w


class AttentionBlock(nn.Module):
    """Pre-norm attention + MLP. ``memory=None`` gives self-attention, otherwise cross-attention to ``memory``."""

    def __init__(self, d: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.nq, self.nk, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = LocalityBiasedAttention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None, dist2: torch.Tensor, need_weights: bool = False):
        xn = self.nq(x)
        kv = self.nk(memory) if memory is not None else xn
        a, w = self.att(xn, kv, dist2, need_weights)
        x = x + a
        return x + self.mlp(self.n2(x)), w


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b(x)
