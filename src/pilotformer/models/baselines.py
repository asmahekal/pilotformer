"""Published baselines, re-implemented from the papers and trained with our data, loss and budget."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from pilotformer.grid import ResourceGrid
from pilotformer.pilots.geometry import PilotGeometry


class ChannelNet(nn.Module):
    """ChannelNet (Soltani et al., IEEE Commun. Lett. 2019): SRCNN + 20-layer DnCNN on the interpolated grid.

    Adaptation: real and imaginary parts are two channels of one network (the paper used two 1-channel networks).
    """

    def __init__(self, grid: ResourceGrid, depth: int = 20, channels: int = 64):
        super().__init__()
        self.grid = grid
        self.sr = nn.Sequential(nn.Conv2d(2, 64, 9, padding=4), nn.ReLU(), nn.Conv2d(64, 32, 1), nn.ReLU(),
                                nn.Conv2d(32, 2, 5, padding=2))
        layers = [nn.Conv2d(2, channels, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU()]
        layers += [nn.Conv2d(channels, 2, 3, padding=1)]
        self.dn = nn.Sequential(*layers)

    def forward(self, h_ls: torch.Tensor, geom: PilotGeometry, s2: torch.Tensor | None = None) -> torch.Tensor:
        K, L = self.grid.K, self.grid.L
        B = h_ls.shape[0]
        with torch.autocast(h_ls.device.type, enabled=False):
            hi = h_ls @ geom.A.T
        x = torch.view_as_real(hi.view(B, K, L)).permute(0, 3, 1, 2)
        y = self.sr(x)
        y = y - self.dn(y)                       # DnCNN predicts the noise
        y = y.float().permute(0, 2, 3, 1).contiguous()
        return torch.view_as_complex(y).reshape(B, K * L)


class Channelformer(nn.Module):
    """Channelformer (Luan and Thompson, IEEE TWC 2023), offline version, 117 587 parameters for 72 pilots.

    Encoder: FC(72 -> 216) -> 2-head attention -> FC -> Add&Norm -> conv pre-network -> Add&Norm.
    Decoder: conv(12 @ 5x5) + 3 residual blocks + FC up-sampling (72 -> K*L) + conv(1 @ 5x5).
    Its fully connected layers fix the number of pilots, so it exists for one layout only.
    """

    def __init__(self, grid: ResourceGrid, n_pilots: int = 72, heads: int = 2, filters: int = 12, blocks: int = 3, ks: int = 5):
        super().__init__()
        self.n_in, self.heads = n_pilots, heads
        self.fc_in = nn.Linear(n_pilots, 3 * n_pilots)
        self.fc_o = nn.Linear(n_pilots, n_pilots)
        self.ln1 = nn.LayerNorm([n_pilots, 2])
        self.ln2 = nn.LayerNorm([n_pilots, 2])
        self.pre = nn.Sequential(nn.Conv2d(1, 5, 2, padding="same"), nn.GELU(), nn.Conv2d(5, 1, 2, padding="same"))
        self.c0 = nn.Conv2d(1, filters, ks, padding="same")
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(filters, filters, ks, padding="same"), nn.ReLU(),
                                                   nn.Conv2d(filters, filters, ks, padding="same")) for _ in range(blocks)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, filters) for _ in range(blocks)])
        self.up = nn.Linear(n_pilots, grid.size)
        self.cout = nn.Conv2d(filters, 1, ks, padding="same")

    def forward(self, h_ls: torch.Tensor, geom: PilotGeometry | None = None, s2: torch.Tensor | None = None) -> torch.Tensor:
        if h_ls.shape[1] != self.n_in:
            raise ValueError(f"Channelformer is fixed to {self.n_in} pilots, got {h_ls.shape[1]}")
        B = h_ls.shape[0]
        x = torch.view_as_real(h_ls).float()
        qkv = self.fc_in(x.transpose(1, 2)).transpose(1, 2)
        q, k, v = qkv.split(self.n_in, 1)
        dh = self.n_in // self.heads
        q, k, v = (t.reshape(B, self.heads, dh, 2) for t in (q, k, v))
        a = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(dh), -1)
        o = (v @ a.transpose(-1, -2)).reshape(B, self.n_in, 2)
        x = self.ln1(x + self.fc_o(o.transpose(1, 2)).transpose(1, 2))
        x = self.ln2(x + self.pre(x[:, None]).squeeze(1))
        y = self.c0(x[:, None])
        for blk, nrm in zip(self.blocks, self.norms, strict=True):
            y = nrm(y + blk(y))
        y = self.up(y.transpose(2, 3)).transpose(2, 3)
        y = self.cout(y).squeeze(1).float().contiguous()
        return torch.view_as_complex(y)
