"""CNN-R: a ReEsNet-style residual CNN with exactly the inputs and output head of PilotFormer-R (attention control)."""

from __future__ import annotations

import torch
import torch.nn as nn

from pilotformer.estimators.lmmse import RobustFilterBank
from pilotformer.grid import ResourceGrid
from pilotformer.models.layers import ResBlock
from pilotformer.models.pilotformer import stack_grids
from pilotformer.pilots.geometry import PilotGeometry


class CNNR(nn.Module):
    def __init__(self, grid: ResourceGrid, bank: RobustFilterBank, channels: int = 64, n_blocks: int = 8):
        super().__init__()
        self.grid, self.bank = grid, bank
        nb = len(bank)
        self.inp = nn.Conv2d(2 + 2 * nb + 1, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.out = nn.Conv2d(channels, 2 + nb, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h_ls: torch.Tensor, geom: PilotGeometry, s2: torch.Tensor) -> torch.Tensor:
        K, L = self.grid.K, self.grid.L
        B, nb = h_ls.shape[0], len(self.bank)
        with torch.autocast(h_ls.device.type, enabled=False):
            hi = h_ls @ geom.A.T
            hb = self.bank(h_ls, geom, s2)
        g = stack_grids([hi[:, None], hb], K, L)
        snr_plane = (torch.log10(s2) / 3.0).view(B, 1, 1, 1).expand(B, 1, K, L).to(g.dtype)
        o = self.out(self.body(self.inp(torch.cat([g, snr_plane], 1)))).float()
        delta = torch.complex(o[:, 0], o[:, 1]).reshape(B, K * L)
        a = torch.softmax(o[:, 2:].reshape(B, nb, K * L), 1)
        return (a * hb).sum(1) + delta
