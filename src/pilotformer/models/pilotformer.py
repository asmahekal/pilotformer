"""PilotFormer-R (Section IV, Algorithm 2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from pilotformer.estimators.lmmse import RobustFilterBank
from pilotformer.grid import ResourceGrid
from pilotformer.models.layers import AttentionBlock, FourierFeatures, SNREmbedding
from pilotformer.pilots.geometry import PilotGeometry


def stack_grids(grids: list[torch.Tensor], K: int, L: int) -> torch.Tensor:
    """List of complex (B, n_i, KL) tensors -> real image (B, 2 * sum n_i, K, L)."""
    g = torch.cat(grids, 1)
    B = g.shape[0]
    return torch.view_as_real(g.view(B, -1, K, L)).permute(0, 1, 4, 2, 3).reshape(B, -1, K, L)


class PilotFormerR(nn.Module):
    """Layout-agnostic set-to-grid Transformer with a robust-LMMSE filter-bank prior.

    Pilots -> tokens (Eq. 11) -> set encoder (N_e self-attention blocks) -> pilot memory X.
    REs   -> queries (Eq. 16) -> grid decoder (N_d cross-attention blocks) -> per-RE head:
             H(q) = sum_b softmax(w_q)_b H_b(q) + Delta_q  (Eq. 17), head zero-initialized.
    No parameter depends on the number or position of the pilots (Proposition 1).
    """

    def __init__(self, grid: ResourceGrid, bank: RobustFilterBank | None, d: int = 128, heads: int = 4,
                 enc_layers: int = 4, dec_layers: int = 3, dec_mlp: int = 2, n_freq: int = 10,
                 use_prior: bool = True, use_snr: bool = True):
        super().__init__()
        self.grid, self.bank = grid, bank
        self.use_prior, self.use_snr = use_prior and bank is not None, use_snr
        nb = len(bank) if self.use_prior else 0
        self.pos = FourierFeatures(n_freq, d)
        self.pin = nn.Linear(2, d)
        self.snr = SNREmbedding(d)
        self.stem = nn.Sequential(nn.Conv2d(2 + 2 * nb, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, d, 1))
        self.enc = nn.ModuleList([AttentionBlock(d, heads) for _ in range(enc_layers)])
        self.dec = nn.ModuleList([AttentionBlock(d, heads, dec_mlp) for _ in range(dec_layers)])
        self.norm = nn.LayerNorm(d)
        self.res = nn.Linear(d, 2)
        nn.init.zeros_(self.res.weight)
        nn.init.zeros_(self.res.bias)
        if self.use_prior:
            self.mix = nn.Linear(d, nb)
            nn.init.zeros_(self.mix.weight)
            nn.init.zeros_(self.mix.bias)
        self.register_buffer("cq", grid.normalized_coordinates(), persistent=False)

    def forward(self, h_ls: torch.Tensor, geom: PilotGeometry, s2: torch.Tensor, return_attention: bool = False):
        K, L = self.grid.K, self.grid.L
        B = h_ls.shape[0]
        with torch.autocast(h_ls.device.type, enabled=False):
            h_interp = h_ls @ geom.A.T
            h_bank = self.bank(h_ls, geom, s2) if self.use_prior else None
        e = self.snr(s2)[:, None] if self.use_snr else 0.0
        x = self.pin(torch.view_as_real(h_ls)) + self.pos(geom.cp)[None] + e
        for blk in self.enc:
            x, _ = blk(x, None, geom.dpp)
        img = stack_grids([h_interp[:, None]] + ([h_bank] if self.use_prior else []), K, L)
        q = self.pos(self.cq)[None] + self.stem(img).permute(0, 2, 3, 1).reshape(B, K * L, -1) + e
        w = None
        for i, blk in enumerate(self.dec):
            q, w = blk(q, x, geom.dqp, need_weights=return_attention and i == len(self.dec) - 1)
        q = self.norm(q)
        delta = torch.view_as_complex(self.res(q).float().contiguous())
        if self.use_prior:
            a = torch.softmax(self.mix(q).float(), -1)
            H = (a.transpose(1, 2) * h_bank).sum(1) + delta
        else:
            H = h_interp + delta
        return (H, w) if return_attention else H
