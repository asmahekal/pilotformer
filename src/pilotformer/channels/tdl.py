"""Tapped-delay-line channel with sum-of-sinusoids (Jakes) Doppler, Eqs. (2)-(3) of the paper."""

from __future__ import annotations

import math

import torch

from pilotformer.channels.profiles import build_profiles
from pilotformer.grid import ResourceGrid


class TDLChannel:
    """Generates H[k, l] on a resource grid; E|H|^2 = 1. ICI is neglected (f_D / delta_f <= 0.067 in our setting)."""

    def __init__(self, grid: ResourceGrid, n_sinusoids: int = 16, tdl_delay_spread_ns: float = 300.0):
        self.grid = grid
        self.n_sin = n_sinusoids
        self.profiles = build_profiles(tdl_delay_spread_ns)
        dev = grid.device
        self._k = torch.arange(grid.K, device=dev, dtype=torch.float32)
        self._t = torch.arange(grid.L, device=dev, dtype=torch.float32) * grid.symbol_duration

    def max_excess_delay(self, profile: str) -> float:
        return self.profiles[profile][0][-1] * 1e-9

    def __call__(self, batch: int, profile: str, doppler_hz: torch.Tensor, delay_scale: torch.Tensor | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Return H with shape (batch, K, L), complex64.

        doppler_hz: (batch,) maximum Doppler shift per slot.
        delay_scale: optional (batch,) factor that stretches the profile delays (delay-spread randomization).
        """
        g_ = self.grid
        dev = g_.device
        tau_ns, p_db = self.profiles[profile]
        tau = torch.tensor(tau_ns, device=dev, dtype=torch.float32) * 1e-9
        tau = tau.expand(batch, -1) * (delay_scale.view(batch, 1) if delay_scale is not None else 1.0)
        pw = 10 ** (torch.tensor(p_db, device=dev) / 10)
        pw = pw / pw.sum()
        P = tau.shape[-1]
        theta = torch.rand(batch, P, self.n_sin, device=dev, generator=generator) * 2 * math.pi
        phi = torch.rand(batch, P, self.n_sin, device=dev, generator=generator) * 2 * math.pi
        phase = 2 * math.pi * doppler_hz.view(batch, 1, 1, 1) * torch.cos(theta)[..., None] * self._t.view(1, 1, 1, g_.L) + phi[..., None]
        gains = torch.polar(torch.ones_like(phase), phase).sum(2) * torch.sqrt(pw / self.n_sin).view(1, P, 1)
        steer = torch.polar(torch.ones(batch, P, g_.K, device=dev),
                            -2 * math.pi * g_.subcarrier_spacing * tau[..., None] * self._k.view(1, 1, g_.K))
        return torch.einsum("bpl,bpk->bkl", gains, steer)


def add_pilot_noise(H: torch.Tensor, pilot_index: torch.Tensor, snr_db: torch.Tensor,
                    generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """LS pilot observations (pilots x = 1), Eq. (4). Returns (h_ls of shape (B, P), noise variance (B,))."""
    B = H.shape[0]
    s2 = 10 ** (-snr_db / 10)
    hp = H.reshape(B, -1)[:, pilot_index]
    n = torch.complex(torch.randn(hp.shape, device=H.device, generator=generator),
                      torch.randn(hp.shape, device=H.device, generator=generator))
    return hp + n * torch.sqrt(s2 / 2).view(B, 1), s2
