"""Classical estimators: LMMSE (Eq. 5), robust covariance (Eqs. 6-7) and the robust-LMMSE filter bank (Eq. 15, Alg. 1)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from scipy.special import j0

from pilotformer.grid import ResourceGrid
from pilotformer.pilots.geometry import PilotGeometry


def robust_covariance(grid: ResourceGrid, tau_max: float, doppler_hz: float) -> np.ndarray:
    """Li-Cimini-Sollenberger design: uniform PDP on [0, tau_max] (x) Jakes spectrum, R = R_f kron R_t."""
    dk = np.arange(grid.K)[:, None] - np.arange(grid.K)[None, :]
    x = 2 * np.pi * dk * grid.subcarrier_spacing * tau_max
    Rf = np.where(dk == 0, 1.0 + 0j, (1 - np.exp(-1j * x)) / (1j * np.where(dk == 0, 1, x)))
    dl = np.arange(grid.L)[:, None] - np.arange(grid.L)[None, :]
    Rt = j0(2 * np.pi * doppler_hz * dl * grid.symbol_duration)
    return np.kron(Rf, Rt)


@torch.no_grad()
def empirical_covariance(sampler: Callable[[int], torch.Tensor], n: int, grid: ResourceGrid, chunk: int = 1000) -> torch.Tensor:
    R = torch.zeros(grid.size, grid.size, dtype=torch.complex64, device=grid.device)
    for _ in range(max(1, n // chunk)):
        H = sampler(chunk).reshape(chunk, -1)
        R += H.T @ H.conj()
    return R / (max(1, n // chunk) * chunk)


def lmmse_estimate(h_ls: torch.Tensor, s2: torch.Tensor, R: torch.Tensor, geom: PilotGeometry) -> torch.Tensor:
    """h_hat = R_GP (R_PP + s2 I)^(-1) h_ls, solved once per distinct noise variance in the batch."""
    idx = geom.idx
    Rhp, Rpp = R[:, idx], R[idx][:, idx]
    out = torch.empty(h_ls.shape[0], R.shape[0], dtype=torch.complex64, device=h_ls.device)
    for s in torch.unique(s2):
        M = Rpp + s * torch.eye(len(idx), device=h_ls.device)
        WT = torch.linalg.solve(M.T, Rhp.T)
        sel = s2 == s
        out[sel] = h_ls[sel] @ WT
    return out


class RobustFilterBank:
    """B robust LMMSE filters with designs (tau_max, f_D,b), evaluated for any layout and any per-slot noise variance.

    Per layout (cached):  R_PP^(b) = U_b Lambda_b U_b^H,  A_b = R_GP^(b) U_b.
    Per slot:             h_b = A_b (Lambda_b + s2 I)^(-1) U_b^H h_ls.
    """

    def __init__(self, grid: ResourceGrid, tau_max: float, doppler_hz: list[float]):
        self.grid = grid
        self.tau_max = tau_max
        self.doppler_hz = list(doppler_hz)
        self._R = [robust_covariance(grid, tau_max, f) for f in self.doppler_hz]
        self._cache: dict[bytes, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self._R)

    def factors(self, geom: PilotGeometry) -> dict[str, torch.Tensor]:
        if geom.key not in self._cache:
            idx = geom.idx.cpu().numpy()
            As, UHs, lams = [], [], []
            for R in self._R:
                lam, U = np.linalg.eigh(R[np.ix_(idx, idx)])
                As.append(R[:, idx] @ U)
                UHs.append(U.conj().T)
                lams.append(np.clip(lam, 0, None))
            dev = self.grid.device
            self._cache[geom.key] = dict(A=torch.tensor(np.stack(As), dtype=torch.complex64, device=dev),
                                         UH=torch.tensor(np.stack(UHs), dtype=torch.complex64, device=dev),
                                         lam=torch.tensor(np.stack(lams), dtype=torch.float32, device=dev))
        return self._cache[geom.key]

    def __call__(self, h_ls: torch.Tensor, geom: PilotGeometry, s2: torch.Tensor) -> torch.Tensor:
        """Return the bank estimates, shape (B, n_filters, KL)."""
        f = self.factors(geom)
        with torch.autocast(h_ls.device.type, enabled=False):
            z = torch.einsum("bp,nqp->bnq", h_ls.to(torch.complex64), f["UH"])
            z = z / (f["lam"][None] + s2.view(-1, 1, 1))
            return torch.einsum("bnp,nkp->bnk", z, f["A"])
