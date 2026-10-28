"""Per-layout geometry: pilot indices, coordinates, squared distances and a 2-D linear interpolator (cached)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from pilotformer.grid import ResourceGrid


@dataclass
class PilotGeometry:
    key: bytes
    idx: torch.Tensor          # (P,) flat pilot indices, q = k * L + l
    A: torch.Tensor            # (KL, P) piecewise-linear interpolation (nearest neighbour outside the convex hull)
    cp: torch.Tensor           # (P, 2) normalized pilot coordinates
    dqp: torch.Tensor          # (KL, P, 2) squared (sub-carrier, symbol) distances RE -> pilot
    dpp: torch.Tensor          # (P, P, 2) squared distances pilot -> pilot

    @property
    def P(self) -> int:
        return int(self.idx.numel())

    # dict-style access keeps the models compatible with the Kaggle notebook
    def __getitem__(self, name: str):
        return self.P if name == "P" else getattr(self, name)


class GeometryCache:
    def __init__(self, grid: ResourceGrid):
        self.grid = grid
        self._cache: dict[bytes, PilotGeometry] = {}
        self._re = grid.re_indices

    def __call__(self, mask: np.ndarray) -> PilotGeometry:
        key = mask.tobytes()
        if key not in self._cache:
            self._cache[key] = self._build(mask, key)
        return self._cache[key]

    def _build(self, mask: np.ndarray, key: bytes) -> PilotGeometry:
        g, re = self.grid, self._re
        idx = np.flatnonzero(mask.ravel())
        pts = re[idx]
        P = len(idx)
        if len(np.unique(pts[:, 1])) == 1 or len(np.unique(pts[:, 0])) == 1:
            A = np.full((g.size, P), np.nan)          # degenerate (collinear) layout: nearest neighbour only
        else:
            A = LinearNDInterpolator(pts, np.eye(P))(re)
        outside = np.isnan(A[:, 0])
        if outside.any():
            nn_ = cKDTree(pts).query(re[outside])[1]
            A[outside] = 0.0
            A[np.flatnonzero(outside), nn_] = 1.0
        coords = re / np.array([g.K - 1, g.L - 1])
        dev = g.device
        return PilotGeometry(
            key=key,
            idx=torch.tensor(idx, device=dev),
            A=torch.tensor(A, dtype=torch.complex64, device=dev),
            cp=torch.tensor(coords[idx], dtype=torch.float32, device=dev),
            dqp=torch.tensor((re[:, None, :] - pts[None]) ** 2, dtype=torch.float32, device=dev),
            dpp=torch.tensor((pts[:, None, :] - pts[None]) ** 2, dtype=torch.float32, device=dev),
        )
