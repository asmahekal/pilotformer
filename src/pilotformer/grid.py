"""OFDM resource grid: dimensions, timing and normalized RE coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pilotformer.config import GridConfig


@dataclass(frozen=True)
class ResourceGrid:
    K: int
    L: int
    subcarrier_spacing: float
    symbol_duration: float
    device: torch.device

    @classmethod
    def from_config(cls, cfg: GridConfig, device: torch.device | str = "cpu") -> ResourceGrid:
        return cls(cfg.n_subcarriers, cfg.n_symbols, cfg.subcarrier_spacing, cfg.slot_duration / cfg.n_symbols, torch.device(device))

    @property
    def size(self) -> int:
        return self.K * self.L

    @property
    def re_indices(self) -> np.ndarray:
        """(KL, 2) array of (sub-carrier, symbol) indices in row-major order, q = k * L + l."""
        return np.stack(np.meshgrid(np.arange(self.K), np.arange(self.L), indexing="ij"), -1).reshape(-1, 2).astype(float)

    def normalized_coordinates(self) -> torch.Tensor:
        """(KL, 2) coordinates in [0, 1]^2 used as attention queries."""
        c = self.re_indices / np.array([self.K - 1, self.L - 1])
        return torch.tensor(c, dtype=torch.float32, device=self.device)
