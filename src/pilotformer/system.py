"""Everything the experiments share: grid, channel generator, pilot geometry, filter bank and test layouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pilotformer.channels.tdl import TDLChannel
from pilotformer.config import ExperimentConfig
from pilotformer.estimators.lmmse import RobustFilterBank
from pilotformer.grid import ResourceGrid
from pilotformer.pilots.geometry import GeometryCache
from pilotformer.pilots.layouts import random_train_layout, test_layouts
from pilotformer.utils import resolve_device


@dataclass
class SimulationSystem:
    cfg: ExperimentConfig
    grid: ResourceGrid
    channel: TDLChannel
    geometry: GeometryCache
    bank: RobustFilterBank
    layouts: dict[str, np.ndarray]

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> SimulationSystem:
        grid = ResourceGrid.from_config(cfg.grid, resolve_device(cfg.device))
        ch = cfg.channel
        channel = TDLChannel(grid, ch.n_sinusoids, ch.tdl_delay_spread_ns)
        # tau_max = largest excess delay of the training channels after delay stretching (5.52 us)
        tau_max = max(channel.max_excess_delay(p) for p in ch.train_profiles) * ch.train_delay_scale[1]
        bank = RobustFilterBank(grid, tau_max, cfg.model.bank_doppler_hz)
        return cls(cfg, grid, channel, GeometryCache(grid), bank, test_layouts(grid.K, grid.L))

    @property
    def device(self) -> torch.device:
        return self.grid.device

    def sample_train_layout(self, rng: np.random.Generator) -> np.ndarray:
        return random_train_layout(rng, self.grid.K, self.grid.L)

    def sample_train_channels(self, batch: int, profile: str, use_delay_scale: bool = True) -> torch.Tensor:
        ch, dev = self.cfg.channel, self.device
        fd = torch.empty(batch, device=dev).uniform_(*ch.train_doppler_hz)
        ds = torch.empty(batch, device=dev).uniform_(*ch.train_delay_scale) if use_delay_scale else None
        return self.channel(batch, profile, fd, delay_scale=ds)
