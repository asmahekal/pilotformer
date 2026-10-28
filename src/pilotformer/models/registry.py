"""Model zoo: every estimator of the paper, built from one config."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from pilotformer.config import ExperimentConfig
from pilotformer.estimators.lmmse import RobustFilterBank
from pilotformer.grid import ResourceGrid
from pilotformer.models.baselines import Channelformer, ChannelNet
from pilotformer.models.cnn_r import CNNR
from pilotformer.models.pilotformer import PilotFormerR


@dataclass
class ModelSpec:
    build: Callable[[ExperimentConfig, ResourceGrid, RobustFilterBank], nn.Module]
    lr: float | None = None                 # None: use train.lr
    fixed_layout: str | None = None         # train on one layout only (and evaluate only there if exclusive)
    exclusive_layout: bool = False
    use_delay_scale: bool = True
    tags: list[str] = field(default_factory=list)


def _pf(cfg, grid, bank, **kw):
    m = cfg.model
    args = dict(d=m.d_model, heads=m.n_heads, enc_layers=m.encoder_layers, dec_layers=m.decoder_layers,
                dec_mlp=m.decoder_mlp_ratio, n_freq=m.fourier_frequencies)
    args.update(kw)
    return PilotFormerR(grid, bank, **args)


MODEL_ZOO: dict[str, ModelSpec] = {
    "PilotFormer": ModelSpec(lambda c, g, b: _pf(c, g, b), tags=["proposed", "seeds"]),
    "CNN": ModelSpec(lambda c, g, b: CNNR(g, b), tags=["control", "seeds"]),
    "PilotFormer-Lite": ModelSpec(lambda c, g, b: _pf(c, g, b, d=64, enc_layers=2, dec_layers=2), tags=["ablation"]),
    "PilotFormer-noPrior": ModelSpec(lambda c, g, b: _pf(c, g, b, use_prior=False), tags=["ablation"]),
    "PilotFormer-fixed": ModelSpec(lambda c, g, b: _pf(c, g, b), fixed_layout="LTE-CRS", tags=["ablation"]),
    "PilotFormer-noDS": ModelSpec(lambda c, g, b: _pf(c, g, b), use_delay_scale=False, tags=["optional"]),
    "ChannelNet": ModelSpec(lambda c, g, b: ChannelNet(g), lr=1e-3, tags=["published"]),
    "Channelformer": ModelSpec(lambda c, g, b: Channelformer(g), lr=2e-3, fixed_layout="NR-DMRS", exclusive_layout=True,
                               tags=["published"]),
}


def build_model(name: str, cfg: ExperimentConfig, grid: ResourceGrid, bank: RobustFilterBank) -> nn.Module:
    if name not in MODEL_ZOO:
        raise KeyError(f"unknown model '{name}', choose from {sorted(MODEL_ZOO)}")
    return MODEL_ZOO[name].build(cfg, grid, bank)
