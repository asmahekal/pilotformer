"""Experiment configuration (dataclasses, loadable from YAML)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GridConfig:
    n_subcarriers: int = 72          # K (6 PRBs)
    n_symbols: int = 14              # L (one 1-ms slot)
    subcarrier_spacing: float = 15e3  # Hz
    slot_duration: float = 1e-3      # s, symbols include the cyclic prefix


@dataclass
class ChannelConfig:
    train_profiles: list[str] = field(default_factory=lambda: ["EPA", "EVA"])
    train_doppler_hz: tuple[float, float] = (0.0, 1000.0)
    train_snr_db: tuple[float, float] = (0.0, 30.0)
    train_delay_scale: tuple[float, float] = (0.5, 2.2)
    n_sinusoids: int = 16
    tdl_delay_spread_ns: float = 300.0


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 3
    decoder_mlp_ratio: int = 2
    fourier_frequencies: int = 10
    bank_doppler_hz: list[float] = field(default_factory=lambda: [100.0, 400.0, 1000.0])


@dataclass
class TrainConfig:
    steps: int = 20000
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    mixed_precision: bool = True
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    baseline_seeds: list[int] = field(default_factory=lambda: [0])
    ablations: bool = True
    log_every: int = 2000


@dataclass
class EvalConfig:
    snrs_db: list[float] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    n_test_slots: int = 1000
    n_covariance_samples: int = 5000
    doppler_sweep_hz: list[float] = field(default_factory=lambda: [0, 100, 200, 300, 400, 500, 600, 750, 900, 1000, 1200, 1400])
    snr_offsets_db: list[float] = field(default_factory=lambda: [-3.0, -1.5, 0.0, 1.5, 3.0])
    conditions: dict[str, list[Any]] = field(default_factory=lambda: {
        "EVA-200Hz (seen)": ["EVA", 200.0],
        "EVA-750Hz (high Doppler)": ["EVA", 750.0],
        "EVA-1200Hz (beyond training)": ["EVA", 1200.0],
        "ETU-200Hz (unseen profile)": ["ETU", 200.0],
        "TDL-A-200Hz (3GPP 38.901)": ["TDL-A", 200.0],
        "TDL-C-200Hz (3GPP 38.901)": ["TDL-C", 200.0],
    })


@dataclass
class ExperimentConfig:
    name: str = "pilotformer_r"
    output_dir: str = "runs/default"
    device: str = "auto"
    grid: GridConfig = field(default_factory=GridConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc_type, data: dict[str, Any] | None):
    obj = dc_type()
    if not data:
        return obj
    known = {f.name: f for f in fields(dc_type)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"unknown config key '{key}' for {dc_type.__name__}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__"):
            value = _merge(type(current), value)
        elif isinstance(current, tuple):
            value = tuple(value)
        setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from YAML; ``overrides`` uses dotted keys, e.g. ``{"train.steps": 100}``."""
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, value in (overrides or {}).items():
        node = data
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return _merge(ExperimentConfig, data)
