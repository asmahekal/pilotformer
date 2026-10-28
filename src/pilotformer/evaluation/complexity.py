"""Parameters, FLOPs per slot (PyTorch FLOP counter) and FP16 latency."""

from __future__ import annotations

import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from pilotformer.estimators.lmmse import lmmse_estimate
from pilotformer.models.registry import MODEL_ZOO
from pilotformer.system import SimulationSystem
from pilotformer.utils import count_parameters


@torch.no_grad()
def _latency(fn, reps: int = 20) -> float:
    for _ in range(3):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t) / reps


def _mflops(fn, batch: int) -> float:
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / batch / 1e6


def complexity_table(system: SimulationSystem, models: dict[str, nn.Module], R_train: torch.Tensor, batch: int = 256) -> pd.DataFrame:
    dev, KL = system.device, system.grid.size
    amp = dev.type == "cuda"
    geoms = {k: system.geometry(v) for k, v in system.layouts.items()}
    s2 = torch.full((batch,), 0.01, device=dev)
    h = {k: torch.randn(batch, g.P, dtype=torch.complex64, device=dev) for k, g in geoms.items()}
    g0 = geoms["LTE-CRS"]

    def fp16(fn):
        def run():
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                return fn()
        return run

    rows = [dict(method="LMMSE (one pattern/SNR, filter precomputed)", params=0, mflops_per_slot=8 * KL * g0.P / 1e6,
                 ms_per_slot=1e3 * _latency(lambda: lmmse_estimate(h["LTE-CRS"], s2, R_train, g0)) / batch),
            dict(method="Robust LMMSE bank (3 filters, any SNR)", params=0,
                 mflops_per_slot=_mflops(lambda: system.bank(h["LTE-CRS"], g0, s2), batch),
                 ms_per_slot=1e3 * _latency(lambda: system.bank(h["LTE-CRS"], g0, s2)) / batch)]
    for name, m in models.items():
        lay = MODEL_ZOO[name].fixed_layout if MODEL_ZOO[name].exclusive_layout else "LTE-CRS"
        def f(m=m, lay=lay):
            return m(h[lay], geoms[lay], s2)

        rows.append(dict(method=name, params=count_parameters(m), mflops_per_slot=_mflops(f, batch),
                         ms_per_slot=1e3 * _latency(fp16(f)) / batch))
    return pd.DataFrame(rows).round(4)
