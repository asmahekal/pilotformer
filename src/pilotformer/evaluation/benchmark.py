"""Benchmarks of Section VI: SNR sweep over channels x layouts, Doppler sweep and SNR-mismatch test."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pilotformer.channels.tdl import add_pilot_noise
from pilotformer.estimators.lmmse import empirical_covariance, lmmse_estimate
from pilotformer.evaluation.metrics import ber_zf, nmse_db
from pilotformer.models.registry import MODEL_ZOO
from pilotformer.pilots.geometry import PilotGeometry
from pilotformer.system import SimulationSystem
from pilotformer.utils import get_logger

log = get_logger()


class Benchmark:
    """Runs every estimator on identical channels, noise and data bits."""

    def __init__(self, system: SimulationSystem, models: dict[str, list[nn.Module]], chunk: int = 250):
        self.sys = system
        self.models = models
        self.chunk = chunk
        ev = system.cfg.eval
        self.amp = system.device.type == "cuda" and system.cfg.train.mixed_precision
        profiles = system.cfg.channel.train_profiles

        def train_sampler(bs: int) -> torch.Tensor:
            prof = profiles[np.random.randint(len(profiles))]
            return system.sample_train_channels(bs, prof)

        self.R_train = empirical_covariance(train_sampler, ev.n_covariance_samples, system.grid)

    # ------------------------------------------------------------------ helpers
    def _oracle_covariance(self, profile: str, fd: float, n: int | None = None) -> torch.Tensor:
        dev = self.sys.device
        return empirical_covariance(lambda bs: self.sys.channel(bs, profile, torch.full((bs,), float(fd), device=dev)),
                                    n or self.sys.cfg.eval.n_covariance_samples, self.sys.grid)

    @torch.no_grad()
    def estimate_all(self, h_ls: torch.Tensor, geom: PilotGeometry, R_oracle: torch.Tensor, s2_true: torch.Tensor,
                     s2_given: torch.Tensor | None = None, layout: str | None = None) -> dict[str, list[torch.Tensor]]:
        sg = s2_true if s2_given is None else s2_given
        hb = self.sys.bank(h_ls, geom, sg)
        out = {"LS+Interp": [h_ls @ geom.A.T],
               "LMMSE (train stats)": [lmmse_estimate(h_ls, sg, self.R_train, geom)],
               "Robust LMMSE": [hb[:, -1]]}
        for i, f in enumerate(self.sys.bank.doppler_hz):
            out[f"Robust LMMSE ({int(f)} Hz)"] = [hb[:, i]]
        out["LMMSE (oracle)"] = [lmmse_estimate(h_ls, s2_true, R_oracle, geom)]
        for name, ms in self.models.items():
            spec = MODEL_ZOO[name]
            if spec.exclusive_layout and spec.fixed_layout != layout:
                continue
            out[name] = []
            for m in ms:
                with torch.autocast(h_ls.device.type, dtype=torch.float16, enabled=self.amp):
                    out[name].append(torch.cat([m(h_ls[i:i + self.chunk], geom, sg[i:i + self.chunk]).to(torch.complex64)
                                                for i in range(0, len(h_ls), self.chunk)]))
        return out

    @staticmethod
    def add_genie(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        """'Robust bank (genie)': per test point, the best robust design chosen with hindsight."""
        rb = df[df.method.str.startswith("Robust LMMSE (")]
        best = rb.loc[rb.groupby(keys)["nmse_db"].idxmin()].assign(method="Robust bank (genie)")
        return pd.concat([df, best], ignore_index=True)

    # ------------------------------------------------------------------ benchmarks
    def snr_sweep(self) -> pd.DataFrame:
        ev, dev, sysm = self.sys.cfg.eval, self.sys.device, self.sys
        rows = []
        for cname, (prof, fd) in ev.conditions.items():
            g = torch.Generator(device=dev).manual_seed(1234)
            R_or = self._oracle_covariance(prof, fd)
            n = ev.n_test_slots
            H = sysm.channel(n, prof, torch.full((n,), float(fd), device=dev), generator=g).reshape(n, -1)
            for lname, mask in sysm.layouts.items():
                geom = sysm.geometry(mask)
                for snr in ev.snrs_db:
                    s2 = 10 ** (-snr / 10)
                    s2v = torch.full((n,), s2, device=dev)
                    gn = torch.Generator(device=dev).manual_seed(int(snr) + 7)
                    h_ls, _ = add_pilot_noise(H.view(-1, sysm.grid.K, sysm.grid.L), geom.idx,
                                              torch.full((n,), float(snr), device=dev), generator=gn)
                    ests = self.estimate_all(h_ls, geom, R_or, s2v, layout=lname)
                    ests["Perfect CSI"] = [H]
                    for mname, lst in ests.items():
                        for si, Hh in enumerate(lst):
                            gb = torch.Generator(device=dev).manual_seed(int(snr) + 99)   # same bits and noise for all
                            rows.append(dict(condition=cname, pattern=lname, pilots=int(mask.sum()), snr=snr, method=mname,
                                             seed=si, nmse_db=nmse_db(Hh, H) if mname != "Perfect CSI" else np.nan,
                                             ber=ber_zf(H, Hh, mask, s2, gb)))
            log.info("SNR sweep done: %s", cname)
        return self.add_genie(pd.DataFrame(rows), ["condition", "pattern", "snr"])

    def doppler_sweep(self, profile: str = "EVA", layout: str = "LTE-CRS", snr_db: float = 20.0) -> pd.DataFrame:
        ev, dev, sysm = self.sys.cfg.eval, self.sys.device, self.sys
        geom, n, rows = sysm.geometry(sysm.layouts[layout]), ev.n_test_slots, []
        s2 = 10 ** (-snr_db / 10)
        for fd in ev.doppler_sweep_hz:
            g = torch.Generator(device=dev).manual_seed(55)
            H = sysm.channel(n, profile, torch.full((n,), float(fd), device=dev), generator=g).reshape(n, -1)
            R_or = self._oracle_covariance(profile, fd, n=2000)
            s2v = torch.full((n,), s2, device=dev)
            h_ls, _ = add_pilot_noise(H.view(-1, sysm.grid.K, sysm.grid.L), geom.idx, torch.full((n,), snr_db, device=dev),
                                      generator=torch.Generator(device=dev).manual_seed(3))
            for mname, lst in self.estimate_all(h_ls, geom, R_or, s2v, layout=layout).items():
                rows += [dict(fd=fd, method=mname, seed=si, nmse_db=nmse_db(Hh, H)) for si, Hh in enumerate(lst)]
        return self.add_genie(pd.DataFrame(rows), ["fd"])

    def snr_mismatch(self, layout: str = "LTE-CRS", snr_db: float = 20.0) -> pd.DataFrame:
        ev, dev, sysm = self.sys.cfg.eval, self.sys.device, self.sys
        geom, n, rows = sysm.geometry(sysm.layouts[layout]), ev.n_test_slots, []
        s2 = 10 ** (-snr_db / 10)
        for cname, (prof, fd) in ev.conditions.items():
            g = torch.Generator(device=dev).manual_seed(77)
            H = sysm.channel(n, prof, torch.full((n,), float(fd), device=dev), generator=g).reshape(n, -1)
            R_or = self._oracle_covariance(prof, fd, n=2000)
            s2v = torch.full((n,), s2, device=dev)
            h_ls, _ = add_pilot_noise(H.view(-1, sysm.grid.K, sysm.grid.L), geom.idx, torch.full((n,), snr_db, device=dev),
                                      generator=torch.Generator(device=dev).manual_seed(5))
            for off in ev.snr_offsets_db:
                ests = self.estimate_all(h_ls, geom, R_or, s2v, s2_given=s2v * 10 ** (-off / 10), layout=layout)
                for mname, lst in ests.items():
                    rows += [dict(condition=cname, offset_db=off, method=mname, seed=si, nmse_db=nmse_db(Hh, H))
                             for si, Hh in enumerate(lst)]
        return pd.DataFrame(rows)


def summary_tables(results: pd.DataFrame, methods: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    """Pivot tables used in the paper (NMSE at 20 dB per channel/layout, NR-DMRS comparison, seed statistics)."""
    t = {}
    t["nmse_20dB"] = results[(results.snr == 20) & ~results.method.str.startswith("Robust LMMSE (")].pivot_table(
        index=["condition", "pattern"], columns="method", values="nmse_db").round(2)
    t["nmse_vs_snr_lte_crs"] = results[results.pattern == "LTE-CRS"].pivot_table(
        index=["condition", "method"], columns="snr", values="nmse_db").round(2)
    t["ber_vs_snr_lte_crs"] = results[results.pattern == "LTE-CRS"].pivot_table(index=["condition", "method"], columns="snr", values="ber")
    t["nrdmrs_published"] = results[results.pattern == "NR-DMRS"].pivot_table(
        index=["condition", "method"], columns="snr", values="nmse_db").round(2)
    g = results[(results.snr == 20) & results.method.isin(["PilotFormer", "CNN"])].groupby(["condition", "pattern", "method"])["nmse_db"]
    t["seeds_mean_std"] = pd.concat({"mean": g.mean().round(2), "std": g.std().fillna(0).round(2)}, axis=1).unstack("method")
    return t
