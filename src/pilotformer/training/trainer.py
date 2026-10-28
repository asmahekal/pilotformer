"""Training loop (Algorithm S1): fresh slots every step, one random pilot layout per step, AdamW + one-cycle, FP16."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pilotformer.channels.tdl import add_pilot_noise
from pilotformer.models.registry import MODEL_ZOO, build_model
from pilotformer.system import SimulationSystem
from pilotformer.training.losses import log_nmse_loss
from pilotformer.utils import get_logger

log = get_logger()


class Trainer:
    def __init__(self, system: SimulationSystem, output_dir: str | Path):
        self.sys = system
        self.out = Path(output_dir)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.out / "logs").mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, name: str, seed: int) -> Path:
        return self.out / "checkpoints" / f"{name}_s{seed}.pt"

    def load_or_train(self, name: str, seed: int, steps: int | None = None) -> nn.Module:
        model = build_model(name, self.sys.cfg, self.sys.grid, self.sys.bank).to(self.sys.device)
        ckpt = self.checkpoint_path(name, seed)
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=self.sys.device))
            log.info("[%s s%d] loaded %s", name, seed, ckpt)
            return model.eval()
        return self.fit(model, name, seed, steps)

    def fit(self, model: nn.Module, name: str, seed: int, steps: int | None = None) -> nn.Module:
        cfg, sysm = self.sys.cfg, self.sys
        spec = MODEL_ZOO[name]
        tc = cfg.train
        steps = steps or tc.steps
        lr = spec.lr or tc.lr
        dev = sysm.device
        amp = tc.mixed_precision and dev.type == "cuda"
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        fixed = sysm.layouts[spec.fixed_layout] if spec.fixed_layout else None

        model.to(dev).train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=min(0.3, max(0.05, 3 / steps)))
        scaler = torch.amp.GradScaler(enabled=amp)
        B, t0, hist = tc.batch_size, time.time(), []
        profiles = cfg.channel.train_profiles
        for it in range(1, steps + 1):
            mask = fixed if fixed is not None else sysm.sample_train_layout(rng)
            geom = sysm.geometry(mask)
            with torch.no_grad():
                prof = profiles[it % len(profiles)]
                fd = torch.empty(B, device=dev).uniform_(*cfg.channel.train_doppler_hz)
                snr = torch.empty(B, device=dev).uniform_(*cfg.channel.train_snr_db)
                ds = torch.empty(B, device=dev).uniform_(*cfg.channel.train_delay_scale) if spec.use_delay_scale else None
                H = sysm.channel(B, prof, fd, delay_scale=ds)
                h_ls, s2 = add_pilot_noise(H, geom.idx, snr)
                H = H.reshape(B, -1)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                H_hat = model(h_ls, geom, s2)
            loss, lin = log_nmse_loss(H_hat.to(torch.complex64), H)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            hist.append(lin.item())
            if it % max(1, min(tc.log_every, steps // 10 or 1)) == 0 or it == steps:
                log.info("[%s s%d] step %6d/%d  train NMSE %6.2f dB  (%.0f s)", name, seed, it, steps,
                         10 * np.log10(np.mean(hist[-200:])), time.time() - t0)
        curve = pd.DataFrame({"step": np.arange(1, steps + 1), "nmse": hist})
        curve.to_csv(self.out / "logs" / f"train_curve_{name}_s{seed}.csv", index=False)
        torch.save(model.state_dict(), self.checkpoint_path(name, seed))
        return model.eval()
