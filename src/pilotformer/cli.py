"""Command-line interface.

    pilotformer info       --config configs/paper.yaml
    pilotformer train      --config configs/paper.yaml --model PilotFormer --seed 0
    pilotformer evaluate   --config configs/paper.yaml
    pilotformer reproduce  --config configs/paper.yaml          # train everything, then evaluate
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import yaml

from pilotformer.config import ExperimentConfig, load_config
from pilotformer.evaluation import Benchmark, complexity_table, plots, summary_tables
from pilotformer.models.registry import MODEL_ZOO
from pilotformer.system import SimulationSystem
from pilotformer.training import Trainer
from pilotformer.utils import count_parameters, get_logger, seed_everything

log = get_logger()


def _parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        out[key] = yaml.safe_load(value)
    return out


def _plan(cfg: ExperimentConfig) -> list[tuple[str, int]]:
    """Which (model, seed) pairs the paper trains."""
    tc = cfg.train
    plan = [(m, s) for m in ("PilotFormer", "CNN") for s in tc.seeds]
    if tc.ablations:
        plan += [(m, tc.seeds[0]) for m in ("PilotFormer-Lite", "PilotFormer-noPrior", "PilotFormer-fixed")]
    plan += [(m, s) for m in ("ChannelNet", "Channelformer") for s in tc.baseline_seeds]
    return plan


def cmd_info(cfg: ExperimentConfig, args) -> None:
    sysm = SimulationSystem.from_config(cfg)
    print(f"device: {sysm.device} | grid {sysm.grid.K} x {sysm.grid.L} | tau_max = {sysm.bank.tau_max * 1e6:.2f} us | "
          f"bank designs {sysm.bank.doppler_hz} Hz")
    for name, mask in sysm.layouts.items():
        print(f"  layout {name:8s}: {int(mask.sum()):3d} pilots ({100 * mask.mean():.1f}% overhead)")
    for name, spec in MODEL_ZOO.items():
        m = spec.build(cfg, sysm.grid, sysm.bank)
        print(f"  model  {name:20s} {count_parameters(m):>10,d} parameters  [{', '.join(spec.tags)}]")


def cmd_train(cfg: ExperimentConfig, args) -> None:
    sysm = SimulationSystem.from_config(cfg)
    trainer = Trainer(sysm, cfg.output_dir)
    for name, seed in ([(args.model, args.seed)] if args.model else _plan(cfg)):
        seed_everything(seed)
        trainer.load_or_train(name, seed)


def cmd_evaluate(cfg: ExperimentConfig, args) -> None:
    sysm = SimulationSystem.from_config(cfg)
    trainer = Trainer(sysm, cfg.output_dir)
    out = Path(cfg.output_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    models: dict[str, list] = {}
    for name, seed in _plan(cfg):
        if trainer.checkpoint_path(name, seed).exists():
            models.setdefault(name, []).append(trainer.load_or_train(name, seed))
    if not models:
        raise SystemExit(f"no checkpoints in {Path(cfg.output_dir) / 'checkpoints'}; run `pilotformer train` first")
    seed_everything(0)
    bench = Benchmark(sysm, models)
    res = bench.snr_sweep()
    res.to_csv(out / "results_all.csv", index=False)
    for key, table in summary_tables(res).items():
        table.to_csv(out / f"table_{key}.csv")
    sweep = bench.doppler_sweep()
    sweep.to_csv(out / "doppler_sweep.csv", index=False)
    bench.snr_mismatch().to_csv(out / "snr_mismatch.csv", index=False)
    complexity_table(sysm, {k: v[0] for k, v in models.items()}, bench.R_train).to_csv(out / "complexity.csv", index=False)
    plots.nmse_vs_snr(res, "LTE-CRS", out / "nmse_vs_snr_lte_crs.png")
    plots.doppler_sweep(sweep, out / "doppler_sweep.png", cfg.channel.train_doppler_hz[1])
    meta = dict(config=cfg.to_dict(), torch=torch.__version__, python=platform.python_version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    (out / "run_info.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("results written to %s", out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pilotformer", description="PilotFormer-R experiments")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("info", "train", "evaluate", "reproduce"):
        s = sub.add_parser(name)
        s.add_argument("--config", default="configs/paper.yaml")
        s.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override, e.g. train.steps=500")
        if name == "train":
            s.add_argument("--model", choices=sorted(MODEL_ZOO), help="train one model only")
            s.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    cfg = load_config(args.config, _parse_overrides(args.set))
    if args.command == "info":
        cmd_info(cfg, args)
    elif args.command == "train":
        cmd_train(cfg, args)
    elif args.command == "evaluate":
        cmd_evaluate(cfg, args)
    else:
        args.model = None
        cmd_train(cfg, args)
        cmd_evaluate(cfg, args)


if __name__ == "__main__":
    main()
