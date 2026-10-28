"""Quick-look figures from the benchmark results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STYLE = {
    "PilotFormer": ("#2a78d6", "-", "o"), "CNN": ("#1baf7a", "-", "s"), "ChannelNet": ("#e87ba4", "-", "D"),
    "Channelformer": ("#6250d6", "-", "v"), "LMMSE (train stats)": ("#eda100", "-", "^"), "Robust LMMSE": ("#008300", "-", "x"),
    "Robust bank (genie)": ("#e34948", "-.", "+"), "LS+Interp": ("#eb6834", "-", "<"), "LMMSE (oracle)": ("#0b0b0b", "--", None),
    "PilotFormer-fixed": ("#2a78d6", ":", "o"),
}


def _draw(ax, s: pd.Series, method: str) -> None:
    c, ls, m = STYLE.get(method, ("#52514e", "-", None))
    ax.plot(s.index, s.values, ls, color=c, marker=m, ms=4, lw=1.6, label=method)


def nmse_vs_snr(results: pd.DataFrame, layout: str, out: Path) -> None:
    conds = list(dict.fromkeys(results.condition))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, c in zip(axes.ravel(), conds, strict=False):
        d = results[(results.condition == c) & (results.pattern == layout)]
        for m in STYLE:
            s = d[d.method == m].groupby("snr")["nmse_db"].mean()
            if len(s):
                _draw(ax, s, m)
        ax.set_title(c)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("SNR (dB)")
    for ax in axes[:, 0]:
        ax.set_ylabel("NMSE (dB)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def doppler_sweep(sweep: pd.DataFrame, out: Path, train_max_hz: float = 1000.0) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvspan(train_max_hz, sweep.fd.max() + 50, color="#f3f2ee", lw=0)
    for m in STYLE:
        s = sweep[sweep.method == m].groupby("fd")["nmse_db"].mean()
        if len(s):
            _draw(ax, s, m)
    ax.set(xlabel="Maximum Doppler shift (Hz)", ylabel="NMSE (dB)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
