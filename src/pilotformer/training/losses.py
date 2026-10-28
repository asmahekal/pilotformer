"""Training objective, Eq. (18): mean per-slot NMSE in the log domain."""

from __future__ import annotations

import torch


def log_nmse_loss(H_hat: torch.Tensor, H: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (loss, linear NMSE). The log domain gives high- and low-SNR slots the same weight."""
    e = ((H_hat - H).abs() ** 2).sum(1).div((H.abs() ** 2).sum(1))
    return torch.log10(e + eps).mean(), e.mean()
