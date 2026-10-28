"""NMSE (Eq. 9) and uncoded 16-QAM BER with zero-forcing equalization."""

from __future__ import annotations

import math

import numpy as np
import torch


def nmse_db(H_hat: torch.Tensor, H: torch.Tensor) -> float:
    nm = ((H_hat - H).abs() ** 2).sum().item() / (H.abs() ** 2).sum().item()
    return float(10 * np.log10(nm + 1e-12))


_PAM = (-3.0, -1.0, 3.0, 1.0)   # Gray: 00, 01, 10, 11


def qam16_modulate(bits: torch.Tensor) -> torch.Tensor:
    pam = torch.tensor(_PAM, device=bits.device) / math.sqrt(10)
    return torch.complex(pam[bits[..., 0] * 2 + bits[..., 1]], pam[bits[..., 2] * 2 + bits[..., 3]])


def _pam_demod(x: torch.Tensor) -> torch.Tensor:
    x = x * math.sqrt(10)
    return torch.stack([(x > 0).long(), (x.abs() < 2).long()], -1)


def ber_zf(H: torch.Tensor, H_hat: torch.Tensor, pilot_mask: np.ndarray, s2: float, generator: torch.Generator) -> float:
    """Transmit random 16-QAM on the data REs, equalize with H_hat (zero forcing) and count bit errors."""
    B, KL = H.shape
    dev = H.device
    data = torch.tensor(~pilot_mask.ravel(), device=dev)
    bits = torch.randint(0, 2, (B, KL, 4), device=dev, generator=generator)
    X = qam16_modulate(bits)
    N = torch.complex(torch.randn(B, KL, device=dev, generator=generator),
                      torch.randn(B, KL, device=dev, generator=generator)) * math.sqrt(s2 / 2)
    Y = H * X + N
    X_hat = Y * H_hat.conj() / (H_hat.abs() ** 2 + 1e-12)
    b_hat = torch.cat([_pam_demod(X_hat.real), _pam_demod(X_hat.imag)], -1)
    return (b_hat != bits)[:, data].float().mean().item()
