import numpy as np
import torch

from pilotformer.estimators import lmmse_estimate, robust_covariance


def test_filter_bank_matches_direct_lmmse(system):
    """Eigen-trick bank estimate (Eq. 15) equals R_GP (R_PP + s2 I)^-1 h for every design and noise level."""
    geom = system.geometry(system.layouts["Sparse"])
    h = torch.randn(3, geom.P, dtype=torch.complex64)
    s2 = torch.tensor([0.01, 0.1, 1.0])
    hb = system.bank(h, geom, s2)
    for b, fd in enumerate(system.bank.doppler_hz):
        R = torch.tensor(robust_covariance(system.grid, system.bank.tau_max, fd), dtype=torch.complex64)
        ref = torch.cat([lmmse_estimate(h[i:i + 1], s2[i:i + 1], R, geom) for i in range(3)])
        assert torch.allclose(hb[:, b], ref, atol=2e-3, rtol=1e-3)


def test_robust_covariance_is_valid(system):
    R = robust_covariance(system.grid, 5.52e-6, 400.0)
    assert np.allclose(np.diag(R), 1.0)
    assert np.allclose(R, R.conj().T)
    assert np.linalg.eigvalsh(R).min() > -1e-8


def test_tau_max_covers_training_delays(system):
    assert abs(system.bank.tau_max - 5.522e-6) < 1e-9
