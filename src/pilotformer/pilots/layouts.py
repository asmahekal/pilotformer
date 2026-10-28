"""Pilot layouts: the four test layouts and the randomized comb family used for training."""

from __future__ import annotations

import numpy as np


def comb_layout(K: int, L: int, symbols: list[int], k_period: int, k_offsets: list[int]) -> np.ndarray:
    """Boolean (K, L) mask with pilots every ``k_period`` sub-carriers on ``symbols``; offsets cycle over the symbols."""
    m = np.zeros((K, L), bool)
    for i, l in enumerate(symbols):
        m[k_offsets[i % len(k_offsets)]::k_period, l] = True
    return m


def test_layouts(K: int = 72, L: int = 14) -> dict[str, np.ndarray]:
    """LTE-CRS-like (48 pilots), NR-DMRS-like (72), Sparse (18) and Block (144); none is in the training family."""
    return {
        "LTE-CRS": comb_layout(K, L, [0, 4, 7, 11], 6, [0, 3]),
        "NR-DMRS": comb_layout(K, L, [2, 11], 2, [0]),
        "Sparse": comb_layout(K, L, [3, 10], 8, [0, 4]),
        "Block": comb_layout(K, L, [0, 7], 1, [0]),
    }


def random_train_layout(rng: np.random.Generator, K: int = 72, L: int = 14) -> np.ndarray:
    """Draw from F_train: symbol period 3..7, sub-carrier period {2,3,4,6}, random offsets, stagger with p = 1/2."""
    lp = int(rng.choice([3, 4, 5, 6, 7]))
    kp = int(rng.choice([2, 3, 4, 6]))
    l0 = int(rng.integers(0, lp))
    k0 = int(rng.integers(0, kp))
    offs = [k0, (k0 + kp // 2) % kp] if rng.random() < 0.5 else [k0]
    return comb_layout(K, L, list(range(l0, L, lp)), kp, offs)
