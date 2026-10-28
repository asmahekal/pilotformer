<div align="center">

 # PilotFormer-R
#    Copyright (c) 2026 Asmaa A. Hekal

**A Layout-Agnostic Transformer with a Robust-LMMSE Prior for OFDM Channel Estimation under Unseen Pilot Patterns**

[![CI](https://github.com/asmahekal/pilotformer/actions/workflows/ci.yml/badge.svg)](https://github.com/asmahekal/pilotformer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23721076.svg)](https://doi.org/10.5281/zenodo.23721076)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

PilotFormer-R estimates the full OFDM channel of a slot from **any pilot layout**, including layouts it never saw during
training, with **one** trained model. The pilots are treated as an unordered set of coordinate-tagged tokens, every
resource element (RE) is a coordinate query decoded by locality-biased cross-attention, and the output is a per-RE mixture
of closed-form robust LMMSE filters plus a learned residual.

```
 received pilots {(k_p, l_p, h_p)}  ──► pilot tokens ──► set encoder (locality-biased self-attention) ──┐
          │                                                                                               ▼
          ├──► robust-LMMSE filter bank (closed form, any layout, any SNR) ──► RE queries ──► grid decoder ──► per-RE head
          │                                                                                               │
 noise variance σ² ───────────────────────────────────────────────────────────────────────────────────────┘
                                   Ĥ(q) = Σ_b softmax(w_q)_b · Ĥ_b(q) + Δ_q
```

## Highlights

| | |
|---|---|
| **Layout agnostic by construction** | no parameter depends on the number or position of the pilots; the estimator is invariant to pilot ordering (Proposition 1, checked in `tests/test_models.py`) |
| **Zero-shot to unseen layouts** | trained on randomized comb layouts only, it serves LTE-CRS-like, NR-DMRS-like, sparse and block pilots |
| **Estimation-theoretic prior** | robust LMMSE filter bank evaluated per layout with one eigendecomposition, then O(KLP) per slot for any noise variance |
| **Accuracy** | most accurate practical estimator in 17 of 24 channel/layout cases at 20 dB; 0.8 to 7.4 dB better than Channelformer on Channelformer's own layout at 200 Hz (20 to 30 dB SNR) |
| **Speed** | 0.49 ms per 1-ms slot on an NVIDIA T4 (FP16) |

The cases where another estimator is better (high Doppler with few pilot symbols, Doppler beyond the training range,
18 sparse pilots on ETU) are reported in the paper.

## Installation

```bash
git clone https://github.com/asmahekal/pilotformer.git
cd pilotformer
pip install -e ".[dev]"
```

## Quick start

```bash
pilotformer info                                   # grid, layouts, filter bank, model sizes
pilotformer reproduce --config configs/smoke.yaml  # end-to-end check on CPU (a few minutes)
pilotformer reproduce --config configs/paper.yaml  # full study of the paper (about 6-7 h on one T4)
```

Train or evaluate a single model, or override any setting from the command line:

```bash
pilotformer train    --config configs/paper.yaml --model PilotFormer --seed 0
pilotformer train    --config configs/paper.yaml --model Channelformer --set train.steps=5000
pilotformer evaluate --config configs/paper.yaml
```

Python API:

```python
import torch
from pilotformer import load_config
from pilotformer.system import SimulationSystem
from pilotformer.models import build_model
from pilotformer.channels import add_pilot_noise

cfg = load_config("configs/paper.yaml")
sysm = SimulationSystem.from_config(cfg)
model = build_model("PilotFormer", cfg, sysm.grid, sysm.bank).to(sysm.device).eval()

geom = sysm.geometry(sysm.layouts["NR-DMRS"])                       # any boolean (K, L) mask works
H = sysm.channel(8, "TDL-A", torch.full((8,), 200.0, device=sysm.device))
h_ls, s2 = add_pilot_noise(H, geom.idx, torch.full((8,), 20.0, device=sysm.device))
with torch.no_grad():
    H_hat = model(h_ls, geom, s2)                                    # (8, K*L) complex
```

## Repository structure

```
pilotformer/
├── configs/
│   ├── paper.yaml              # settings of the paper
│   └── smoke.yaml              # tiny end-to-end run used by CI
├── src/pilotformer/
│   ├── config.py               # typed experiment configuration (YAML + dotted overrides)
│   ├── grid.py                 # OFDM resource grid
│   ├── system.py               # shared simulation context
│   ├── channels/               # EPA/EVA/ETU (TS 36.104), TDL-A/C (TR 38.901), sum-of-sinusoids Doppler
│   ├── pilots/                 # test layouts, randomized training family, per-layout geometry and interpolation
│   ├── estimators/             # LMMSE, robust covariance, robust-LMMSE filter bank
│   ├── models/                 # PilotFormer-R, CNN-R, ChannelNet, Channelformer, model registry
│   ├── training/               # log-NMSE loss and trainer
│   ├── evaluation/             # NMSE / BER, benchmarks, complexity, plots
│   └── cli.py                  # `pilotformer` command
├── tests/                      # unit tests (layouts, channels, filter bank, invariance, training)
├── notebooks/
│   └── pilotformer_r_kaggle.ipynb   # single-notebook version for Kaggle
├── docs/method.md              # method and experiment details
└── .github/workflows/ci.yml    # lint + tests + smoke run
```

## Models

| Name | Role | Parameters |
|---|---|---:|
| `PilotFormer` | proposed model (3 seeds) | 1.24 M |
| `PilotFormer-Lite` | d = 64, 2 + 2 blocks | 0.18 M |
| `PilotFormer-noPrior` | ablation without the filter bank | 1.24 M |
| `PilotFormer-fixed` | ablation trained on LTE-CRS only | 1.24 M |
| `CNN` | CNN-R, same inputs and head (attention control) | 0.60 M |
| `ChannelNet` | Soltani et al., IEEE Commun. Lett. 2019 | 0.68 M |
| `Channelformer` | Luan and Thompson, IEEE TWC 2023 (fixed to 72 pilots) | 0.12 M |

## Running on Kaggle

`notebooks/pilotformer_r_kaggle.ipynb` contains the same experiment in one notebook: choose *Accelerator: GPU T4* and run
*Save & Run All*. Set the environment variables `SEEDS=0` and `STEPS=3000` for a quick trial. The package and the notebook
implement the same models and channel generator: their outputs agree exactly for the same weights and seed, and checkpoints
saved by the notebook (`<name>_s<seed>.pt`) load into the package unchanged when placed in `<output_dir>/checkpoints/`.

## Tests

```bash
pytest          # 45 tests, CPU, about 2 minutes
ruff check src tests
```

The tests check, among other things, that the eigendecomposition-based filter bank equals the direct LMMSE formula, that
the training family contains 750 layouts with 24 to 180 pilots and none of the test layouts, that the model is invariant
to pilot permutations, and that the zero-initialized head starts from the average of the filter bank.

## Citation

The paper is under review. Citation details will be added here (see `CITATION.cff`).

## License

MIT, see [LICENSE](LICENSE).
