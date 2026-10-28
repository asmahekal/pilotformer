import pytest
import torch

from pilotformer.models import MODEL_ZOO, build_model
from pilotformer.utils import count_parameters


def _inputs(system, layout, B=2):
    geom = system.geometry(system.layouts[layout])
    h = torch.randn(B, geom.P, dtype=torch.complex64)
    return h, geom, torch.full((B,), 0.05)


@pytest.mark.parametrize("name", [n for n in MODEL_ZOO if n != "Channelformer"])
@pytest.mark.parametrize("layout", ["LTE-CRS", "NR-DMRS", "Sparse", "Block"])
def test_layout_agnostic_models_run_on_every_layout(system, name, layout):
    m = build_model(name, system.cfg, system.grid, system.bank).eval()
    h, geom, s2 = _inputs(system, layout)
    with torch.no_grad():
        out = m(h, geom, s2)
    assert out.shape == (2, system.grid.size) and torch.isfinite(torch.view_as_real(out)).all()


def test_channelformer_is_fixed_to_72_pilots(system):
    m = build_model("Channelformer", system.cfg, system.grid, system.bank)
    assert count_parameters(m) == 117_587
    with pytest.raises(ValueError):
        m(*_inputs(system, "LTE-CRS"))


def test_permutation_invariance(system):
    """Proposition 1: permuting the pilots (indices, coordinates, LS values) does not change the estimate."""
    torch.manual_seed(0)
    m = build_model("PilotFormer", system.cfg, system.grid, system.bank).eval()
    for p in m.parameters():                                   # move away from the zero-initialized head
        torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else None
    mask = system.layouts["LTE-CRS"]
    geom = system.geometry(mask)
    h = torch.randn(2, geom.P, dtype=torch.complex64)
    s2 = torch.full((2,), 0.05)
    perm = torch.randperm(geom.P)
    from dataclasses import replace
    gp = replace(geom, idx=geom.idx[perm], A=geom.A[:, perm], cp=geom.cp[perm], dqp=geom.dqp[:, perm],
                 dpp=geom.dpp[perm][:, perm], key=b"permuted")
    with torch.no_grad():
        a, b = m(h, geom, s2), m(h[:, perm], gp, s2)
    assert torch.allclose(a, b, atol=1e-4)


def test_zero_initialized_head_returns_bank_average(system):
    m = build_model("PilotFormer", system.cfg, system.grid, system.bank).eval()
    h, geom, s2 = _inputs(system, "NR-DMRS")
    with torch.no_grad():
        out = m(h, geom, s2)
    assert torch.allclose(out, system.bank(h, geom, s2).mean(1), atol=1e-5)


def test_paper_model_size(system):
    from pilotformer.config import load_config
    cfg = load_config(None, {"device": "cpu"})
    assert count_parameters(build_model("PilotFormer", cfg, system.grid, system.bank)) == 1_239_005
