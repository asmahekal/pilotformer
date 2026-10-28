import torch

from pilotformer.channels import add_pilot_noise


def test_shape_and_unit_power(system):
    H = system.channel(4000, "EVA", torch.full((4000,), 200.0))
    assert H.shape == (4000, 72, 14) and H.dtype == torch.complex64
    assert abs((H.abs() ** 2).mean().item() - 1.0) < 0.05


def test_static_channel_is_constant_in_time(system):
    H = system.channel(8, "ETU", torch.zeros(8))
    assert torch.allclose(H, H[..., :1].expand_as(H), atol=1e-5)


def test_all_profiles_available(system):
    for p in ("EPA", "EVA", "ETU", "TDL-A", "TDL-C"):
        assert system.channel(2, p, torch.full((2,), 100.0)).shape == (2, 72, 14)


def test_pilot_noise_level(system):
    H = system.channel(512, "EVA", torch.full((512,), 100.0))
    geom = system.geometry(system.layouts["NR-DMRS"])
    h_ls, s2 = add_pilot_noise(H, geom.idx, torch.full((512,), 10.0))
    err = (h_ls - H.reshape(512, -1)[:, geom.idx]).abs().pow(2).mean().item()
    assert abs(err - 0.1) < 0.01 and torch.allclose(s2, torch.full((512,), 0.1))
