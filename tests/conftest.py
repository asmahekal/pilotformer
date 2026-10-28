import pytest
import torch

from pilotformer.config import load_config
from pilotformer.system import SimulationSystem


@pytest.fixture(scope="session")
def system():
    cfg = load_config(None, {"device": "cpu", "model.d_model": 32, "model.encoder_layers": 1, "model.decoder_layers": 1,
                       "train.batch_size": 16, "train.mixed_precision": False})
    torch.manual_seed(0)
    return SimulationSystem.from_config(cfg)
