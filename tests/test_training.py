import torch

from pilotformer.training import Trainer, log_nmse_loss


def test_log_nmse_loss():
    H = torch.ones(4, 10, dtype=torch.complex64)
    loss, lin = log_nmse_loss(H * 1.1, H)
    assert abs(lin.item() - 0.01) < 1e-6 and abs(loss.item() + 2.0) < 1e-3


def test_training_writes_and_reloads_checkpoint(system, tmp_path):
    trainer = Trainer(system, tmp_path)
    model = trainer.load_or_train("CNN", seed=0, steps=10)
    assert trainer.checkpoint_path("CNN", 0).exists()
    assert (tmp_path / "logs" / "train_curve_CNN_s0.csv").exists()
    reloaded = trainer.load_or_train("CNN", seed=0)
    for a, b in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True):
        assert torch.equal(a, b)
