import numpy as np

from pilotformer.pilots import random_train_layout


def test_test_layout_pilot_counts(system):
    counts = {k: int(v.sum()) for k, v in system.layouts.items()}
    assert counts == {"LTE-CRS": 48, "NR-DMRS": 72, "Sparse": 18, "Block": 144}


def test_training_family_size_and_range():
    rng = np.random.default_rng(0)
    layouts = {random_train_layout(rng).tobytes() for _ in range(20000)}
    sizes = {int(np.frombuffer(k, bool).sum()) for k in layouts}
    assert len(layouts) == 750
    assert min(sizes) == 24 and max(sizes) == 180


def test_test_layouts_are_unseen():
    rng = np.random.default_rng(1)
    family = {random_train_layout(rng).tobytes() for _ in range(20000)}
    from pilotformer.pilots import test_layouts
    assert not any(m.tobytes() in family for m in test_layouts().values())


def test_interpolator_reproduces_pilots(system):
    geom = system.geometry(system.layouts["LTE-CRS"])
    A = geom.A.real.numpy()
    np.testing.assert_allclose(A[geom.idx.numpy()], np.eye(geom.P), atol=1e-6)
    np.testing.assert_allclose(A.sum(1), 1.0, atol=1e-6)
