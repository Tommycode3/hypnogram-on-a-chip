import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from tinysleep.metrics import report


def test_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 2000)
    p = np.where(rng.random(2000) < 0.7, y, rng.integers(0, 5, 2000))
    r = report(y, p)
    assert abs(r["kappa"] - cohen_kappa_score(y, p)) < 1e-12
    assert abs(r["macro_f1"] - f1_score(y, p, average="macro")) < 1e-12
    assert np.allclose(r["f1"], f1_score(y, p, average=None))
    assert abs(r["acc"] - np.mean(y == p)) < 1e-12
    assert sum(map(sum, r["confusion"])) == 2000


def test_perfect_and_chance():
    y = np.arange(5).repeat(100)
    assert report(y, y)["kappa"] == 1.0
    assert abs(report(y, np.roll(y, 1))["kappa"] - (0.99 - 0.2) / 0.8) < 1e-12
