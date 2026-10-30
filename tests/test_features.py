import numpy as np

from tinysleep.features import FEATURE_NAMES, add_context, epoch_features, hjorth, spectral_entropy, psd

FS = 100
t = np.arange(3000) / FS


def test_alpha_sine_lands_in_alpha_band():
    x = 30 * np.sin(2 * np.pi * 10 * t)[None]
    F = epoch_features(x)[0]
    rel = {n: F[FEATURE_NAMES.index(n)] for n in FEATURE_NAMES if n.startswith("rel_")}
    assert max(rel, key=rel.get) == "rel_alpha" and rel["rel_alpha"] > 0.95


def test_hjorth_mobility_of_sine():
    # for a sampled sine, var(diff x)/var(x) = 2 - 2 cos(w) -> mobility = 2 sin(w/2)
    f = 5.0
    x = np.sin(2 * np.pi * f * t)[None]
    _, mob, comp = hjorth(x)
    assert abs(mob[0] - 2 * np.sin(np.pi * f / FS)) < 1e-3
    assert abs(comp[0] - 1.0) < 1e-2  # a pure tone has complexity ~1


def test_spectral_entropy_orders_noise_above_tone():
    rng = np.random.default_rng(0)
    X = np.stack([np.sin(2 * np.pi * 10 * t), rng.standard_normal(3000)])
    f, P = psd(X)
    h = spectral_entropy(f, P)
    assert h[0] < 0.3 < 0.9 < h[1] <= 1.0


def test_causal_context():
    F = np.arange(4, dtype=float)[:, None]
    C = add_context(F, past=2)
    assert C.tolist() == [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 2, 1]]
