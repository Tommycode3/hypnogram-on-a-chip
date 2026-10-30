"""Hand-crafted per-epoch EEG features for the classical baseline.

Per 30 s epoch x[n] (n = 0..N-1, fs = 100 Hz):
  * Welch PSD (2 s Hann windows, 50 % overlap) -> absolute log band powers and
    relative band powers for delta/theta/alpha/sigma/beta;
  * spectral entropy of the normalised 0.5-30 Hz PSD;
  * Hjorth activity / mobility / complexity;
  * a few ratios that track stage transitions (theta/alpha, delta/beta, ...).
Context features: the same vector for epochs t-1 and t+1 (and t-2 / t+2) can be
appended with :func:`add_context`; for a causal (real-time) model use only past
epochs.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

FS = 100
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 15.0),
    "beta": (15.0, 30.0),
}
EPS = 1e-12


def psd(X: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per epoch. X: (n, N) -> f: (F,), P: (n, F) in uV^2/Hz."""
    f, P = welch(X, fs=fs, window="hann", nperseg=2 * fs, noverlap=fs, axis=-1)
    return f, P


def band_power(f: np.ndarray, P: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (f >= lo) & (f < hi)
    df = f[1] - f[0]
    return P[:, m].sum(axis=1) * df


def spectral_entropy(f: np.ndarray, P: np.ndarray, lo: float = 0.5, hi: float = 30.0) -> np.ndarray:
    """H = -sum p_k log2 p_k / log2 K, with p_k = P_k / sum P (0..1)."""
    m = (f >= lo) & (f < hi)
    p = P[:, m] / (P[:, m].sum(axis=1, keepdims=True) + EPS)
    return -(p * np.log2(p + EPS)).sum(axis=1) / np.log2(m.sum())


def hjorth(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Activity = var(x); mobility = sqrt(var(x')/var(x)); complexity = mob(x')/mob(x)."""
    dx = np.diff(X, axis=-1)
    ddx = np.diff(dx, axis=-1)
    v0, v1, v2 = X.var(axis=-1), dx.var(axis=-1), ddx.var(axis=-1)
    mob = np.sqrt(v1 / (v0 + EPS))
    comp = np.sqrt(v2 / (v1 + EPS)) / (mob + EPS)
    return v0, mob, comp


FEATURE_NAMES = (
    [f"log_{b}" for b in BANDS]
    + [f"rel_{b}" for b in BANDS]
    + ["log_total", "spec_entropy", "log_activity", "mobility", "complexity"]
    + ["theta_alpha", "delta_beta", "sigma_rel_theta", "alpha_rel_delta", "log_ptp", "zero_cross"]
)


def epoch_features(X: np.ndarray) -> np.ndarray:
    """X: (n, 3000) uV -> (n, len(FEATURE_NAMES)) float32."""
    X = X - X.mean(axis=-1, keepdims=True)
    f, P = psd(X)
    bp = np.stack([band_power(f, P, lo, hi) for lo, hi in BANDS.values()], axis=1)
    total = band_power(f, P, 0.5, 30.0)
    rel = bp / (total[:, None] + EPS)
    act, mob, comp = hjorth(X)
    d, t, a, s, b = bp.T
    feats = [
        np.log10(bp + EPS),
        rel,
        np.log10(total + EPS)[:, None],
        spectral_entropy(f, P)[:, None],
        np.log10(act + EPS)[:, None],
        mob[:, None],
        comp[:, None],
        np.log10((t + EPS) / (a + EPS))[:, None],
        np.log10((d + EPS) / (b + EPS))[:, None],
        np.log10((s + EPS) / (t + EPS))[:, None],
        np.log10((a + EPS) / (d + EPS))[:, None],
        np.log10(np.ptp(X, axis=-1) + EPS)[:, None],
        (np.diff(np.signbit(X), axis=-1).sum(axis=-1) / X.shape[-1])[:, None],
    ]
    return np.concatenate(feats, axis=1).astype(np.float32)


def add_context(F: np.ndarray, past: int = 2, future: int = 0) -> np.ndarray:
    """Concatenate features of neighbouring epochs within one night (edge-padded)."""
    cols = [F]
    for k in range(1, past + 1):
        cols.append(np.concatenate([np.repeat(F[:1], k, 0), F[:-k]], 0))
    for k in range(1, future + 1):
        cols.append(np.concatenate([F[k:], np.repeat(F[-1:], k, 0)], 0))
    return np.concatenate(cols, axis=1)
