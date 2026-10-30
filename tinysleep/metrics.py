"""Metrics, written out explicitly (and cross-checked against sklearn in tests)."""
from __future__ import annotations

import numpy as np

K = 5


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = K) -> np.ndarray:
    """C[i, j] = number of epochs with true class i predicted as j."""
    C = np.zeros((k, k), dtype=np.int64)
    np.add.at(C, (y_true, y_pred), 1)
    return C


def cohen_kappa(C: np.ndarray) -> float:
    """kappa = (p_o - p_e) / (1 - p_e)."""
    n = C.sum()
    po = np.trace(C) / n
    pe = (C.sum(0) * C.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe))


def per_class_f1(C: np.ndarray) -> np.ndarray:
    tp = np.diag(C).astype(float)
    prec = tp / np.maximum(C.sum(0), 1)
    rec = tp / np.maximum(C.sum(1), 1)
    return np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    C = confusion(y_true, y_pred)
    f1 = per_class_f1(C)
    return {
        "n": int(C.sum()),
        "acc": float(np.trace(C) / C.sum()),
        "macro_f1": float(f1.mean()),
        "kappa": cohen_kappa(C),
        "f1": [float(v) for v in f1],
        "confusion": C.tolist(),
    }
