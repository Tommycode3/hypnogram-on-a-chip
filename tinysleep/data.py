"""Sleep-EDF Expanded (sleep-cassette) -> 30 s Fpz-Cz epochs with AASM labels.

Conventions (same as most single-channel papers on this dataset):
  * channel ``EEG Fpz-Cz``, 100 Hz, 30 s epochs (3000 samples), units = microvolts;
  * R&K stages 3 and 4 merged into AASM N3;
  * ``Movement time`` and ``?`` epochs dropped;
  * wake trimmed to 30 min before the first and after the last sleep epoch
    (otherwise ~70 % of each 20 h recording is wake and accuracy is inflated).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS = 100
EPOCH_S = 30
EPOCH_N = FS * EPOCH_S
STAGES = ["W", "N1", "N2", "N3", "REM"]
ANNOT_TO_LABEL = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
WAKE_PAD_EPOCHS = 60  # 30 min
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


@dataclass
class Night:
    subject: int
    night: int
    X: np.ndarray  # (n_epochs, 3000) float32, microvolts
    y: np.ndarray  # (n_epochs,) int64 in 0..4


def labels_from_annotations(onsets, durations, descriptions, n_epochs: int) -> np.ndarray:
    """Expand hypnogram annotations to one label per 30 s epoch (-1 = unscored)."""
    y = np.full(n_epochs, -1, dtype=np.int64)
    for on, dur, desc in zip(onsets, durations, descriptions):
        lab = ANNOT_TO_LABEL.get(desc, -1)
        first = int(round(on / EPOCH_S))
        count = int(round(dur / EPOCH_S))
        y[first : min(first + count, n_epochs)] = lab
    return y


def trim_wake(y: np.ndarray, pad: int = WAKE_PAD_EPOCHS) -> slice:
    sleep = np.flatnonzero((y >= 1) & (y <= 4))
    if sleep.size == 0:
        return slice(0, 0)
    return slice(max(sleep[0] - pad, 0), min(sleep[-1] + pad + 1, len(y)))


def epoch_signal(sig_uv: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(sig_uv) // EPOCH_N, len(y))
    X = sig_uv[: n * EPOCH_N].reshape(n, EPOCH_N).astype(np.float32)
    y = y[:n]
    sl = trim_wake(y)
    X, y = X[sl], y[sl]
    keep = y >= 0
    return X[keep], y[keep]


def load_night(psg: str, hyp: str) -> tuple[np.ndarray, np.ndarray]:
    import mne

    raw = mne.io.read_raw_edf(psg, include=["EEG Fpz-Cz"], preload=True, verbose="error")
    assert int(raw.info["sfreq"]) == FS
    ann = mne.read_annotations(hyp)
    sig = raw.get_data()[0] * 1e6  # V -> uV
    n_epochs = len(sig) // EPOCH_N
    y = labels_from_annotations(ann.onset, ann.duration, ann.description, n_epochs)
    return epoch_signal(sig, y)


def cache_path(subject: int, night: int) -> Path:
    return CACHE / f"sc{subject:02d}_n{night}.npz"


def build_cache(subjects: list[int]) -> None:
    from mne.datasets.sleep_physionet import age

    CACHE.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        for n in (1, 2):
            out = cache_path(s, n)
            if out.exists():
                continue
            files = age.fetch_data(subjects=[s], recording=[n], on_missing="ignore", verbose="error")
            if not files:
                continue
            X, y = load_night(files[0][0], files[0][1])
            np.savez_compressed(out, X=X, y=y)
            print(f"subject {s:2d} night {n}: {len(y)} epochs", flush=True)


def load_nights(subjects: list[int]) -> list[Night]:
    out = []
    for s in subjects:
        for n in (1, 2):
            p = cache_path(s, n)
            if p.exists():
                d = np.load(p)
                out.append(Night(s, n, d["X"], d["y"]))
    return out


# Subject-wise k-fold CV over the first 20 sleep-cassette subjects (the
# "Sleep-EDF-20" subset used by DeepSleepNet/TinySleepNet). Both nights of a
# subject are always in the same fold: a model is never evaluated on a person
# it has seen during training.
ALL = list(range(20))
N_FOLDS = 5


def fold_split(fold: int, subjects: list[int] = ALL, n_folds: int = N_FOLDS, n_val: int = 2):
    """Return (train, val, test) subject lists for ``fold``.

    test = every n_folds-th block of subjects; val = the first ``n_val`` subjects of
    the next block (used only for early stopping / model selection); train = rest.
    """
    size = len(subjects) // n_folds
    blocks = [subjects[i * size : (i + 1) * size] for i in range(n_folds)]
    test = blocks[fold]
    val = blocks[(fold + 1) % n_folds][:n_val]
    train = [s for s in subjects if s not in test and s not in val]
    return train, val, test
