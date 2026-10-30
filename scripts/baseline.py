"""Classical baseline: hand-crafted features + logistic regression / gradient boosting.

Subject-wise 5-fold CV; predictions of the 5 test folds are pooled before scoring,
so every one of the 20 subjects is scored by a model that never saw them.
"""
from __future__ import annotations

import os

# sklearn's OpenMP pool stalls badly on this 16-core macOS box; 1 thread is ~100x faster.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tinysleep.data import ALL, N_FOLDS, fold_split, load_nights
from tinysleep.features import FEATURE_NAMES, add_context, epoch_features
from tinysleep.metrics import report

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 4  # past epochs, causal, same as the CNN context head


def main() -> None:
    nights = load_nights(ALL)
    F = {(n.subject, n.night): epoch_features(n.X) for n in nights}
    Y = {(n.subject, n.night): n.y for n in nights}
    print(f"{len(FEATURE_NAMES)} features per epoch")

    models = {
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, class_weight="balanced", random_state=0
        ),
    }
    results = {}
    for ctx in (0, CONTEXT):
        for name, make in models.items():
            t0 = time.time()
            yt, yp = [], []
            for fold in range(N_FOLDS):
                train, val, test = fold_split(fold)
                train = train + val  # no early stopping needed here

                def xy(subs):
                    keys = [k for k in F if k[0] in subs]
                    X = np.concatenate([add_context(F[k], past=ctx) for k in keys])
                    return X, np.concatenate([Y[k] for k in keys])

                Xtr, ytr = xy(train)
                Xte, yte = xy(test)
                clf = make().fit(Xtr, ytr)
                yt.append(yte)
                yp.append(clf.predict(Xte))
            r = report(np.concatenate(yt), np.concatenate(yp))
            key = f"{name}_ctx{ctx}"
            results[key] = r
            print(f"{key:12s} acc={r['acc']:.3f} mf1={r['macro_f1']:.3f} kappa={r['kappa']:.3f} "
                  f"f1={np.round(r['f1'], 2).tolist()} ({time.time() - t0:.0f}s)", flush=True)
    out = ROOT / "results" / "baseline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
