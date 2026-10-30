"""Train tiny CNNs with subject-wise 5-fold CV.

Stage 1: encoder + context-free head, end-to-end, early stopping on the validation
         subjects' Cohen's kappa.
Stage 2: encoder frozen; a linear head over [e_t, e_{t-1}, ..., e_{t-K}] is fitted
         for each K (cheap: it only sees D-dim embeddings).
Writes models/<cfg>_f<fold>_k<K>.pt and results/cnn.json (pooled test metrics).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tinysleep.data import ALL, fold_split, load_nights
from tinysleep.metrics import cohen_kappa, confusion, report
from tinysleep.models import CONFIGS, TinySleepCNN, cost, preprocess, stack_context

ROOT = Path(__file__).resolve().parents[1]


def parse(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def kappa(y, p) -> float:
    return cohen_kappa(confusion(np.asarray(y), np.asarray(p)))


@torch.no_grad()
def embed(enc, X, bs=1024):
    enc.eval()
    return torch.cat([enc(X[i : i + bs]) for i in range(0, len(X), bs)])


def class_weights(y: np.ndarray) -> torch.Tensor:
    freq = np.bincount(y, minlength=5) / len(y)
    w = 1 / np.sqrt(freq)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def train_stage1(cfg, Xtr, ytr, Xva, yva, epochs, seed, dev, bs=128, lr=2e-3):
    torch.manual_seed(seed)
    model = TinySleepCNN(cfg, context=0).to(dev)
    Xtr, ytr, Xva = Xtr.to(dev), ytr.to(dev), Xva.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(Xtr) // bs + 1))
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr.cpu().numpy()).to(dev))
    best, best_state, g = -1.0, None, torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs].to(dev)
            x = Xtr[idx]
            # augmentation: random gain (electrode impedance) + small white noise
            x = x * torch.empty(len(x), 1, device=dev).uniform_(0.8, 1.25) + 0.01 * torch.randn_like(x)
            loss = lossf(model(x.clamp(-1, 1)), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            k = kappa(yva.numpy(), model.head(embed(model.encoder, Xva)).argmax(1).cpu().numpy())
        if k > best:
            best, best_state = k, {n: v.clone() for n, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.cpu(), best


def fit_head(D, K, E_tr_nights, y_tr, E_va_nights, y_va, seed, steps=600, lr=1e-2):
    torch.manual_seed(seed)
    Ztr = torch.cat([stack_context(E, K) for E in E_tr_nights])
    Zva = torch.cat([stack_context(E, K) for E in E_va_nights])
    head, drop = nn.Linear((K + 1) * D, 5), nn.Dropout(0.2)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(weight=class_weights(y_tr.numpy()))
    best, best_state = -1.0, None
    for step in range(steps):
        loss = lossf(head(drop(Ztr)), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 19:
            with torch.no_grad():
                k = kappa(y_va.numpy(), head(Zva).argmax(1).numpy())
            if k > best:
                best, best_state = k, {n: v.clone() for n, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--folds", default="0-4")
    ap.add_argument("--contexts", default="0,2,4,8")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    contexts = parse(args.contexts)

    nights = load_nights(ALL)
    data = {(n.subject, n.night): (preprocess(torch.from_numpy(n.X)), torch.from_numpy(n.y)) for n in nights}
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results" / "preds").mkdir(parents=True, exist_ok=True)
    out_json = ROOT / "results" / "cnn.json"
    results = json.loads(out_json.read_text()) if out_json.exists() else {}

    for cfg in args.configs.split(","):
        preds = {K: {} for K in contexts}
        for fold in parse(args.folds):
            t0 = time.time()
            train, val, test = fold_split(fold)
            keys = lambda subs: sorted(k for k in data if k[0] in subs)  # noqa: E731
            cat = lambda ks, j: torch.cat([data[k][j] for k in ks])  # noqa: E731
            ktr, kva, kte = keys(train), keys(val), keys(test)
            model, kva_best = train_stage1(cfg, cat(ktr, 0), cat(ktr, 1), cat(kva, 0), cat(kva, 1),
                                           args.epochs, seed=fold, dev=args.device)
            enc = model.encoder
            E = {k: embed(enc, data[k][0]) for k in ktr + kva + kte}
            for K in contexts:
                head = fit_head(enc.dim, K, [E[k] for k in ktr], cat(ktr, 1), [E[k] for k in kva], cat(kva, 1),
                                seed=fold)
                full = TinySleepCNN(cfg, context=K)
                full.encoder.load_state_dict(enc.state_dict())
                full.head.load_state_dict(head.state_dict())
                torch.save(full.state_dict(), ROOT / "models" / f"{cfg}_f{fold}_k{K}.pt")
                with torch.no_grad():
                    for k in kte:
                        preds[K][f"{k[0]}_{k[1]}"] = head(stack_context(E[k], K)).argmax(1).numpy()
            print(f"[{cfg}] fold {fold}: val kappa (K=0) {kva_best:.3f}  {time.time() - t0:.0f}s", flush=True)
        for K in contexts:
            ks = sorted(preds[K])
            yt = np.concatenate([data[tuple(map(int, k.split('_')))][1].numpy() for k in ks])
            yp = np.concatenate([preds[K][k] for k in ks])
            np.savez(ROOT / "results" / "preds" / f"{cfg}_k{K}.npz", **{k: preds[K][k] for k in ks})
            r = report(yt, yp)
            r.update(cost(cfg, K))
            results[f"{cfg}_k{K}"] = r
            print(f"  {cfg} K={K}: acc={r['acc']:.3f} mf1={r['macro_f1']:.3f} kappa={r['kappa']:.3f} "
                  f"params={r['params']}", flush=True)
        out_json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
