"""Make every figure in figures/ from results/*.json, models/ and the epoch cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tinysleep.data import ALL, STAGES, fold_split, load_nights  # noqa: E402
from tinysleep.features import psd  # noqa: E402
from tinysleep.models import TinySleepCNN, preprocess, stack_context  # noqa: E402
from tinysleep.quant import int8_embed, int8_head, quantize_input, quantize_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
COL = {"W": "#e8a33d", "N1": "#8fb8de", "N2": "#3c78b5", "N3": "#173f6b", "REM": "#c4473a"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def psd_per_stage(nights):
    X = np.concatenate([n.X for n in nights])
    y = np.concatenate([n.y for n in nights])
    f, P = psd(X - X.mean(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    m = (f >= 0.5) & (f <= 30)
    for i, s in enumerate(STAGES):
        lp = 10 * np.log10(P[y == i][:, m])
        med = np.median(lp, 0)
        q1, q3 = np.percentile(lp, [25, 75], 0)
        ax.plot(f[m], med, color=COL[s], label=f"{s} (n={np.sum(y == i)})", lw=1.6)
        ax.fill_between(f[m], q1, q3, color=COL[s], alpha=0.12, lw=0)
    for edge in (4, 8, 12, 15):
        ax.axvline(edge, color="0.85", lw=0.8, zorder=0)
    ax.set(xlabel="frequency [Hz]", ylabel="PSD [dB re 1 µV²/Hz]",
           title="Fpz-Cz Welch PSD per stage (median, IQR) — all 39 nights")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "psd_per_stage.png")


def pareto(cnn, quant, base):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, xkey, xlabel in ((axes[0], "params", "parameters"), (axes[1], "flash", "flash on Cortex-M4F [bytes]")):
        for K, marker in ((0, "o"), (4, "s")):
            keys = [k for k in quant if k.endswith(f"_k{K}")]
            xs = [quant[k]["params"] if xkey == "params" else quant[k]["c_fold0"]["arm_m4f"]["flash"] for k in keys]
            kf = [quant[k]["float"]["kappa"] for k in keys]
            kq = [quant[k]["int8"]["kappa"] for k in keys]
            order = np.argsort(xs)
            xs, kf, kq = np.array(xs)[order], np.array(kf)[order], np.array(kq)[order]
            lab = "no context" if K == 0 else f"K={K} past epochs"
            ax.plot(xs, kf, marker=marker, mfc="white", color="0.55", ls="--", lw=1, label=f"float, {lab}")
            ax.plot(xs, kq, marker=marker, color="#3c78b5" if K else "#e8a33d", lw=1.6, label=f"int8, {lab}")
            for x, k, name in zip(xs, kq, np.array(keys)[order]):
                ax.annotate(name.split("_")[0], (x, k), textcoords="offset points", xytext=(4, -10), fontsize=7)
        for name, ls in (("hgb_ctx4", ":"), ("logreg_ctx4", "-.")):
            if name in base:
                ax.axhline(base[name]["kappa"], color="0.3", ls=ls, lw=1,
                           label=f"{name.replace('_ctx4', '')} + 21 features, K=4")
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        ax.set_xticks([1e3, 2e3, 5e3, 1e4, 2e4, 5e4])
        ax.set(xlabel=xlabel, ylabel="Cohen's κ (pooled 5-fold, 20 subjects)")
    axes[0].legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.suptitle("Accuracy vs size — every point is a subject-wise held-out score", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "pareto.png")


def confusion(r, name):
    C = np.array(r["confusion"], float)
    R = C / C.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{R[i, j]:.2f}\n{int(C[i, j])}", ha="center", va="center", fontsize=7,
                    color="white" if R[i, j] > 0.55 else "black")
    ax.set_xticks(range(5), STAGES)
    ax.set_yticks(range(5), STAGES)
    ax.set(xlabel="predicted", ylabel="true", title=f"{name}\nκ={r['kappa']:.3f}, MF1={r['macro_f1']:.3f}")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "confusion.png")


def hypnogram(nights, cfg, K):
    _, _, test = fold_split(0)
    train = [n for n in nights if n.subject not in test and n.subject not in fold_split(0)[1]]
    night = next(n for n in nights if n.subject == test[0] and n.night == 1)
    m = TinySleepCNN(cfg, context=K)
    m.load_state_dict(torch.load(ROOT / "models" / f"{cfg}_f0_k{K}.pt", weights_only=True))
    m.eval()
    with torch.no_grad():
        pf = m.classify(stack_context(m.encoder(preprocess(torch.from_numpy(night.X))), K)).argmax(1).numpy()
    pool = np.concatenate([n.X for n in train])
    q = quantize_model(m, pool[np.random.default_rng(0).choice(len(pool), 2000, replace=False)])
    pq = int8_head(q, int8_embed(q, quantize_input(night.X, q.inv_s_in))).argmax(1)
    level = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
    t = np.arange(len(night.y)) / 120  # hours
    fig, axes = plt.subplots(3, 1, figsize=(10, 4.6), sharex=True)
    for ax, y, title in ((axes[0], night.y, "expert (R&K → AASM)"), (axes[1], pf, f"float {cfg}, K={K}"),
                         (axes[2], pq, f"int8 (bit-exact with the C kernel), K={K}")):
        lv = np.vectorize(level.get)(y)
        ax.step(t, lv, where="post", color="0.25", lw=0.8)
        rem = y == 4
        ax.scatter(t[rem], lv[rem], s=2, color=COL["REM"], zorder=3)
        ax.set_yticks(range(5), ["N3", "N2", "N1", "REM", "W"])
        acc = np.mean(y == night.y)
        ax.set_title(f"{title}" + ("" if y is night.y else f" — epoch accuracy {acc:.1%}"), fontsize=9, loc="left")
    axes[-1].set_xlabel("hours since start of trimmed recording")
    fig.suptitle(f"Held-out night: Sleep-EDF SC subject {night.subject}, night {night.night} (never seen in training)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "hypnogram.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="m_k4")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    nights = load_nights(ALL)
    cnn = json.loads((ROOT / "results" / "cnn.json").read_text())
    quant = json.loads((ROOT / "results" / "quant.json").read_text())
    base = json.loads((ROOT / "results" / "baseline.json").read_text())
    psd_per_stage(nights)
    pareto(cnn, quant, base)
    confusion(quant[args.best]["int8"], f"int8 {args.best.replace('_k', ', K=')} (pooled 20 subjects)")
    cfg, K = args.best.split("_k")
    hypnogram(nights, cfg, int(K))


if __name__ == "__main__":
    main()
