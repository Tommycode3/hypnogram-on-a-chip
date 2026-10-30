"""Print the README tables from results/*.json (so every number is traceable)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = {p.stem: json.loads(p.read_text()) for p in (ROOT / "results").glob("*.json")}
CPM = {"ref": 16, "simd": 2}  # assumed Cortex-M4 cycles per MAC, see README
CORE_MHZ = (64, 128)


def row(name, r, extra=""):
    f1 = " | ".join(f"{v:.2f}" for v in r["f1"])
    return f"| {name} | {r['acc']:.3f} | {r['macro_f1']:.3f} | {r['kappa']:.3f} | {f1} |{extra}"


print("| model | acc | MF1 | κ | F1 W | F1 N1 | F1 N2 | F1 N3 | F1 REM |")
print("|---|---|---|---|---|---|---|---|---|")
b = R["baseline"]
for k, lab in (("logreg_ctx0", "LogReg, 21 features"), ("hgb_ctx0", "Gradient boosting, 21 features"),
               ("logreg_ctx4", "LogReg, 21 features × (t … t−4)"), ("hgb_ctx4", "Gradient boosting, 21 features × (t … t−4)")):
    print(row(lab, b[k]))
q = R["quant"]
for k in sorted(q, key=lambda k: q[k]["params"]):
    cfg, K = k.split("_k")
    print(row(f"CNN-{cfg}, K={K}, float", q[k]["float"]))
    print(row(f"**CNN-{cfg}, K={K}, int8**", q[k]["int8"]))

print("\n| model | K | params | MACs/epoch | flash M4F | flash M33 | RAM | κ float | κ int8 | int8=float | C=NumPy | est. ms @64 / 128 MHz (ref kernel) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(q, key=lambda k: (int(k.split('_k')[1]), q[k]["params"])):
    r = q[k]
    c = r["c_fold0"]
    ms = [r["macs"] * CPM["ref"] / (f * 1e3) for f in CORE_MHZ]
    print(f"| {k.split('_k')[0]} | {r['context'] if 'context' in r else k.split('_k')[1]} | {r['params']:,} | {r['macs'] / 1e6:.2f} M | "
          f"{c['arm_m4f']['flash'] / 1024:.1f} KiB | {c['arm_m33']['flash'] / 1024:.1f} KiB | {c['arm_m4f']['ram'] / 1024:.1f} KiB | "
          f"{r['float']['kappa']:.3f} | {r['int8']['kappa']:.3f} | {100 * r['int8_vs_float_agreement']:.1f} % | "
          f"{100 * c['c_vs_numpy_int8_logits_bitexact']:.1f} % | {ms[0]:.0f} / {ms[1]:.0f} |")

print("\ncontext ablation (float, pooled):")
cnn = R["cnn"]
for cfg in ("xs", "s", "m", "l", "xl"):
    vals = [f"K={K}: {cnn[f'{cfg}_k{K}']['kappa']:.3f}" for K in (0, 2, 4, 8) if f"{cfg}_k{K}" in cnn]
    print(cfg, ", ".join(vals))


# Subject-level bootstrap of kappa for the float CNNs (resample the 20 subjects).
import numpy as np  # noqa: E402

from tinysleep.data import load_nights  # noqa: E402
from tinysleep.metrics import cohen_kappa, confusion  # noqa: E402

Y = {f"{n.subject}_{n.night}": n.y for n in load_nights(list(range(20)))}
rng = np.random.default_rng(0)
B = 2000
boot_idx = [rng.integers(0, 20, 20) for _ in range(B)]


def subject_boot(preds):
    per_subj = {}
    for k, p in preds.items():
        s = int(k.split("_")[0])
        per_subj.setdefault(s, []).append(confusion(Y[k], p))
    C = {s: sum(v) for s, v in per_subj.items()}
    subs = sorted(C)
    ks = [cohen_kappa(sum(C[subs[i]] for i in idx)) for idx in boot_idx]
    return np.percentile(ks, [2.5, 97.5]), ks


print("\nsubject-bootstrap 95% CI of kappa (float):")
boots = {}
for f in sorted((ROOT / "results" / "preds").glob("*_k[04].npz")):
    d = dict(np.load(f))
    ci, ks = subject_boot(d)
    boots[f.stem] = np.array(ks)
    print(f"{f.stem}: [{ci[0]:.3f}, {ci[1]:.3f}]")
if "xs_k4" in boots and "l_k4" in boots:
    diff = boots["l_k4"] - boots["xs_k4"]
    print(f"paired l_k4 - xs_k4: median {np.median(diff):.3f}, 95% CI [{np.percentile(diff, 2.5):.3f}, {np.percentile(diff, 97.5):.3f}]")
if "m_k4" in boots and "m_k0" in boots:
    diff = boots["m_k4"] - boots["m_k0"]
    print(f"paired m_k4 - m_k0: median {np.median(diff):.3f}, 95% CI [{np.percentile(diff, 2.5):.3f}, {np.percentile(diff, 97.5):.3f}]")
