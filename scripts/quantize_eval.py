"""Post-training int8 quantisation of every trained CNN, and the C cross-check.

1. For each (config, K) and each fold: calibrate on 2000 epochs of that fold's
   *training* subjects, quantise, run the bit-exact NumPy int8 reference on the
   fold's test nights. Pool -> int8 metrics + agreement with the float model.
2. Fold 0 only: export a C header, compile c/ with the host `cc`, run it on the
   same test nights (raw float uV in, int8 logits out) and compare with both the
   NumPy int8 reference (should be bit-exact) and the float model.
3. Cross-compile the kernel + model with arm-none-eabi-gcc for Cortex-M4F and
   Cortex-M33 and read flash/RAM with arm-none-eabi-size (if the toolchain exists).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from tinysleep.data import ALL, N_FOLDS, fold_split, load_nights
from tinysleep.metrics import report
from tinysleep.models import TinySleepCNN, cost
from tinysleep.quant import buffer_bytes, export_c_header, int8_embed, int8_head, quantize_input, quantize_model

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
CFLAGS = ["-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off"]


def load(cfg, fold, K):
    m = TinySleepCNN(cfg, context=K)
    m.load_state_dict(torch.load(ROOT / "models" / f"{cfg}_f{fold}_k{K}.pt", weights_only=True))
    return m.eval()


def write_bin(nights, path):
    with open(path, "wb") as f:
        np.array([len(nights)], np.int32).tofile(f)
        for n in nights:
            np.array([len(n.y)], np.int32).tofile(f)
            n.X.astype(np.float32).tofile(f)


def run_c(header: Path, inp: Path, tag: str) -> np.ndarray:
    exe = BUILD / f"ts_{tag}"
    subprocess.run(["cc", *CFLAGS, f'-DMODEL_H="{header.name}"', f"-I{BUILD}", f"-I{ROOT / 'c'}",
                    str(ROOT / "c" / "main.c"), str(ROOT / "c" / "tinysleep.c"), "-o", str(exe)], check=True)
    out = BUILD / f"logits_{tag}.bin"
    subprocess.run([str(exe), str(inp), str(out)], check=True, capture_output=True)
    return np.fromfile(out, dtype=np.int8).reshape(-1, 5)


def arm_size(header: Path, cpu: str) -> dict | None:
    gcc = shutil.which("arm-none-eabi-gcc")
    if gcc is None:
        return None
    fpu = {"cortex-m4": ["-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"], "cortex-m33": ["-mfpu=fpv5-sp-d16", "-mfloat-abi=hard"]}[cpu]
    objs = []
    for src in ("tinysleep.c", "mcu_entry.c"):
        o = BUILD / f"{header.stem}_{cpu}_{src}.o"
        subprocess.run([gcc, f"-mcpu={cpu}", "-mthumb", *fpu, "-Os", "-std=c99", "-ffunction-sections",
                        "-fdata-sections", "-ffp-contract=off", f'-DMODEL_H="{header.name}"', f"-I{BUILD}",
                        f"-I{ROOT / 'c'}", "-c", str(ROOT / "c" / src), "-o", str(o)], check=True)
        objs.append(str(o))
    txt = subprocess.run(["arm-none-eabi-size", "-B", *objs], check=True, capture_output=True, text=True).stdout
    text = data = bss = 0
    for line in txt.strip().splitlines()[1:]:
        t, d, b = (int(v) for v in line.split()[:3])
        text, data, bss = text + t, data + d, bss + b
    return {"flash": text + data, "ram": data + bss}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="xs,s,m,l,xl")
    ap.add_argument("--contexts", default="0,4")
    args = ap.parse_args()
    BUILD.mkdir(exist_ok=True)
    nights = {(n.subject, n.night): n for n in load_nights(ALL)}
    rng = np.random.default_rng(0)
    out_json = ROOT / "results" / "quant.json"
    results = json.loads(out_json.read_text()) if out_json.exists() else {}

    _, _, test0 = fold_split(0)
    test0_nights = [nights[k] for k in sorted(nights) if k[0] in test0]
    inp0 = BUILD / "test_fold0.bin"
    if not inp0.exists():
        write_bin(test0_nights, inp0)

    for cfg in args.configs.split(","):
        for K in (int(k) for k in args.contexts.split(",")):
            yt, yf, yq = [], [], []
            fold0 = {}
            for fold in range(N_FOLDS):
                train, val, test = fold_split(fold)
                model = load(cfg, fold, K)
                tr = [nights[k] for k in sorted(nights) if k[0] in train]
                pool = np.concatenate([n.X for n in tr])
                calib = pool[rng.choice(len(pool), 2000, replace=False)]
                q = quantize_model(model, calib)
                for k in sorted(nights):
                    if k[0] not in test:
                        continue
                    n = nights[k]
                    with torch.no_grad():
                        from tinysleep.models import preprocess, stack_context
                        E = model.encoder(preprocess(torch.from_numpy(n.X)))
                        pf = model.head(stack_context(E, K)).argmax(1).numpy()
                    Lq = int8_head(q, int8_embed(q, quantize_input(n.X, q.inv_s_in)))
                    yt.append(n.y)
                    yf.append(pf)
                    yq.append(Lq.argmax(1))
                    if fold == 0:
                        fold0[k] = (pf, Lq)
                if fold == 0:
                    tag = f"{cfg}_k{K}"
                    header = BUILD / f"model_{tag}.h"
                    export_c_header(q, str(header))
                    Lc = run_c(header, inp0, tag)
                    Lpy = np.concatenate([fold0[(n.subject, n.night)][1] for n in test0_nights])
                    pf0 = np.concatenate([fold0[(n.subject, n.night)][0] for n in test0_nights])
                    c_info = {
                        "epochs": int(len(Lc)),
                        "c_vs_numpy_int8_logits_bitexact": float(np.mean(np.all(Lc == Lpy, axis=1))),
                        "c_vs_float_pred_agreement": float(np.mean(Lc.argmax(1) == pf0)),
                        "weights_bytes": q.nbytes_weights(),
                        "act_buffers_bytes": 2 * buffer_bytes(q),
                        "ring_bytes": int(q.head_w.shape[1]),
                        "arm_m4f": arm_size(header, "cortex-m4"),
                        "arm_m33": arm_size(header, "cortex-m33"),
                    }
            yt, yf, yq = map(np.concatenate, (yt, yf, yq))
            r = {
                "float": report(yt, yf),
                "int8": report(yt, yq),
                "int8_vs_float_agreement": float(np.mean(yf == yq)),
                "c_fold0": c_info,
                **cost(cfg, K),
            }
            results[f"{cfg}_k{K}"] = r
            print(f"{cfg} K={K}: float kappa {r['float']['kappa']:.3f} | int8 kappa {r['int8']['kappa']:.3f} | "
                  f"agree {r['int8_vs_float_agreement']:.4f} | C bit-exact {c_info['c_vs_numpy_int8_logits_bitexact']:.4f} "
                  f"| C-vs-float {c_info['c_vs_float_pred_agreement']:.4f} | m4f {c_info['arm_m4f']}", flush=True)
            out_json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
