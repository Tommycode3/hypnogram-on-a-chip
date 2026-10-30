"""int8 quantisation: fixed-point maths, float agreement, and C == NumPy bit-exactness."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tinysleep.models import TinySleepCNN, preprocess, stack_context
from tinysleep.quant import (export_c_header, int8_embed, int8_head, quantize_input, quantize_model,
                             quantize_multiplier, requant)

ROOT = Path(__file__).resolve().parents[1]


def test_quantize_multiplier_roundtrip():
    M = np.array([1e-5, 3.3e-3, 0.25, 0.9999, 1.7, 12.0])
    M0, sh = quantize_multiplier(M)
    assert np.all((M0 >= 2**30) & (M0 < 2**31))
    assert np.allclose(M0 / 2.0**sh, M, rtol=1e-9)
    acc = np.array([-100000, -3, 0, 7, 123456])
    for m, m0, s in zip(M, M0, sh):
        assert np.all(np.abs(requant(acc, m0, s) - acc * m) <= 0.5 + 1e-6)


def synthetic_eeg(n, seed=0):
    """Pink-ish noise + bursts at 2/10/13 Hz, ~tens of uV: enough to exercise every layer."""
    rng = np.random.default_rng(seed)
    t = np.arange(3000) / 100
    X = np.cumsum(rng.standard_normal((n, 3000)), axis=1)
    X -= X.mean(1, keepdims=True)
    X = 20 * X / X.std(1, keepdims=True)
    for f in (2, 10, 13):
        X += rng.uniform(0, 40, (n, 1)) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6, (n, 1)))
    return X.astype(np.float32)


@pytest.fixture(scope="module")
def model_and_q():
    torch.manual_seed(0)
    m = TinySleepCNN("s", context=2)
    # give BatchNorm non-trivial statistics so folding is actually tested
    m.train()
    with torch.no_grad():
        for _ in range(5):
            m.encoder(preprocess(torch.from_numpy(synthetic_eeg(64, seed=1))))
    m.eval()
    q = quantize_model(m, synthetic_eeg(256, seed=2))
    return m, q


def test_int8_matches_float(model_and_q):
    m, q = model_and_q
    X = synthetic_eeg(300, seed=3)
    with torch.no_grad():
        Lf = m.classify(stack_context(m.encoder(preprocess(torch.from_numpy(X))), 2)).numpy()
    Lq = int8_head(q, int8_embed(q, quantize_input(X, q.inv_s_in)))
    Lq_real = q.s_logit * (Lq - q.z_logit)
    assert np.corrcoef(Lf.ravel(), Lq_real.ravel())[0, 1] > 0.98
    assert np.mean(Lf.argmax(1) == Lq.argmax(1)) > 0.9


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_kernel_is_bit_exact(model_and_q, tmp_path):
    _, q = model_and_q
    X = synthetic_eeg(40, seed=4)
    export_c_header(q, str(tmp_path / "model.h"))
    with open(tmp_path / "in.bin", "wb") as f:
        np.array([2], np.int32).tofile(f)  # two "nights" -> also tests ts_reset()
        for part in (X[:25], X[25:]):
            np.array([len(part)], np.int32).tofile(f)
            part.tofile(f)
    exe = tmp_path / "ts"
    subprocess.run(["cc", "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                    '-DMODEL_H="model.h"', f"-I{tmp_path}", f"-I{ROOT / 'c'}",
                    str(ROOT / "c" / "main.c"), str(ROOT / "c" / "tinysleep.c"), "-o", str(exe)], check=True)
    subprocess.run([str(exe), str(tmp_path / "in.bin"), str(tmp_path / "out.bin")], check=True,
                   capture_output=True)
    Lc = np.fromfile(tmp_path / "out.bin", dtype=np.int8).reshape(-1, 5)
    Lpy = np.concatenate([int8_head(q, int8_embed(q, quantize_input(p, q.inv_s_in))) for p in (X[:25], X[25:])])
    np.testing.assert_array_equal(Lc, Lpy)
