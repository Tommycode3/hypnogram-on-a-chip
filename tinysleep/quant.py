"""Post-training int8 quantisation + a bit-exact NumPy reference of the C kernel.

Scheme (TFLite-style, but written from scratch):
  real value  x ~= s * (q - z),  q = clamp(round(x / s) + z, -128, 127)
  * weights: int8, symmetric (z = 0), one scale per output channel  s_w[c] = max|w_c| / 127
  * bias:    int32, scale s_in * s_w[c], z = 0
  * activations: int8, one (s, z) per tensor; post-ReLU tensors use z = -128 so the
    whole [0, a_max] range maps onto 256 codes
  * requantisation of an int32 accumulator to the next int8 tensor:
        q_out = clamp(rshift_round(acc * M0[c], 31 - e[c]) + z_out)
    where M[c] = s_in * s_w[c] / s_out = M0[c] * 2^(e[c] - 31), M0 in [2^30, 2^31)
    (integer-only, as on a Cortex-M without touching the FPU).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .models import INPUT_CLIP_UV, TinySleepCNN

QMIN, QMAX = -128, 127


def quantize_multiplier(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M (float, >0) -> (M0 int32 in [2^30, 2^31), total right shift)."""
    M = np.atleast_1d(np.asarray(M, dtype=np.float64))
    m, e = np.frexp(M)  # M = m * 2^e, m in [0.5, 1)
    M0 = np.round(m * (1 << 31)).astype(np.int64)
    over = M0 == (1 << 31)
    M0[over] //= 2
    e[over] += 1
    shift = 31 - e
    assert np.all((shift > 0) & (shift < 63)), shift
    return M0, shift.astype(np.int64)


def requant(acc: np.ndarray, M0: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """round(acc * M) with integer arithmetic; ties round towards +inf (same as C)."""
    acc = acc.astype(np.int64)
    return (acc * M0 + (np.int64(1) << (shift - 1))) >> shift


@dataclass
class QConv:
    w: np.ndarray  # (cout, cin, k) int8
    b: np.ndarray  # (cout,) int32
    M0: np.ndarray
    shift: np.ndarray
    z_in: int
    z_out: int
    k: int
    stride: int
    pool: int
    s_out: float


@dataclass
class QModel:
    cfg_name: str
    context: int
    s_in: float  # input scale, in uV per LSB
    inv_s_in: float = 0.0  # float32(1 / s_in), what C actually multiplies by
    convs: list[QConv] = field(default_factory=list)
    gap_M0: int = 0
    gap_shift: int = 0
    z_emb: int = -128
    s_emb: float = 1.0
    head_w: np.ndarray | None = None
    head_b: np.ndarray | None = None
    head_M0: np.ndarray | None = None
    head_shift: np.ndarray | None = None
    z_logit: int = 0
    s_logit: float = 1.0

    def nbytes_weights(self) -> int:
        n = 0
        for c in self.convs:
            n += c.w.size + 4 * c.b.size + 4 * c.M0.size + c.shift.size
        n += self.head_w.size + 4 * self.head_b.size + 4 * self.head_M0.size + self.head_shift.size
        return n


# ---------------------------------------------------------------- float folding

def fold_bn(model: TinySleepCNN) -> list[tuple[np.ndarray, np.ndarray, int, int, int]]:
    """Return [(W, b, k, stride, pool)] with BatchNorm folded into the conv."""
    out = []
    mods = list(model.encoder.body)
    for i in range(0, len(mods), 4):
        conv, bn, _, pool = mods[i : i + 4]
        W = conv.weight.detach().double().numpy()
        g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double().numpy()
        b = (bn.bias - bn.running_mean * bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double().numpy()
        out.append((W * g[:, None, None], b, conv.kernel_size[0], conv.stride[0], pool.kernel_size))
    return out


def _windows(x: np.ndarray, k: int, stride: int, pad_value) -> np.ndarray:
    """x: (B, C, N) -> (B, C, Nout, k) zero/z-padded sliding windows."""
    p = k // 2
    xp = np.pad(x, ((0, 0), (0, 0), (p, p)), constant_values=pad_value)
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)
    return win[:, :, ::stride, :]


def float_forward_folded(layers, x: np.ndarray, record: list | None = None) -> np.ndarray:
    """Float64 forward through folded conv layers. x: (B, N) normalised. Returns (B, D)."""
    h = x[:, None, :]
    for W, b, k, s, p in layers:
        a = np.einsum("bcnk,ock->bon", _windows(h, k, s, 0.0), W, optimize=True) + b[None, :, None]
        a = np.maximum(a, 0)
        n = a.shape[2] // p
        h = a[:, :, : n * p].reshape(a.shape[0], a.shape[1], n, p).max(-1)
        if record is not None:
            record.append(h)
    return h.mean(-1)


def preprocess_np(X_uv: np.ndarray) -> np.ndarray:
    """float32 uV -> normalised float64 in [-1, 1] (matches models.preprocess)."""
    X = X_uv.astype(np.float64)
    X = X - X.mean(-1, keepdims=True)
    return np.clip(X, -INPUT_CLIP_UV, INPUT_CLIP_UV) / INPUT_CLIP_UV


def context_np(E: np.ndarray, K: int) -> np.ndarray:
    n = E.shape[0]
    idx = np.arange(n)
    return np.concatenate([E[np.clip(idx - k, 0, None)] for k in range(K + 1)], axis=1)


# ---------------------------------------------------------------- calibration

def quantize_model(model: TinySleepCNN, calib_X_uv: np.ndarray, pct: float = 99.99) -> QModel:
    """Post-training quantisation with activation ranges from a calibration set."""
    model.eval()
    layers = fold_bn(model)
    rec: list[np.ndarray] = []
    E = float_forward_folded(layers, preprocess_np(calib_X_uv), rec)
    Wh = model.head.weight.detach().double().numpy()
    bh = model.head.bias.detach().double().numpy()
    K = model.context
    # Context: embeddings of previous epochs come from the same distribution, so the
    # logit range can be calibrated by feeding the calibration batch as a sequence.
    logits = context_np(E, K) @ Wh.T + bh

    q = QModel(model.cfg_name, K, s_in=INPUT_CLIP_UV / 127.0)
    q.inv_s_in = float(np.float32(127.0 / INPUT_CLIP_UV))  # stored as float32 in C
    s_prev, z_prev = 1.0 / 127.0, 0  # input in normalised units: x_norm = s * q
    for (W, b, k, s, p), h in zip(layers, rec):
        a_max = float(np.percentile(h, pct)) or 1e-6
        s_out, z_out = a_max / 255.0, -128
        s_w = np.abs(W).reshape(W.shape[0], -1).max(1) / 127.0
        s_w[s_w == 0] = 1.0
        Wq = np.clip(np.round(W / s_w[:, None, None]), -127, 127).astype(np.int8)
        bq = np.round(b / (s_prev * s_w)).astype(np.int32)
        M0, sh = quantize_multiplier(s_prev * s_w / s_out)
        q.convs.append(QConv(Wq, bq, M0, sh, z_prev, z_out, k, s, p, s_out))
        s_prev, z_prev = s_out, z_out
    # GAP: q_e = round(sum(q - z_a) * s_a / (L * s_e)) + z_e; L depends on the input length.
    e_max = float(np.percentile(E, pct)) or 1e-6
    q.s_emb, q.z_emb = e_max / 255.0, -128
    L = rec[-1].shape[-1]
    M0, sh = quantize_multiplier(np.array([s_prev / (L * q.s_emb)]))
    q.gap_M0, q.gap_shift = int(M0[0]), int(sh[0])
    # Head
    s_w = np.abs(Wh).max(1) / 127.0
    s_w[s_w == 0] = 1.0
    q.head_w = np.clip(np.round(Wh / s_w[:, None]), -127, 127).astype(np.int8)
    q.head_b = np.round(bh / (q.s_emb * s_w)).astype(np.int32)
    lo, hi = float(np.percentile(logits, 100 - pct)), float(np.percentile(logits, pct))
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    q.s_logit = (hi - lo) / 255.0
    q.z_logit = int(np.round(QMIN - lo / q.s_logit))
    q.head_M0, q.head_shift = quantize_multiplier(q.s_emb * s_w / q.s_logit)
    return q


# ---------------------------------------------------------------- int8 reference

def quantize_input(X_uv: np.ndarray, inv_s: float) -> np.ndarray:
    """Bit-exact with ts_quantize_input() in C (float32, sequential mean, no FMA).

    q = clamp(floor((x - mean) * inv_s + 0.5), -127, 127)
    """
    X = X_uv.astype(np.float32)
    mean = np.cumsum(X, axis=-1, dtype=np.float32)[:, -1] / np.float32(X.shape[-1])
    v = (X - mean[:, None].astype(np.float32)) * np.float32(inv_s)
    v = v + np.float32(0.5)
    return np.clip(np.floor(v), -127, 127).astype(np.int32)


def int8_embed(q: QModel, xq: np.ndarray, batch: int = 512) -> np.ndarray:
    """xq: (B, N) int input codes -> (B, D) int8 embedding codes."""
    outs = []
    for i in range(0, len(xq), batch):
        h = xq[i : i + batch, None, :].astype(np.int64)
        for c in q.convs:
            win = _windows(h - c.z_in, c.k, c.stride, 0)
            acc = np.einsum("bcnk,ock->bon", win, c.w.astype(np.int64), optimize=True) + c.b[None, :, None]
            y = requant(acc, c.M0[None, :, None], c.shift[None, :, None]) + c.z_out
            y = np.clip(y, c.z_out, QMAX)  # ReLU == clamp at the zero point
            n = y.shape[2] // c.pool
            h = y[:, :, : n * c.pool].reshape(y.shape[0], y.shape[1], n, c.pool).max(-1)
        S = (h - q.convs[-1].z_out).sum(-1)
        e = requant(S, np.int64(q.gap_M0), np.int64(q.gap_shift)) + q.z_emb
        outs.append(np.clip(e, QMIN, QMAX))
    return np.concatenate(outs).astype(np.int64)


def int8_head(q: QModel, Eq: np.ndarray) -> np.ndarray:
    """Eq: (n, D) embedding codes of one night in time order -> (n, 5) int8 logits."""
    Z = context_np(Eq, q.context) - q.z_emb
    acc = Z @ q.head_w.astype(np.int64).T + q.head_b[None, :]
    y = requant(acc, q.head_M0[None, :], q.head_shift[None, :]) + q.z_logit
    return np.clip(y, QMIN, QMAX)


def int8_predict_night(q: QModel, X_uv: np.ndarray) -> np.ndarray:
    return int8_head(q, int8_embed(q, quantize_input(X_uv, q.inv_s_in))).argmax(1)


# ---------------------------------------------------------------- C export

def _arr(name: str, ctype: str, a: np.ndarray) -> str:
    vals = ", ".join(str(int(v)) for v in np.asarray(a).ravel())
    return f"static const {ctype} {name}[{np.asarray(a).size}] = {{{vals}}};\n"


def buffer_bytes(q: QModel, n: int = 3000) -> int:
    """Size of each of the two ping-pong activation buffers (mirrors ts_buffer_bytes)."""
    best = n
    for c in q.convs:
        n = ((n + 2 * (c.k // 2) - c.k) // c.stride + 1) // c.pool
        best = max(best, n * c.w.shape[0])
    return best


def export_c_header(q: QModel, path: str) -> None:
    lines = [
        "/* Auto-generated by tinysleep.quant.export_c_header -- do not edit. */\n",
        "#pragma once\n#include \"tinysleep.h\"\n\n",
        f"/* model '{q.cfg_name}', context K={q.context}, input scale {q.s_in:.6g} uV/LSB */\n",
        f"#define TS_BUF_BYTES {buffer_bytes(q)}\n",
        f"#define TS_RING_BYTES {q.head_w.shape[1]}\n\n",
    ]
    for i, c in enumerate(q.convs):
        lines += [
            _arr(f"L{i}_w", "int8_t", c.w),
            _arr(f"L{i}_b", "int32_t", c.b),
            _arr(f"L{i}_m0", "int32_t", c.M0),
            _arr(f"L{i}_sh", "int8_t", c.shift),
        ]
    lines += [
        _arr("H_w", "int8_t", q.head_w),
        _arr("H_b", "int32_t", q.head_b),
        _arr("H_m0", "int32_t", q.head_M0),
        _arr("H_sh", "int8_t", q.head_shift),
        "\nstatic const ts_conv_t TS_LAYERS[] = {\n",
    ]
    for i, c in enumerate(q.convs):
        lines.append(
            f"  {{{c.w.shape[1]}, {c.w.shape[0]}, {c.k}, {c.stride}, {c.pool}, "
            f"L{i}_w, L{i}_b, L{i}_m0, L{i}_sh, {c.z_in}, {c.z_out}}},\n"
        )
    lines += [
        "};\n\n",
        "static const ts_model_t TS_MODEL = {\n",
        f"  {len(q.convs)}, TS_LAYERS, {q.inv_s_in!r}f,\n",
        f"  {q.gap_M0}, {q.gap_shift}, {q.z_emb},\n",
        f"  {q.head_w.shape[1]}, {q.head_w.shape[0]}, {q.context}, H_w, H_b, H_m0, H_sh, {q.z_logit},\n",
        "};\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)
