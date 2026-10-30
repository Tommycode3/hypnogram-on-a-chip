"""Tiny 1-D CNNs for single-channel sleep staging.

Every block is   Conv1d(k, stride, pad=k//2, no bias) -> BatchNorm -> ReLU -> MaxPool(p)
so that after BN folding each block is one int8 conv + requantise + clamp + maxpool,
which is exactly what ``c/tinysleep.c`` implements. A global average pool turns the
last feature map into a D-dim epoch embedding e_t; the classifier is a single
linear layer over [e_t, e_{t-1}, ..., e_{t-K}] (K = 0 -> no context). Only past
epochs are used, so the model is causal and can run on-line, one epoch every 30 s,
keeping K embeddings (K*D bytes) in a ring buffer.
"""
from __future__ import annotations

import torch
from torch import nn

INPUT_CLIP_UV = 200.0  # inputs are clipped to +/-200 uV and divided by this

# name -> list of (out_channels, kernel, stride, pool)
CONFIGS: dict[str, list[tuple[int, int, int, int]]] = {
    "xs": [(4, 15, 2, 4), (8, 7, 1, 4), (8, 5, 1, 4), (16, 5, 1, 2)],
    "s": [(8, 15, 2, 4), (12, 7, 1, 4), (16, 5, 1, 4), (24, 5, 1, 2)],
    "m": [(8, 15, 2, 4), (16, 7, 1, 4), (32, 5, 1, 4), (32, 5, 1, 2)],
    "l": [(16, 15, 2, 4), (24, 7, 1, 4), (48, 5, 1, 4), (48, 5, 1, 2)],
    "xl": [(16, 15, 2, 4), (32, 7, 1, 4), (64, 5, 1, 4), (96, 5, 1, 2)],
}


def conv_out_len(n: int, k: int, s: int) -> int:
    return (n + 2 * (k // 2) - k) // s + 1


def receptive_field(cfg) -> int:
    """r_L = r_{L-1} + (k_L - 1) * j_{L-1};  j_L = j_{L-1} * stride_L (pools count as layers)."""
    r, j = 1, 1
    for _, k, s, p in cfg:
        r += (k - 1) * j
        j *= s
        r += (p - 1) * j
        j *= p
    return r


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        blocks, cin = [], 1
        for cout, k, s, p in cfg:
            blocks += [
                nn.Conv1d(cin, cout, k, stride=s, padding=k // 2, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(),
                nn.MaxPool1d(p),
            ]
            cin = cout
        self.body = nn.Sequential(*blocks)
        self.dim = cin

    def forward(self, x):  # x: (B, 3000) normalised
        return self.body(x.unsqueeze(1)).mean(dim=-1)  # (B, D)


class TinySleepCNN(nn.Module):
    def __init__(self, cfg_name: str = "m", context: int = 0, n_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.cfg_name = cfg_name
        self.context = context
        self.encoder = Encoder(CONFIGS[cfg_name])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear((context + 1) * self.encoder.dim, n_classes)

    def classify(self, emb_ctx):  # (B, (K+1)*D)
        return self.head(self.drop(emb_ctx))

    def forward(self, x):  # x: (B, 3000), context-free path
        return self.classify(self.encoder(x))


def stack_context(E: torch.Tensor, K: int) -> torch.Tensor:
    """E: (n, D) embeddings of one night in time order -> (n, (K+1)*D) = [e_t, e_{t-1}, ...].

    Missing past epochs at the start of the night are filled with e_0, matching the
    ring buffer initialisation in C.
    """
    n = E.shape[0]
    idx = torch.arange(n)
    cols = [E[(idx - k).clamp(min=0)] for k in range(K + 1)]
    return torch.cat(cols, dim=1)


def preprocess(X_uv: torch.Tensor) -> torch.Tensor:
    """Remove per-epoch mean, clip to +/-INPUT_CLIP_UV, scale to [-1, 1]."""
    X = X_uv - X_uv.mean(dim=-1, keepdim=True)
    return X.clamp(-INPUT_CLIP_UV, INPUT_CLIP_UV) / INPUT_CLIP_UV


def cost(cfg_name: str, context: int = 0, n_in: int = 3000, n_classes: int = 5) -> dict:
    """Parameters and multiply-accumulates per 30 s epoch (after BN folding)."""
    cfg = CONFIGS[cfg_name]
    params = macs = 0
    n, cin, max_act = n_in, 1, n_in
    for cout, k, s, p in cfg:
        nc = conv_out_len(n, k, s)
        params += cout * cin * k + cout  # folded bias
        macs += nc * cout * cin * k
        max_act = max(max_act, n * cin + (nc // p) * cout)  # in + pooled out (conv+pool fused in C)
        n, cin = nc // p, cout
    D = cin
    macs += n * D  # GAP adds
    params += (context + 1) * D * n_classes + n_classes
    macs += (context + 1) * D * n_classes
    return {"params": params, "macs": macs, "emb_dim": D, "max_live_act": max_act, "rf": receptive_field(cfg)}
