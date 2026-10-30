/*
 * tinysleep.c -- reference int8 inference kernel. Plain C99, no dynamic memory.
 * Written for clarity and bit-exactness with tinysleep/quant.py, not for speed:
 * no SIMD (a CMSIS-NN style SMLAD inner loop would be the next step).
 */
#include "tinysleep.h"

#include <string.h>

static inline int32_t requant(int32_t acc, int32_t m0, int sh) {
    int64_t p = (int64_t)acc * (int64_t)m0 + ((int64_t)1 << (sh - 1));
    return (int32_t)(p >> sh); /* arithmetic shift on all supported compilers */
}

static inline int8_t clamp8(int32_t v, int32_t lo) {
    if (v < lo) v = lo;
    if (v > 127) v = 127;
    return (int8_t)v;
}

static int conv_out_len(int n, int k, int s) { return (n + 2 * (k / 2) - k) / s + 1; }

int ts_emb_dim(const ts_model_t *m) { return m->layers[m->n_layers - 1].cout; }

int ts_buffer_bytes(const ts_model_t *m) {
    int n = TS_EPOCH_LEN, best = TS_EPOCH_LEN;
    for (int i = 0; i < m->n_layers; i++) {
        const ts_conv_t *L = &m->layers[i];
        n = conv_out_len(n, L->k, L->stride) / L->pool;
        if (n * L->cout > best) best = n * L->cout;
    }
    return best;
}

void ts_reset(ts_state_t *st) { st->n_seen = 0; }

void ts_quantize_input(const ts_model_t *m, const float *x_uv, int8_t *q) {
    float sum = 0.0f;
    for (int i = 0; i < TS_EPOCH_LEN; i++) sum += x_uv[i];
    const float mean = sum / (float)TS_EPOCH_LEN;
    for (int i = 0; i < TS_EPOCH_LEN; i++) {
        float v = x_uv[i] - mean;
        v = v * m->inv_s_in;
        v = v + 0.5f;
        int32_t r = (int32_t)v; /* truncates toward zero ... */
        if ((float)r > v) r -= 1; /* ... so fix up to floor() without libm */
        if (r < -127) r = -127;
        if (r > 127) r = 127;
        q[i] = (int8_t)r;
    }
}

/* Conv1d (zero padding in the real domain == z_in in the code domain, i.e. a
 * skipped tap) + requantise + ReLU (clamp at z_out) + MaxPool, fused so the
 * un-pooled feature map is never stored. Layout: [channel][time]. */
static int conv_block(const ts_conv_t *L, const int8_t *in, int n_in, int8_t *out) {
    const int pad = L->k / 2;
    const int n_conv = conv_out_len(n_in, L->k, L->stride);
    const int n_out = n_conv / L->pool;
    for (int co = 0; co < L->cout; co++) {
        const int8_t *wc = L->w + co * L->cin * L->k;
        for (int j = 0; j < n_out; j++) {
            int32_t best = -128;
            for (int p = 0; p < L->pool; p++) {
                const int t0 = (j * L->pool + p) * L->stride - pad;
                int32_t acc = L->b[co];
                for (int ci = 0; ci < L->cin; ci++) {
                    const int8_t *x = in + ci * n_in;
                    const int8_t *w = wc + ci * L->k;
                    for (int kk = 0; kk < L->k; kk++) {
                        const int t = t0 + kk;
                        if (t >= 0 && t < n_in) acc += (int32_t)w[kk] * ((int32_t)x[t] - L->z_in);
                    }
                }
                int32_t v = requant(acc, L->m0[co], L->sh[co]) + L->z_out;
                if (v > best) best = v;
            }
            out[co * n_out + j] = clamp8(best, L->z_out);
        }
    }
    return n_out;
}

int ts_infer_epoch(const ts_model_t *m, ts_state_t *st, const float *x_uv, int8_t logits[TS_N_CLASSES]) {
    int8_t *a = st->buf_a, *b = st->buf_b;
    ts_quantize_input(m, x_uv, a);
    int n = TS_EPOCH_LEN;
    for (int i = 0; i < m->n_layers; i++) {
        n = conv_block(&m->layers[i], a, n, b);
        int8_t *t = a; a = b; b = t;
    }
    /* Global average pool -> embedding e_t, written to slot 0 of the history. */
    const ts_conv_t *last = &m->layers[m->n_layers - 1];
    const int D = last->cout;
    int8_t *ring = st->ring;
    const int K = m->context;
    memmove(ring + D, ring, (size_t)(K * D)); /* shift history: [e_t, e_{t-1}, ...] */
    for (int c = 0; c < D; c++) {
        int32_t s = 0;
        for (int j = 0; j < n; j++) s += (int32_t)a[c * n + j] - last->z_out;
        ring[c] = clamp8(requant(s, m->gap_m0, m->gap_sh) + m->z_emb, -128);
    }
    if (st->n_seen == 0) /* no history yet: replicate e_0 (matches training) */
        for (int k = 1; k <= K; k++) memcpy(ring + k * D, ring, (size_t)D);
    st->n_seen++;

    int best = 0;
    for (int o = 0; o < m->head_out; o++) {
        int32_t acc = m->hb[o];
        const int8_t *w = m->hw + o * m->head_in;
        for (int i = 0; i < m->head_in; i++) acc += (int32_t)w[i] * ((int32_t)ring[i] - m->z_emb);
        logits[o] = clamp8(requant(acc, m->hm0[o], m->hsh[o]) + m->z_logit, -128);
        if (logits[o] > logits[best]) best = o;
    }
    return best;
}
