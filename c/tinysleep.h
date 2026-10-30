/*
 * tinysleep -- int8 single-channel EEG sleep staging, pure C99, no malloc, no libm
 * in the integer path. One call per 30 s epoch (3000 samples @ 100 Hz).
 *
 * Quantisation: x ~= s * (q - z). Weights int8 per-output-channel symmetric,
 * activations int8 per-tensor, int32 accumulators, fixed-point requantisation
 * q_out = ((acc * m0 + 2^(sh-1)) >> sh) + z_out.
 */
#ifndef TINYSLEEP_H
#define TINYSLEEP_H

#include <stdint.h>

#define TS_EPOCH_LEN 3000
#define TS_N_CLASSES 5

typedef struct {
    int cin, cout, k, stride, pool;
    const int8_t *w;     /* [cout][cin][k] */
    const int32_t *b;    /* [cout] */
    const int32_t *m0;   /* [cout] Q31 multiplier */
    const int8_t *sh;    /* [cout] total right shift */
    int32_t z_in, z_out; /* zero points of input / output tensors */
} ts_conv_t;

typedef struct {
    int n_layers;
    const ts_conv_t *layers;
    float inv_s_in; /* 1 / input scale, LSB per uV */
    int32_t gap_m0;
    int32_t gap_sh;
    int32_t z_emb;
    int head_in, head_out, context; /* head_in = (context + 1) * D */
    const int8_t *hw;
    const int32_t *hb;
    const int32_t *hm0;
    const int8_t *hsh;
    int32_t z_logit;
} ts_model_t;

/* Caller-owned state. Size the buffers with ts_arena_bytes(). */
typedef struct {
    int8_t *buf_a, *buf_b; /* ping-pong activation buffers, each ts_arena_bytes()/2 */
    int8_t *ring;          /* (context + 1) * D embedding history */
    int n_seen;            /* epochs processed since ts_reset() */
} ts_state_t;

int ts_buffer_bytes(const ts_model_t *m); /* bytes needed by EACH of buf_a / buf_b */
int ts_emb_dim(const ts_model_t *m);
void ts_reset(ts_state_t *st);

/* uV float epoch -> int8 codes: subtract mean (float32, sequential), floor(x * inv_s + 0.5), clamp.
 * Compile with -ffp-contract=off so no FMA changes the rounding vs. the Python reference. */
void ts_quantize_input(const ts_model_t *m, const float *x_uv, int8_t *q);

/* Run one epoch. Writes int8 logits, returns argmax stage (0=W,1=N1,2=N2,3=N3,4=REM). */
int ts_infer_epoch(const ts_model_t *m, ts_state_t *st, const float *x_uv, int8_t logits[TS_N_CLASSES]);

#endif
