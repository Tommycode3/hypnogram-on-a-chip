/*
 * Minimal firmware-side entry point, used only to measure the real flash/RAM
 * footprint with arm-none-eabi-size (no startup code, no libc, no HAL).
 * Buffers are sized exactly for the selected model.
 */
#include "tinysleep.h"
#include MODEL_H

static int8_t buf_a[TS_BUF_BYTES], buf_b[TS_BUF_BYTES], ring[TS_RING_BYTES];
static ts_state_t st = {buf_a, buf_b, ring, 0};

void tinysleep_reset(void) { ts_reset(&st); }

int tinysleep_epoch(const float *x_uv, int8_t logits[TS_N_CLASSES]) {
    return ts_infer_epoch(&TS_MODEL, &st, x_uv, logits);
}
