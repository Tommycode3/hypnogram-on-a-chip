/*
 * Host test harness: runs the int8 kernel over a binary dump of test nights.
 *   input : int32 n_nights, then per night { int32 n_epochs, float32 x[n_epochs][3000] }
 *   output: per epoch { int8 logits[5] }   (predictions = argmax, computed by the caller)
 * Build:  cc -O2 -std=c99 -Wall -Wextra -DMODEL_H='"model_m_k4.h"' -Ibuild -Ic c/main.c c/tinysleep.c
 */
#include <stdio.h>
#include <stdlib.h>

#include "tinysleep.h"
#include MODEL_H

static int8_t buf_a[16384], buf_b[16384], ring[4096];

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s in.bin out.bin\n", argv[0]);
        return 2;
    }
    const ts_model_t *m = &TS_MODEL;
    const int need = ts_buffer_bytes(m);
    const int D = ts_emb_dim(m);
    if (need > (int)sizeof buf_a || (m->context + 1) * D > (int)sizeof ring) {
        fprintf(stderr, "buffers too small\n");
        return 1;
    }
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    if (!fi || !fo) return 1;
    int32_t n_nights;
    if (fread(&n_nights, 4, 1, fi) != 1) return 1;
    static float x[TS_EPOCH_LEN];
    ts_state_t st = {buf_a, buf_b, ring, 0};
    long total = 0;
    for (int nn = 0; nn < n_nights; nn++) {
        int32_t n_ep;
        if (fread(&n_ep, 4, 1, fi) != 1) return 1;
        ts_reset(&st);
        for (int e = 0; e < n_ep; e++) {
            if (fread(x, sizeof(float), TS_EPOCH_LEN, fi) != TS_EPOCH_LEN) return 1;
            int8_t logits[TS_N_CLASSES];
            ts_infer_epoch(m, &st, x, logits);
            fwrite(logits, 1, TS_N_CLASSES, fo);
            total++;
        }
    }
    fclose(fi);
    fclose(fo);
    printf("epochs=%ld activation_buffers=2x%d B ring=%d B\n", total, need, (m->context + 1) * D);
    return 0;
}
