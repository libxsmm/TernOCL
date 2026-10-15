// MoE prefill: tokens grouped by expert, then per-expert M-tiled int2 GEMMs
// (the tile loop of int2_fp16_upcvt.cl's int2_fp16_upcvt_gemm_mt), so an expert's
// weights are read once per row tile instead of once per token. Routing stays on the
// device:
//
//   int2_moe_route    one work-group: rows per expert, offsets, row permutation,
//                     row-tile offsets (meta) and the inverse permutation
//   int2_moe_gather   Xs[p] = X[perm[p] / TOPK]                      (expert-sorted rows)
//   int2_moe_gemm_mt  C[p, n] = epi(Xs[p] . B_e[:, n]) for the rows p of expert e,
//                     one work-group per (column block, row tile of some expert)
//   int2_moe_combine  O[t, n] = sum_j rw[g] * Y[inv[g], n],  g = t * TOPK + j
//
// Gate / up / down: up = gemm(POSTOP 0), H = gemm(gate, POSTOP 1, Other = up)
// = silu(gate) * up, Y = gemm(H, down, OUT_F32). Row tiles are WG_M * MT_M rows.
//
//   meta (int): [0, E]      row offset of each expert in the sorted order
//               [E+1, 2E+1] row-tile offset of each expert

#include "int2_fp16_upcvt.cl"

#ifndef TOPK
#define TOPK 8
#endif
#ifndef NUM_EXPERTS
#define NUM_EXPERTS 128
#endif
#define ROWS_PER_TILE (MT_M * WG_M)

__kernel void int2_moe_route(const __global int *ids, __global int *meta, __global int *perm,
        __global int *inv, int G) {
    __local int cnt[NUM_EXPERTS];
    __local int off[NUM_EXPERTS + 1];
    const int lid = get_local_id(0), nl = get_local_size(0);
    for (int e = lid; e < NUM_EXPERTS; e += nl) cnt[e] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int g = lid; g < G; g += nl) atomic_inc(&cnt[ids[g]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int r = 0, t = 0;
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            off[e] = r;
            meta[e] = r;
            meta[NUM_EXPERTS + 1 + e] = t;
            r += cnt[e];
            t += (cnt[e] + ROWS_PER_TILE - 1) / ROWS_PER_TILE;
            cnt[e] = 0;
        }
        off[NUM_EXPERTS] = r;
        meta[NUM_EXPERTS] = r;
        meta[2 * NUM_EXPERTS + 1] = t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int g = lid; g < G; g += nl) {
        const int e = ids[g];
        const int p = off[e] + atomic_inc(&cnt[e]);
        perm[p] = g;
        inv[g] = p;
    }
}

__kernel void int2_moe_gather(const __global dt *X, const __global int *perm, __global dt *Xs, int K) {
    const int p = get_global_id(1);
    const __global uint *src = (const __global uint *)(X + (size_t)(perm[p] / TOPK) * K);
    __global uint *dst = (__global uint *)(Xs + (size_t)p * K);
    for (int k = get_global_id(0); k < K / 2; k += get_global_size(0)) dst[k] = src[k];
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16 * WG_N * WG_M, 1, 1)))
__kernel void int2_moe_gemm_mt(const __global int *meta, const __global dt *A_all,
        const __global uint *B_all, const __global dt *S_all, __global ct *C_all,
        const __global dt *Other_all, int N, int K) {
    // this work-group's expert: the last one whose first row tile is <= tile
    const int tile = get_group_id(1);
    if (tile >= meta[2 * NUM_EXPERTS + 1]) return;
    int lo = 0, hi = NUM_EXPERTS - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (meta[NUM_EXPERTS + 1 + mid] <= tile) lo = mid;
        else hi = mid - 1;
    }
    const int e = lo;
    const int row0 = meta[e];
    const int M = meta[e + 1] - row0;
    const __global dt *A = A_all + (size_t)row0 * K;
    const __global uint *B = B_all + (size_t)e * (K / 16) * N;
    const __global dt *S = S_all + (size_t)e * (K / GS) * N;
    __global ct *C = C_all + (size_t)row0 * N;
    const __global dt *Other = Other_all + (size_t)row0 * N;
    const __global ct *Bias = 0;

    // from here on, int2_fp16_upcvt_gemm_mt with m0 relative to the expert's rows
    const int sg = get_sub_group_id();
    const int m0 = ((tile - meta[NUM_EXPERTS + 1 + e]) * WG_M + sg / WG_N) * MT_M;
    const int n0 = (get_group_id(0) * WG_N + sg % WG_N) * MT_N;

    float8 acc[MB][NB];
    for (int i = 0; i < MB; ++i)
        for (int j = 0; j < NB; ++j) acc[i][j] = 0.0f;

    for (int s = 0; s < K / GS; ++s) {
        uint w[NB][8];
        uint s2[NB];
        for (int j = 0; j < NB; ++j) {
            intel_sub_group_2d_block_read_32b_8r16x1c((__global void *)B, N * 4,
                    K / 16, N * 4, (int2)(n0 + 16 * j, s * 8), w[j]);
            ushort sc;
            intel_sub_group_2d_block_read_16b_1r16x1c((__global void *)S, N * 2,
                    K / GS, N * 2, (int2)(n0 + 16 * j, s), &sc);
            s2[j] = (uint)sc * 0x10001u;
        }
#pragma unroll
        for (int c2 = 0; c2 < GS / AK; ++c2) {
            ushort a[MT_M / AR][AK / 16 * AR];
            for (int i = 0; i < MT_M / AR; ++i)
                A_READ((__global void *)A, K * 2, M, K * 2,
                        (int2)(s * GS + AK * c2, m0 + AR * i), a[i]);
#pragma unroll
            for (int h = 0; h < AK / 16; ++h)
                for (int j = 0; j < NB; ++j) {
                    const int8 b = dq_word(w[j][AK / 16 * c2 + h], s2[j]);
                    for (int i = 0; i < MB; ++i)
                        acc[i][j] = MAD(as_short8(vload8(0,
                                &a[i / (AR / 8)][h * AR + 8 * (i % (AR / 8))])), b, acc[i][j]);
                }
        }
    }

    for (int i = 0; i < MB; ++i)
        for (int j = 0; j < NB; ++j) {
            float8 v = acc[i][j];
            __asm__ volatile("" : "+rw"(v));
            epi_store8x16(C, Other, Bias, M, N, m0 + 8 * i, n0 + 16 * j, v);
        }
}

__kernel void int2_moe_combine(const __global float *Y, const __global int *inv, const __global dt *RW,
        __global dt *O, int N) {
    const int n = get_global_id(0);
    const int t = get_global_id(1);
    if (n >= N) return;
    float s = 0.0f;
    for (int j = 0; j < TOPK; ++j) {
        const int g = t * TOPK + j;
        s += TOF(RW[g]) * Y[(size_t)inv[g] * N + n];
    }
    O[(size_t)t * N + n] = TO_DT(s);
}
