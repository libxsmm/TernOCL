// MoE prefill on the int2 x int8 DPAS GEMM (int2_int8_dpas.cl): the same expert
// grouping as int2_moe_prefill.cl, with activations quantized to int8 per
// (row, 128-group) and multiplied on the native s8 x s2 DPAS.
//
//   int2_moe_route / combine            as in int2_moe_prefill.cl
//   QMODE 1 (in-GEMM quantization):
//     int2_moe_gather                   fp16 rows, then quant_a writes SA [K/128, LDSA(G)]
//   QMODE 0 (upfront, build with -DWRITE_Q):
//     quant_a on the T token rows       int8 Xq and SA [K/128, LDSA(T)]
//     int2_moe_gather_q                 int8 rows and their scales into expert order
//   int2_moe_gemm_mt_i8                 int2_int8_gemm_mt per expert segment
//
// up = gemm(X, POSTOP 0), H = gemm(X, gate, POSTOP 1, Other = up), Y = gemm(H, down, OUT_F32);
// H is quantized with quant_a (G rows) before down.

#include "int2_int8_dpas.cl"

#ifndef TOPK
#define TOPK 8
#endif
#ifndef NUM_EXPERTS
#define NUM_EXPERTS 128
#endif
#define ROWS_PER_TILE (MT_M * WG_M)

#if !defined(TERNOCL_SELECT_KERNELS) || defined(TERNOCL_KERNEL_INT2_MOE_ROUTE)
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
#endif

#if !defined(TERNOCL_SELECT_KERNELS) || defined(TERNOCL_KERNEL_INT2_MOE_GATHER)
__kernel void int2_moe_gather(const __global dt *X, const __global int *perm, __global dt *Xs, int K) {
    const int p = get_global_id(1);
    const __global uint *src = (const __global uint *)(X + (size_t)(perm[p] / TOPK) * K);
    __global uint *dst = (__global uint *)(Xs + (size_t)p * K);
    for (int k = get_global_id(0); k < K / 2; k += get_global_size(0)) dst[k] = src[k];
}
#endif

// Xq [T, K] int8, SAt [K/128, LDSA(T)] -> Xsq [G, K], SA [K/128, LDSA(G)] in expert order
#if !defined(TERNOCL_SELECT_KERNELS) || defined(TERNOCL_KERNEL_INT2_MOE_GATHER_Q)
__kernel void int2_moe_gather_q(const __global char *Xq, const __global dt *SAt, const __global int *perm,
        __global char *Xsq, __global dt *SA, int T, int G, int K) {
    const int p = get_global_id(1);
    const int t = perm[p] / TOPK;
    const __global uint *src = (const __global uint *)(Xq + (size_t)t * K);
    __global uint *dst = (__global uint *)(Xsq + (size_t)p * K);
    for (int k = get_global_id(0); k < K / 4; k += get_global_size(0)) dst[k] = src[k];
    for (int g = get_global_id(0); g < K / GS; g += get_global_size(0))
        SA[(size_t)g * LDSA(G) + p] = SAt[(size_t)g * LDSA(T) + t];
}
#endif

#if !defined(TERNOCL_SELECT_KERNELS) || defined(TERNOCL_KERNEL_INT2_MOE_GEMM_MT_I8)
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16 * WG_N * WG_M, 1, 1)))
__kernel void int2_moe_gemm_mt_i8(const __global int *meta, const __global dt *A_all,
        const __global char *Aq_all, const __global dt *SA, const __global uint *B_all,
        const __global dt *SB_all, __global ct *C_all, const __global dt *Other_all, int G, int N, int K) {
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
    const __global char *Aq = Aq_all + (size_t)row0 * K;
    const __global uint *B = B_all + (size_t)e * (K / 16) * N;
    const __global dt *SB = SB_all + (size_t)e * (K / GS) * N;
    __global ct *C = C_all + (size_t)row0 * N;
    const __global dt *Other = Other_all + (size_t)row0 * N;
    const __global ct *Bias = 0;

    // from here on, int2_int8_gemm_mt (QMODE 1) with m0 relative to the expert's rows
    const int sg = get_sub_group_id();
    const int m0 = ((tile - meta[NUM_EXPERTS + 1 + e]) * WG_M + sg / WG_N) * MT_M;
    const int n0 = (get_group_id(0) * WG_N + sg % WG_N) * MT_N;

    float8 acc[MB][NB];
    for (int i = 0; i < MB; ++i)
        for (int j = 0; j < NB; ++j) acc[i][j] = 0.0f;

    for (int s = 0; s < K / GS; ++s) {
        uint w[NB][8];
        float sb[NB];
        for (int j = 0; j < NB; ++j)
            intel_sub_group_2d_block_read_32b_8r16x1c((__global void *)B, N * 4,
                    K / 16, N * 4, (int2)(n0 + 16 * j, s * 8), w[j]);
        {
            ushort t[NB];
            for (int j = 0; j < NB; j += 2) {
                if (j + 1 < NB)
                    intel_sub_group_2d_block_read_16b_1r16x2c((__global void *)SB, N * 2,
                            K / GS, N * 2, (int2)(n0 + 16 * j, s), &t[j]);
                else
                    intel_sub_group_2d_block_read_16b_1r16x1c((__global void *)SB, N * 2,
                            K / GS, N * 2, (int2)(n0 + 16 * j, s), &t[j]);
            }
            for (int j = 0; j < NB; ++j) sb[j] = TOF(t[j]);
        }
#pragma unroll
        for (int i = 0; i < MB; ++i) {
            const int mr = m0 + 8 * i;
            short8 aq[4];
            float inv[8];
#if QMODE == 0
            ushort t[2][16];
            for (int h = 0; h < 2; ++h)
                intel_sub_group_2d_block_read_16b_8r16x2c((__global void *)Aq, K, M, K,
                        (int2)(s * GS / 2 + 32 * h, mr), t[h]);
            for (int c = 0; c < 4; ++c) aq[c] = as_short8(vload8(0, &t[c / 2][8 * (c % 2)]));
#else
            uint a[4][8];
            for (int c = 0; c < 4; ++c)
                intel_sub_group_2d_block_read_32b_8r16x1c((__global void *)A, K * 2, M,
                        K * 2, (int2)(s * GS / 2 + 16 * c, mr), a[c]);
#endif
            // segment rows start anywhere (odd row0 breaks load_sa's 2D read alignment): plain
            // per-lane load; rows past the segment read another expert's scale, their dot is 0
            // and the store clips them
            const float sal = TOF(SA[(size_t)s * LDSA(G) + min(row0 + mr + (int)get_sub_group_local_id(), G - 1)]);
#if QMODE != 0
            for (int c = 0; c < 4; ++c) aq[c] = quant8x32(vload8(0, a[c]), sal);
#endif
            const float invl = native_recip(sal);
            for (int r = 0; r < 8; ++r) inv[r] = sub_group_broadcast(invl, r);
            for (int j = 0; j < NB; ++j) {
                int8 ia = 0;
                for (int c = 0; c < 4; ++c)
                    ia = intel_sub_group_i8_i2_matrix_mad_k32(aq[c],
                            (int2)(w[j][2 * c], w[j][2 * c + 1]), ia);
                const float8 fi = convert_float8(ia);
                for (int r = 0; r < 8; ++r) acc[i][j][r] += fi[r] * (sb[j] * inv[r]);
            }
        }
    }

    for (int i = 0; i < MB; ++i)
        for (int j = 0; j < NB; ++j)
            epi_store8x16(C, Other, Bias, M, N, m0 + 8 * i, n0 + 16 * j, acc[i][j]);
}
#endif

#if !defined(TERNOCL_SELECT_KERNELS) || defined(TERNOCL_KERNEL_INT2_MOE_COMBINE)
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
#endif
