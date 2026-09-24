// OpenCL port of the xetla int2 x fp16/bf16 "upcvt" GEMM/GEMV
// (xetla/include/experimental/group/gemm/impl/int2_fp16_upcvt_xmx_xe.hpp):
//
//   C[m,n] = DT( sum_k A[m,k] * code(B[k,n]) * S[k/128, n] )
//
//   A : DT [M, K] row-major
//   B : uint32 [K/16, N], word (kp, n) holds the 2-bit codes of K rows
//       16*kp .. 16*kp+15 of column n (code of row 16*kp+j at bits 2j..2j+1)
//   S : DT [K/128, N]
//   C : DT [M, N]
//
// DT is fp16, or bf16 with -DBF16 (xetla's XT template parameter).
// Codes are {0, 1, 3} = {0, +1, -1}. As in xetla, the scale is folded into
// the upconvert with integer ops only -- (scale ^ sign) & magnitude -- so the
// B tile comes out in whichever 16-bit float DT is and goes straight into
// the DT DPAS, fp32 accumulate.
//
// Work decomposition (compile-time via -D):
//   SGM    rows per sub-group (DPAS repeat count: 1, 2, 4, 8)
//   NSG_N  sub-groups along N per work-group (WG covers 16*NSG_N columns)
//   LS     local k-slicing: sub-groups per column block splitting K, reduced
//          through SLM
// A sub-group owns 16 columns (one per lane) and walks its K slice in steps
// of 128 (one scale group). Per step: 8 packed B rows (one 2D block read),
// one scale row, SGM A rows, 8 DPAS of K=16.
//
// Requires N % 16 == 0 and K % 128 == 0.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef BF16
#define MAD intel_sub_group_bf16_bf16_matrix_mad_k16
#define TOF(h) as_float((uint)(h) << 16)
#define TOF8(h8) as_float8(convert_uint8(h8) << 16)
#define TO_DT(x) intel_convert_bfloat16_as_ushort(x)
#define TO_DT8(x) intel_convert_bfloat168_as_ushort8(x)
#else
#define MAD intel_sub_group_f16_f16_matrix_mad_k16
#define TOF(h) convert_float(as_half(h))
#define TOF8(h8) convert_float8(as_half8(h8))
#define TO_DT(x) as_ushort(convert_half(x))
#define TO_DT8(x) as_ushort8(convert_half8(x))
#endif
typedef ushort dt;  // raw DT bits

#include "epilogue.clh"

#ifndef SGM
#define SGM 1
#endif
#ifndef NSG_N
#define NSG_N 2
#endif
#ifndef LS
#define LS 4
#endif
// k-steps (of 128) whose loads are issued together before any compute
#ifndef U
#define U 2
#endif
// prefetch B this many k-steps ahead (0 = off)
#ifndef PF
#define PF 0
#endif
#define SG_N 16
#define GS 128

#if SGM == 1
typedef short a_t;
typedef float acc_t;
#elif SGM == 2
typedef short2 a_t;
typedef float2 acc_t;
#elif SGM == 4
typedef short4 a_t;
typedef float4 acc_t;
#elif SGM == 8
typedef short8 a_t;
typedef float8 acc_t;
#else
#error "SGM must be 1, 2, 4 or 8"
#endif

// Two consecutive K codes (one nibble of the word, low bits of x) -> one VNNI
// dword holding the two DT weights, each 0, +scale or -scale (sign = bit 15).
inline int dq_pair(uint x, uint s2) {
    uint mask = ((uint)(((int)(x << 31)) >> 31) & 0x0000FFFFu)
              | ((uint)(((int)(x << 29)) >> 31) & 0xFFFF0000u);
    uint sign = ((x & 2u) << 14) | ((x & 8u) << 28);
    return (int)((s2 ^ sign) & mask);
}

inline int8 dq_word(uint w, uint s2) {
    int8 b;
    b.s0 = dq_pair(w, s2);
    b.s1 = dq_pair(w >> 4, s2);
    b.s2 = dq_pair(w >> 8, s2);
    b.s3 = dq_pair(w >> 12, s2);
    b.s4 = dq_pair(w >> 16, s2);
    b.s5 = dq_pair(w >> 20, s2);
    b.s6 = dq_pair(w >> 24, s2);
    b.s7 = dq_pair(w >> 28, s2);
    return b;
}

inline void load_b(const __global uint *B, int N, int K, int n0, int s,
        __private uint *w) {
#if defined(cl_intel_subgroup_2d_block_io) && !defined(NO_2D) && !defined(B_BLK8)
    // one 2D message per row, as xetla: the first DPAS waits for 64 B, not the whole
    // 512 B tile (5-9% on B70 GEMV, 8B o_proj 13.0 -> 11.7 us)
    for (int r = 0; r < 8; ++r)
        intel_sub_group_2d_block_read_32b_1r16x1c((__global void *)B, N * 4, K / 16,
                N * 4, (int2)(n0, s * 8 + r), &w[r]);
#elif defined(cl_intel_subgroup_2d_block_io) && !defined(NO_2D)
    intel_sub_group_2d_block_read_32b_8r16x1c((__global void *)B, N * 4, K / 16,
            N * 4, (int2)(n0, s * 8), w);
#else
    for (int r = 0; r < 8; ++r)
        w[r] = intel_sub_group_block_read(B + (size_t)(s * 8 + r) * N + n0);
#endif
}

inline void prefetch_b(const __global uint *B, int N, int K, int n0, int s) {
#if PF > 0 && defined(cl_intel_subgroup_2d_block_io)
    intel_sub_group_2d_block_prefetch_32b_8r16x1c((__global void *)B, N * 4,
            K / 16, N * 4, (int2)(n0, s * 8));
#endif
}

inline acc_t step(acc_t acc, const __private uint *w, uint sc,
        const __private ushort8 *ar) {
    const uint s2 = sc | (sc << 16);
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        a_t a;
#if SGM == 1
        a = as_short(ar[0][c]);
#else
        for (int r = 0; r < SGM; ++r) a[r] = as_short(ar[r][c]);
#endif
        acc = MAD(a, dq_word(w[c], s2), acc);
    }
    return acc;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(SG_N * NSG_N * LS, 1, 1)))
__kernel void int2_fp16_upcvt_gemm(const __global dt *A,
        const __global uint *B, const __global dt *S, __global ct *C, EPI_ARGS,
        int M, int N, int K) {
    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int sgn = sg % NSG_N;
    const int sgk = sg / NSG_N;
    const int n0 = (get_group_id(0) * NSG_N + sgn) * SG_N;
    const int m0 = get_group_id(1) * SGM;

    const int nsteps = K / GS;
    const int per = (nsteps + LS - 1) / LS;
    const int s_begin = sgk * per;
    const int s_end = min(nsteps, s_begin + per);

    acc_t acc = 0.0f;
    if (n0 < N) {
        int s = s_begin;
#if PF > 0
        for (int p = 0; p < PF && s_begin + p < s_end; ++p)
            prefetch_b(B, N, K, n0, s_begin + p);
#endif
        for (; s + U <= s_end; s += U) {
            uint w[U][8];
            uint sc[U];
            ushort8 ar[U][SGM];
#pragma unroll
            for (int u = 0; u < U; ++u) {
#if PF > 0
                if (s + u + PF < s_end) prefetch_b(B, N, K, n0, s + u + PF);
#endif
                load_b(B, N, K, n0, s + u, w[u]);
                sc[u] = intel_sub_group_block_read_us(
                        (const __global ushort *)(S + (size_t)(s + u) * N + n0));
                for (int r = 0; r < SGM; ++r)
                    ar[u][r] = (SGM == 1 || m0 + r < M)
                            ? intel_sub_group_block_read_us8(
                                    (const __global ushort *)(A
                                            + (size_t)(m0 + r) * K + (s + u) * GS))
                            : (ushort8)0;
            }
#pragma unroll
            for (int u = 0; u < U; ++u) acc = step(acc, w[u], sc[u], ar[u]);
        }
        for (; s < s_end; ++s) {
            uint w[8];
            load_b(B, N, K, n0, s, w);
            const uint sc = intel_sub_group_block_read_us(
                    (const __global ushort *)(S + (size_t)s * N + n0));
            ushort8 ar[SGM];
            for (int r = 0; r < SGM; ++r)
                ar[r] = (SGM == 1 || m0 + r < M)
                        ? intel_sub_group_block_read_us8(
                                (const __global ushort *)(A
                                        + (size_t)(m0 + r) * K + s * GS))
                        : (ushort8)0;
            acc = step(acc, w, sc, ar);
        }
    }

#if LS > 1
    __local float red[(LS - 1) * NSG_N * SGM * SG_N];
    if (sgk > 0) {
        __local float *dst = red + (((sgk - 1) * NSG_N + sgn) * SGM) * SG_N;
#if SGM == 1
        dst[lane] = acc;
#else
        for (int r = 0; r < SGM; ++r) dst[r * SG_N + lane] = acc[r];
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgk > 0) return;
    for (int j = 0; j < LS - 1; ++j) {
        const __local float *src = red + ((j * NSG_N + sgn) * SGM) * SG_N;
#if SGM == 1
        acc += src[lane];
#else
        for (int r = 0; r < SGM; ++r) acc[r] += src[r * SG_N + lane];
#endif
    }
#endif

    if (n0 >= N) return;
    for (int r = 0; r < SGM; ++r)
        if (m0 + r < M) {
            const size_t i = (size_t)(m0 + r) * N + n0 + lane;
#if SGM == 1
            const float v = acc;
#else
            const float v = acc[r];
#endif
            C[i] = CT_STORE(epi(v, epi_in(Other, Bias, i, n0 + lane)));
        }
}

// ---------------------------------------------------------------------------
// Large-M GEMM. A sub-group computes an MT_M x MT_N tile (MT_M % 8 == 0,
// MT_N % 16 == 0): per K16 it dequantizes B once per 16-column block and
// reuses it for all MT_M/8 DPAS row blocks. A work-group is WG_M x WG_N
// sub-groups. A/B/C go through 2D block I/O, which zero-fills reads and clips
// writes outside the matrix, so any M works.
#ifndef MT_M
#define MT_M 32
#endif
#ifndef MT_N
#define MT_N 32
#endif
#ifndef WG_M
#define WG_M 4
#endif
#ifndef WG_N
#define WG_N 2
#endif
#define MB (MT_M / 8)
#define NB (MT_N / 16)

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16 * WG_N * WG_M, 1, 1)))
__kernel void int2_fp16_upcvt_gemm_mt(const __global dt *A,
        const __global uint *B, const __global dt *S, __global ct *C, EPI_ARGS,
        int M, int N, int K) {
    const int sg = get_sub_group_id();
    const int m0 = (get_group_id(1) * WG_M + sg / WG_N) * MT_M;
    const int n0 = (get_group_id(0) * WG_N + sg % WG_N) * MT_N;

    float8 acc[MB][NB];
    for (int i = 0; i < MB; ++i)
        for (int j = 0; j < NB; ++j) acc[i][j] = 0.0f;

    for (int s = 0; s < K / GS; ++s) {
        uint w[NB][8];
        uint s2[NB];
        for (int j = 0; j < NB; ++j) {
#ifdef MT_B_ROWS
            for (int r = 0; r < 8; ++r)
                intel_sub_group_2d_block_read_32b_1r16x1c((__global void *)B, N * 4,
                        K / 16, N * 4, (int2)(n0 + 16 * j, s * 8 + r), &w[j][r]);
#else
            intel_sub_group_2d_block_read_32b_8r16x1c((__global void *)B, N * 4,
                    K / 16, N * 4, (int2)(n0 + 16 * j, s * 8), w[j]);
#endif
            const uint sc = (n0 + 16 * j < N) ? intel_sub_group_block_read_us(
                    (const __global ushort *)(S + (size_t)s * N + n0 + 16 * j)) : 0u;
            s2[j] = sc | (sc << 16);
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            ushort a[MB][8];
            for (int i = 0; i < MB; ++i)
                intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)A,
                        K * 2, M, K * 2, (int2)(s * GS + 16 * c, m0 + 8 * i), a[i]);
            for (int j = 0; j < NB; ++j) {
                const int8 b = dq_word(w[j][c], s2[j]);
                for (int i = 0; i < MB; ++i)
                    acc[i][j] = MAD(
                            as_short8(vload8(0, a[i])), b, acc[i][j]);
            }
        }
    }

    for (int i = 0; i < MB; ++i)
        for (int j = 0; j < NB; ++j) {
            // barrier: otherwise IGC folds the fp16 conversion into the K loop's
            // accumulator chain and spills at MT_M*MT_N >= 2048 (bf16 unaffected)
            float8 v = acc[i][j];
            __asm__ volatile("" : "+rw"(v));
            epi_store8x16(C, Other, Bias, M, N, m0 + 8 * i, n0 + 16 * j, v);
        }
}
