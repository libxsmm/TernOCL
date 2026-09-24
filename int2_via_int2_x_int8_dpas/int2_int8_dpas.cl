// OpenCL port of the xetla int2 x int8 DPAS GEMM/GEMV
// (xetla/include/experimental/group/gemm/impl/int2_fp16_dpas_xmx_xe.hpp and
//  int2_bf16_dpas_xmx_xe.hpp, driver xetla/int2_fp16_dpas_fp16scales_fast_test):
//
//   C[m,n] = DT( sum_g  float(sum_{k in g} q[m,k] * code(B[k,n]))
//                       * (SB[g,n] * (1 / SA[g,m])) )
//   q[m,k] = sat_int8_rtz(A[m,k] * SA[g,m]),  SA[g,m] = DT(127 / absmax_g(A[m,:]))
//
//   A  : DT [M, K] row-major             SA : DT [K/128, LDSA(M)]
//   B  : uint32 [K/16, N], word (kp, n) holds the 2-bit two's-complement codes
//        of K rows 16*kp .. 16*kp+15 of column n (row 16*kp+j at bits 2j..2j+1)
//   SB : DT [K/128, N]                   C  : DT [M, N]
//   Aq : int8 [M, K] (QMODE 0 only)
//
// DT is fp16, or bf16 with -DBF16 (xetla's fp16 / bf16 variants).
// B never gets unpacked: the DPAS takes it natively (s8 x s2,
// intel_sub_group_i8_i2_matrix_mad_k32). Per lane (= column) the int2 B
// operand is two consecutive words of the layout above, i.e. K = 32.
// int32 accumulation within a 128-group, fp32 rescale per group, DT store.
//
// QMODE (A quantization):
//   0  upfront : quant_a writes SA and int8 Aq; the GEMM reads int8 A
//      (pure OpenCL)
//   1  xetla   : quant_a writes SA only (xetla external_scale_a_calc=1); the
//      GEMM quantizes the DT A tile on the fly with xetla's instruction
//      sequence (inline vISA, quant8x32)
// A quantized A tile is reused for every 16-column DPAS block of the
// sub-group tile, and B (never converted) for every 8-row block.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef ushort dt;  // raw DT bits
#ifdef BF16
#define TOF(h) as_float((uint)(h) << 16)
#define TOF2(h2) ((float2)(as_float((h2) << 16), as_float((h2) & 0xffff0000u)))
#define TOF8(h8) as_float8(convert_uint8(h8) << 16)
#define TO_DT(x) intel_convert_bfloat16_as_ushort(x)
#define TO_DT8(x) intel_convert_bfloat168_as_ushort8(x)
#else
#define TOF(h) convert_float(as_half(h))
#define TOF2(h2) convert_float2(as_half2(h2))
#define TOF8(h8) convert_float8(as_half8(h8))
#define TO_DT(x) as_ushort(convert_half(x))
#define TO_DT8(x) as_ushort8(convert_half8(x))
#endif

#include "epilogue.clh"

#ifndef QMODE
#define QMODE 1
#endif
#define GS 128
// SA row pitch, padded so SA is a valid 2D block surface (pitch % 64 B == 0)
#define LDSA(M) (((M) + 31) & ~31)
#define EPS 1.1920928955078125e-07f  /* FLT_EPSILON, xetla absmax init */

// GEMV path: clamp + RTZ convert lowers to one mov.sat per element
inline short q2(uint h2, float sa) {
    return as_short(convert_char2(clamp(TOF2(h2) * sa, -128.0f, 127.0f)));
}

// xetla's elemwise_scale_{fp16,bf16}_to_int8 on one 8-row x 32-K block, on the
// row-contiguous registers (SIMT code would split and re-pair every pair).
// a: row r = a[r] (lane l holds K 2l, 2l+1); lane r of sal = scale of row r.
// SIMD32 NoMask like xetla: call only from convergent code.
#ifdef BF16
#define A2F(r, g) "shl (M1_NM, 32) TD(" #g ",0)<1> AW(" #r ",0)<1;1,0> 0x10:uw\n"
#else
#define A2F(r, g) "mov (M1_NM, 32) T(" #g ",0)<1> AH(" #r ",0)<1;1,0>\n"
#endif
inline short8 quant8x32(uint8 a, float sal) {
    short8 q;
    __asm__ volatile("{\n"
        ".decl AH v_type=G type=hf num_elts=256 align=GRF alias=<%1,0>\n"
        ".decl AW v_type=G type=uw num_elts=256 align=GRF alias=<%1,0>\n"
        ".decl QB v_type=G type=b num_elts=256 align=GRF alias=<%0,0>\n"
        ".decl T v_type=G type=f num_elts=256 align=GRF\n"
        ".decl TD v_type=G type=ud num_elts=256 align=GRF alias=<T,0>\n"
        ".decl TB v_type=G type=b num_elts=1024 align=GRF alias=<T,0>\n"
#define QROW(r, g, orow, ocol) \
        A2F(r, g) \
        "mul (M1_NM, 32) T(" #g ",0)<1> T(" #g ",0)<1;1,0> %2(0," #r ")<0;1,0>\n" \
        "mov.sat (M1_NM, 32) TB(" #g ",0)<4> T(" #g ",0)<1;1,0>\n" \
        "mov (M1_NM, 32) QB(" #orow "," #ocol ")<1> TB(" #g ",0)<4;1,0>\n"
        QROW(0, 0, 0,0) QROW(1, 2, 0,32) QROW(2, 4, 1,0) QROW(3, 6, 1,32)
        QROW(4, 8, 2,0) QROW(5, 10, 2,32) QROW(6, 12, 3,0) QROW(7, 14, 3,32)
#undef QROW
        "}\n" : "=rw"(q) : "rw"(a), "rw"(sal));
    return q;
}

// ---------------------------------------------------------------------------
// Pre-kernel: one sub-group per (row, 128-group). SA always, Aq if WRITE_Q.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void quant_a(const __global dt *A, __global dt *SA,
        __global char *Aq, int M, int K) {
    const int lane = get_sub_group_local_id();
    const int g = get_global_id(0) / 16;
    const int m = get_global_id(1);
    if (g >= K / GS || m >= M) return;
    const size_t off = (size_t)m * K + g * GS + 8 * lane;
    const float8 a = TOF8(vload8(0, A + off));
    const float8 f = fabs(a);
    float mx = fmax(fmax(fmax(f.s0, f.s1), fmax(f.s2, f.s3)),
            fmax(fmax(f.s4, f.s5), fmax(f.s6, f.s7)));
    mx = sub_group_reduce_max(mx);
    const dt sh = TO_DT(127.0f / fmax(mx, EPS));
    if (lane == 0) SA[(size_t)g * LDSA(M) + m] = sh;
#ifdef WRITE_Q
    vstore8(convert_char8_sat(a * TOF(sh)), 0, Aq + off);
#endif
}

// ---------------------------------------------------------------------------
// GEMV / small M. A sub-group owns 16 columns and SGM rows and walks its K
// slice in 128-steps: one 2D block read of 8 B words (4 DPAS of K = 32), one
// SB row, SGM A rows (quantized in registers for QMODE 1).
//   SGM rows per sub-group (1, 2, 4, 8)   NSG_N sub-groups along N
//   LS  K-slices per column block, reduced through SLM
//   U   128-steps whose loads are issued before any compute
#ifndef SGM
#define SGM 1
#endif
#ifndef NSG_N
#define NSG_N 2
#endif
#ifndef LS
#define LS 1
#endif
#ifndef U
#define U 2
#endif

#if SGM == 1
typedef short a_t;
typedef int ia_t;
typedef float fa_t;
#define CVT_F convert_float
#elif SGM == 2
typedef short2 a_t;
typedef int2 ia_t;
typedef float2 fa_t;
#define CVT_F convert_float2
#elif SGM == 4
typedef short4 a_t;
typedef int4 ia_t;
typedef float4 fa_t;
#define CVT_F convert_float4
#elif SGM == 8
typedef short8 a_t;
typedef int8 ia_t;
typedef float8 fa_t;
#define CVT_F convert_float8
#else
#error "SGM must be 1, 2, 4 or 8"
#endif

#if SGM == 1
#define EL(v, r) (v)
#else
#define EL(v, r) ((v)[r])
#endif

// Loads + quantizes the SGM x 128 A tile of step s: aq[c] = K 32c..32c+31,
// inv[r] = 1 / SA of row r.
inline void load_a(const __global dt *A, const __global char *Aq,
        const __global dt *SA, int M, int K, int m0, int s,
        __private a_t *aq, __private float *inv) {
#pragma unroll
    for (int r = 0; r < SGM; ++r) {
        const bool ok = m0 + r < M;
#if QMODE == 0
        const ushort4 v = ok ? intel_sub_group_block_read_us4(
                (const __global ushort *)(Aq + (size_t)(m0 + r) * K + s * GS)) : (ushort4)0;
        for (int c = 0; c < 4; ++c) EL(aq[c], r) = as_short(v[c]);
        inv[r] = ok ? native_recip(TOF(SA[(size_t)s * LDSA(M) + m0 + r])) : 0.0f;
#else
        const uint4 v = ok ? intel_sub_group_block_read4(
                (const __global uint *)(A + (size_t)(m0 + r) * K + s * GS)) : (uint4)0;
        const float sa = ok ? TOF(SA[(size_t)s * LDSA(M) + m0 + r]) : 0.0f;
        for (int c = 0; c < 4; ++c) EL(aq[c], r) = q2(v[c], sa);
        inv[r] = ok ? native_recip(sa) : 0.0f;
#endif
    }
}

inline fa_t step(fa_t acc, const __private uint *w, float sb,
        const __private a_t *aq, const __private float *inv) {
    ia_t ia = 0;
#pragma unroll
    for (int c = 0; c < 4; ++c)
        ia = intel_sub_group_i8_i2_matrix_mad_k32(aq[c], (int2)(w[2 * c], w[2 * c + 1]), ia);
    const fa_t fi = CVT_F(ia);
#pragma unroll
    for (int r = 0; r < SGM; ++r) EL(acc, r) += EL(fi, r) * (sb * inv[r]);
    return acc;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16 * NSG_N * LS, 1, 1)))
__kernel void int2_int8_gemv(const __global dt *A, const __global char *Aq,
        const __global dt *SA, const __global uint *B,
        const __global dt *SB, __global ct *C, EPI_ARGS, int M, int N, int K) {
    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int sgn = sg % NSG_N;
    const int sgk = sg / NSG_N;
    const int n0 = (get_group_id(0) * NSG_N + sgn) * 16;
    const int m0 = get_group_id(1) * SGM;

    const int nsteps = K / GS;
    const int per = (nsteps + LS - 1) / LS;
    const int s_begin = sgk * per;
    const int s_end = min(nsteps, s_begin + per);

    fa_t acc = 0.0f;
    if (n0 < N) {
        int s = s_begin;
        for (; s + U <= s_end; s += U) {
            uint w[U][8];
            float sb[U];
            a_t aq[U][4];
            float inv[U][SGM];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                intel_sub_group_2d_block_read_32b_8r16x1c((__global void *)B, N * 4,
                        K / 16, N * 4, (int2)(n0, (s + u) * 8), w[u]);
                sb[u] = TOF(intel_sub_group_block_read_us(
                        (const __global ushort *)(SB + (size_t)(s + u) * N + n0)));
                load_a(A, Aq, SA, M, K, m0, s + u, aq[u], inv[u]);
            }
#pragma unroll
            for (int u = 0; u < U; ++u) acc = step(acc, w[u], sb[u], aq[u], inv[u]);
        }
        for (; s < s_end; ++s) {
            uint w[8];
            a_t aq[4];
            float inv[SGM];
            intel_sub_group_2d_block_read_32b_8r16x1c((__global void *)B, N * 4,
                    K / 16, N * 4, (int2)(n0, s * 8), w);
            const float sb = TOF(intel_sub_group_block_read_us(
                    (const __global ushort *)(SB + (size_t)s * N + n0)));
            load_a(A, Aq, SA, M, K, m0, s, aq, inv);
            acc = step(acc, w, sb, aq, inv);
        }
    }

#if LS > 1
    __local float red[(LS - 1) * NSG_N * SGM * 16];
    if (sgk > 0) {
        __local float *dst = red + (((sgk - 1) * NSG_N + sgn) * SGM) * 16;
        for (int r = 0; r < SGM; ++r) dst[r * 16 + lane] = EL(acc, r);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgk > 0) return;
    for (int j = 0; j < LS - 1; ++j) {
        const __local float *src = red + ((j * NSG_N + sgn) * SGM) * 16;
        for (int r = 0; r < SGM; ++r) EL(acc, r) += src[r * 16 + lane];
    }
#endif

    if (n0 >= N) return;
    for (int r = 0; r < SGM; ++r)
        if (m0 + r < M) {
            const size_t i = (size_t)(m0 + r) * N + n0 + lane;
            C[i] = CT_STORE(epi(EL(acc, r), epi_in(Other, Bias, i, n0 + lane)));
        }
}

// ---------------------------------------------------------------------------
// Large-M GEMM. A sub-group computes an MT_M x MT_N tile (MT_M % 8 == 0,
// MT_N % 16 == 0); a work-group is WG_M x WG_N sub-groups. Per 128-group:
// B (8 words per 16-column block) and SB are loaded once and reused for all
// MT_M/8 row blocks; each 8 x 128 A block is loaded (and for QMODE 1
// quantized) once and reused for all MT_N/16 column blocks. 2D block I/O
// zero-fills out-of-range reads and clips writes, so any M works.
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

// lane l gets SA[s, m + l] with one 2D read (pad columns past M are junk)
inline float load_sa(const __global dt *SA, int M, int K, int m, int s) {
    ushort t;
    intel_sub_group_2d_block_read_16b_1r16x1c((__global void *)SA, LDSA(M) * 2,
            K / GS, LDSA(M) * 2, (int2)(m, s), &t);
    return TOF(t);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16 * WG_N * WG_M, 1, 1)))
__kernel void int2_int8_gemm_mt(const __global dt *A, const __global char *Aq,
        const __global dt *SA, const __global uint *B,
        const __global dt *SB, __global ct *C, EPI_ARGS, int M, int N, int K) {
    const int sg = get_sub_group_id();
    const int m0 = (get_group_id(1) * WG_M + sg / WG_N) * MT_M;
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
            const float sal = load_sa(SA, M, K, mr, s);
            // rows >= M get inf here, but their int32 dot is 0 and the store clips them
            for (int r = 0; r < 8; ++r) inv[r] = native_recip(sub_group_broadcast(sal, r));
#else
            uint a[4][8];
            for (int c = 0; c < 4; ++c)
                intel_sub_group_2d_block_read_32b_8r16x1c((__global void *)A, K * 2, M,
                        K * 2, (int2)(s * GS / 2 + 16 * c, mr), a[c]);
            const float sal = load_sa(SA, M, K, mr, s);
            for (int c = 0; c < 4; ++c) aq[c] = quant8x32(vload8(0, a[c]), sal);
            for (int r = 0; r < 8; ++r) inv[r] = native_recip(sub_group_broadcast(sal, r));
#endif
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
