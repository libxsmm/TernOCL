// 1-bit (binary, {+1, -1} x per-group scale) weight GEMM/GEMV with fp16/bf16
// activations, the 1-bit sibling of int2_fp16_upcvt.cl:
//
//   C[m,n] = DT( sum_k A[m,k] * sgn(B[k,n]) * S[k/128, n] ),  bit 0 -> +1, 1 -> -1
//
//   A : DT [M, K] row-major
//   B : uint32 [K/32, N] select masks; within K32 row kp and 16-column block
//       n0, word (kp, n0 + p) holds the bits of row pair p (rows 32kp + 2p,
//       32kp + 2p + 1) for columns n0..n0+15: bit j = (row 32kp + 2p + (j & 1),
//       column n0 + j/2)
//   S : DT [K/128, N]
//   C : DT [M, N]
//
// DT is fp16, or bf16 with -DBF16. The B operand of the f16/bf16 DPAS is
// VNNI2: dword c of lane n holds rows 2c (low half) and 2c+1 (high half) of
// column n, so register c of the operand, read as 32 16-bit channels, is
// (column j/2, row 2c + (j & 1)) for channel j -- the bit order of one mask
// word. Each DPAS register is then one predicated select between the
// hoisted -s and +s pairs, with the weight word as the predicate:
//   setp P <- w(lane c);  (P) sel (32) B[c]:uw  -s2, +s2
// (inline vISA). No shifts, shuffles or SLM.
//
// -DSHL keeps the earlier shift path with a per-column VNNI2 bit order (word
// (kp, n): row 32kp + 2i at bit i, 32kp + 2i + 1 at bit 16 + i), where
// s2 ^ ((w << (15 - i)) & 0x80008000) builds dword i (shl + bfn, plain C).
//
// Work decomposition (compile-time via -D), as in int2_fp16_upcvt.cl:
//   SGM    rows per sub-group (DPAS repeat count: 1, 2, 4, 8)
//   NSG_N  sub-groups along N per work-group (WG covers 16*NSG_N columns)
//   LS     local k-slicing: sub-groups per column block splitting K, reduced
//          through SLM
//   U      k-steps (of 128) whose loads are issued before any compute
// A sub-group owns 16 columns (one per lane) and walks its K slice in steps
// of 128 (one scale group). Per step: 4 packed B rows, one scale row, SGM A
// rows, 8 DPAS of K=16.
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
#ifndef U
#define U 2
#endif
// B rows (of K32) per 2D block read in the GEMV: 1, 2 or 4
#ifndef BR
#define BR 1
#endif
#define SG_N 16
#define GS 128
#define WPS (GS / 32)  // B words per lane per k-step

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

#ifndef SHL
// predicate of register c = mask word of row pair p, held by lane p
#define SEL_ROW(c, p) \
    "setp (M1_NM, 32) P" #c " %1(0," #p ")<0;1,0>\n" \
    "(P" #c ") sel (M1_NM, 32) BW(" #c ",0)<1> NW(0,0)<1;1,0> PW(0,0)<1;1,0>\n"
#define SEL_DECLS \
    ".decl P0 v_type=P num_elts=32\n.decl P1 v_type=P num_elts=32\n" \
    ".decl P2 v_type=P num_elts=32\n.decl P3 v_type=P num_elts=32\n" \
    ".decl P4 v_type=P num_elts=32\n.decl P5 v_type=P num_elts=32\n" \
    ".decl P6 v_type=P num_elts=32\n.decl P7 v_type=P num_elts=32\n" \
    ".decl BW v_type=G type=uw num_elts=256 align=GRF alias=<%0,0>\n" \
    ".decl PW v_type=G type=uw num_elts=32 align=GRF alias=<%2,0>\n" \
    ".decl NW v_type=G type=uw num_elts=32 align=GRF alias=<%3,0>\n"

inline int8 dq_half(uint w, uint s2, int h) {
    const uint n2 = s2 ^ 0x80008000u;
    int8 b;
    if (h == 0)
        __asm__("{\n" SEL_DECLS SEL_ROW(0, 0) SEL_ROW(1, 1) SEL_ROW(2, 2) SEL_ROW(3, 3)
                SEL_ROW(4, 4) SEL_ROW(5, 5) SEL_ROW(6, 6) SEL_ROW(7, 7) "}\n"
                : "=rw"(b) : "rw"(w), "rw"(s2), "rw"(n2));
    else
        __asm__("{\n" SEL_DECLS SEL_ROW(0, 8) SEL_ROW(1, 9) SEL_ROW(2, 10) SEL_ROW(3, 11)
                SEL_ROW(4, 12) SEL_ROW(5, 13) SEL_ROW(6, 14) SEL_ROW(7, 15) "}\n"
                : "=rw"(b) : "rw"(w), "rw"(s2), "rw"(n2));
    return b;
}
#else
// K rows 16h .. 16h+15 of a B word (h = 0, 1) -> the int8 VNNI2 B operand,
// each DT weight +s or -s.
inline int8 dq_half(uint w, uint s2, int h) {
    int8 b;
#pragma unroll
    for (int c = 0; c < 8; ++c)
        b[c] = (int)(s2 ^ ((w << (15 - 8 * h - c)) & 0x80008000u));
    return b;
}
#endif

inline void load_b(const __global uint *B, int N, int K, int n0, int s,
        __private uint *w) {
#if BR == 4
    intel_sub_group_2d_block_read_32b_4r16x1c((__global void *)B, N * 4, K / 32,
            N * 4, (int2)(n0, s * WPS), w);
#elif BR == 2
    for (int r = 0; r < WPS; r += 2)
        intel_sub_group_2d_block_read_32b_2r16x1c((__global void *)B, N * 4, K / 32,
                N * 4, (int2)(n0, s * WPS + r), &w[r]);
#else
    // one message per row: the first DPAS waits for 64 B, not the whole tile
    for (int r = 0; r < WPS; ++r)
        intel_sub_group_2d_block_read_32b_1r16x1c((__global void *)B, N * 4, K / 32,
                N * 4, (int2)(n0, s * WPS + r), &w[r]);
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
        acc = MAD(a, dq_half(w[c / 2], s2, c % 2), acc);
    }
    return acc;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(SG_N * NSG_N * LS, 1, 1)))
__kernel void int1_fp16_upcvt_gemm(const __global dt *A,
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
        for (; s + U <= s_end; s += U) {
            uint w[U][WPS];
            uint sc[U];
            ushort8 ar[U][SGM];
#pragma unroll
            for (int u = 0; u < U; ++u) {
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
            uint w[WPS];
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
// A rows per 2D block read (8, 16 or 32, dividing MT_M) and K per read (AK = 32
// or 16; 32 K of A for MT_M > 64 does not fit in 256 GRF next to the accumulators)
#ifndef AR
#define AR (MT_M % 32 == 0 ? 32 : (MT_M % 16 == 0 ? 16 : 8))
#endif
#ifndef AK
#define AK (MT_M <= 64 ? 32 : 16)
#endif
#if AK == 32
#if AR == 32
#define A_READ intel_sub_group_2d_block_read_16b_32r16x2c
#elif AR == 16
#define A_READ intel_sub_group_2d_block_read_16b_16r16x2c
#else
#define A_READ intel_sub_group_2d_block_read_16b_8r16x2c
#endif
#else
#if AR == 32
#define A_READ intel_sub_group_2d_block_read_16b_32r16x1c
#elif AR == 16
#define A_READ intel_sub_group_2d_block_read_16b_16r16x1c
#else
#define A_READ intel_sub_group_2d_block_read_16b_8r16x1c
#endif
#endif

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16 * WG_N * WG_M, 1, 1)))
__kernel void int1_fp16_upcvt_gemm_mt(const __global dt *A,
        const __global uint *B, const __global dt *S, __global ct *C, EPI_ARGS,
        int M, int N, int K) {
    const int sg = get_sub_group_id();
    const int m0 = (get_group_id(1) * WG_M + sg / WG_N) * MT_M;
    const int n0 = (get_group_id(0) * WG_N + sg % WG_N) * MT_N;

    float8 acc[MB][NB];
    for (int i = 0; i < MB; ++i)
        for (int j = 0; j < NB; ++j) acc[i][j] = 0.0f;

    for (int s = 0; s < K / GS; ++s) {
        uint w[NB][WPS];
        uint s2[NB];
        for (int j = 0; j < NB; ++j) {
            intel_sub_group_2d_block_read_32b_4r16x1c((__global void *)B, N * 4,
                    K / 32, N * 4, (int2)(n0 + 16 * j, s * WPS), w[j]);
            ushort sc;
            intel_sub_group_2d_block_read_16b_1r16x1c((__global void *)S, N * 2,
                    K / GS, N * 2, (int2)(n0 + 16 * j, s), &sc);
            s2[j] = (uint)sc * 0x10001u;
        }
        // one A read of AR rows x AK K feeds AK/16 K16 steps (half a B word each)
#pragma unroll
        for (int c2 = 0; c2 < GS / AK; ++c2) {
            ushort a[MT_M / AR][AK / 16 * AR];
            for (int i = 0; i < MT_M / AR; ++i)
                A_READ((__global void *)A, K * 2, M, K * 2,
                        (int2)(s * GS + AK * c2, m0 + AR * i), a[i]);
#pragma unroll
            for (int h = 0; h < AK / 16; ++h)
                for (int j = 0; j < NB; ++j) {
                    const int k16 = AK / 16 * c2 + h;
                    const int8 b = dq_half(w[j][k16 / 2], s2[j], k16 % 2);
                    for (int i = 0; i < MB; ++i)
                        acc[i][j] = MAD(as_short8(vload8(0,
                                &a[i / (AR / 8)][h * AR + 8 * (i % (AR / 8))])), b, acc[i][j]);
                }
        }
    }

    for (int i = 0; i < MB; ++i)
        for (int j = 0; j < NB; ++j) {
            // keeps IGC from folding the fp16 conversion into the K loop (spills)
            float8 v = acc[i][j];
            __asm__ volatile("" : "+rw"(v));
            epi_store8x16(C, Other, Bias, M, N, m0 + 8 * i, n0 + 16 * j, v);
        }
}
