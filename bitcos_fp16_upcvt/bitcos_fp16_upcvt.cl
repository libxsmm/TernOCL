// OpenCL port of the xetla BITCOS ternary GEMM/GEMV with 16-bit activations
// (xetla/include/experimental/group/gemm/impl/bitcos_fp16_upcvt_xmx_xe.hpp as
// built by the vLLM plugin: BITCOS_FP16_LUT, BITCOS_SIGN_GATHER4/3):
//
//   C[m,n] = DT( sum_k A[m,k] * code(k,n) * S[k/128, n] ),  code in {-1, 0, +1}
//
//   A : DT [M, K] row-major, S : DT [K/128, N], C : ct [M, N] (epilogue.clh)
//   B : uint32 buffer, three planes back to back (common/bitcos.hpp):
//       bitmap  [K/32][N]  bit c of word (kp, n) = weight(kp*32+c, n) != 0
//       offsets [N]        first sign word of column n
//       signs   bitstream  one bit per non-zero, column-major in k, 1 -> -1
//   SR: slice_ranks [LS-1][N], non-zeros of column n above slice boundary s*K/LS
//
// Unpack, per lane (= output column) and 64-k step, as xetla:
//   - one 2D block read gives the two 32-row bitmap words;
//   - the column's rank (its cursor into the sign stream) advances by one
//     popcount per 32-row block, and one 3-dword gather at word rank/32 covers
//     the sign windows of both blocks (a block consumes at most 32 bits);
//   - each block is decoded as 8 four-row groups: the presence nibble and the
//     next four sign bits index a 256-entry SLM table of four DT codes {0, +1,
//     -1} in VNNI order (a 4-bit pdep), the window then shifts by
//     popcount(nibble), and the codes are scaled. fp16 (default): xetla's
//     two SIMD32 hf multiplies on the looked-up dwords, as inline vISA
//     (VISA_HMUL; SIMT_HMUL = plain OpenCL half4 * half, which IGC de- and
//     re-interleaves, 369 vs 213 loop instructions). INT_APPLY (bf16, which
//     has no 16-bit multiply; optional for fp16, bit-identical): the table
//     holds 0 / 0x7fff / 0xffff masks and one AND with (scale | 0x8000) makes
//     0 / +s / -s exactly;
//   - two DPAS of k = 16 per block, fp32 accumulate.
// Work decomposition: SGM rows per sub-group (DPAS repeat count), NSG_N
// sub-groups along N, LS sub-groups splitting K (reduced through SLM; slice s
// starts at the column's rank SR[s-1]). Requires N % 16 == 0 and
// K % (64 * LS) == 0.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef BF16
#define MAD intel_sub_group_bf16_bf16_matrix_mad_k16
#define TOF(h) as_float((uint)(h) << 16)
#define TOF8(h8) as_float8(convert_uint8(h8) << 16)
#define TO_DT(x) intel_convert_bfloat16_as_ushort(x)
#define TO_DT8(x) intel_convert_bfloat168_as_ushort8(x)
#ifndef INT_APPLY
#define INT_APPLY
#endif
#else
#define MAD intel_sub_group_f16_f16_matrix_mad_k16
#define TOF(h) convert_float(as_half(h))
#define TOF8(h8) convert_float8(as_half8(h8))
#define TO_DT(x) as_ushort(convert_half(x))
#define TO_DT8(x) as_ushort8(convert_half8(x))
#if !defined(INT_APPLY) && !defined(SIMT_HMUL) && !defined(VISA_HMUL)
#define VISA_HMUL
#endif
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
#define GS 128
#define STEP 64

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

#ifdef INT_APPLY
#define CODE_POS 0x7fffu
#define CODE_NEG 0xffffu
#else
#define CODE_POS 0x3c00u  // fp16 +1
#define CODE_NEG 0xbc00u  // fp16 -1
#endif

#ifndef NO_SIGN_CACHE_HINT
// IGC's LSC load with an explicit cache control: XeTLA gathers the sign words
// L1-cached / L3-cached (load.ugm.d32x3.ca.ca); a column's next steps re-read them.
enum LSC_LDCC { LSC_LDCC_DEFAULT = 0, LSC_LDCC_L1UC_L3UC = 1, LSC_LDCC_L1UC_L3C = 2,
    LSC_LDCC_L1C_L3UC = 3, LSC_LDCC_L1C_L3C = 4 };
uint3 __builtin_IB_lsc_load_global_uint3(const __global uint3 *base, int imm_elem_off,
        enum LSC_LDCC cache);
#define LOAD_SIGNS(p) __builtin_IB_lsc_load_global_uint3((const __global uint3 *)(p), 0, LSC_LDCC_L1C_L3C)
#else
#define LOAD_SIGNS(p) vload3(0, (p))
#endif

// Entry (m, s), m = presence nibble of rows 4g..4g+3, s = next four compact
// sign bits: row i is 0 if absent, else the sign bit of rank sum(m_j, j < i).
inline void build_lut(__local uint2 *lut) {
    const int n = get_local_size(0);
    for (int idx = get_local_id(0); idx < 256; idx += n) {
        const uint m = (uint)idx >> 4, s = (uint)idx & 15u;
        uint c[4];
        uint r = 0;
        for (int i = 0; i < 4; ++i) {
            const uint p = (m >> i) & 1u;
            c[i] = p ? (((s >> r) & 1u) ? CODE_NEG : CODE_POS) : 0u;
            r += p;
        }
        lut[idx] = (uint2)(c[0] | (c[1] << 16), c[2] | (c[3] << 16));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Sign window at bit `rank` from the gathered words q (q.x holds word w0 =
// rank0/32 of this step; rank/32 - w0 is 0 or 1).
inline uint sign_window(uint3 q, uint w0, uint rank) {
    const uint m = (rank >> 5) - w0;
    const uint lo = m ? q.y : q.x, hi = m ? q.z : q.y;
    const uint sh = rank & 31u;
    return (lo >> sh) | ((hi << ((32u - sh) & 31u)) & (sh ? 0xffffffffu : 0u));
}

// One block: bitmap word bmp, sign window w -> two k16 VNNI B operands.
#ifdef VISA_HMUL
// xetla's scale apply: the two looked-up dwords per lane (4 interleaved hf codes)
// times the lane's scale replicated in both halves, as two SIMD32 hf muls in place
// (SIMT half4 * half would de-interleave into 4 SIMD16 muls and re-interleave).
inline uint2 hmul2(uint2 e, uint s2) {
    uint2 r;
    __asm__("{\n"
        ".decl EH v_type=G type=hf num_elts=64 align=GRF alias=<%1,0>\n"
        ".decl RH v_type=G type=hf num_elts=64 align=GRF alias=<%0,0>\n"
        ".decl SH v_type=G type=hf num_elts=32 align=GRF alias=<%2,0>\n"
        "mul (M1_NM, 32) RH(0,0)<1> EH(0,0)<1;1,0> SH(0,0)<1;1,0>\n"
        "mul (M1_NM, 32) RH(1,0)<1> EH(1,0)<1;1,0> SH(0,0)<1;1,0>\n"
        "}\n" : "=rw"(r) : "rw"(e), "rw"(s2));
    return r;
}
#endif

inline void unpack_block(__local const uint2 *lut, uint bmp, uint w, uint sx, int8 *b0, int8 *b1) {
    uint d[16];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        const uint m4 = (bmp >> (4 * g)) & 15u;
        const uint2 e = lut[(m4 << 4) | (w & 15u)];
        w >>= popcount(m4);
#ifdef INT_APPLY
        d[2 * g] = e.x & sx;
        d[2 * g + 1] = e.y & sx;
#elif defined(VISA_HMUL)
        const uint2 v = hmul2(e, sx);
        d[2 * g] = v.x;
        d[2 * g + 1] = v.y;
#else
        const half4 v = as_half4(e) * as_half2(sx).x;
        d[2 * g] = as_uint2(v).x;
        d[2 * g + 1] = as_uint2(v).y;
#endif
    }
    *b0 = as_int8((uint8)(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]));
    *b1 = as_int8((uint8)(d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]));
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16 * NSG_N * LS, 1, 1)))
__kernel void bitcos_fp16_upcvt_gemv(const __global dt *A, const __global uint *B,
        const __global dt *S, const __global uint *SR, __global ct *C, EPI_ARGS,
        int M, int N, int K) {
    __local uint2 lut[256];
    build_lut(lut);

    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int sgn = sg % NSG_N;
    const int sgk = sg / NSG_N;
    const int n0 = (get_group_id(0) * NSG_N + sgn) * 16;
    const int m0 = get_group_id(1) * SGM;
    const int kslice = K / LS;
    const int k_begin = sgk * kslice, k_end = k_begin + kslice;

    const size_t bw = (size_t)(K / 32) * N;
    const __global uint *signs = B + bw + N;
    const uint off = n0 < N ? B[bw + n0 + lane] : 0u;
    uint rank = (LS > 1 && sgk > 0 && n0 < N) ? SR[(size_t)(sgk - 1) * N + n0 + lane] : 0u;

    acc_t acc = 0.0f;
    if (n0 < N) {
        for (int k = k_begin; k < k_end; k += STEP) {
            uint bm[2];
            intel_sub_group_2d_block_read_32b_2r16x1c((__global void *)B, N * 4, K / 32, N * 4,
                    (int2)(n0, k / 32), bm);
            const uint sc = intel_sub_group_block_read_us(
                    (const __global ushort *)(S + (size_t)(k / GS) * N + n0));
            ushort4 ar[SGM];
            for (int r = 0; r < SGM; ++r)
                ar[r] = (SGM == 1 || m0 + r < M)
                        ? intel_sub_group_block_read_us4(
                                  (const __global ushort *)(A + (size_t)(m0 + r) * K + k))
                        : (ushort4)0;
            const uint r0 = rank, r1 = r0 + popcount(bm[0]);
            rank = r1 + popcount(bm[1]);
            const uint w0 = r0 >> 5;
            const uint3 q = LOAD_SIGNS(signs + off + w0);
#ifdef INT_APPLY
            const uint sx = (sc | 0x8000u) * 0x10001u;
#elif defined(VISA_HMUL)
            const uint sx = sc * 0x10001u;
#else
            const uint sx = sc;
#endif
#pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                int8 b0, b1;
                unpack_block(lut, bm[ii], sign_window(q, w0, ii ? r1 : r0), sx, &b0, &b1);
                a_t a0, a1;
#if SGM == 1
                a0 = as_short(ar[0][2 * ii]);
                a1 = as_short(ar[0][2 * ii + 1]);
#else
                for (int r = 0; r < SGM; ++r) {
                    a0[r] = as_short(ar[r][2 * ii]);
                    a1[r] = as_short(ar[r][2 * ii + 1]);
                }
#endif
                acc = MAD(a0, b0, acc);
                acc = MAD(a1, b1, acc);
            }
        }
    }

#if LS > 1
    __local float red[(LS - 1) * NSG_N * SGM * 16];
    if (sgk > 0) {
        __local float *dst = red + (((sgk - 1) * NSG_N + sgn) * SGM) * 16;
#if SGM == 1
        dst[lane] = acc;
#else
        for (int r = 0; r < SGM; ++r) dst[r * 16 + lane] = acc[r];
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgk > 0) return;
    for (int j = 0; j < LS - 1; ++j) {
        const __local float *src = red + ((j * NSG_N + sgn) * SGM) * 16;
#if SGM == 1
        acc += src[lane];
#else
        for (int r = 0; r < SGM; ++r) acc[r] += src[r * 16 + lane];
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
