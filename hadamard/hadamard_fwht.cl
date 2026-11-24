// Fused sign flip + blockwise normalised Walsh-Hadamard transform:
//   y = H_HB (s * x) / sqrt(HB)   per HB-wide block along K of a [rows, K] DT tensor.
// Input pre-transform of rotated-basis ternary checkpoints (Bonsai 2). HB is a
// power of two in [256, 4096] (-DHB=..., default 1024). One work-group of HB/8
// items per block: the log2(HB) radix-2 stages run as radix-8 passes through SLM
// and, when log2(HB) is not a multiple of 3, a final radix-2 or radix-4 pass that
// stores; fp32 butterflies, DT only at load/store. For HB = 1024 the stage order
// is that of the SYCL kernel, so results match it bit for bit.
//   global = rows * K / HB * HB / 8, local = HB / 8; -DBF16 for bf16 activations.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef HB
#define HB 1024
#endif

#if HB == 256
#define LOG2_HB 8
#define SCALE 0.0625f
#elif HB == 512
#define LOG2_HB 9
#define SCALE 0.044194173824159216f
#elif HB == 1024
#define LOG2_HB 10
#define SCALE 0.03125f
#elif HB == 2048
#define LOG2_HB 11
#define SCALE 0.022097086912079608f
#elif HB == 4096
#define LOG2_HB 12
#define SCALE 0.015625f
#else
#error "HB must be a power of two in [256, 4096]"
#endif

#define WG (HB / 8)
#define NP8 (LOG2_HB / 3)  // radix-8 passes
#define REM (LOG2_HB % 3)  // trailing radix-2 (1) or radix-4 (2) stages

#ifdef BF16
#define LOAD_F(p, i) as_float((uint)(p)[i] << 16)
#define STORE_DT(v) intel_convert_bfloat16_as_ushort(v)
#else
#define LOAD_F(p, i) convert_float(as_half((p)[i]))
#define STORE_DT(v) as_ushort(convert_half(v))
#endif

inline void wht8(float *v) {
#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
        }
    }
}

// -DFWHT_QUANT: also write the int2 x int8 GEMM's activation scales (quant_a, QMODE 1) of the
// rotated rows, SA[g, m] = DT(127 / absmax over the 128-group g of row m), from the stored DT values.
#ifdef FWHT_QUANT
#define GS 128
#define LDSA(M) (((M) + 31) & ~31)
#define EPS 1.1920928955078125e-07f
__attribute__((intel_reqd_sub_group_size(16)))
#endif
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void hadamard_fwht(const __global ushort *x, const __global char *signs,
        __global ushort *y, int K, int use_signs
#ifdef FWHT_QUANT
        , __global ushort *SA, int M
#endif
        ) {
    __local float slm[HB];
    const int lid = get_local_id(0);
    const size_t blk = get_group_id(0);
    const size_t bpr = K / HB;
    const size_t off = (blk / bpr) * K + (blk % bpr) * HB;
    const __global ushort *xb = x + off;
    __global ushort *yb = y + off;
    const __global char *sb = signs + (blk % bpr) * HB;

    // pass p combines 8 elements at stride s = 8^p
    float v[8];
    int s = 1;
    int base = lid * 8;
    for (int j = 0; j < 8; ++j)
        v[j] = use_signs ? LOAD_F(xb, base + j) * (float)sb[base + j] : LOAD_F(xb, base + j);
    wht8(v);
#pragma unroll
    for (int p = 1; p < NP8; ++p) {
        for (int j = 0; j < 8; ++j) slm[base + j * s] = v[j];
        barrier(CLK_LOCAL_MEM_FENCE);
        s <<= 3;
        base = (lid & (s - 1)) | ((lid / s) * (s << 3));
        for (int j = 0; j < 8; ++j) v[j] = slm[base + j * s];
        wht8(v);
    }

#ifdef FWHT_QUANT
#define OUT(i, val) do { const ushort o_ = STORE_DT(val); yb[i] = o_; slm[i] = LOAD_F(&o_, 0); } while (0)
#else
#define OUT(i, val) yb[i] = STORE_DT(val)
#endif
#if REM == 0
    for (int j = 0; j < 8; ++j) OUT(base + j * s, v[j] * SCALE);
#else
    for (int j = 0; j < 8; ++j) slm[base + j * s] = v[j];
    barrier(CLK_LOCAL_MEM_FENCE);
#if REM == 1
    for (int j = 0; j < 4; ++j) {
        const int i = lid + j * WG;
        const float a = slm[i], b = slm[i + HB / 2];
        OUT(i, (a + b) * SCALE);
        OUT(i + HB / 2, (a - b) * SCALE);
    }
#else
    for (int j = 0; j < 2; ++j) {
        const int i = lid + j * WG, q = HB / 4;
        const float a0 = slm[i], a1 = slm[i + q], a2 = slm[i + 2 * q], a3 = slm[i + 3 * q];
        const float t0 = a0 + a1, t1 = a0 - a1, t2 = a2 + a3, t3 = a2 - a3;
        OUT(i, (t0 + t2) * SCALE);
        OUT(i + q, (t1 + t3) * SCALE);
        OUT(i + 2 * q, (t0 - t2) * SCALE);
        OUT(i + 3 * q, (t1 - t3) * SCALE);
    }
#endif
#endif
#ifdef FWHT_QUANT
    // 16 consecutive items (one sub-group) own one contiguous 128-group of the block
    barrier(CLK_LOCAL_MEM_FENCE);
    float mx = 0.0f;
    for (int j = 0; j < 8; ++j) mx = fmax(mx, fabs(slm[lid * 8 + j]));
    mx = sub_group_reduce_max(mx);
    if (get_sub_group_local_id() == 0) {
        const size_t g = (blk % bpr) * (HB / GS) + lid / 16;
        SA[g * LDSA(M) + blk / bpr] = STORE_DT(127.0f / fmax(mx, EPS));
    }
#endif
}
