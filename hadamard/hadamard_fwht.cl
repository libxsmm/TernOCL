// Fused sign flip + blockwise (1024) normalised Walsh-Hadamard transform:
//   y = H_1024 (s * x) / 32   per 1024-wide block along K of a [rows, K] DT tensor.
// Input pre-transform of rotated-basis ternary checkpoints (Bonsai 2). One
// work-group of 128 items per block: the 10 radix-2 stages run as three radix-8
// passes through SLM and a final radix-2 pass that stores; fp32 butterflies,
// DT only at load/store. Same stage order as the SYCL kernel, so results match
// it bit for bit.
//   global = rows * K / 1024 * 128, local = 128; -DBF16 for bf16 activations.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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

__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void hadamard_fwht_1024(const __global ushort *x, const __global char *signs,
        __global ushort *y, int K, int use_signs) {
    __local float slm[1024];
    const int lid = get_local_id(0);
    const size_t blk = get_group_id(0);
    const size_t bpr = K / 1024;
    const size_t off = (blk / bpr) * K + (blk % bpr) * 1024;
    const __global ushort *xb = x + off;
    __global ushort *yb = y + off;
    const __global char *sb = signs + (blk % bpr) * 1024;

    float v[8];
    int base = lid * 8;
    for (int j = 0; j < 8; ++j)
        v[j] = use_signs ? LOAD_F(xb, base + j) * (float)sb[base + j] : LOAD_F(xb, base + j);
    wht8(v);
    for (int j = 0; j < 8; ++j) slm[base + j] = v[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    base = (lid & 7) | ((lid >> 3) << 6);
    for (int j = 0; j < 8; ++j) v[j] = slm[base + (j << 3)];
    wht8(v);
    for (int j = 0; j < 8; ++j) slm[base + (j << 3)] = v[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    base = (lid & 63) | ((lid >> 6) << 9);
    for (int j = 0; j < 8; ++j) v[j] = slm[base + (j << 6)];
    wht8(v);
    for (int j = 0; j < 8; ++j) slm[base + (j << 6)] = v[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    const float scale = 1.0f / 32.0f;
    for (int j = 0; j < 4; ++j) {
        const int i = lid + j * 128;
        const float a = slm[i], b = slm[i + 512];
        yb[i] = STORE_DT((a + b) * scale);
        yb[i + 512] = STORE_DT((a - b) * scale);
    }
}
