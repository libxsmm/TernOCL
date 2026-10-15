// MoE expert GEMVs on the int2 x fp16 up-convert kernel (int2_fp16_upcvt.cl):
// one launch covers every (token, expert) pair, the expert index comes from a
// device buffer, and the inner loop is that file's load_b / step (unchanged),
// so the weight decode is the same predicated-select vISA.
//
//   ids : int [G], G = T * TOPK; row g belongs to token g / TOPK
//   B*  : uint [E, K/16, N] per projection, S* : DT [E, K/128, N]
//
//   int2_moe_gate_up: H[g, n] = silu(x_t . gate_e[:, n]) * (x_t . up_e[:, n]),  x_t = X[g / TOPK]
//   int2_moe_down:    Y[g, n] = rw[g] * (H[g] . down_e[:, n])                    (fp32)
//   int2_moe_reduce:  O[t, n] = sum_j Y[t * TOPK + j, n]
//
// Same knobs as the decode GEMV (NSG_N, LS, U); SGM is 1 since every row has its
// own expert. Requires N % 16 == 0 and K % 128 == 0.

#define SGM 1
#include "int2_fp16_upcvt.cl"

#ifndef TOPK
#define TOPK 8
#endif

// sum over this sub-group's K slice of A[0:K] . B[:, n0 + lane]
inline float moe_dot(const __global dt *A, const __global uint *B, const __global dt *S,
        int N, int K, int n0, int s_begin, int s_end) {
    float acc = 0.0f;
    int s = s_begin;
    for (; s + U <= s_end; s += U) {
        uint w[U][8];
        uint sc[U];
        ushort8 ar[U][1];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            load_b(B, N, K, n0, s + u, w[u]);
            sc[u] = intel_sub_group_block_read_us((const __global ushort *)(S + (size_t)(s + u) * N + n0));
            ar[u][0] = intel_sub_group_block_read_us8((const __global ushort *)(A + (s + u) * GS));
        }
#pragma unroll
        for (int u = 0; u < U; ++u) acc = step(acc, w[u], sc[u], ar[u]);
    }
    for (; s < s_end; ++s) {
        uint w[8];
        load_b(B, N, K, n0, s, w);
        const uint sc = intel_sub_group_block_read_us((const __global ushort *)(S + (size_t)s * N + n0));
        ushort8 ar[1];
        ar[0] = intel_sub_group_block_read_us8((const __global ushort *)(A + s * GS));
        acc = step(acc, w, sc, ar);
    }
    return acc;
}

// sub-group sgk > 0 hands its partial to sgk == 0 through SLM; returns false on the others
inline bool moe_reduce_ls(__local float *red, float *acc, int sgk, int sgn, int lane) {
#if LS > 1
    if (sgk > 0) red[((sgk - 1) * NSG_N + sgn) * SG_N + lane] = *acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgk > 0) return false;
    for (int j = 0; j < LS - 1; ++j) *acc += red[(j * NSG_N + sgn) * SG_N + lane];
#endif
    return true;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(SG_N * NSG_N * LS, 1, 1)))
__kernel void int2_moe_gate_up(const __global int *ids, const __global dt *X,
        const __global uint *Bg, const __global dt *Sg, const __global uint *Bu, const __global dt *Su,
        __global dt *H, int N, int K) {
    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int sgn = sg % NSG_N;
    const int sgk = sg / NSG_N;
    const int n0 = (get_group_id(0) * NSG_N + sgn) * SG_N;
    const int g = get_group_id(1);
    const size_t e = (size_t)ids[g];

    const int nsteps = K / GS;
    const int per = (nsteps + LS - 1) / LS;
    const int s_begin = sgk * per;
    const int s_end = min(nsteps, s_begin + per);

    float ag = 0.0f, au = 0.0f;
    if (n0 < N) {
        const __global dt *x = X + (size_t)(g / TOPK) * K;
        ag = moe_dot(x, Bg + e * (K / 16) * N, Sg + e * nsteps * N, N, K, n0, s_begin, s_end);
        au = moe_dot(x, Bu + e * (K / 16) * N, Su + e * nsteps * N, N, K, n0, s_begin, s_end);
    }
#if LS > 1
    __local float red_g[(LS - 1) * NSG_N * SG_N];
    __local float red_u[(LS - 1) * NSG_N * SG_N];
    moe_reduce_ls(red_g, &ag, sgk, sgn, lane);
    if (!moe_reduce_ls(red_u, &au, sgk, sgn, lane)) return;
#endif
    if (n0 >= N) return;
    H[(size_t)g * N + n0 + lane] = TO_DT(ag * sigmoid_x(ag) * au);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(SG_N * NSG_N * LS, 1, 1)))
__kernel void int2_moe_down(const __global int *ids, const __global dt *H, const __global dt *RW,
        const __global uint *B, const __global dt *S, __global float *Y, int N, int K) {
    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int sgn = sg % NSG_N;
    const int sgk = sg / NSG_N;
    const int n0 = (get_group_id(0) * NSG_N + sgn) * SG_N;
    const int g = get_group_id(1);
    const size_t e = (size_t)ids[g];

    const int nsteps = K / GS;
    const int per = (nsteps + LS - 1) / LS;
    const int s_begin = sgk * per;
    const int s_end = min(nsteps, s_begin + per);

    float acc = 0.0f;
    if (n0 < N)
        acc = moe_dot(H + (size_t)g * K, B + e * (K / 16) * N, S + e * nsteps * N, N, K, n0, s_begin, s_end);
#if LS > 1
    __local float red[(LS - 1) * NSG_N * SG_N];
    if (!moe_reduce_ls(red, &acc, sgk, sgn, lane)) return;
#endif
    if (n0 >= N) return;
    Y[(size_t)g * N + n0 + lane] = acc * TOF(RW[g]);
}

__kernel void int2_moe_reduce(const __global float *Y, __global dt *O, int N) {
    const int n = get_global_id(0);
    const int t = get_global_id(1);
    if (n >= N) return;
    float s = 0.0f;
    for (int j = 0; j < TOPK; ++j) s += Y[((size_t)t * TOPK + j) * N + n];
    O[(size_t)t * N + n] = TO_DT(s);
}

// int2_moe_down + int2_moe_reduce in one launch: a work-group owns a column block
// of one token and walks its TOPK experts, O[t, n] = sum_j rw[g] * (H[g] . down_e[:, n])
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(SG_N * NSG_N * LS, 1, 1)))
__kernel void int2_moe_down_sum(const __global int *ids, const __global dt *H, const __global dt *RW,
        const __global uint *B, const __global dt *S, __global dt *O, int N, int K) {
    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int sgn = sg % NSG_N;
    const int sgk = sg / NSG_N;
    const int n0 = (get_group_id(0) * NSG_N + sgn) * SG_N;
    const int t = get_group_id(1);

    const int nsteps = K / GS;
    const int per = (nsteps + LS - 1) / LS;
    const int s_begin = sgk * per;
    const int s_end = min(nsteps, s_begin + per);

    float acc = 0.0f;
    if (n0 < N)
        for (int j = 0; j < TOPK; ++j) {
            const int g = t * TOPK + j;
            const size_t e = (size_t)ids[g];
            acc += TOF(RW[g]) * moe_dot(H + (size_t)g * K, B + e * (K / 16) * N, S + e * nsteps * N,
                    N, K, n0, s_begin, s_end);
        }
#if LS > 1
    __local float red[(LS - 1) * NSG_N * SG_N];
    if (!moe_reduce_ls(red, &acc, sgk, sgn, lane)) return;
#endif
    if (n0 >= N) return;
    O[(size_t)t * N + n0 + lane] = TO_DT(acc);
}
