// Host side of the BITCOS ternary layout (xetla_vllm_plugin.pack_bitcos):
// one uint32 buffer holding three planes back to back,
//   [0, K*N/32)          bitmap  : bit c of word kp*N+n = weight(kp*32+c, n) != 0
//   [K*N/32, +N)         offsets : first sign word of column n
//   [K*N/32+N, ...)      signs   : one bit per non-zero, column-major in k, 1 -> -1
// plus pad words (the kernels' sign window gathers up to two words past a run),
// and slice_ranks [LS-1, N]: non-zeros of column n above each local-K-slice
// boundary s*K/LS, needed by k-sliced kernels to enter the sign stream.
#pragma once

#include <omp.h>

#include <cstdint>
#include <random>
#include <vector>

struct Bitcos {
    int K = 0, N = 0;
    std::vector<uint32_t> buf;          // bitmap | offsets | signs | pad
    std::vector<uint32_t> slice_ranks;  // [(LS-1) * N]
    size_t bitmap_words() const { return (size_t)K / 32 * N; }
    size_t offsets_base() const { return bitmap_words(); }
    size_t signs_base() const { return bitmap_words() + N; }
    size_t bytes() const { return buf.size() * 4; }
};

inline constexpr int kBitcosPad = 4;

// Random ternary weights at zero density z (P(0) = z, P(+1) = P(-1) = (1-z)/2),
// packed directly (no [K, N] code matrix).
inline Bitcos bitcos_random(int K, int N, double z, uint64_t seed) {
    Bitcos b;
    b.K = K;
    b.N = N;
    const size_t bw = b.bitmap_words();
    std::vector<uint32_t> bmp(bw);
    const uint32_t thr = (uint32_t)((1.0 - z) * 4294967296.0 > 4294967295.0 ? 4294967295.0
                                                                            : (1.0 - z) * 4294967296.0);
#pragma omp parallel
    {
        std::mt19937 g((uint32_t)(seed * 7919u + omp_get_thread_num()));
#pragma omp for schedule(static)
        for (long i = 0; i < (long)bw; ++i) {
            uint32_t w = 0;
            for (int c = 0; c < 32; ++c) w |= (uint32_t)(g() < thr) << c;
            bmp[i] = w;
        }
    }
    std::vector<uint32_t> nnz(N, 0);
#pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        uint32_t s = 0;
        for (int kp = 0; kp < K / 32; ++kp) s += __builtin_popcount(bmp[(size_t)kp * N + n]);
        nnz[n] = s;
    }
    std::vector<uint32_t> off(N);
    size_t words = 0;
    for (int n = 0; n < N; ++n) { off[n] = (uint32_t)words; words += (nnz[n] + 31) / 32; }
    b.buf.assign(bw + N + words + kBitcosPad, 0);
    std::copy(bmp.begin(), bmp.end(), b.buf.begin());
    std::copy(off.begin(), off.end(), b.buf.begin() + bw);
    uint32_t *sg = b.buf.data() + b.signs_base();
#pragma omp parallel
    {
        std::mt19937 g((uint32_t)(seed * 104729u + 17u + omp_get_thread_num()));
#pragma omp for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint32_t nw = (nnz[n] + 31) / 32;
            for (uint32_t w = 0; w < nw; ++w) {
                uint32_t v = g();
                const uint32_t bits = std::min<uint32_t>(32, nnz[n] - 32 * w);
                if (bits < 32) v &= (1u << bits) - 1u;  // pack_bitcos leaves the tail zero
                sg[off[n] + w] = v;
            }
        }
    }
    return b;
}

// slice_ranks for LS local K slices of K/LS rows each (xetla: ceil(K/LS)).
inline std::vector<uint32_t> bitcos_slice_ranks(const Bitcos &b, int ls) {
    std::vector<uint32_t> r((size_t)(ls > 1 ? ls - 1 : 0) * b.N);
    const int slice_k = (b.K + ls - 1) / ls;
#pragma omp parallel for
    for (int n = 0; n < b.N; ++n) {
        uint32_t s = 0;
        int kp = 0;
        for (int sl = 1; sl < ls; ++sl) {
            for (; kp < sl * slice_k / 32; ++kp) s += __builtin_popcount(b.buf[(size_t)kp * b.N + n]);
            r[(size_t)(sl - 1) * b.N + n] = s;
        }
    }
    return r;
}

// Decode column n into codes[k] in {-1, 0, +1}.
inline void bitcos_decode_col(const Bitcos &b, int n, int8_t *codes) {
    const uint32_t *sg = b.buf.data() + b.signs_base() + b.buf[b.offsets_base() + n];
    uint32_t rank = 0;
    for (int k = 0; k < b.K; ++k) {
        const bool p = (b.buf[(size_t)(k / 32) * b.N + n] >> (k % 32)) & 1u;
        if (!p) { codes[k] = 0; continue; }
        codes[k] = ((sg[rank / 32] >> (rank % 32)) & 1u) ? -1 : 1;
        ++rank;
    }
}

// Measured zero density of the bitmap.
inline double bitcos_zero_density(const Bitcos &b) {
    uint64_t s = 0;
#pragma omp parallel for reduction(+ : s)
    for (long i = 0; i < (long)b.bitmap_words(); ++i) s += __builtin_popcount(b.buf[i]);
    return 1.0 - (double)s / ((double)b.K * b.N);
}
