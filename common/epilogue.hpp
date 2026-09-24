// Host side of the fused epilogues (see epilogue.clh for the device side).
#pragma once

#include "dt16.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Epilogue {
    int postop = 0;  // 0 none, 1 silu(acc)*other, 2 acc+other, 3 acc+bias[n], 4 sigmoid(acc)
    bool out_f32 = false;  // fp32 C and bias
    bool needs_other() const { return postop == 1 || postop == 2; }
    bool needs_bias() const { return postop == 3; }
    size_t out_size() const { return out_f32 ? 4 : 2; }
    std::string opts() const {
        return " -DPOSTOP=" + std::to_string(postop) + (out_f32 ? " -DOUT_F32" : "");
    }
    std::string name() const {
        static const char *n[] = {"none", "silu(acc)*other", "acc+other", "acc+bias", "sigmoid(acc)"};
        return std::string(n[postop]) + (out_f32 ? ", fp32 out" : "");
    }
    // sigmoid output is in (0, 1): compare it on that scale, not with abs_tol = 8
    double abs_tol() const {
        if (postop == 4) return dt_is_bf16() && !out_f32 ? 0.03 : 0.004;
        return dt_abs_tol();
    }
    uint32_t ulp_tol() const { return postop == 4 ? 4u : dt_ulp_tol(); }
};

inline float sigmoid_ref(float x) { return x <= -10.0f ? 0.0f : 1.0f / (std::exp(-x) + 1.0f); }  // xetla_sigmoid

inline void fill_epilogue_inputs(const Epilogue &e, std::vector<dt16> &other,
        std::vector<unsigned char> &bias, int M, int N, std::mt19937 &gen) {
    std::uniform_real_distribution<float> uo(-4.0f, 4.0f), ub(-8.0f, 8.0f);
    other.assign(e.needs_other() ? (size_t)M * N : 0, 0);
    for (auto &o : other) o = fromf(uo(gen));
    bias.assign(e.needs_bias() ? (size_t)N * e.out_size() : 0, 0);
    for (int n = 0; n < (int)(bias.size() / e.out_size()); ++n) {
        const float b = ub(gen);
        if (e.out_f32) std::memcpy(&bias[4 * (size_t)n], &b, 4);
        else { const dt16 h = fromf(b); std::memcpy(&bias[2 * (size_t)n], &h, 2); }
    }
}

// Epilogue on the fp32 gold accumulator, then cast to the output type.
inline void epilogue_ref(const Epilogue &e, const float *acc, const dt16 *other,
        const unsigned char *bias, unsigned char *out, int M, int N) {
#pragma omp parallel for
    for (long i = 0; i < (long)M * N; ++i) {
        float v = acc[i];
        const int n = (int)(i % N);
        if (e.postop == 1) v = v * sigmoid_ref(v) * tof(other[i]);
        else if (e.postop == 2) v += tof(other[i]);
        else if (e.postop == 3) {
            float b;
            if (e.out_f32) std::memcpy(&b, bias + 4 * (size_t)n, 4);
            else { dt16 h; std::memcpy(&h, bias + 2 * (size_t)n, 2); b = tof(h); }
            v += b;
        } else if (e.postop == 4) v = sigmoid_ref(v);
        if (e.out_f32) std::memcpy(out + 4 * (size_t)i, &v, 4);
        else { const dt16 h = fromf(v); std::memcpy(out + 2 * (size_t)i, &h, 2); }
    }
}

// fp32 outputs: |d| <= 0.05 || rel <= 2e-4 (only fp32 reduction order differs)
inline bool compare_f32(const float *C, const float *G, size_t n, const std::string &label) {
    size_t bad = 0;
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)C[i] - G[i]), r = d / std::max(1e-30, (double)std::fabs(G[i]));
        max_abs = std::max(max_abs, std::isnan(d) ? INFINITY : d);
        max_rel = std::max(max_rel, std::isnan(r) ? INFINITY : r);
        if (!(d <= 0.05 || r <= 2e-4)) {
            if (++bad <= 10) std::cout << "\tidx " << i << " data " << C[i] << " gold " << G[i] << "\n";
        }
    }
    std::cout << label << ": max abs diff " << max_abs << ", max rel diff " << max_rel
              << ", pass rate " << 100.0 * (n - bad) / n << "%\n";
    return bad == 0;
}
