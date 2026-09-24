// Host side of the 16-bit activation/scale/output dtype (fp16 or bf16),
// stored as raw bits like the device buffers. Conversions round to nearest
// even, matching convert_half() / intel_convert_bfloat16_as_ushort().
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

using dt16 = uint16_t;

inline bool &dt_is_bf16() {
    static bool bf16 = false;
    return bf16;
}
inline const char *dt_name() { return dt_is_bf16() ? "bf16" : "fp16"; }

inline float tof(dt16 h) {
    if (dt_is_bf16()) {
        const uint32_t u = (uint32_t)h << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
    _Float16 x;
    std::memcpy(&x, &h, 2);
    return (float)x;
}

inline dt16 fromf(float f) {
    if (dt_is_bf16()) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        if ((u & 0x7fffffffu) > 0x7f800000u) return (dt16)((u >> 16) | 0x40);  // NaN
        u += 0x7fffu + ((u >> 16) & 1u);
        return (dt16)(u >> 16);
    }
    const _Float16 x = (_Float16)f;
    dt16 h;
    std::memcpy(&h, &x, 2);
    return h;
}

// Pass rule, per element: |d| <= abs_tol || ulp <= ulp_tol || rel <= 1e-3.
// fp16 as xetla_buff_cmp(ulp_tol=64, abs_tol=8); bf16 as the xetla bf16
// harness (ulp_tol=32, abs_tol=16) since bf16 keeps 8 mantissa bits.
inline double dt_abs_tol() { return dt_is_bf16() ? 16.0 : 8.0; }
inline uint32_t dt_ulp_tol() { return dt_is_bf16() ? 32u : 64u; }
