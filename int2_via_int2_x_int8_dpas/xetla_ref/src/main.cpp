/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <chrono>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <random>
#include "../tests/unit/tile_row_reduction/kernel_func.hpp"
#include "utils/buff_compare.hpp"
#include "utils/profiling.hpp"
#include "xetla.hpp"
#include "variant_common.hpp"

// Templated kernel-name to avoid duplicate KernelInfo specializations and
// to mirror the GEMM kernel naming methodology used elsewhere in this TU.
template <typename KernelTag, int TileW>
class compute_scale_kernel;

// Kernel wrapper structs to avoid mangled name collisions with different post-op configurations
template<typename T, bool EnableBias, bool EnableSiLU, int TileSize> 
struct compute_scale_kernel_wrapper {};

template<typename T, bool EnableBias, bool EnableSiLU> 
struct gemm_kernel_wrapper {};

using namespace gpu::xetla;

// Scale element type for both A and B scales. This variant of the int2 fp16
// DPAS test stores both scale tensors in fp16 (the original test stores them
// in fp32). The kernel-side compute is still promoted to float internally.
using scale_dtype = fp16;

// Per-row-per-group absmax reduction that writes the FINAL scale value
// (127 / absmax) directly into the device buffer with arbitrary output dtype.
// This is the fp16-scales analogue of `absmax_row_reduction` from
// tests/unit/tile_row_reduction/kernel_func.hpp; that helper writes raw
// absmax as float, which is incompatible with both (a) the fp16 storage
// requested by this test and (b) the GEMM's expectation that scaleA already
// equals 127/absmax (the GEMM multiplies activations by scaleA and saturates
// to int8, so scaleA must be the forward-quant factor, not the inverse).
template <typename in_dtype, typename out_dtype, int twidth, int theight>
struct absmax_to_inv_scale_reduction {
    static KERNEL_FUNC inline void run(sycl::nd_item<1> it, in_dtype *a_base, int swidth, int sheight, int spitch, int scale_bs, out_dtype *c_base) {
        const int group_id = static_cast<int>(it.get_group(0));
        const int local_id = static_cast<int>(it.get_local_id(0));
        const int total_rows = sheight;
        const int total_cols = swidth;
        const int tiles_per_row = ((total_cols + twidth - 1) / twidth) / scale_bs;
        const int group_base = group_id * theight;
        const int row = group_base + local_id;
        if (row >= total_rows) return;

        using row_tile_desc_t = subgroup::tile_desc_t<twidth, 1, twidth, 1, reg_layout::tiled>;
        using row_tile_t = subgroup::tile_t<in_dtype, row_tile_desc_t>;
        using row_payload_t = subgroup::mem_payload_t<mem_desc_t<in_dtype, mem_layout::row_major, mem_space::global>, row_tile_desc_t, msg_type_v<row_tile_desc_t, mem_space::global>, gpu_arch::Xe>;
        using row_prefetch_payload_t = subgroup::prefetch_payload_t<mem_desc_t<in_dtype, mem_layout::row_major, mem_space::global>, row_tile_desc_t, twidth, gpu_arch::Xe>;

        row_tile_t row_tile;
        row_payload_t row_payload(a_base + row * spitch, swidth, 1, spitch, 0, 0);
        row_prefetch_payload_t row_payload_prefetch(a_base + row * spitch, swidth, 1, spitch, 0, 0);

        xetla_vector<float, twidth> src_reg_f32;
        xetla_mask<row_tile_t::tile_size_x> mask_max;

        for (int igs = 0; igs < scale_bs; igs++) {
            xetla_vector<float, twidth> acc_vec = FLT_EPSILON;
#pragma unroll
            for (int t = 0; t < tiles_per_row; ++t) {
                subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(row_payload_prefetch);
                row_payload_prefetch.template update_tdesc<tdesc_update_dir::x_dir>(row_tile_t::tile_size_x);
            }

#pragma unroll
            for (int t = 0; t < tiles_per_row; ++t) {
                subgroup::tile_load<cache_hint::cached, cache_hint::cached>(row_tile, row_payload);
                row_payload.template update_tdesc<tdesc_update_dir::x_dir>(row_tile_t::tile_size_x);
                src_reg_f32 = xetla_abs<float, row_tile_t::tile_size_x>(xetla_cvt<float, in_dtype, row_tile_t::tile_size_x>(row_tile.reg.xetla_select<row_tile_t::tile_size_x, 1>(0)));
                mask_max = src_reg_f32 > acc_vec;
                acc_vec.xetla_merge(src_reg_f32, acc_vec, mask_max);
            }

            float row_max = xetla_reduce<float, float, twidth, reduce_op::max>(acc_vec);
            // Store 127/absmax cast to the output dtype (fp16 in this test).
            // This matches the host-side formula and the GEMM's expectation.
            float scale_val = 127.0f / row_max;
            c_base[igs * sheight + row] = static_cast<out_dtype>(scale_val);
        }
    }
};

// Add the missing get_time function
double get_time() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    return millis * 1e-9; // Convert to seconds
}

// Add missing random number generation functions
uint32_t random_uint32() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);
    return dis(gen);
}

// Global variables for runtime configuration
bool g_enable_validation = true;
int g_num_iters = 20; // Number of times the kernel is executed (default: 20)
size_t g_mat_m = 1; // Default matrix M dimension
size_t g_mat_n = 8192; // Default matrix N dimension
size_t g_mat_k = 8192; // Default matrix K dimension
uint32_t g_global_kslicing = 1; // Default global K-slicing
// Runtime override for scale group size (0 = use Test::scale_gs)
uint32_t g_scale_gs = 0;

// Runtime tile-size selection (no recompilation required). Defaults chosen to match precompiled variants.
uint32_t g_wg_m = 128;
uint32_t g_sg_m = 8;
uint32_t g_wg_n = 128;
uint32_t g_sg_n = 128;
uint32_t g_sg_k = 32;  // sg_k selection for GEMVs: 32, 128, or 160

// Post-op fusion flags
bool g_enable_bias = false;
bool g_enable_silu = false;

// Scale A calculation mode: 1 = external (separate kernel), 0 = internal (SLM within GEMM)
uint32_t g_external_scale_a_calc = 1;

// Template base class for int2_dequant with configurable global_kslicing, tile sizes, and data types
// int2_dequant_base class template is now defined in variant_common.hpp
// All variants are precompiled in precompiled_gemm_*.cpp files based on variant_common.hpp configuration

// bitcos_paper/opencl_int2_int8dpas copy: rotating sets sized like the OpenCL
// driver -- distinct weights (B uint32 words + SB fp16) >= g_weights_gib GiB,
// or g_sets if given. Everything else is the original harness.
double g_weights_gib = 2.0;
int g_sets = 0;
int calculate_n_sets_runtime() {
    if (g_sets > 0) return g_sets;
    const double w = (double)g_mat_k * g_mat_n / 16 * 4 + (double)(g_mat_k / 128) * g_mat_n * 2;
    return std::max(1, (int)std::ceil(g_weights_gib * (1ULL << 30) / w));
}

// Legacy template function for compile-time calculation (kept for compatibility)
template <class Test>
constexpr int calculate_n_sets() {
    constexpr size_t target_memory_gb = 2; // Target 2GB
    constexpr size_t bytes_per_gb = 1024ULL * 1024ULL * 1024ULL;
    constexpr size_t target_memory_bytes = target_memory_gb * bytes_per_gb;

    // Calculate memory per matrix set
    constexpr size_t matrix_m = 1; // Default fallback
    constexpr size_t matrix_n = 8192; // Default fallback
    constexpr size_t matrix_k = 8192; // Default fallback

    constexpr size_t size_a = matrix_m * matrix_k * sizeof(typename Test::data_type_a); // int8 = 1 byte
    constexpr size_t size_b = matrix_k * matrix_n / 16 * sizeof(typename Test::data_type_b); // int2x16 = 2 bytes (16 values packed)
    constexpr size_t size_c = matrix_m * matrix_n * sizeof(typename Test::data_type_c); // int32 = 4 bytes
    constexpr size_t size_acc = matrix_m * matrix_n * sizeof(int32_t); // 4 bytes
    constexpr size_t size_cnt = matrix_m * matrix_n * sizeof(uint32_t); // 4 bytes

    constexpr size_t memory_per_set = (size_a + size_b + size_c + size_acc + size_cnt);

    // Calculate number of sets to reach target memory
    constexpr int n_sets = static_cast<int>((target_memory_bytes + memory_per_set - 1) / memory_per_set);

    return n_sets > 0 ? n_sets : 1; // Ensure at least 1 set
}

//The number of matrix sets to benchmark (will be recalculated based on command line matrix dimensions)
int N_SETS = 1; // Default value, will be updated in main() via calculate_n_sets_runtime()

template <typename data_type_a, typename data_type_b, typename data_type_c, typename data_type_acc = int32_t>
int gemm_result_validate(data_type_a *A, data_type_b *B, data_type_c *C, uint32_t m, uint32_t k, uint32_t n, sycl::context &context, uint32_t scale_gs, const scale_dtype *scaleA, size_t size_scale_a, const scale_dtype *scaleB, size_t size_scale_b,
        const data_type_c *bias = nullptr, bool apply_silu = false,
        mem_layout mem_layout_a_ = mem_layout::row_major, mem_layout mem_layout_b_ = mem_layout::row_major) {
    buff_cmp::buff_vals<data_type_c> data(C, m, n, n);

    // compute mapping parameters for scale tensors
    const size_t ks_groups = scale_gs;
    const size_t expected_size_scale_a = std::max<size_t>(1, m * ks_groups);
    const size_t expected_size_scale_b = std::max<size_t>(1, n * ks_groups);

    // Compute gold reference in float and then convert to data_type_c (may be fp16)
    float *gold_f = static_cast<float *>(malloc_host(m * n * sizeof(float), context));
    int8_t *quant_A = static_cast<int8_t *>(malloc_host(m * k * sizeof(int8_t), context));

    // Let's quantize A using the scaleA tensor.
    // Match the kernel: round-toward-zero (truncation) + saturating cast to int8.
    // ESIMD saturate<int8_t>(float) uses RTZ, not RTE, so we MUST truncate.
    for (int i = 0; i < m; i++) {
        for (int gk = 0; gk < ks_groups; gk++) {
            for (int _ik = 0; _ik < (k / scale_gs); _ik++) {
                int ik = gk * (k / scale_gs) + _ik;
                float cur_scale_a = static_cast<float>(scaleA[gk * m + i]);
                float prod_f = static_cast<float>(A[i * k + ik]) * cur_scale_a;
                long q_l = static_cast<long>(prod_f); // truncate (RTZ)
                if (q_l >  127) q_l =  127;
                if (q_l < -128) q_l = -128;
                quant_A[i * k + ik] = static_cast<int8_t>(q_l);
            }
        }
    }

// B is in vnni-16
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float accum = 0.0f;
            for (int gk = 0; gk < ks_groups; gk++) {
                int32_t acc_int = 0;
                float cur_scale_a = frcp(static_cast<float>(scaleA[gk * m + i]));
                float cur_scale_b = static_cast<float>(scaleB[gk * n + j]);
                float prod = cur_scale_b * cur_scale_a;
                for (int _ik = 0; _ik < (k / scale_gs); _ik++) {
                    int ik = gk * (k / scale_gs) + _ik;
                    acc_int += static_cast<int8_t>(quant_A[i * k + ik]) * static_cast<int8_t>(B[(ik / 16) * n * 16 + j * 16 + (ik % 16)]);
                }
                accum += static_cast<float>(acc_int) * prod;
            }
            gold_f[i * n + j] = accum;
        }
    }

    // Apply bias if provided
    if (bias != nullptr) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                gold_f[i * n + j] += static_cast<float>(bias[j]);
            }
        }
    }

    // Apply SiLU if requested: x * sigmoid(x)
    if (apply_silu) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float x = gold_f[i * n + j];
                float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
                gold_f[i * n + j] = x * sigmoid_x;
            }
        }
    }

    // Convert float gold into data_type_c buffer for comparison
    data_type_c *gold_C = static_cast<data_type_c *>(malloc_host(m * n * sizeof(data_type_c), context));
    for (size_t idx = 0; idx < static_cast<size_t>(m) * static_cast<size_t>(n); ++idx) {
        gold_C[idx] = static_cast<data_type_c>(gold_f[idx]);
    }
    free(gold_f, context);
    free(quant_A, context);

    buff_cmp::buff_vals<data_type_c, data_type_c> other(gold_C, m, n, n);

    // fp16 GEMM tolerances. With activations in [-5, 5] and ternary weights in
    // {-1, 0, 1}, partial products can cancel across the K-dimension, so noise
    // sources are: (a) fp32 reduction order across the 32 K-groups, and
    // (b) the final fp32 -> fp16 cast. For matrix_k = 4096 and outputs in the
    // O(1e3..1e4) range, 1 fp16 ULP can be up to ~4 in absolute terms; for
    // outputs near zero, catastrophic cancellation makes ULP/relative diffs
    // large but still bounded in absolute terms by the per-K-group fp32 ULP
    // accumulated 32 times.
    //
    // The comparator passes a sample if ANY of {abs_diff <= abs_tol,
    // ulp_diff <= ulp_tol, rel_diff <= 0.01} holds. Setting:
    //   - abs_tol = 8.0  (covers 2 fp16 ULPs at magnitude ~4000)
    //   - ulp_tol = 64   (covers cancellation cases with fp16 magnitude > ~10)
    //   - rel_diff (1%)  built into comparator
    bool result = buff_cmp::xetla_buff_cmp(data, other, "int2 gemm validation",
            /*diff_elems_tol*/ 0.001, /*ulp_tol*/ 64, /*abs_tol*/ 8.0);

    std::cout << (!result ? "FAILED\n" : "PASSED\n");

#if 0
    // Print all values of gold_C and C
    std::cout << "Computed C VS gold C\n";
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            std::cout << C[i * n + j] << " " << gold_C[i * n + j] << " ";
            std::cout << "\n";
        }
    }
#endif

    // Clean up dynamically allocated memory
    free(gold_C, context);

    return result ? 0 : 1;
}

template <class Test, bool EnableBias = false, bool EnableSiLU = false>
void int2_dequantize_gemm_run(int iter, int n_sets, bool enable_validation) {
    using namespace gpu;
    //Accept incoming parameters
    const size_t matrix_m = Test::mat_m();
    const size_t matrix_n = Test::mat_n();
    const size_t matrix_k = Test::mat_k();
    constexpr uint32_t global_kslicing = Test::global_kslicing;
    constexpr uint32_t local_kslicing = Test::local_kslicing;
    // runtime-selectable scale group size: if g_scale_gs != 0 use it, else fall back to Test::scale_gs
    uint32_t scale_gs = g_scale_gs ? g_scale_gs : Test::scale_gs;

    constexpr size_t wg_tile_m = Test::wg_m;
    constexpr size_t wg_tile_n = Test::wg_n;
    constexpr size_t sg_tile_m = Test::sg_m;
    constexpr size_t sg_tile_n = Test::sg_n;
    constexpr size_t sg_tile_k = Test::sg_k;

    // Print tile size configuration:
    std::cout << "Tile size configuration:\n";
    std::cout << "  wg_m = " << wg_tile_m << ", wg_n = " << wg_tile_n << "\n";
    std::cout << "  sg_m = " << sg_tile_m << ", sg_n = " << sg_tile_n << ", sg_k = " << sg_tile_k << "\n";
    std::cout << "  scale_gs = " << scale_gs << "\n";

    using data_type_a = typename Test::data_type_a;
    using data_type_b = typename Test::data_type_b;
    using data_type_acc = typename Test::data_type_acc; // Accumulator type (internal computation)
    using data_type_c = typename Test::data_type_c; // Output C matrix type
    using data_type_acc_in = data_type_acc;

    const size_t size_a = matrix_m * matrix_k;
    const size_t size_b = matrix_k * matrix_n / 16; // We have int2x16 datatype
    const size_t size_c = matrix_m * matrix_n;
    uint32_t lda = matrix_k;
    uint32_t ldb = matrix_n;
    uint32_t ldc = matrix_n;

    // compute scale tensor sizes per user request: size_scale_a = M x (K/scale_gs), size_scale_b = (K/scale_gs) x N
    const size_t ks_groups = scale_gs;
    const size_t size_scale_a = std::max<size_t>(1, matrix_m * ks_groups);
    const size_t size_scale_b = std::max<size_t>(1, matrix_n * ks_groups);
    std::cout << "Scale tensor sizes: size_scale_a=" << size_scale_a << ", size_scale_b=" << size_scale_b << " (ks_groups=" << ks_groups << ")\n";

    //Turn on the enable_profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running int2 dequantization on " << device.get_info<info::device::name>() << "\n";
    static constexpr gpu_arch arch_tag = gpu_arch::Xe;
    using tile_shape = xetla::group::tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;
    static constexpr uint32_t periodic_sync_interval = 1;
    static constexpr uint32_t prefetch_distance = 4;

    using mem_desc_a_t = xetla::mem_desc_t<data_type_a, mem_layout::row_major, mem_space::global>;
    using mem_desc_b_t = xetla::mem_desc_t<data_type_b, mem_layout::row_major, mem_space::global>;
    using mem_desc_c_t = xetla::mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>;

    // For int8 XMX operations
    using data_type_mma_a = int8_t;
    using data_type_mma_b = int8_t;
    using compute_attr = xetla::group::compute_attr_t<data_type_mma_a, data_type_mma_b, data_type_acc>;
    using perf_tuning_knob = xetla::group::perf_tuning_knob_t<sg_tile_k, prefetch_distance, periodic_sync_interval>;
    // scale_gs is a runtime field in gemm arguments; compute_policy signature omits it here
    using compute_policy = xetla::group::compute_policy_int2_fp16_dpas_xmx<compute_attr, perf_tuning_knob, scale_dtype, scale_dtype, static_cast<int>(Test::mma_xmx_m), arch_tag>;
    using gemm_t = xetla::group::gemm_t<compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;

    // Post-op fusion in kernel epilogue using compile-time template parameters
    using bias_op_t = xetla::subgroup::bias_add_op_t<data_type_c, arch_tag>;
    using silu_op_t = xetla::subgroup::silu_op_t;
    using bias_silu_op_t = xetla::subgroup::chained_tile_op_t<bias_op_t, silu_op_t>;
    using none_op_t = xetla::subgroup::none_op_t;
    
    // Select tile_op based on template parameters (compile-time constants)
    using tile_op_t = typename std::conditional<EnableBias && EnableSiLU, bias_silu_op_t,
        typename std::conditional<EnableBias, bias_op_t,
        typename std::conditional<EnableSiLU, silu_op_t, none_op_t>::type>::type>::type;
    
    // Use tile_op epilogue policy if any post-ops are enabled, otherwise use default
    using epilogue_policy_t = typename std::conditional<EnableBias || EnableSiLU,
        xetla::group::epilogue_policy_tile_op<tile_op_t, arch_tag>,
        xetla::group::epilogue_policy_default<arch_tag>>::type;
    
    using epilogue_t = xetla::group::epilogue_t<epilogue_policy_t, tile_shape, mem_desc_c_t>;
    using group_swizzle = xetla::kernel::group_swizzle_default<arch_tag>;
    constexpr bool use_external_scale_a = Test::use_external_scale_a;
    using gemm_op_t = xetla::kernel::gemm_universal_t<gpu::xetla::kernel::dispatch_policy_int2_fp16_dpas_kslicing<group_swizzle, global_kslicing, local_kslicing, use_external_scale_a>, gemm_t, epilogue_t>;

    size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);

    // Print memory usage information
    const size_t size_a_bytes = size_a * sizeof(data_type_a);
    const size_t size_b_bytes = size_b * sizeof(data_type_b);
    const size_t size_c_bytes = size_c * sizeof(data_type_c);
    size_t size_acc_bytes = size_acc * sizeof(data_type_acc);
    size_t size_cnt_bytes = size_cnt * sizeof(uint32_t);

    size_t memory_per_set = (size_a_bytes + size_b_bytes + size_c_bytes + size_acc_bytes + size_cnt_bytes);
    size_t total_memory = memory_per_set * n_sets;

    std::cout << "Matrix dimensions: " << matrix_m << "x" << matrix_k << " * " << matrix_k << "x" << matrix_n << "\n";
    std::cout << "Memory per matrix set: " << memory_per_set / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Number of matrix sets (auto-calculated for ~2GB): " << n_sets << "\n";
    std::cout << "Total memory usage: " << total_memory / (1024.0 * 1024.0 * 1024.0) << " GB\n";

    //Define and initialize the data required for the calculation
    std::vector<data_type_a *> A_h(n_sets), A_d(n_sets);
    std::vector<data_type_b *> B_h(n_sets), B_packed_h(n_sets), B_d(n_sets);
    std::vector<data_type_c *> C_h(n_sets), C_d(n_sets);
    std::vector<data_type_acc *> Acc_h(n_sets), Acc_d(n_sets);
    std::vector<uint32_t *> Cnt_h(n_sets), Cnt_d(n_sets);
    // Scale buffers (fp16) per set
    std::vector<scale_dtype *> ScaleA_h(n_sets), ScaleA_d(n_sets);
    std::vector<scale_dtype *> ScaleB_h(n_sets), ScaleB_d(n_sets);
    // Bias buffers (only allocate if bias is enabled)
    std::vector<data_type_c *> Bias_h(n_sets), Bias_d(n_sets);
    const size_t size_bias = EnableBias ? matrix_n : 0;

    std::cout << "Starting allocations\n";
    if constexpr (EnableBias) {
        std::cout << "Bias enabled: allocating bias buffers of size " << size_bias << " elements\n";
    }
    if constexpr (EnableSiLU) {
        std::cout << "SiLU post-op enabled\n";
    }

    // Allocate memory for each matrix set
    for (int set = 0; set < n_sets; ++set) {
        A_h[set] = static_cast<data_type_a *>(malloc_host(size_a * sizeof(data_type_a), context));
        B_h[set] = static_cast<data_type_b *>(malloc_host(size_b * sizeof(data_type_b), context));
        B_packed_h[set] = static_cast<data_type_b *>(malloc_host(size_b * sizeof(data_type_b), context));
        C_h[set] = static_cast<data_type_c *>(malloc_host(size_c * sizeof(data_type_c), context));
        Acc_h[set] = static_cast<data_type_acc *>(malloc_host(size_acc * sizeof(data_type_acc), context));
        Cnt_h[set] = static_cast<uint32_t *>(malloc_host(size_cnt * sizeof(uint32_t), context));
        // allocate scale host buffers
        ScaleA_h[set] = static_cast<scale_dtype *>(malloc_host(size_scale_a * sizeof(scale_dtype), context));
        ScaleB_h[set] = static_cast<scale_dtype *>(malloc_host(size_scale_b * sizeof(scale_dtype), context));

        A_d[set] = static_cast<data_type_a *>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT, size_a * sizeof(data_type_a), device, context));
        B_d[set] = static_cast<data_type_b *>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT, size_b * sizeof(data_type_b), device, context));
        C_d[set] = static_cast<data_type_c *>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT, size_c * sizeof(data_type_c), device, context));
        Acc_d[set] = static_cast<data_type_acc *>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT, size_acc * sizeof(data_type_acc), device, context));
        Cnt_d[set] = static_cast<uint32_t *>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT, size_cnt * sizeof(uint32_t), device, context));
        // allocate scale device buffers
        ScaleA_d[set] = static_cast<scale_dtype *>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT, size_scale_a * sizeof(scale_dtype), device, context));
        ScaleB_d[set] = static_cast<scale_dtype *>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT, size_scale_b * sizeof(scale_dtype), device, context));
        
        // allocate bias buffers if enabled
        if constexpr (EnableBias) {
            Bias_h[set] = static_cast<data_type_c *>(malloc_host(size_bias * sizeof(data_type_c), context));
            Bias_d[set] = static_cast<data_type_c *>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT, size_bias * sizeof(data_type_c), device, context));
        }
    }

    // Done with allocations
    std::cout << "Done with allocations\n";

    // Initialize data for each matrix set
    for (int set = 0; set < n_sets; ++set) {
#pragma omp parallel for
        for (unsigned i = 0; i < size_a; ++i) {
            // Activations: random fp16 in [-5, 5]. Cancellation is bounded by
            // restricting the weights below to {-1, 0, 1}.
            A_h[set][i] = static_cast<fp16>(generate_real_random<float>(-5.0f, 5.0f));
        }
#pragma omp parallel for
        for (unsigned i = 0; i < size_b; ++i) {
            // Weights: restrict every int2 code to a value in {-1, 0, 1} to
            // bound output magnitudes and minimize fp32 reduction-order noise
            // in the host vs device comparison. The unpacking map is
            //   code 0 -> 0,  1 -> 1,  2 -> -2,  3 -> -1
            // so we must avoid emitting code 2 in any 2-bit pair. We pick
            // from {0, 1, 3} only:
            //   - low-bit  l = "is non-zero"  (random)
            //   - high-bit h = "is negative"  (random, AND-ed with l)
            // Pair pattern (h<<1)|l:
            //   l=0      -> 00 = code 0 -> value  0
            //   l=1,h=0  -> 01 = code 1 -> value +1
            //   l=1,h=1  -> 11 = code 3 -> value -1
            uint32_t r1 = random_uint32();
            uint32_t r2 = random_uint32();
            uint32_t low_bits  = r1 & 0x55555555u;            // bit 2k
            uint32_t high_bits = (r1 & r2) & 0x55555555u;     // bit 2k, only if l=1
            uint32_t a = low_bits | (high_bits << 1);
            B_h[set][i] = a;
            B_packed_h[set][i] = B_h[set][i];
        }

        // initialize ScaleA_h with the maximum of each row of A (per group)
        for (int rowA = 0; rowA < matrix_m; rowA++) {
            for (int igs = 0; igs < scale_gs; igs++) {
                float absmax_row = FLT_EPSILON;
                for (int _colA = 0; _colA < matrix_k / scale_gs; _colA++) {
                    int colA = igs * (matrix_k / scale_gs) + _colA;
                    absmax_row = std::max(absmax_row, std::abs(static_cast<float>(A_h[set][rowA * matrix_k + colA])));
                }
                // Match device formula exactly: single division 127/absmax (NOT 127 * (1/absmax)),
                // otherwise host vs device scale_a may differ by 1 ULP, which flips the quantization
                // of the absmax element from 127 to 126 and produces coherent fp32 errors.
                ScaleA_h[set][igs * matrix_m + rowA] = static_cast<scale_dtype>(127.0f / absmax_row);
            }
        }

        // Fake-quantize A so that A is bit-exactly representable as q/scale_a.
        // Match the kernel's saturating cast xetla_sat<int8_t,float>, which
        // lowers to ESIMD __ESIMD_NS::saturate. Per the IEEE-754 default and
        // the SPIR-V/oneAPI float->int conversion semantics used by ESIMD,
        // the rounding mode is round-to-nearest-even (RTE). Use std::lrintf,
        // which respects the current rounding mode (default FE_TONEAREST=RTE).
        for (int rowA = 0; rowA < matrix_m; rowA++) {
            for (int igs = 0; igs < scale_gs; igs++) {
                float scale_a = static_cast<float>(ScaleA_h[set][igs * matrix_m + rowA]);
                float inv_scale_a = frcp(scale_a);
                for (int _colA = 0; _colA < matrix_k / scale_gs; _colA++) {
                    int colA = igs * (matrix_k / scale_gs) + _colA;
                    float v = static_cast<float>(A_h[set][rowA * matrix_k + colA]);
                    long q_l = std::lrintf(v * scale_a); // round-to-nearest-even
                    if (q_l >  127) q_l =  127;
                    if (q_l < -128) q_l = -128;
                    int8_t q = static_cast<int8_t>(q_l);
                    A_h[set][rowA * matrix_k + colA] = static_cast<fp16>(static_cast<float>(q) * inv_scale_a);
                }
            }
        }

        // After fake-quant, the per-group absmax of A may have shifted by one
        // fp16 ULP. Recompute ScaleA_h from the *modified* A so that the host
        // gold reference uses exactly the same scale that the kernel will
        // recompute internally (when use_external_scale_a == false) and the
        // same scale we ship through global memory (when use_external_scale_a
        // == true). Without this, the kernel sees a slightly different scale
        // from the host, producing systematic ~1% errors.
        for (int rowA = 0; rowA < matrix_m; rowA++) {
            for (int igs = 0; igs < scale_gs; igs++) {
                float absmax_row = FLT_EPSILON;
                for (int _colA = 0; _colA < matrix_k / scale_gs; _colA++) {
                    int colA = igs * (matrix_k / scale_gs) + _colA;
                    absmax_row = std::max(absmax_row, std::abs(static_cast<float>(A_h[set][rowA * matrix_k + colA])));
                }
                // Match device formula exactly: single division 127/absmax.
                ScaleA_h[set][igs * matrix_m + rowA] = static_cast<scale_dtype>(127.0f / absmax_row);
            }
        }
        //for (size_t si = 0; si < size_scale_a; ++si) ScaleA_h[set][si] = random_float(); // example A-scale
        for (size_t si = 0; si < size_scale_b; ++si) {
            ScaleB_h[set][si] = static_cast<scale_dtype>(generate_real_random<float>(0.0f, 15.0f) + 0.75f); // example B-scale
        }

        // Initialize bias if enabled
        if constexpr (EnableBias) {
#pragma omp parallel for
            for (unsigned i = 0; i < size_bias; ++i) {
                Bias_h[set][i] = static_cast<data_type_c>(generate_real_random<float>(-1.0f, 1.0f));
            }
        }

#pragma omp parallel for
        for (unsigned i = 0; i < size_c; ++i) {
            C_h[set][i] = 0;
        }
#pragma omp parallel for
        for (unsigned i = 0; i < size_acc; ++i) {
            Acc_h[set][i] = 0;
        }
#pragma omp parallel for
        for (unsigned i = 0; i < size_cnt; ++i) {
            Cnt_h[set][i] = 0;
        }
    }

    // Done with initialization
    std::cout << "Done with initialization\n";

    // Copy data to device for each matrix set
    for (int set = 0; set < n_sets; ++set) {
        queue.memcpy((void *)A_d[set], (void *)A_h[set], size_a * sizeof(data_type_a)).wait();
        queue.memcpy((void *)B_d[set], (void *)B_packed_h[set], size_b * sizeof(data_type_b)).wait();
        queue.memcpy((void *)C_d[set], (void *)C_h[set], size_c * sizeof(data_type_c)).wait();
        queue.memcpy((void *)Acc_d[set], (void *)Acc_h[set], size_acc * sizeof(data_type_acc)).wait();
        queue.memcpy((void *)Cnt_d[set], (void *)Cnt_h[set], size_cnt * sizeof(uint32_t)).wait();
        // copy scales to device
        //queue.memcpy((void *)ScaleA_d[set], (void *)ScaleA_h[set], size_scale_a * sizeof(scale_dtype)).wait();
        queue.memcpy((void *)ScaleB_d[set], (void *)ScaleB_h[set], size_scale_b * sizeof(scale_dtype)).wait();
        
        // copy bias to device if enabled
        if constexpr (EnableBias) {
            queue.memcpy((void *)Bias_d[set], (void *)Bias_h[set], size_bias * sizeof(data_type_c)).wait();
        }
    }

    // Helper lambda to create epilogue arguments based on enabled post-ops
    auto create_epilogue_args = [&](data_type_c* bias_ptr) {
        if constexpr (EnableBias && EnableSiLU) {
            // Both bias and SiLU enabled
            typename bias_op_t::shape_t bias_shape(matrix_n, 1, matrix_n);
            typename bias_op_t::arguments_t bias_args(bias_ptr, bias_shape);
            typename silu_op_t::arguments_t silu_args{};
            return typename epilogue_t::arguments_t(typename tile_op_t::arguments_t(bias_args, silu_args));
        } else if constexpr (EnableBias) {
            // Only bias enabled
            typename bias_op_t::shape_t bias_shape(matrix_n, 1, matrix_n);
            typename bias_op_t::arguments_t bias_args(bias_ptr, bias_shape);
            return typename epilogue_t::arguments_t(bias_args);
        } else if constexpr (EnableSiLU) {
            // Only SiLU enabled
            typename silu_op_t::arguments_t silu_args{};
            return typename epilogue_t::arguments_t(silu_args);
        } else {
            // No post-ops enabled
            return typename epilogue_t::arguments_t();
        }
    };

    // set up gemm arguments for first set to check compatibility
    uint32_t scale_a_ld = matrix_m;
    uint32_t scale_b_ld = matrix_n;
    
    // Create gemm_arg with epilogue args if post-ops are enabled
    typename gemm_op_t::arguments_t gemm_arg_test = [&]() {
        if constexpr (use_external_scale_a) {
            // External mode: pass scale_a pointer from device memory
            if constexpr (EnableBias || EnableSiLU) {
                auto epilogue_args = create_epilogue_args(Bias_d[0]);
                return typename gemm_op_t::arguments_t(matrix_m, matrix_k, matrix_n, A_d[0], matrix_k, B_d[0], matrix_n, C_d[0], matrix_n, ScaleA_d[0], scale_a_ld, ScaleB_d[0], scale_b_ld, Acc_d[0], Cnt_d[0], epilogue_args);
            } else {
                return typename gemm_op_t::arguments_t(matrix_m, matrix_k, matrix_n, A_d[0], matrix_k, B_d[0], matrix_n, C_d[0], matrix_n, ScaleA_d[0], scale_a_ld, ScaleB_d[0], scale_b_ld, Acc_d[0], Cnt_d[0]);
            }
        } else {
            // Internal SLM mode: pass empty scale_a_base_t (computed on-the-fly in kernel)
            typename gemm_op_t::scale_a_base_t empty_scale_a{};
            if constexpr (EnableBias || EnableSiLU) {
                auto epilogue_args = create_epilogue_args(Bias_d[0]);
                return typename gemm_op_t::arguments_t(matrix_m, matrix_k, matrix_n, A_d[0], matrix_k, B_d[0], matrix_n, C_d[0], matrix_n, empty_scale_a, scale_a_ld, ScaleB_d[0], scale_b_ld, Acc_d[0], Cnt_d[0], epilogue_args);
            } else {
                return typename gemm_op_t::arguments_t(matrix_m, matrix_k, matrix_n, A_d[0], matrix_k, B_d[0], matrix_n, C_d[0], matrix_n, empty_scale_a, scale_a_ld, ScaleB_d[0], scale_b_ld, Acc_d[0], Cnt_d[0]);
            }
        }
    }();
    
    // set runtime scale_gs in arguments
    gemm_arg_test.scale_gs = scale_gs;

    sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg_test);
    if (!gemm_op_t::can_implement(gemm_arg_test)) {
        std::cout << "The arguments cannot be supported, aborting ... " << std::endl;
        FAIL();
    }

    size_t ops = 2 * matrix_m * matrix_n * matrix_k * n_sets; // Total ops for all sets
    profiling_helper prof("int2_dequantize_gemm", ops, "gflops");
    std::string compile_str
            = " -vc-codegen  "
              " -vc-disable-indvars-opt "
              " -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' ";

    compile_str += " -doubleGRF ";
    // Timer start to measure the performance
    double start_time = 0.0;
    double end_time = 0.0;

    // Variables for device time measurement (declare outside try block)
    std::vector<sycl::event> events;
    std::vector<sycl::event> events_absmax;
    std::vector<sycl::event> events_gemm;

    double total_device_time_ns = 0.0;
    double avg_device_time_ms = 0.0;
    double total_device_absmax_ns = 0.0;
    double total_device_gemm_ns = 0.0;
    double avg_device_absmax_time_ms = 0.0;
    double avg_device_gemm_time_ms = 0.0;

    try {
        // Create unique kernel type based on Test and post-op flags
        using kernel_name_t = gemm_kernel_wrapper<Test, EnableBias, EnableSiLU>;

        // Single scale-kernel uniqueness tag (TILE_X = 128 only).
        using scale_kernel_128_t = compute_scale_kernel_wrapper<Test, EnableBias, EnableSiLU, 128>;

        // Conditionally include scale kernels only if external scale A is used
        std::vector<kernel_id> kernelId;
        if constexpr (use_external_scale_a) {
            kernelId = {get_kernel_id<kernel_name_t>(),
                        get_kernel_id<scale_kernel_128_t>()};
        } else {
            // Only include GEMM kernel (scale computation is done internally in SLM)
            kernelId = {get_kernel_id<kernel_name_t>()};
        }
        
        auto inputBundle = get_kernel_bundle<bundle_state::input>(context, kernelId);
        setenv("SYCL_PROGRAM_COMPILE_OPTIONS", compile_str.c_str(), 1);
        kernel_bundle<bundle_state::executable> exeBundle = build(inputBundle);
        unsetenv("SYCL_PROGRAM_COMPILE_OPTIONS");

        for (int i = 0; i < iter; i++) {
            if (i == 1) start_time = get_time();
            prof.cpu_start();

            // Process all matrix sets in this iteration
            for (int set = 0; set < n_sets; ++set) {
                // set up gemm arguments for current set with epilogue args if post-ops enabled
                typename gemm_op_t::arguments_t gemm_arg = [&]() {
                    if constexpr (use_external_scale_a) {
                        // External mode: pass scale_a pointer from device memory
                        if constexpr (EnableBias || EnableSiLU) {
                            auto epilogue_args = create_epilogue_args(Bias_d[set]);
                            return typename gemm_op_t::arguments_t(matrix_m, matrix_k, matrix_n, A_d[set], matrix_k, B_d[set], matrix_n, C_d[set], matrix_n, ScaleA_d[set], scale_a_ld, ScaleB_d[set], scale_b_ld, Acc_d[set], Cnt_d[set], epilogue_args);
                        } else {
                            return typename gemm_op_t::arguments_t(matrix_m, matrix_k, matrix_n, A_d[set], matrix_k, B_d[set], matrix_n, C_d[set], matrix_n, ScaleA_d[set], scale_a_ld, ScaleB_d[set], scale_b_ld, Acc_d[set], Cnt_d[set]);
                        }
                    } else {
                        // Internal SLM mode: pass empty scale_a_base_t (computed on-the-fly in kernel)
                        typename gemm_op_t::scale_a_base_t empty_scale_a{};
                        if constexpr (EnableBias || EnableSiLU) {
                            auto epilogue_args = create_epilogue_args(Bias_d[set]);
                            return typename gemm_op_t::arguments_t(matrix_m, matrix_k, matrix_n, A_d[set], matrix_k, B_d[set], matrix_n, C_d[set], matrix_n, empty_scale_a, scale_a_ld, ScaleB_d[set], scale_b_ld, Acc_d[set], Cnt_d[set], epilogue_args);
                        } else {
                            return typename gemm_op_t::arguments_t(matrix_m, matrix_k, matrix_n, A_d[set], matrix_k, B_d[set], matrix_n, C_d[set], matrix_n, empty_scale_a, scale_a_ld, ScaleB_d[set], scale_b_ld, Acc_d[set], Cnt_d[set]);
                        }
                    }
                }();
                gemm_arg.scale_gs = scale_gs;

                // Compute per-row absolute max on device (only if external scale A is used)
                if constexpr (use_external_scale_a) {
                    const size_t rows = matrix_m;
                    const size_t cols = matrix_k;
                    // Single-tile-width specialization: with K=4096 and scale_gs=32,
                    // group_k = K / scale_gs = 128. Use TILE_X = 128 so each
                    // inner load tile covers exactly one ks-group's K-stride.
                    constexpr uint32_t TILE_X = 128;
                    constexpr uint32_t TILE_Y = 32;

                    const size_t local_size = TILE_Y; // threads per group
                    const size_t num_groups = (rows + TILE_Y - 1) / TILE_Y;
                    const size_t global_size = num_groups * local_size;

                    auto A_dev = A_d[set];
                    auto Scale_dev = ScaleA_d[set];

                    auto e_scale = queue.submit([&](handler &cgh) {
                        cgh.parallel_for<scale_kernel_128_t>(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)), [=](sycl::nd_item<1> it) {
                            absmax_to_inv_scale_reduction<data_type_a, scale_dtype, TILE_X, TILE_Y>::run(it, A_dev, static_cast<int>(cols), static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(scale_gs), Scale_dev);
                        });
                    });

                    // ensure scale written before GEMM reads it
                    e_scale.wait();
                    prof.add_gpu_event(e_scale);

                    if (i >= 1) {
                        events.push_back(e_scale);
                        events_absmax.push_back(e_scale);
                    }
                }

                auto e_esimd = queue.submit([&](handler &cgh) {
                    cgh.use_kernel_bundle(exeBundle);
                    cgh.parallel_for<kernel_name_t>(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                        // allocate slm and nbarrier resource
                        slm_barrier_init<gemm_op_t>();
                        gemm_op_t gemm_op;
                        gemm_op(item, gemm_arg);
                    });
                });
                e_esimd.wait();
                prof.add_gpu_event(e_esimd);

                // Store events for device time measurement (skip warmup iteration)
                if (i >= 1) {
                    events.push_back(e_esimd);
                    events_gemm.push_back(e_esimd);
                }
            }

            prof.cpu_end();
            if (i == iter - 1) end_time = get_time();
        }

        // Calculate total device execution time using event profilingmak
        for (const auto &event : events) {
            auto start_time_device = event.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end_time_device = event.get_profiling_info<sycl::info::event_profiling::command_end>();
            total_device_time_ns += (end_time_device - start_time_device);
        }

        for (const auto &event : events_absmax) {
            auto start_time_device = event.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end_time_device = event.get_profiling_info<sycl::info::event_profiling::command_end>();
            total_device_absmax_ns += (end_time_device - start_time_device);
        }

        for (const auto &event : events_gemm) {
            auto start_time_device = event.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end_time_device = event.get_profiling_info<sycl::info::event_profiling::command_end>();
            total_device_gemm_ns += (end_time_device - start_time_device);
        }

        // Convert to milliseconds and average per iteration and per set
        // Note: events contains (iter-1) * n_sets events, so we divide to get average per iteration
        // and then normalize to per-set by dividing by n_sets.
        avg_device_time_ms = total_device_time_ns / 1e6 / (iter - 1) / n_sets;
        avg_device_absmax_time_ms = total_device_absmax_ns / 1e6 / (iter - 1) / n_sets;
        avg_device_gemm_time_ms = total_device_gemm_ns / 1e6 / (iter - 1) / n_sets;
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    //performance
    //prof.print_profiling_result(profiling_selector::GPU);

    // Calculate performance metrics (will be printed after validation)
    double total_time_ms = (end_time - start_time) * 1e3;
    // Normalize host average time per iteration and per set
    double host_avg_time_ms_per_set = total_time_ms / (iter - 1) / n_sets;

    // Host-side bandwidth / gflops per set
    double bytes_per_set = (double)((matrix_k / 4) * matrix_n); // original driver's bandwidth measure
    double avg_bw_per_set = bytes_per_set / (1024.0 * 1024.0 * 1024.0) / (host_avg_time_ms_per_set * 1e-3);
    double avg_gflops_per_set = (double)((2.0 * matrix_m * matrix_k * matrix_n) / (1000.0 * 1000.0 * 1000.0) / (host_avg_time_ms_per_set * 1e-3));

    // Device-only performance metrics (normalize to per-set)
    double device_bytes_per_set = (double)((matrix_k / 4) * matrix_n + matrix_k * matrix_m * sizeof(data_type_a) + (size_scale_a + size_scale_b) * sizeof(scale_dtype));
    double device_avg_bw_per_set = device_bytes_per_set / (1024.0 * 1024.0 * 1024.0) / (avg_device_time_ms * 1e-3);
    double device_avg_gemm_bw_per_set = device_bytes_per_set / (1024.0 * 1024.0 * 1024.0) / (avg_device_gemm_time_ms * 1e-3);
    double device_avg_absmax_bw_per_set = (double)(matrix_k * matrix_m * sizeof(data_type_a)) / (1024.0 * 1024.0 * 1024.0) / (avg_device_absmax_time_ms * 1e-3);

    double device_avg_gflops_per_set = (double)((2.0 * matrix_m * matrix_k * matrix_n) / (1000.0 * 1000.0 * 1000.0) / (avg_device_time_ms * 1e-3));
    double device_avg_gemm_gflops_per_set = (double)((2.0 * matrix_m * matrix_k * matrix_n) / (1000.0 * 1000.0 * 1000.0) / (avg_device_gemm_time_ms * 1e-3));

    // int2 validation is simplified since there are no scales
    if (enable_validation) {
        std::cout << "Running int2 validation...\n";

        // Print validation memory requirements
        size_t dequantize_b_size = matrix_k * matrix_n * sizeof(int8_t);
        size_t gold_C_size = matrix_m * matrix_n * sizeof(int32_t);
        std::cout << "Validation memory per set: dequantize_b=" << dequantize_b_size / (1024.0 * 1024.0) << " MB, gold_C=" << gold_C_size / (1024.0 * 1024.0) << " MB\n";

        for (int set = 0; set < n_sets; ++set) {
            std::cout << "Allocating validation buffers for set " << set + 1 << "/" << n_sets << "...\n";

            // Use dynamic allocation for large matrices to avoid stack overflow
            int8_t *dequantize_b = static_cast<int8_t *>(malloc_host(matrix_k * matrix_n * sizeof(int8_t), context));

            if (!dequantize_b) {
                std::cout << "Failed to allocate dequantize_b buffer\n";
                continue;
            }

            std::cout << "Creating dequantized reference...\n";
// Create dequantized reference for validation
#pragma omp parallel for
            for (int64_t j = 0; j < size_b; j++) {
                uint32_t packed_vals = B_h[set][j];
                // Unpack the 16 int2 values in the dequantized_b array
                for (int idx = 0; idx < 16; idx++) {
                    int8_t val = (packed_vals >> (2 * idx)) & 0x03;
                    if (val == 3)
                        val = -1;
                    else if (val == 2)
                        val = -2;
                    else if (val == 1)
                        val = 1;
                    else
                        val = 0;
                    dequantize_b[j * 16 + idx] = val;
                }
            }

            std::cout << "Copying results from device...\n";
            queue.memcpy((void *)C_h[set], (void *)C_d[set], size_c * sizeof(data_type_c)).wait();
            std::cout << "Validating matrix set " << set + 1 << "/" << n_sets << ": ";
            ASSERT_EQ(0, gemm_result_validate(A_h[set], dequantize_b, C_h[set], matrix_m, matrix_k, matrix_n, context, scale_gs, ScaleA_h[set], size_scale_a, ScaleB_h[set], size_scale_b,
                    EnableBias ? Bias_h[set] : nullptr, EnableSiLU));

            std::cout << "Freeing validation buffers...\n";
            // Free the dynamically allocated dequantize_b buffer
            free(dequantize_b, context);
        }
    } else {
        std::cout << "Int2 validation skipped (disabled via command line argument)\n";
    }

    // Print performance statistics after validation
    std::cout << "Benchmarking " << n_sets << " int2 matrix sets\n";
    std::cout << "Host-side timing (over " << iter << " iterations):\n";
    std::cout << "  Average time per set: " << host_avg_time_ms_per_set << " ms\n";
    std::cout << "  Average Bandwidth per set: " << avg_bw_per_set << " GB/s\n";
    std::cout << "  Average GFLOPS per set: " << avg_gflops_per_set << " GFLOPS\n";
    std::cout << "Device-only timing (per set, averaged over iterations):\n";
    std::cout << "  Average device time per set: " << avg_device_time_ms << " ms\n";
    std::cout << "  Average device GEMM time per set: " << avg_device_gemm_time_ms << " ms\n";
    std::cout << "  Average device GEMM flops per set: " << device_avg_gemm_gflops_per_set << " GFLOPS\n";
    std::cout << "  Average device GEMM Bandwidth per set: " << device_avg_gemm_bw_per_set << " GB/s\n";
    std::cout << "  Average device AbsMax time per set: " << avg_device_absmax_time_ms << " ms\n";
    std::cout << "  Average device AbsMax Bandwidth per set: " << device_avg_absmax_bw_per_set << " GB/s\n";
    std::cout << "  Average device Bandwidth per set: " << device_avg_bw_per_set << " GB/s\n";
    std::cout << "  Average device GFLOPS per set: " << device_avg_gflops_per_set << " GFLOPS\n";
    std::cout << "Device efficiency: " << (avg_device_time_ms / host_avg_time_ms_per_set) * 100.0 << "%\n";

    // Free memory for all matrix sets
    for (int set = 0; set < n_sets; ++set) {
        free(A_h[set], context);
        free(B_h[set], context);
        free(B_packed_h[set], context);
        free(C_h[set], context);
        free(A_d[set], context);
        free(B_d[set], context);
        free(C_d[set], context);
        free(Acc_h[set], context);
        free(Cnt_h[set], context);
        free(Acc_d[set], context);
        free(Cnt_d[set], context);
        // free scale buffers
        free(ScaleA_h[set], context);
        free(ScaleB_h[set], context);
        free(ScaleA_d[set], context);
        free(ScaleB_d[set], context);
        // free bias buffers if allocated
        if constexpr (EnableBias) {
            free(Bias_h[set], context);
            free(Bias_d[set], context);
        }
    }
}

template <typename T>
class int2_dequantize_gemm_test : public ::testing::Test {};

// Forward declaration
static bool find_and_run_selected_variant(bool use_gemv);

void run_int2_test_with_kslicing() {
    // Choose optimized templates based on matrix M dimension
    bool use_gemv_kernel = (g_mat_m == 1);

    // Use precompiled variant selection instead of hardcoded type aliases
    if (!find_and_run_selected_variant(use_gemv_kernel)) {
        std::cout << "No precompiled variant matched the requested configuration.\n";
        std::cout << "Requested: wg_m=" << g_wg_m << " sg_m=" << g_sg_m << " wg_n=" << g_wg_n 
                  << " sg_n=" << g_sg_n << " sg_k=" << g_sg_k << " gks=" << g_global_kslicing << "\n";
        FAIL();
    }
}

// Precompiled variant configuration (constexpr so templates can be instantiated at compile time)
// Set PRECOMPILE_FULL_AUTOTUNE=1 to compile all variants for autotuning, or 0 for optimized set only
#ifndef PRECOMPILE_FULL_AUTOTUNE
#define PRECOMPILE_FULL_AUTOTUNE 0
#endif

// Precompiled variant configuration now defined in variant_common.hpp

// Forward declare print_precompile_plan (implementation below)
//static void print_precompile_plan();

// Find and run matching precompiled variant based on runtime knobs.
// Returns true if a matching variant was found and invoked.
static bool find_and_run_selected_variant(bool use_gemv) {
    const auto &variants = get_precompiled_variants();
    for (const auto &var : variants) {
        if (var.wg_m == g_wg_m && var.sg_m == g_sg_m && var.wg_n == g_wg_n && var.sg_n == g_sg_n && var.sg_k == g_sg_k && var.gks == g_global_kslicing && var.is_gemv == use_gemv) {
            std::cout << "Found matching precompiled variant: wg_m=" << var.wg_m << " sg_m=" << var.sg_m << " wg_n=" << var.wg_n << " sg_n=" << var.sg_n << " sg_k=" << var.sg_k << " gks=" << var.gks << " is_gemv=" << var.is_gemv << std::endl;
            std::cout << "Scale A calculation mode: " << (g_external_scale_a_calc ? "external (separate kernel)" : "internal (SLM)") << std::endl;
            var.fn();
            return true;
        }
    }
    std::cout << "No matching precompiled variant found for the requested tile configuration: wg_m=" << g_wg_m << " sg_m=" << g_sg_m << " wg_n=" << g_wg_n << " sg_n=" << g_sg_n << " sg_k=" << g_sg_k << " gks=" << g_global_kslicing << "\n";
    return false;
}

// All test types now come from precompiled variants in variant_registry.cpp

int main(int argc, char **argv) {
    // Parse command line arguments before initializing Google Test
    bool select_variant = false;
    auto parse_uint64 = [&](const std::string &arg, const std::string &key, uint64_t &out) -> bool {
        size_t eq = arg.find('=');
        if (eq == std::string::npos) return false;
        std::string k = arg.substr(0, eq + 1);
        if (k != key + "=") return false;
        std::string val = arg.substr(eq + 1);
        if (val.empty()) {
            std::cout << "Missing value for " << key << "\n";
            return false;
        }
        try {
            out = std::stoull(val);
            return true;
        } catch (const std::exception &e) {
            std::cout << "Invalid numeric value for " << key << ": '" << val << "' (" << e.what() << ")\n";
            return false;
        }
    };

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--validation=0" || arg == "-v0") {
            g_enable_validation = false;
            std::cout << "Validation disabled via command line argument\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1];
            argc--; i--; continue;
        }
        if (arg == "--validation=1" || arg == "-v1") {
            g_enable_validation = true;
            std::cout << "Validation enabled via command line argument\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1];
            argc--; i--; continue;
        }
        uint64_t tmp = 0;
        if (parse_uint64(arg, "--mat_m", tmp)) {
            g_mat_m = tmp;
            std::cout << "Matrix M dimension set to: " << g_mat_m << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--mat_n", tmp)) {
            g_mat_n = tmp;
            std::cout << "Matrix N dimension set to: " << g_mat_n << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--mat_k", tmp)) {
            g_mat_k = tmp;
            std::cout << "Matrix K dimension set to: " << g_mat_k << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--global_kslicing", tmp)) {
            g_global_kslicing = static_cast<uint32_t>(tmp);
            if (g_global_kslicing != 1 && g_global_kslicing != 2 && g_global_kslicing != 4 && g_global_kslicing != 8) {
                std::cout << "Error: global_kslicing must be 1, 2, 4, or 8. Got: " << g_global_kslicing << "\n";
                return 1;
            }
            std::cout << "Global K-slicing set to: " << g_global_kslicing << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--scale_gs", tmp)) {
            g_scale_gs = static_cast<uint32_t>(tmp);
            std::cout << "Runtime scale_gs set to: " << g_scale_gs << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--external_scale_a_calc", tmp)) {
            g_external_scale_a_calc = static_cast<uint32_t>(tmp);
            if (g_external_scale_a_calc > 1) {
                std::cout << "Error: external_scale_a_calc must be 0 or 1. Got: " << g_external_scale_a_calc << "\n";
                return 1;
            }
            // Both modes are now supported in the fp16-scales variant: the
            // kslicing kernel's internal SLM scale_a path was templated on
            // dtype_scale_a so it stores fp16 (not float) when the scale
            // dtype is fp16.  See compute_scale_a_to_slm in
            // include/experimental/kernel/gemm/impl/int2_fp16_dpas_kslicing_xe.hpp.
            std::cout << "Scale A calculation mode set to: " << (g_external_scale_a_calc ? "external (separate kernel)" : "internal (SLM within GEMM)") << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--wg_m", tmp)) {
            g_wg_m = static_cast<uint32_t>(tmp);
            select_variant = true;
            std::cout << "Requested wg_m=" << g_wg_m << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--sg_m", tmp)) {
            g_sg_m = static_cast<uint32_t>(tmp);
            select_variant = true;
            std::cout << "Requested sg_m=" << g_sg_m << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--wg_n", tmp)) {
            g_wg_n = static_cast<uint32_t>(tmp);
            select_variant = true;
            std::cout << "Requested wg_n=" << g_wg_n << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--sg_n", tmp)) {
            g_sg_n = static_cast<uint32_t>(tmp);
            select_variant = true;
            std::cout << "Requested sg_n=" << g_sg_n << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--sg_k", tmp)) {
            g_sg_k = static_cast<uint32_t>(tmp);
            select_variant = true;
            std::cout << "Requested sg_k=" << g_sg_k << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--sets", tmp)) {
            g_sets = static_cast<int>(tmp);
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (arg.rfind("--weights_gib=", 0) == 0) {
            g_weights_gib = std::atof(arg.c_str() + 14);
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (parse_uint64(arg, "--iters", tmp) || parse_uint64(arg, "--num_iters", tmp)) {
            g_num_iters = static_cast<int>(tmp);
            if (g_num_iters < 1) {
                std::cout << "Error: num_iters must be at least 1. Got: " << g_num_iters << "\n";
                return 1;
            }
            std::cout << "Number of iterations set to: " << g_num_iters << "\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1]; argc--; i--; continue;
        }
        if (arg == "--bias" || arg == "--enable_bias") {
            g_enable_bias = true;
            std::cout << "Bias post-op enabled\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1];
            argc--; i--; continue;
        }
        if (arg == "--silu" || arg == "--enable_silu") {
            g_enable_silu = true;
            std::cout << "SiLU post-op enabled\n";
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1];
            argc--; i--; continue;
        }
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --validation=0, -v0    Disable validation\n";
            std::cout << "  --validation=1, -v1    Enable validation (default)\n";
            std::cout << "  --mat_m=M              Set matrix M dimension (default: 1)\n";
            std::cout << "  --mat_n=N              Set matrix N dimension (default: 8192)\n";
            std::cout << "  --mat_k=K              Set matrix K dimension (default: 8192)\n";
            std::cout << "  --global_kslicing=K    Set global K-slicing factor (1,2,4,8) (default: 1)\n";
            std::cout << "  --scale_gs=S           Runtime scale group size (default: Test::scale_gs)\n";
            std::cout << "  --external_scale_a_calc=X  Scale A calc mode: 1=external kernel, 0=internal SLM (default: 1)\n";
            std::cout << "  --wg_m=W               Select work-group M tile size from precompiled variants\n";
            std::cout << "  --sg_m=S               Select subgroup M tile size from precompiled variants\n";
            std::cout << "  --wg_n=W               Select work-group N tile size from precompiled variants\n";
            std::cout << "  --sg_n=S               Select subgroup N tile size from precompiled variants\n";
            std::cout << "  --sg_k=K               Select subgroup K tile size (32,128,160) for GEMVs\n";
            std::cout << "  --iters=N, --num_iters=N   Set number of kernel iterations (default: 20)\n";
            std::cout << "  --bias, --enable_bias  Enable bias post-op fusion\n";
            std::cout << "  --silu, --enable_silu  Enable SiLU post-op fusion\n";
            std::cout << "  --help, -h             Show this help message\n";
            std::cout << "\nCurrent settings:\n";
            std::cout << "  Matrix dimensions: " << g_mat_m << "x" << g_mat_k << " * " << g_mat_k << "x" << g_mat_n << "\n";
            std::cout << "  Global K-slicing: " << g_global_kslicing << "\n";
            std::cout << "  Matrix sets: auto-calculated for ~2GB based on dimensions\n";
            std::cout << "  Iterations: " << g_num_iters << "\n";
            return 0;
        }
    }

    // Print precompile plan for templates (no actual instantiation beyond PRECOMPILED_VARIANTS)
    //print_precompile_plan();

    // Recalculate N_SETS based on the final matrix dimensions
    N_SETS = calculate_n_sets_runtime();
    std::cout << "Matrix dimensions: " << g_mat_m << "x" << g_mat_k << " * " << g_mat_k << "x" << g_mat_n << "\n";
    std::cout << "Global K-slicing: " << g_global_kslicing << "\n";
    std::cout << "Subgroup K tile: " << g_sg_k << (g_mat_m == 1 ? " (GEMV)" : " (GEMM, always 32)") << "\n";
    std::cout << "Auto-calculated matrix sets for ~2GB: " << N_SETS << "\n";
    std::cout << "Running int2 test with global_kslicing=" << g_global_kslicing << "\n";

    try {
        bool use_gemv_kernel = (g_mat_m == 1);
        if (select_variant) {
            // Try to find and run the requested precompiled variant
            if (!find_and_run_selected_variant(use_gemv_kernel)) {
                std::cout << "No precompiled variant matched the requested tiles. Aborting.\n";
                return 1;
            }
        } else {
            // Default behavior: select kernel based on kslicing and gemv/gemm choice
            run_int2_test_with_kslicing();
        }

        std::cout << "Test PASSED\n";
        return 0;
    } catch (const std::exception &e) {
        std::cout << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}

#if 0
// Variant descriptor and helpers to instantiate Test types from PRE_* tuples
static void print_precompile_plan() {
    size_t gemm_count = PRE_GEMM_WG_M.size() * PRE_GEMM_WG_N_PAIRS.size();
    size_t gemv_count = PRE_GEMV_WG_N_PAIRS.size() * PRE_GEMV_SG_K.size();

    std::cout << "Planned precompiled GEMM instantiations (" << gemm_count << "):\n";
    for (size_t im = 0; im < PRE_GEMM_WG_M.size(); ++im) {
        uint32_t wg_m = PRE_GEMM_WG_M[im];
        uint32_t sg_m = PRE_GEMM_SG_M;
        uint32_t sg_k = PRE_GEMM_SG_K;
        for (size_t in = 0; in < PRE_GEMM_WG_N_PAIRS.size(); ++in) {
            uint32_t wg_n = PRE_GEMM_WG_N_PAIRS[in].first;
            uint32_t sg_n = PRE_GEMM_WG_N_PAIRS[in].second;
            std::cout << "  int2_dequant_base<" << PRECOMP_GKS << ", " << wg_m << ", " << sg_m << ", " << wg_n << ", " << sg_n << ", " << sg_k << ", " << sg_m << ", int32_t, fp16>\n";
        }
    }

    std::cout << "Planned precompiled GEMV instantiations (" << gemv_count << "):\n";
    for (size_t in = 0; in < PRE_GEMV_WG_N_PAIRS.size(); ++in) {
        uint32_t wg_n = PRE_GEMV_WG_N_PAIRS[in].first;
        uint32_t sg_n = PRE_GEMV_WG_N_PAIRS[in].second;
        for (size_t ik = 0; ik < PRE_GEMV_SG_K.size(); ++ik) {
            uint32_t sg_k = PRE_GEMV_SG_K[ik];
            std::cout << "  int2_dequant_base<" << PRECOMP_GKS << ", " << PRE_GEMV_WG_M << ", " << PRE_GEMV_SG_M << ", " << wg_n << ", " << sg_n << ", " << sg_k << ", 1, int32_t, fp16>\n";
        }
    }
}
#endif


// Explicit template instantiations for precompiled variants.
// Both UseExternalScaleA={true,false} branches are instantiated: the
// kslicing kernel's SLM scale_a path was templated on dtype_scale_a so it
// stores fp16 (not float) when the scale dtype is fp16, which makes the
// internal-SLM mode (--external_scale_a_calc=0) functional for this variant.
#define INSTANTIATE_ALL_POSTOPS_ESA(GKS, WgM, SgM, WgN, SgN, SgK, XMXM, AccType, CType, ESA) \
    template void int2_dequantize_gemm_run<int2_dequant_base<GKS, WgM, SgM, WgN, SgN, SgK, XMXM, AccType, CType, ESA>, false, false>(int, int, bool); \
    template void int2_dequantize_gemm_run<int2_dequant_base<GKS, WgM, SgM, WgN, SgN, SgK, XMXM, AccType, CType, ESA>, true, false>(int, int, bool); \
    template void int2_dequantize_gemm_run<int2_dequant_base<GKS, WgM, SgM, WgN, SgN, SgK, XMXM, AccType, CType, ESA>, false, true>(int, int, bool); \
    template void int2_dequantize_gemm_run<int2_dequant_base<GKS, WgM, SgM, WgN, SgN, SgK, XMXM, AccType, CType, ESA>, true, true>(int, int, bool);

#define INSTANTIATE_ALL_POSTOPS(GKS, WgM, SgM, WgN, SgN, SgK, XMXM, AccType, CType) \
    INSTANTIATE_ALL_POSTOPS_ESA(GKS, WgM, SgM, WgN, SgN, SgK, XMXM, AccType, CType, true)  \
    INSTANTIATE_ALL_POSTOPS_ESA(GKS, WgM, SgM, WgN, SgN, SgK, XMXM, AccType, CType, false)

INSTANTIATE_ALL_POSTOPS(1u, 64u, 8u, 256u, 128u, 128u, 8u, int32_t, fp16)

// Additional precompiled tile tuples (matching variant_common.hpp PRE_GEMM_TUPLES / PRE_GEMV_TUPLES).
// Each line expands to 16 explicit instantiations
// (UseExternalScaleA={true,false} x 4 post-op combos x ... etc).

// GEMM tuples (sg_m=8, mma_xmx_m=8, GKS=1)
INSTANTIATE_ALL_POSTOPS(1u,  64u, 8u, 128u, 128u,  32u, 8u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u,  32u, 8u, 256u, 128u,  64u, 8u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 256u, 8u, 160u, 160u,  32u, 8u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u,  32u, 8u, 160u, 160u,  32u, 8u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u,  64u, 8u, 160u, 160u,  32u, 8u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u,  64u, 8u, 256u, 128u,  64u, 8u, int32_t, fp16)

// GEMV tuples (wg_m=1, sg_m=1, mma_xmx_m=1, GKS=1).
// Cartesian product of wg_n in {64, 128, 256}, sg_n in {16, 32}, sg_k in
// {32, 64, 128} (18 shapes total).  The kernel bypasses the standard
// tile_load for the 1-element fp16 scaleA case (global path) with a
// single-lane lsc_gather (SYCL_EXTERNAL) instead of the missing fp16
// 1-element block_load_impl specialization in oneAPI 2025.3.
// wg_n=64
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u,  64u, 16u,  32u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u,  64u, 16u,  64u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u,  64u, 16u, 128u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u,  64u, 32u,  32u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u,  64u, 32u,  64u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u,  64u, 32u, 128u, 1u, int32_t, fp16)
// wg_n=128
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 128u, 16u,  32u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 128u, 16u,  64u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 128u, 16u, 128u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 128u, 32u,  32u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 128u, 32u,  64u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 128u, 32u, 128u, 1u, int32_t, fp16)
// wg_n=256
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 256u, 16u,  32u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 256u, 16u,  64u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 256u, 16u, 128u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 256u, 32u,  32u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 256u, 32u,  64u, 1u, int32_t, fp16)
INSTANTIATE_ALL_POSTOPS(1u, 1u, 1u, 256u, 32u, 128u, 1u, int32_t, fp16)
