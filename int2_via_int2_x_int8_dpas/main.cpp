// Standalone OpenCL driver for int2_int8_dpas.cl, mirroring
// xetla/int2_fp16_dpas_fp16scales_fast_test/src/main.cpp:
//   - same operation, layouts and input generation: A (--dtype fp16|bf16) in [-5, 5]
//     fake-quantized per (row, 128-group) so that A = q / SA exactly
//     (lrintf), SA = DT(127 / absmax) recomputed from the fake-quantized A,
//     B codes {0, 1, 3} = {0, +1, -1}, SB DT in [0.75, 15.75];
//   - same host gold: q = sat_int8(trunc(A * SA)), int32 dot per group,
//     acc += float(dot) * (SB * (1 / SA)), cast to DT;
//   - same cache-defeat methodology as xetla_ref (the xetla harness with the
//     identical rule): rotate over distinct device buffer sets until the
//     distinct weights (B + SB) reach --weights-gib (default 2 GiB), warm up
//     every set, time with device events -- the A pre-kernel and the GEMM
//     separately, like xetla's "AbsMax" and "GEMM" times;
//   - same pass rule as xetla_buff_cmp: fp16 (ulp_tol=64, abs_tol=8), bf16 (32, 16).
// Tile / mode parameters are OpenCL -D options: no rebuild for a sweep.

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "dt16.hpp"
#include "epilogue.hpp"
static constexpr int kGS = 128;

#define CL_CHECK(x)                                                         \
    do {                                                                    \
        cl_int err_ = (x);                                                  \
        if (err_ != CL_SUCCESS) {                                           \
            std::cerr << #x << " failed: " << err_ << " (" << __FILE__      \
                      << ":" << __LINE__ << ")\n";                          \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

struct RunConfig {
    int m = 1, n = 4096, k = 4096, iters = 50;
    int qmode = 1;  // 0 upfront int8 A, 1 xetla scheme (scale pre-kernel, quantize in GEMM)
    bool validate = true, distinct_sets = false;
    int num_sets = 0;
    double weights_gib = 2.0;
    int sgm = 0, nsg = 0, ls = 0, u = 0;  // GEMV kernel, 0 = default
    int mt_m = 0, mt_n = 128, wg_m = 8, wg_n = 2;  // large-M kernel
    Epilogue epi;
    std::string cl_path, extra_opts;
    bool print_build_log = false;
};

static uint32_t rnd_u32() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return std::uniform_int_distribution<uint32_t>(0, UINT32_MAX)(gen);
}
static float rnd_f(float lo, float hi) {
    static thread_local std::mt19937 gen(std::random_device{}());
    return std::uniform_real_distribution<float>(lo, hi)(gen);
}
static int code_to_value(uint32_t c) { return (int)(int8_t)((int8_t)(c << 6) >> 6); }
// SA row pitch (elements), same as LDSA() in the .cl
static int ldsa(int M) { return (M + 31) & ~31; }

// Per (row, group) scale, xetla formula: DT(127 / max(FLT_EPSILON, absmax)).
static void compute_scale_a(const dt16 *A, dt16 *SA, int M, int K) {
#pragma omp parallel for
    for (int m = 0; m < M; ++m)
        for (int g = 0; g < K / kGS; ++g) {
            float mx = FLT_EPSILON;
            for (int p = 0; p < kGS; ++p)
                mx = std::max(mx, std::fabs(tof(A[(size_t)m * K + g * kGS + p])));
            SA[(size_t)g * ldsa(M) + m] = fromf(127.0f / mx);
        }
}

// fp32 accumulator gold; the epilogue and output cast follow in epilogue_ref()
static void compute_gold(const dt16 *A, const uint32_t *B, const dt16 *SA,
        const dt16 *SB, float *C, int M, int K, int N) {
    const int groups = K / kGS;
    std::vector<int8_t> q((size_t)M * K);
#pragma omp parallel for
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            long v = (long)(tof(A[(size_t)m * K + k]) * tof(SA[(size_t)(k / kGS) * ldsa(M) + m]));
            q[(size_t)m * K + k] = (int8_t)std::min(127L, std::max(-128L, v));
        }
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int g = 0; g < groups; ++g) {
                int32_t dot = 0;
                for (int p = 0; p < kGS; ++p) {
                    const int ik = g * kGS + p;
                    const uint32_t code = (B[(size_t)(ik / 16) * N + j] >> (2 * (ik % 16))) & 3u;
                    dot += (int)q[(size_t)i * K + ik] * code_to_value(code);
                }
                const float sa = tof(SA[(size_t)g * ldsa(M) + i]);
                const float inv = sa == 0.0f ? 0.0f : 1.0f / sa;
                acc += (float)dot * (tof(SB[(size_t)g * N + j]) * inv);
            }
            C[(size_t)i * N + j] = acc;
        }
}

static bool compare(const dt16 *C, const dt16 *G, size_t n, const std::string &label,
        double abs_tol, uint32_t ulp_tol) {
    size_t bad = 0, ulp_idx = 0, abs_idx = 0;
    uint32_t max_ulp = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const uint16_t a = C[i], g = G[i];
        uint32_t ulp = a > g ? a - g : g - a;
        float act = tof(C[i]), des = tof(G[i]);
        double d = std::fabs((double)act - des);
        if (ulp > max_ulp) { max_ulp = ulp; ulp_idx = i; }
        if (d > max_abs || std::isnan(d)) { max_abs = d; abs_idx = i; }
        if (!(d <= abs_tol || ulp <= ulp_tol || std::fabs((des - act) / des) <= 0.001)) {
            if (++bad <= 10)
                std::cout << "\tidx " << i << " data " << act << " gold " << des << "\n";
        }
    }
    std::cout << label << ": max abs diff " << max_abs << " (idx " << abs_idx
              << "), max ULP diff " << max_ulp << " (idx " << ulp_idx
              << "), pass rate " << 100.0 * (n - bad) / n << "%\n";
    return bad == 0;
}

static std::string read_file(const std::string &p) {
    std::ifstream f(p);
    if (!f) { std::cerr << "cannot open " << p << "\n"; std::exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void default_tiles(RunConfig &c) {
    if (c.sgm == 0) c.sgm = c.m == 1 ? 1 : (c.m <= 2 ? 2 : (c.m <= 4 ? 4 : 8));
    if (c.nsg == 0) c.nsg = 2;
    if (c.ls == 0) c.ls = c.n <= 8192 ? 4 : 2;
    if (c.u == 0) c.u = 2;
}

static double ev_ns(cl_event e) {
    cl_ulong st, en;
    CL_CHECK(clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof st, &st, nullptr));
    CL_CHECK(clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof en, &en, nullptr));
    return (double)(en - st);
}

static void run(RunConfig cfg) {
    const int M = cfg.m, N = cfg.n, K = cfg.k;
    if (K % kGS || N % 16) { std::cerr << "need K % 128 == 0 and N % 16 == 0\n"; std::exit(1); }
    if (cfg.qmode < 0 || cfg.qmode > 1) { std::cerr << "--qmode 0|1\n"; std::exit(1); }
    const bool mt = cfg.mt_m > 0 || M >= 64;
    if (mt && cfg.mt_m == 0) cfg.mt_m = 8;
    default_tiles(cfg);

    cl_platform_id plats[8];
    cl_uint np = 0;
    CL_CHECK(clGetPlatformIDs(8, plats, &np));
    cl_device_id dev = nullptr;
    for (cl_uint i = 0; i < np && !dev; ++i)
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) != CL_SUCCESS) dev = nullptr;
    if (!dev) { std::cerr << "no OpenCL GPU\n"; std::exit(1); }
    char name[256];
    CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, nullptr));
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err);
    CL_CHECK(err);

    static const char *qname[] = {"upfront (int8 A + scale pre-kernel)",
            "xetla scheme (scale pre-kernel, quantize in GEMM)"};
    std::cout << "Device: " << name << "\n"
              << "Problem: M=" << M << " N=" << N << " K=" << K << " scale_gs=" << kGS
              << " dtype=" << dt_name() << " epilogue=" << cfg.epi.name() << "\n"
              << "A quant: qmode=" << cfg.qmode << " " << qname[cfg.qmode] << "\n";
    if (mt)
        std::cout << "Tile (mt): sg " << cfg.mt_m << "x" << cfg.mt_n << ", wg " << cfg.wg_m
                  << "x" << cfg.wg_n << " sub-groups, 256 GRF\n";
    else
        std::cout << "Tile: sg_m=" << cfg.sgm << " wg_n=" << 16 * cfg.nsg << " (nsg=" << cfg.nsg
                  << ") ls=" << cfg.ls << " u=" << cfg.u << "\n";

    std::string src = read_file(cfg.cl_path);
    const char *srcp = src.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &srcp, nullptr, &err);
    CL_CHECK(err);
    std::string opts = std::string("-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt")
            + (dt_is_bf16() ? " -DBF16" : "") + " -DQMODE="
            + std::to_string(cfg.qmode) + (cfg.qmode == 0 ? " -DWRITE_Q" : "") + " -DSGM="
            + std::to_string(cfg.sgm) + " -DNSG_N=" + std::to_string(cfg.nsg) + " -DLS="
            + std::to_string(cfg.ls) + " -DU=" + std::to_string(cfg.u) + " -DMT_M="
            + std::to_string(cfg.mt_m ? cfg.mt_m : 32) + " -DMT_N=" + std::to_string(cfg.mt_n)
            + " -DWG_M=" + std::to_string(cfg.wg_m) + " -DWG_N=" + std::to_string(cfg.wg_n)
            + (mt ? " -cl-intel-256-GRF-per-thread" : "") + cfg.epi.opts() + " -I "
            + cfg.cl_path.substr(0, cfg.cl_path.find_last_of('/') + 1) + ". " + cfg.extra_opts;
    err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS || cfg.print_build_log) {
        size_t len = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        std::cout << "Build options: " << opts << "\nBuild log:\n" << log << "\n";
        CL_CHECK(err);
    }
    cl_kernel kern = clCreateKernel(prog, mt ? "int2_int8_gemm_mt" : "int2_int8_gemv", &err);
    CL_CHECK(err);
    cl_kernel kq = clCreateKernel(prog, "quant_a", &err);
    CL_CHECK(err);

    const size_t size_a = (size_t)M * K, size_b = (size_t)(K / 16) * N;
    const size_t size_c = (size_t)M * N, size_sb = (size_t)(K / kGS) * N;
    const size_t size_sa = (size_t)(K / kGS) * ldsa(M);
    const Epilogue &ep = cfg.epi;
    const size_t os = ep.out_size();
    const double bytes_set = size_a * 3.0 + size_b * 4.0 + size_c * (double)os + (size_sb + size_sa) * 2.0
            + (ep.needs_other() ? size_c * 2.0 : 0.0) + (ep.needs_bias() ? N * (double)os : 0.0);
    const double wbytes_set = size_b * 4.0 + size_sb * 2.0;
    int sets = cfg.num_sets > 0 ? cfg.num_sets
                                : std::max(1, (int)std::ceil(cfg.weights_gib * (1 << 30) / wbytes_set));
    std::cout << "Per-set bytes: " << bytes_set / (1 << 20) << " MiB (weights " << wbytes_set / (1 << 20)
              << " MiB); using " << sets << " distinct sets -> weights " << std::fixed
              << std::setprecision(3) << wbytes_set * sets / (1 << 30) << " GiB, total "
              << bytes_set * sets / (1 << 30) << " GiB\n";

    std::vector<dt16> A(size_a), SA(size_sa), SB(size_sb), Oth;
    std::vector<unsigned char> Bias, Ch(size_c * os);
    std::vector<uint32_t> B(size_b);
    static std::mt19937 egen(std::random_device{}());
    auto fill = [&] {
#pragma omp parallel for
        for (size_t i = 0; i < size_a; ++i) A[i] = fromf(rnd_f(-5.0f, 5.0f));
#pragma omp parallel for
        for (size_t i = 0; i < size_b; ++i) {
            uint32_t r1 = rnd_u32(), r2 = rnd_u32();
            uint32_t lo = r1 & 0x55555555u, hi = (r1 & r2) & 0x55555555u;
            B[i] = lo | (hi << 1);  // codes {0, 1, 3}
        }
        for (size_t i = 0; i < size_sb; ++i) SB[i] = fromf(rnd_f(0.0f, 15.0f) + 0.75f);
        // fake-quantize A (RTE, as xetla), then recompute SA from the result
        compute_scale_a(A.data(), SA.data(), M, K);
#pragma omp parallel for
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) {
                const float sa = tof(SA[(size_t)(k / kGS) * ldsa(M) + m]);
                long v = std::lrintf(tof(A[(size_t)m * K + k]) * sa);
                v = std::min(127L, std::max(-128L, v));
                A[(size_t)m * K + k] = fromf((float)v * (1.0f / sa));
            }
        compute_scale_a(A.data(), SA.data(), M, K);
        fill_epilogue_inputs(ep, Oth, Bias, M, N, egen);
    };
    fill();

    const int ngold = (cfg.validate && cfg.distinct_sets) ? sets : 1;
    std::vector<std::vector<unsigned char>> gold(cfg.validate ? ngold : 0);
    std::vector<float> acc_gold;
    auto make_gold = [&](std::vector<unsigned char> &g) {
        acc_gold.resize(size_c);
        compute_gold(A.data(), B.data(), SA.data(), SB.data(), acc_gold.data(), M, K, N);
        g.resize(size_c * os);
        epilogue_ref(ep, acc_gold.data(), Oth.data(), Bias.data(), g.data(), M, N);
    };
    if (cfg.validate) make_gold(gold[0]);

    // Other / Bias are only read by their epilogues; bind a dummy otherwise
    cl_mem dummy = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 64, nullptr, &err);
    CL_CHECK(err);
    std::vector<cl_mem> dA(sets), dAq(sets), dSA(sets), dB(sets), dSB(sets), dC(sets);
    std::vector<cl_mem> dO(sets, dummy), dBias(sets, dummy);
    for (int s = 0; s < sets; ++s) {
        if (cfg.distinct_sets && s > 0) {
            fill();
            if (cfg.validate) make_gold(gold[s]);
        }
        if (ep.needs_other()) {
            dO[s] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_c * 2, Oth.data(), &err);
            CL_CHECK(err);
        }
        if (ep.needs_bias()) {
            dBias[s] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Bias.size(), Bias.data(), &err);
            CL_CHECK(err);
        }
        dA[s] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_a * 2, A.data(), &err);
        CL_CHECK(err);
        dB[s] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_b * 4, B.data(), &err);
        CL_CHECK(err);
        dSB[s] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_sb * 2, SB.data(), &err);
        CL_CHECK(err);
        // SA / Aq are produced on the device (garbage-filled so a missing
        // write cannot pass validation)
        dSA[s] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_sa * 2, nullptr, &err);
        CL_CHECK(err);
        dAq[s] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, cfg.qmode == 0 ? size_a : 64, nullptr, &err);
        CL_CHECK(err);
        dC[s] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_c * os, nullptr, &err);
        CL_CHECK(err);
        const cl_uchar junk = 0x5a, zero = 0;
        CL_CHECK(clEnqueueFillBuffer(q, dSA[s], &junk, 1, 0, size_sa * 2, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueFillBuffer(q, dAq[s], &junk, 1, 0, cfg.qmode == 0 ? size_a : 64, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueFillBuffer(q, dC[s], &zero, 1, 0, size_c * os, 0, nullptr, nullptr));
    }
    CL_CHECK(clFinish(q));

    const size_t wg_n = 16 * cfg.nsg;
    size_t local[2] = {wg_n * cfg.ls, 1};
    size_t global[2] = {((N + wg_n - 1) / wg_n) * local[0], (size_t)((M + cfg.sgm - 1) / cfg.sgm)};
    if (mt) {
        const size_t tn = (size_t)cfg.mt_n * cfg.wg_n, tm = (size_t)cfg.mt_m * cfg.wg_m;
        local[0] = 16 * cfg.wg_n * cfg.wg_m;
        global[0] = ((N + tn - 1) / tn) * local[0];
        global[1] = (M + tm - 1) / tm;
    }
    const size_t qlocal[2] = {16, 1}, qglobal[2] = {(size_t)(K / kGS) * 16, (size_t)M};
    std::cout << "Global: {" << global[0] << ", " << global[1] << "} Local: {" << local[0]
              << ", " << local[1] << "}\n";

    double gemm_ns = 0.0, pre_ns = 0.0, host_ms = 0.0;
    int timed = 0;
    for (int it = 0; it < sets + cfg.iters; ++it) {
        const int s = it % sets;
        cl_event eq, e;
        auto t0 = std::chrono::high_resolution_clock::now();
        {
            CL_CHECK(clSetKernelArg(kq, 0, sizeof(cl_mem), &dA[s]));
            CL_CHECK(clSetKernelArg(kq, 1, sizeof(cl_mem), &dSA[s]));
            CL_CHECK(clSetKernelArg(kq, 2, sizeof(cl_mem), &dAq[s]));
            CL_CHECK(clSetKernelArg(kq, 3, sizeof(int), &M));
            CL_CHECK(clSetKernelArg(kq, 4, sizeof(int), &K));
            CL_CHECK(clEnqueueNDRangeKernel(q, kq, 2, nullptr, qglobal, qlocal, 0, nullptr, &eq));
        }
        const cl_mem args[8] = {dA[s], dAq[s], dSA[s], dB[s], dSB[s], dC[s], dO[s], dBias[s]};
        for (int a = 0; a < 8; ++a) CL_CHECK(clSetKernelArg(kern, a, sizeof(cl_mem), &args[a]));
        CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &M));
        CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(kern, 10, sizeof(int), &K));
        CL_CHECK(clEnqueueNDRangeKernel(q, kern, 2, nullptr, global, local, 0, nullptr, &e));
        CL_CHECK(clWaitForEvents(1, &e));
        auto t1 = std::chrono::high_resolution_clock::now();
        if (it >= sets) {
            gemm_ns += ev_ns(e);
            pre_ns += ev_ns(eq);
            host_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ++timed;
        }
        clReleaseEvent(e);
        if (eq) clReleaseEvent(eq);
    }
    if (timed) {
        const double gms = gemm_ns / 1e6 / timed, pms = pre_ns / 1e6 / timed, hms = host_ms / timed;
        const double tms = gms + pms;
        // same byte model as the upcvt driver (A fp16 + B + C + scales)
        const double bytes = (double)M * K * 2 + (double)K * N / 4.0 + (double)M * N * os
                + (double)(K / kGS) * (N + M) * 2 + (ep.needs_other() ? (double)M * N * 2 : 0.0)
                + (ep.needs_bias() ? (double)N * os : 0.0);
        const double gib = 1024.0 * 1024.0 * 1024.0, flop = 2.0 * M * N * K;
        std::cout << std::fixed << std::setprecision(5)
                  << "Avg host  time: " << hms << " ms (" << flop / (hms * 1e-3) / 1e9 << " GFLOPS)\n"
                  << "Avg dev   pre : " << pms << " ms\n"
                  << "Avg dev   gemm: " << gms << " ms (" << flop / (gms * 1e-3) / 1e9 << " GFLOPS, "
                  << bytes / (gms * 1e-3) / gib << " GiB/s)\n"
                  << "Avg dev   time: " << tms << " ms (" << flop / (tms * 1e-3) / 1e9 << " GFLOPS, "
                  << bytes / (tms * 1e-3) / gib << " GiB/s)\n"
                  << "Bytes/GEMM: " << bytes / (1 << 20) << " MiB\n";
    }

    if (cfg.validate) {
        int passed = 0;
        std::vector<unsigned char> C0;
        {  // device scale_a of the last set vs host (last filled) one
            std::vector<dt16> SAh(size_sa);
            CL_CHECK(clEnqueueReadBuffer(q, dSA[sets - 1], CL_TRUE, 0, size_sa * 2, SAh.data(), 0, nullptr, nullptr));
            size_t diff = 0;
            for (int g = 0; g < K / kGS; ++g)
                for (int m = 0; m < M; ++m) {
                    const size_t i = (size_t)g * ldsa(M) + m;
                    diff += std::memcmp(&SAh[i], &SA[i], 2) != 0;
                }
            std::cout << "scale_a: " << diff << "/" << (size_t)(K / kGS) * M << " differ from host\n";
        }
        for (int s = 0; s < sets; ++s) {
            CL_CHECK(clEnqueueReadBuffer(q, dC[s], CL_TRUE, 0, size_c * os, Ch.data(), 0, nullptr, nullptr));
            const auto &g = gold[cfg.distinct_sets ? s : 0];
            const std::string label = "validation [set " + std::to_string(s) + "/" + std::to_string(sets) + "]";
            bool ok;
            if (!cfg.distinct_sets && s > 0 && std::memcmp(Ch.data(), C0.data(), size_c * os) == 0)
                ok = true;
            else if (ep.out_f32)
                ok = compare_f32((const float *)Ch.data(), (const float *)g.data(), size_c, label);
            else
                ok = compare((const dt16 *)Ch.data(), (const dt16 *)g.data(), size_c, label,
                        ep.abs_tol(), ep.ulp_tol());
            if (s == 0) C0 = Ch;
            passed += ok;
            if (!ok) std::cout << "FAILED at set " << s << "\n";
            if (!cfg.distinct_sets && s == 0 && !ok) break;
        }
        std::cout << "Validation summary: " << passed << "/" << sets << " sets PASSED\n";
    }

    for (int s = 0; s < sets; ++s) {
        clReleaseMemObject(dA[s]); clReleaseMemObject(dAq[s]); clReleaseMemObject(dSA[s]);
        clReleaseMemObject(dB[s]); clReleaseMemObject(dSB[s]); clReleaseMemObject(dC[s]);
        if (dO[s] != dummy) clReleaseMemObject(dO[s]);
        if (dBias[s] != dummy) clReleaseMemObject(dBias[s]);
    }
    clReleaseMemObject(dummy);
    clReleaseKernel(kern); clReleaseKernel(kq); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
}

int main(int argc, char **argv) {
    RunConfig cfg;
    cfg.cl_path = std::string(argv[0]);
    cfg.cl_path = cfg.cl_path.substr(0, cfg.cl_path.find_last_of('/') + 1) + "int2_int8_dpas.cl";
    auto ival = [&](int &i) { return std::atoi(argv[++i]); };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--m" && i + 1 < argc) cfg.m = ival(i);
        else if (a == "--n" && i + 1 < argc) cfg.n = ival(i);
        else if (a == "--k" && i + 1 < argc) cfg.k = ival(i);
        else if (a == "--dtype" && i + 1 < argc) {
            const std::string d = argv[++i];
            if (d != "fp16" && d != "bf16") { std::cerr << "--dtype fp16|bf16\n"; return 1; }
            dt_is_bf16() = d == "bf16";
        }
        else if (a == "--qmode" && i + 1 < argc) cfg.qmode = ival(i);
        else if (a == "--postop" && i + 1 < argc) {
            cfg.epi.postop = ival(i);
            if (cfg.epi.postop < 0 || cfg.epi.postop > 4) { std::cerr << "--postop 0..4\n"; return 1; }
        }
        else if (a == "--out-f32") cfg.epi.out_f32 = true;
        else if (a == "--iters" && i + 1 < argc) cfg.iters = ival(i);
        else if (a == "--sets" && i + 1 < argc) cfg.num_sets = ival(i);
        else if (a == "--weights-gib" && i + 1 < argc) cfg.weights_gib = std::atof(argv[++i]);
        else if (a == "--no-validate") cfg.validate = false;
        else if (a == "--distinct-sets") cfg.distinct_sets = true;
        else if (a == "--sgm" && i + 1 < argc) cfg.sgm = ival(i);
        else if (a == "--nsg" && i + 1 < argc) cfg.nsg = ival(i);
        else if (a == "--wgn" && i + 1 < argc) cfg.nsg = ival(i) / 16;
        else if (a == "--ls" && i + 1 < argc) cfg.ls = ival(i);
        else if (a == "--u" && i + 1 < argc) cfg.u = ival(i);
        else if (a == "--mt-m" && i + 1 < argc) cfg.mt_m = ival(i);
        else if (a == "--mt-n" && i + 1 < argc) cfg.mt_n = ival(i);
        else if (a == "--wg-m" && i + 1 < argc) cfg.wg_m = ival(i);
        else if (a == "--wg-n" && i + 1 < argc) cfg.wg_n = ival(i);
        else if (a == "--cl" && i + 1 < argc) cfg.cl_path = argv[++i];
        else if (a == "--opts" && i + 1 < argc) cfg.extra_opts = argv[++i];
        else if (a == "--build-log") cfg.print_build_log = true;
        else {
            std::cout << "Usage: " << argv[0]
                      << " [--m M] [--n N] [--k K] [--dtype fp16|bf16] [--qmode 0|1] [--iters N] [--no-validate]\n"
                         "       [--distinct-sets] [--sets N | --weights-gib G]\n"
                         "       [--sgm 1|2|4|8] [--wgn W | --nsg S] [--ls L] [--u U]\n"
                         "       [--mt-m 8k --mt-n 16k --wg-m W --wg-n W]\n"
                         "       [--cl file.cl] [--opts \"-D...\"] [--build-log]\n"
                         "       [--postop 0|1|2|3|4] [--out-f32]   epilogue: 0 none, 1 silu(acc)*other,\n"
                         "       2 acc+other, 3 acc+bias[n], 4 sigmoid(acc); --out-f32 = fp32 C/bias\n"
                         "qmode: 0 upfront int8 A, 1 xetla scheme (quantize A in the GEMM)\n";
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }
    run(cfg);
    return 0;
}
