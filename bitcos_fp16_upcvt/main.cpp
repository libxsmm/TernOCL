// Standalone OpenCL driver for bitcos_fp16_upcvt.cl, same methodology as the
// int2 drivers: random inputs (ternary weights at zero density --z, packed as
// xetla_vllm_plugin.pack_bitcos does), rotation over enough distinct device
// buffer sets that the distinct weights (BITCOS buffer + scales) reach
// --weights-gib (default 2 GiB, same rule in xetla_ref), device-event timing,
// fp32 host gold (fp32 accumulate per 128-group, x scale) and the
// xetla_buff_cmp pass rule. Tiles are -D options, so a sweep needs no rebuild.

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <omp.h>

#include <algorithm>
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

#include "bitcos.hpp"
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
    double z = 0.40;
    bool validate = true, distinct_sets = false, int_apply = false, simt_mul = false;
    int num_sets = 0;
    double weights_gib = 2.0;
    int sgm = 0, nsg = 0, ls = 0;  // 0 = default
    int mt_m = 0, mt_n = 16, wg_m = 1, wg_n = 4;  // mt_m > 0: bitcos_fp16_upcvt_gemm_mt
    std::string cl_path, extra_opts;
    bool print_build_log = false;
    Epilogue epi;
};

static float rnd_f(float lo, float hi) {
    static thread_local std::mt19937 gen(std::random_device{}());
    return std::uniform_real_distribution<float>(lo, hi)(gen);
}

// fp32 accumulator gold from the decoded codes; epilogue_ref() follows
static void compute_gold(const dt16 *A, const Bitcos &b, const dt16 *S, float *C, int M) {
    const int K = b.K, N = b.N;
#pragma omp parallel
    {
        std::vector<int8_t> codes(K);
#pragma omp for schedule(dynamic, 16)
        for (int j = 0; j < N; ++j) {
            bitcos_decode_col(b, j, codes.data());
            for (int i = 0; i < M; ++i) {
                float acc = 0.0f;
                for (int g = 0; g < K / kGS; ++g) {
                    float partial = 0.0f;
                    for (int p = 0; p < kGS; ++p) {
                        const int k = g * kGS + p;
                        if (codes[k]) partial += tof(A[(size_t)i * K + k]) * (float)codes[k];
                    }
                    acc += partial * tof(S[(size_t)g * N + j]);
                }
                C[(size_t)i * N + j] = acc;
            }
        }
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
    if (c.ls == 0) c.ls = 4;
    while (c.ls > 1 && c.k % (64 * c.ls)) c.ls /= 2;
}

static void run(RunConfig cfg) {
    const int M = cfg.m, N = cfg.n, K = cfg.k;
    default_tiles(cfg);
    if (K % kGS || N % 16 || K % (64 * cfg.ls)) {
        std::cerr << "need K % 128 == 0, N % 16 == 0 and K % (64 * LS) == 0\n";
        std::exit(1);
    }

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

    std::cout << "Device: " << name << "\n"
              << "Problem: M=" << M << " N=" << N << " K=" << K << " scale_gs=" << kGS
              << " z=" << cfg.z << " dtype=" << dt_name() << " epilogue=" << cfg.epi.name() << "\n"
              << "Tile: sg_m=" << cfg.sgm << " wg_n=" << 16 * cfg.nsg << " ls=" << cfg.ls
              << " apply=" << (cfg.int_apply || dt_is_bf16() ? "int-and" : cfg.simt_mul ? "simt-hmul" : "visa-hmul") << "\n";

    std::string src = read_file(cfg.cl_path);
    const char *srcp = src.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &srcp, nullptr, &err);
    CL_CHECK(err);
    std::string opts = std::string("-cl-std=CL3.0") + (dt_is_bf16() ? " -DBF16" : "")
            + (cfg.int_apply ? " -DINT_APPLY" : "") + (cfg.simt_mul ? " -DSIMT_HMUL" : "") + " -DSGM=" + std::to_string(cfg.sgm)
            + " -DNSG_N=" + std::to_string(cfg.nsg) + " -DLS=" + std::to_string(cfg.ls)
            + (cfg.mt_m ? " -DMT_M=" + std::to_string(cfg.mt_m) + " -DMT_N=" + std::to_string(cfg.mt_n) + " -DWG_M="
                          + std::to_string(cfg.wg_m) + " -DWG_N=" + std::to_string(cfg.wg_n) + " -cl-intel-256-GRF-per-thread"
                        : std::string())
            + cfg.epi.opts() + " -I " + cfg.cl_path.substr(0, cfg.cl_path.find_last_of('/') + 1)
            + ". " + cfg.extra_opts;
    err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS || cfg.print_build_log) {
        size_t len = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        std::cout << "Build options: " << opts << "\nBuild log:\n" << log << "\n";
        CL_CHECK(err);
    }
    cl_kernel kern = clCreateKernel(prog, cfg.mt_m ? "bitcos_fp16_upcvt_gemm_mt" : "bitcos_fp16_upcvt_gemv", &err);
    CL_CHECK(err);

    const Epilogue &ep = cfg.epi;
    const size_t os = ep.out_size();
    const size_t size_a = (size_t)M * K, size_c = (size_t)M * N, size_s = (size_t)(K / kGS) * N;
    static std::mt19937 egen(std::random_device{}());
    std::vector<dt16> A(size_a), S(size_s), Oth;
    std::vector<unsigned char> Bias, Ch(size_c * os);
    Bitcos B;
    std::vector<uint32_t> SR;
    uint64_t seed = std::random_device{}();
    auto fill = [&] {
#pragma omp parallel for
        for (size_t i = 0; i < size_a; ++i) A[i] = fromf(rnd_f(-5.0f, 5.0f));
        for (size_t i = 0; i < size_s; ++i) S[i] = fromf(rnd_f(0.0f, 15.0f) + 0.75f);
        B = bitcos_random(K, N, cfg.z, seed++);
        SR = bitcos_slice_ranks(B, cfg.ls);
        if (SR.empty()) SR.assign(1, 0);
        fill_epilogue_inputs(ep, Oth, Bias, M, N, egen);
    };
    fill();
    const double wbytes_set = (double)B.bytes() + size_s * 2.0;
    const int sets = cfg.num_sets > 0 ? cfg.num_sets
                                      : std::max(1, (int)std::ceil(cfg.weights_gib * (1 << 30) / wbytes_set));
    std::cout << "BITCOS: measured z " << std::setprecision(4) << bitcos_zero_density(B) << ", "
              << 8.0 * B.bytes() / ((double)K * N) << " bits/weight (+scales); weights "
              << wbytes_set / (1 << 20) << " MiB/set, " << sets << " distinct sets -> " << std::fixed
              << std::setprecision(3) << wbytes_set * sets / (1 << 30) << " GiB\n";

    const int ngold = (cfg.validate && cfg.distinct_sets) ? sets : 1;
    std::vector<std::vector<unsigned char>> gold(cfg.validate ? ngold : 0);
    std::vector<float> acc_gold;
    auto make_gold = [&](std::vector<unsigned char> &g) {
        acc_gold.resize(size_c);
        compute_gold(A.data(), B, S.data(), acc_gold.data(), M);
        g.resize(size_c * os);
        epilogue_ref(ep, acc_gold.data(), Oth.data(), Bias.data(), g.data(), M, N);
    };
    if (cfg.validate) make_gold(gold[0]);

    cl_mem dummy = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 64, nullptr, &err);
    CL_CHECK(err);
    std::vector<cl_mem> dA(sets), dB(sets), dS(sets), dSR(sets), dC(sets), dO(sets, dummy), dBias(sets, dummy);
    std::vector<size_t> bbytes(sets);
    for (int s = 0; s < sets; ++s) {
        if (cfg.distinct_sets && s > 0) {
            fill();
            if (cfg.validate) make_gold(gold[s]);
        }
        bbytes[s] = B.bytes();
        dA[s] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_a * 2, A.data(), &err);
        CL_CHECK(err);
        dB[s] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B.bytes(), B.buf.data(), &err);
        CL_CHECK(err);
        dS[s] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_s * 2, S.data(), &err);
        CL_CHECK(err);
        dSR[s] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SR.size() * 4, SR.data(), &err);
        CL_CHECK(err);
        dC[s] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_c * os, nullptr, &err);
        CL_CHECK(err);
        if (ep.needs_other()) {
            dO[s] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_c * 2, Oth.data(), &err);
            CL_CHECK(err);
        }
        if (ep.needs_bias()) {
            dBias[s] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Bias.size(), Bias.data(), &err);
            CL_CHECK(err);
        }
        const cl_uchar zero = 0;
        CL_CHECK(clEnqueueFillBuffer(q, dC[s], &zero, 1, 0, size_c * os, 0, nullptr, nullptr));
    }
    CL_CHECK(clFinish(q));

    const size_t wg_n = 16 * cfg.nsg;
    size_t local[2] = {wg_n * cfg.ls, 1};
    size_t global[2] = {((N + wg_n - 1) / wg_n) * local[0], (size_t)((M + cfg.sgm - 1) / cfg.sgm)};
    if (cfg.mt_m) {
        const size_t tn = (size_t)cfg.mt_n * cfg.wg_n, tm = (size_t)cfg.mt_m * cfg.wg_m;
        local[0] = 16 * (size_t)cfg.wg_n * cfg.wg_m;
        global[0] = ((N + tn - 1) / tn) * local[0];
        global[1] = (M + tm - 1) / tm;
    }

    double dev_ns = 0.0, host_ms = 0.0, bsum = 0.0;
    int timed = 0;
    for (int it = 0; it < sets + cfg.iters; ++it) {
        const int s = it % sets;
        const cl_mem args[7] = {dA[s], dB[s], dS[s], dSR[s], dC[s], dO[s], dBias[s]};
        for (int a = 0; a < 7; ++a) CL_CHECK(clSetKernelArg(kern, a, sizeof(cl_mem), &args[a]));
        CL_CHECK(clSetKernelArg(kern, 7, sizeof(int), &M));
        CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &K));
        cl_event e;
        auto t0 = std::chrono::high_resolution_clock::now();
        CL_CHECK(clEnqueueNDRangeKernel(q, kern, 2, nullptr, global, local, 0, nullptr, &e));
        CL_CHECK(clWaitForEvents(1, &e));
        auto t1 = std::chrono::high_resolution_clock::now();
        if (it >= sets) {
            cl_ulong st, en;
            CL_CHECK(clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof st, &st, nullptr));
            CL_CHECK(clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof en, &en, nullptr));
            dev_ns += (double)(en - st);
            host_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            bsum += (double)bbytes[s];
            ++timed;
        }
        clReleaseEvent(e);
    }
    if (timed) {
        const double dms = dev_ns / 1e6 / timed, hms = host_ms / timed;
        // actual bytes: A + BITCOS buffer + scales + C (+ epilogue operands)
        const double bytes = (double)M * K * 2 + bsum / timed + (double)(K / kGS) * N * 2
                + (double)M * N * os + (ep.needs_other() ? (double)M * N * 2 : 0.0)
                + (ep.needs_bias() ? (double)N * os : 0.0);
        // int2-equivalent bytes (2 bits/weight), the throughput at the same weight count
        const double bytes_i2 = bytes - bsum / timed + (double)K * N / 4.0;
        const double gib = 1024.0 * 1024.0 * 1024.0, flop = 2.0 * M * N * K;
        std::cout << std::fixed << std::setprecision(5)
                  << "Avg host  time: " << hms << " ms (" << flop / (hms * 1e-3) / 1e9 << " GFLOPS)\n"
                  << "Avg dev   time: " << dms << " ms (" << flop / (dms * 1e-3) / 1e9
                  << " GFLOPS, " << bytes / (dms * 1e-3) / gib << " GiB/s, int2-equiv "
                  << bytes_i2 / (dms * 1e-3) / gib << " GiB/s)\n"
                  << "Bytes/GEMM: " << bytes / (1 << 20) << " MiB\n";
    }

    if (cfg.validate) {
        int passed = 0;
        std::vector<unsigned char> C0;
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
        clReleaseMemObject(dA[s]); clReleaseMemObject(dB[s]); clReleaseMemObject(dS[s]);
        clReleaseMemObject(dSR[s]); clReleaseMemObject(dC[s]);
        if (dO[s] != dummy) clReleaseMemObject(dO[s]);
        if (dBias[s] != dummy) clReleaseMemObject(dBias[s]);
    }
    clReleaseMemObject(dummy);
    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
}

int main(int argc, char **argv) {
    RunConfig cfg;
    cfg.cl_path = std::string(argv[0]);
    cfg.cl_path = cfg.cl_path.substr(0, cfg.cl_path.find_last_of('/') + 1) + "bitcos_fp16_upcvt.cl";
    auto ival = [&](int &i) { return std::atoi(argv[++i]); };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--m" && i + 1 < argc) cfg.m = ival(i);
        else if (a == "--n" && i + 1 < argc) cfg.n = ival(i);
        else if (a == "--k" && i + 1 < argc) cfg.k = ival(i);
        else if (a == "--z" && i + 1 < argc) cfg.z = std::atof(argv[++i]);
        else if (a == "--dtype" && i + 1 < argc) {
            const std::string d = argv[++i];
            if (d != "fp16" && d != "bf16") { std::cerr << "--dtype fp16|bf16\n"; return 1; }
            dt_is_bf16() = d == "bf16";
        }
        else if (a == "--iters" && i + 1 < argc) cfg.iters = ival(i);
        else if (a == "--sets" && i + 1 < argc) cfg.num_sets = ival(i);
        else if (a == "--weights-gib" && i + 1 < argc) cfg.weights_gib = std::atof(argv[++i]);
        else if (a == "--no-validate") cfg.validate = false;
        else if (a == "--distinct-sets") cfg.distinct_sets = true;
        else if (a == "--int-apply") cfg.int_apply = true;
        else if (a == "--simt-mul") cfg.simt_mul = true;
        else if (a == "--sgm" && i + 1 < argc) cfg.sgm = ival(i);
        else if (a == "--nsg" && i + 1 < argc) cfg.nsg = ival(i);
        else if (a == "--wgn" && i + 1 < argc) cfg.nsg = ival(i) / 16;
        else if (a == "--ls" && i + 1 < argc) cfg.ls = ival(i);
        else if (a == "--mt-m" && i + 1 < argc) cfg.mt_m = ival(i);
        else if (a == "--mt-n" && i + 1 < argc) cfg.mt_n = ival(i);
        else if (a == "--wg-m" && i + 1 < argc) cfg.wg_m = ival(i);
        else if (a == "--wg-n" && i + 1 < argc) cfg.wg_n = ival(i);
        else if (a == "--cl" && i + 1 < argc) cfg.cl_path = argv[++i];
        else if (a == "--opts" && i + 1 < argc) cfg.extra_opts = argv[++i];
        else if (a == "--postop" && i + 1 < argc) {
            cfg.epi.postop = ival(i);
            if (cfg.epi.postop < 0 || cfg.epi.postop > 4) { std::cerr << "--postop 0..4\n"; return 1; }
        }
        else if (a == "--out-f32") cfg.epi.out_f32 = true;
        else if (a == "--build-log") cfg.print_build_log = true;
        else {
            std::cout << "Usage: " << argv[0]
                      << " [--m M] [--n N] [--k K] [--z Z] [--dtype fp16|bf16] [--iters N] [--no-validate]\n"
                         "       [--distinct-sets] [--sets N | --weights-gib G] [--sgm 1|2|4|8] [--wgn W | --nsg S]\n"
                         "       [--ls L] [--int-apply | --simt-mul] [--cl file.cl] [--opts \"-D...\"] [--build-log]\n"
                         "       [--postop 0|1|2|3|4] [--out-f32]\n";
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }
    run(cfg);
    return 0;
}
