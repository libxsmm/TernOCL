// Driver for the MoE prefill path (int2_moe_prefill.cl): T tokens routed to TOPK of E
// random ternary experts, a host reference on the first --check tokens, and device-event
// timing of route / gather / up / gate / down / combine over SETS distinct layers.
//
//   make && ./build/int2_moe_prefill_ocl --t 1024 --tile 32,32,2,4
#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "dt16.hpp"

#define CL_CHECK(x)                                                                         \
    do {                                                                                    \
        cl_int e_ = (x);                                                                    \
        if (e_ != CL_SUCCESS) {                                                             \
            std::cerr << #x << " failed: " << e_ << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

constexpr int kGS = 128;

struct Tile {
    int mt_m = 32, mt_n = 32, wg_m = 2, wg_n = 4;
    std::string opts() const {
        return " -DMT_M=" + std::to_string(mt_m) + " -DMT_N=" + std::to_string(mt_n) + " -DWG_M=" + std::to_string(wg_m) +
               " -DWG_N=" + std::to_string(wg_n) + " -cl-intel-256-GRF-per-thread";
    }
};

static std::string read_file(const std::string &p) {
    std::ifstream f(p);
    if (!f) {
        std::cerr << "cannot read " << p << "\n";
        std::exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static int code_to_value(uint32_t c) { return (int)(int8_t)((int8_t)(c << 6) >> 6); }

struct Proj {
    int E, K, N;
    std::vector<uint32_t> B;
    std::vector<dt16> S;
    cl_mem dB = nullptr, dS = nullptr;
    size_t b_words() const { return (size_t)E * (K / 16) * N; }
    size_t s_elems() const { return (size_t)E * (K / kGS) * N; }
    float w(int e, int k, int n) const {
        const uint32_t word = B[((size_t)e * (K / 16) + k / 16) * N + n];
        return code_to_value((word >> (2 * (k % 16))) & 3u) * tof(S[((size_t)e * (K / kGS) + k / kGS) * N + n]);
    }
};

static void fill_proj(Proj &p, std::mt19937 &rng) {
    p.B.resize(p.b_words());
    p.S.resize(p.s_elems());
    std::uniform_int_distribution<uint32_t> u32;
    std::uniform_real_distribution<float> us(0.005f, 0.05f);
    for (auto &w : p.B) {
        const uint32_t r1 = u32(rng), r2 = u32(rng);
        w = (r1 & 0x55555555u) | (((r1 & r2) & 0x55555555u) << 1);
    }
    for (auto &s : p.S) s = fromf(us(rng));
}

int main(int argc, char **argv) {
    int T = 1024, E = 128, TOPK = 8, H = 2048, I = 768, sets = 8, iters = 10, check = 4;
    Tile tile;
    std::string dir = ".";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if (a == "--t") T = std::stoi(next());
        else if (a == "--e") E = std::stoi(next());
        else if (a == "--topk") TOPK = std::stoi(next());
        else if (a == "--h") H = std::stoi(next());
        else if (a == "--i") I = std::stoi(next());
        else if (a == "--sets") sets = std::stoi(next());
        else if (a == "--iters") iters = std::stoi(next());
        else if (a == "--check") check = std::stoi(next());
        else if (a == "--tile") {
            char c;
            std::istringstream in(next());
            in >> tile.mt_m >> c >> tile.mt_n >> c >> tile.wg_m >> c >> tile.wg_n;
        } else if (a == "--dir") dir = next();
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--t T] [--e E] [--topk K] [--h H] [--i I] [--sets S] [--iters N] [--check C]"
                         " [--tile mt_m,mt_n,wg_m,wg_n] [--dir kernel_dir]\n";
            return 1;
        }
    }
    const int G = T * TOPK, RPT = tile.mt_m * tile.wg_m;

    cl_platform_id plats[8];
    cl_uint np = 0;
    CL_CHECK(clGetPlatformIDs(8, plats, &np));
    cl_device_id dev = nullptr;
    for (cl_uint i = 0; i < np && !dev; ++i)
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) != CL_SUCCESS) dev = nullptr;
    if (!dev) {
        std::cerr << "no OpenCL GPU\n";
        return 1;
    }
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err);
    CL_CHECK(err);
    std::cout << "Prefill: T=" << T << " E=" << E << " topk=" << TOPK << " H=" << H << " I=" << I << " tile " << tile.mt_m
              << "x" << tile.mt_n << " wg " << tile.wg_m << "x" << tile.wg_n << "\n";

    const std::string src = read_file(dir + "/int2_moe_prefill.cl");
    auto build = [&](const std::string &extra) {
        const char *p = src.c_str();
        cl_program prog = clCreateProgramWithSource(ctx, 1, &p, nullptr, &err);
        CL_CHECK(err);
        const std::string opts = "-cl-std=CL3.0 -I " + dir + " -DTOPK=" + std::to_string(TOPK) + " -DNUM_EXPERTS=" +
                                 std::to_string(E) + tile.opts() + extra;
        err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t len = 0;
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            std::string log(len, '\0');
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
            std::cout << "Build options: " << opts << "\n" << log << "\n";
            CL_CHECK(err);
        }
        return prog;
    };
    cl_program p_up = build(""), p_gate = build(" -DPOSTOP=1"), p_down = build(" -DOUT_F32");
    auto kern = [&](cl_program p, const char *n) {
        cl_kernel k = clCreateKernel(p, n, &err);
        CL_CHECK(err);
        return k;
    };
    cl_kernel k_route = kern(p_up, "int2_moe_route"), k_gather = kern(p_up, "int2_moe_gather"),
              k_up = kern(p_up, "int2_moe_gemm_mt"), k_gate = kern(p_gate, "int2_moe_gemm_mt"),
              k_down = kern(p_down, "int2_moe_gemm_mt"), k_comb = kern(p_down, "int2_moe_combine");

    std::mt19937 rng(42);
    std::vector<Proj> gate(sets, Proj{E, H, I}), up(sets, Proj{E, H, I}), down(sets, Proj{E, I, H});
    auto upload = [&](Proj &p, const Proj &s) {
        if (&p == &s) fill_proj(p, rng);
        p.dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, p.b_words() * 4, (void *)s.B.data(), &err);
        CL_CHECK(err);
        p.dS = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, p.s_elems() * 2, (void *)s.S.data(), &err);
        CL_CHECK(err);
    };
    for (int l = 0; l < sets; ++l) {
        upload(gate[l], gate[0]);
        upload(up[l], up[0]);
        upload(down[l], down[0]);
    }

    std::vector<dt16> X((size_t)T * H), RW(G);
    std::vector<int> ids(G);
    std::uniform_real_distribution<float> ux(-1.0f, 1.0f), ur(0.05f, 1.0f);
    for (auto &x : X) x = fromf(ux(rng));
    // skewed routing, as real models: expert e drawn with weight 1 / (1 + e / 8)
    std::vector<double> pw(E);
    for (int e = 0; e < E; ++e) pw[e] = 1.0 / (1.0 + e / 8.0);
    for (int t = 0; t < T; ++t) {
        std::vector<double> w = pw;
        float sum = 0.0f;
        std::vector<float> rw(TOPK);
        for (int j = 0; j < TOPK; ++j) {
            std::discrete_distribution<int> d(w.begin(), w.end());
            const int e = d(rng);
            w[e] = 0.0;
            ids[t * TOPK + j] = e;
            sum += (rw[j] = ur(rng));
        }
        for (int j = 0; j < TOPK; ++j) RW[t * TOPK + j] = fromf(rw[j] / sum);
    }
    auto buf = [&](size_t bytes, const void *host = nullptr) {
        cl_mem m = clCreateBuffer(ctx, host ? CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR : CL_MEM_READ_WRITE, bytes,
                                  (void *)host, &err);
        CL_CHECK(err);
        return m;
    };
    cl_mem dX = buf(X.size() * 2, X.data()), dRW = buf(RW.size() * 2, RW.data()), dIds = buf(ids.size() * 4, ids.data());
    cl_mem dMeta = buf((2 * E + 2) * 4), dPerm = buf(G * 4), dInv = buf(G * 4);
    cl_mem dXs = buf((size_t)G * H * 2), dU = buf((size_t)G * I * 2), dH = buf((size_t)G * I * 2),
           dY = buf((size_t)G * H * 4), dO = buf((size_t)T * H * 2);

    const size_t max_tiles = (size_t)(G + RPT - 1) / RPT + E;
    const size_t wg = 16 * tile.wg_m * tile.wg_n;
    auto gemm_global = [&](int N) {
        const int tn = tile.mt_n * tile.wg_n;
        return std::array<size_t, 2>{(size_t)((N + tn - 1) / tn) * wg, max_tiles};
    };
    auto set = [&](cl_kernel k, int i, auto v) { CL_CHECK(clSetKernelArg(k, i, sizeof(v), &v)); };
    auto enq = [&](cl_kernel k, std::array<size_t, 2> g, std::array<size_t, 2> l, cl_event *ev) {
        CL_CHECK(clEnqueueNDRangeKernel(q, k, 2, nullptr, g.data(), l.data(), 0, nullptr, ev));
    };
    auto run_layer = [&](int l, cl_event *ev) {
        set(k_route, 0, dIds), set(k_route, 1, dMeta), set(k_route, 2, dPerm), set(k_route, 3, dInv), set(k_route, 4, G);
        enq(k_route, {1024, 1}, {1024, 1}, ev ? &ev[0] : nullptr);
        set(k_gather, 0, dX), set(k_gather, 1, dPerm), set(k_gather, 2, dXs), set(k_gather, 3, H);
        enq(k_gather, {256, (size_t)G}, {256, 1}, ev ? &ev[1] : nullptr);
        cl_mem none = dXs;
        // up, then gate with silu(gate) * up
        set(k_up, 0, dMeta), set(k_up, 1, dXs), set(k_up, 2, up[l].dB), set(k_up, 3, up[l].dS), set(k_up, 4, dU),
            set(k_up, 5, none), set(k_up, 6, I), set(k_up, 7, H);
        enq(k_up, gemm_global(I), {wg, 1}, ev ? &ev[2] : nullptr);
        set(k_gate, 0, dMeta), set(k_gate, 1, dXs), set(k_gate, 2, gate[l].dB), set(k_gate, 3, gate[l].dS),
            set(k_gate, 4, dH), set(k_gate, 5, dU), set(k_gate, 6, I), set(k_gate, 7, H);
        enq(k_gate, gemm_global(I), {wg, 1}, ev ? &ev[3] : nullptr);
        set(k_down, 0, dMeta), set(k_down, 1, dH), set(k_down, 2, down[l].dB), set(k_down, 3, down[l].dS),
            set(k_down, 4, dY), set(k_down, 5, none), set(k_down, 6, H), set(k_down, 7, I);
        enq(k_down, gemm_global(H), {wg, 1}, ev ? &ev[4] : nullptr);
        set(k_comb, 0, dY), set(k_comb, 1, dInv), set(k_comb, 2, dRW), set(k_comb, 3, dO), set(k_comb, 4, H);
        enq(k_comb, {(size_t)(H + 63) / 64 * 64, (size_t)T}, {64, 1}, ev ? &ev[5] : nullptr);
    };

    run_layer(0, nullptr);
    std::vector<dt16> O((size_t)T * H);
    CL_CHECK(clEnqueueReadBuffer(q, dO, CL_TRUE, 0, O.size() * 2, O.data(), 0, nullptr, nullptr));
    double max_err = 0.0, max_ref = 0.0;
    for (int t = 0; t < std::min(T, check); ++t) {
        std::vector<double> o(H, 0.0);
        for (int j = 0; j < TOPK; ++j) {
            const int g = t * TOPK + j, e = ids[g];
            std::vector<float> h(I);
            for (int n = 0; n < I; ++n) {
                double sg = 0.0, su = 0.0;
                for (int k = 0; k < H; ++k) {
                    const double x = tof(X[(size_t)t * H + k]);
                    sg += x * gate[0].w(e, k, n);
                    su += x * up[0].w(e, k, n);
                }
                h[n] = tof(fromf((float)(sg / (1.0 + std::exp(-sg)) * tof(fromf((float)su)))));
            }
            for (int n = 0; n < H; ++n) {
                double s = 0.0;
                for (int k = 0; k < I; ++k) s += h[k] * down[0].w(e, k, n);
                o[n] += s * tof(RW[g]);
            }
        }
        for (int n = 0; n < H; ++n) {
            max_ref = std::max(max_ref, std::fabs(o[n]));
            max_err = std::max(max_err, std::fabs(o[n] - tof(O[(size_t)t * H + n])));
        }
    }
    const double rel = max_err / std::max(max_ref, 1e-30);
    std::cout << "check (" << std::min(T, check) << " tokens): max_abs_err " << std::scientific << std::setprecision(3)
              << max_err << " (max |ref| " << max_ref << ", rel " << rel << ") " << (rel < 1e-2 ? "PASS" : "FAIL")
              << std::fixed << "\n";

    for (int w = 0; w < 10; ++w)
        for (int l = 0; l < sets; ++l) run_layer(l, nullptr);
    CL_CHECK(clFinish(q));
    const char *names[6] = {"route", "gather", "up", "gate", "down", "combine"};
    std::vector<std::vector<double>> t(6);
    for (int it = 0; it < iters; ++it)
        for (int l = 0; l < sets; ++l) {
            cl_event ev[6];
            run_layer(l, ev);
            CL_CHECK(clWaitForEvents(6, ev));
            for (int k = 0; k < 6; ++k) {
                cl_ulong s0, s1;
                clGetEventProfilingInfo(ev[k], CL_PROFILING_COMMAND_START, sizeof s0, &s0, nullptr);
                clGetEventProfilingInfo(ev[k], CL_PROFILING_COMMAND_END, sizeof s1, &s1, nullptr);
                t[k].push_back((s1 - s0) * 1e-3);
                clReleaseEvent(ev[k]);
            }
        }
    double total = 0.0;
    std::cout << std::setprecision(1);
    for (int k = 0; k < 6; ++k) {
        std::sort(t[k].begin(), t[k].end());
        const double m = t[k][t[k].size() / 2];
        total += m;
        std::cout << names[k] << " " << m << " us  ";
    }
    const double flops = 2.0 * G * (2.0 * H * I + (double)I * H);
    std::cout << "| layer " << total << " us, " << flops / (total * 1e-6) / 1e12 << " TFLOPS, x48 = " << 48 * total / 1e3
              << " ms\n";
    return 0;
}
