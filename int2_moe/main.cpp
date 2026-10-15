// Driver for the MoE expert GEMVs (int2_moe.cl): random ternary experts, a host
// reference on one layer, and device-event timing over SETS distinct layers of
// weights (as in a model, nothing is reused from cache between layers).
//
//   source /swtools/intel/2026.0/oneapi-vars.sh --force
//   make && ./build/int2_moe_ocl --t 1 --e 128 --topk 8 --h 2048 --i 768
#include <CL/cl.h>

#include <algorithm>
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
    int nsg = 2, ls = 4, u = 2;
    std::string opts() const {
        return " -DNSG_N=" + std::to_string(nsg) + " -DLS=" + std::to_string(ls) + " -DU=" + std::to_string(u);
    }
    std::string str() const { return "nsg=" + std::to_string(nsg) + " ls=" + std::to_string(ls) + " u=" + std::to_string(u); }
};

static Tile parse_tile(const std::string &s) {
    Tile t;
    char c;
    std::istringstream in(s);
    in >> t.nsg >> c >> t.ls >> c >> t.u;
    return t;
}

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

// one projection of all experts: B [E, K/16, N], S [E, K/128, N]
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
        const uint32_t lo = r1 & 0x55555555u, hi = (r1 & r2) & 0x55555555u;
        w = lo | (hi << 1);  // codes {0, 1, 3}
    }
    for (auto &s : p.S) s = fromf(us(rng));
}

int main(int argc, char **argv) {
    int T = 1, E = 128, TOPK = 8, H = 2048, I = 768, sets = 48, iters = 20;
    bool fused = false, distinct = false;
    Tile gu, dn;
    dn.ls = 6;
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
        else if (a == "--gu") gu = parse_tile(next());
        else if (a == "--dn") dn = parse_tile(next());
        else if (a == "--dir") dir = next();
        else if (a == "--fused") fused = true;
        else if (a == "--distinct") distinct = true;
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--t T] [--e E] [--topk K] [--h H] [--i I] [--sets S] [--iters N]"
                         " [--gu nsg,ls,u] [--dn nsg,ls,u] [--dir kernel_dir] [--fused]\n";
            return 1;
        }
    }
    const int G = T * TOPK;

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
    char name[256];
    CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, nullptr));
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err);
    CL_CHECK(err);
    std::cout << "Device: " << name << "\nProblem: T=" << T << " E=" << E << " topk=" << TOPK << " H=" << H
              << " I=" << I << " sets=" << sets << "\nTiles: gate_up " << gu.str() << ", down " << dn.str() << "\n";

    const std::string src = read_file(dir + "/int2_moe.cl");
    auto build = [&](const Tile &t) {
        const char *p = src.c_str();
        cl_program prog = clCreateProgramWithSource(ctx, 1, &p, nullptr, &err);
        CL_CHECK(err);
        const std::string opts = "-cl-std=CL3.0 -I " + dir + " -DTOPK=" + std::to_string(TOPK) + t.opts();
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
    cl_program pgu = build(gu), pdn = build(dn);
    cl_kernel k_gu = clCreateKernel(pgu, "int2_moe_gate_up", &err);
    CL_CHECK(err);
    cl_kernel k_dn = clCreateKernel(pdn, "int2_moe_down", &err);
    CL_CHECK(err);
    cl_kernel k_rd = clCreateKernel(pdn, "int2_moe_reduce", &err);
    CL_CHECK(err);
    cl_kernel k_ds = clCreateKernel(pdn, "int2_moe_down_sum", &err);
    CL_CHECK(err);

    // layers of weights: random data for layer 0 (kept for the reference), the
    // other layers upload the same values into their own buffers
    std::mt19937 rng(42);
    std::vector<Proj> gate(sets, Proj{E, H, I}), up(sets, Proj{E, H, I}), down(sets, Proj{E, I, H});
    auto upload = [&](Proj &p, const Proj &src) {
        if (&p == &src || distinct) fill_proj(p, rng);
        const Proj &d = distinct ? p : src;
        p.dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, p.b_words() * 4, (void *)d.B.data(), &err);
        CL_CHECK(err);
        p.dS = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, p.s_elems() * 2, (void *)d.S.data(), &err);
        CL_CHECK(err);
        if (distinct && &p != &src) {
            std::vector<uint32_t>().swap(p.B);
            std::vector<dt16>().swap(p.S);
        }
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
    for (int t = 0; t < T; ++t) {
        std::vector<int> perm(E);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        float sum = 0.0f;
        std::vector<float> w(TOPK);
        for (int j = 0; j < TOPK; ++j) sum += (w[j] = ur(rng));
        for (int j = 0; j < TOPK; ++j) {
            ids[t * TOPK + j] = perm[j];
            RW[t * TOPK + j] = fromf(w[j] / sum);
        }
    }
    cl_mem dX = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, X.size() * 2, X.data(), &err);
    cl_mem dRW = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, RW.size() * 2, RW.data(), &err);
    cl_mem dIds = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ids.size() * 4, ids.data(), &err);
    cl_mem dH = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)G * I * 2, nullptr, &err);
    cl_mem dY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)G * H * 4, nullptr, &err);
    cl_mem dO = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)T * H * 2, nullptr, &err);
    CL_CHECK(err);

    const size_t gu_global[2] = {(size_t)((I + 16 * gu.nsg - 1) / (16 * gu.nsg)) * 16 * gu.nsg * gu.ls, (size_t)G};
    const size_t gu_local[2] = {(size_t)16 * gu.nsg * gu.ls, 1};
    const size_t dn_global[2] = {(size_t)((H + 16 * dn.nsg - 1) / (16 * dn.nsg)) * 16 * dn.nsg * dn.ls, (size_t)G};
    const size_t dn_local[2] = {(size_t)16 * dn.nsg * dn.ls, 1};
    const size_t rd_global[2] = {(size_t)(H + 63) / 64 * 64, (size_t)T};
    const size_t rd_local[2] = {64, 1};
    const size_t ds_global[2] = {dn_global[0], (size_t)T};
    const int n_ev = fused ? 2 : 3;

    auto run_layer = [&](int l, cl_event *ev) {
        CL_CHECK(clSetKernelArg(k_gu, 0, sizeof(cl_mem), &dIds));
        CL_CHECK(clSetKernelArg(k_gu, 1, sizeof(cl_mem), &dX));
        CL_CHECK(clSetKernelArg(k_gu, 2, sizeof(cl_mem), &gate[l].dB));
        CL_CHECK(clSetKernelArg(k_gu, 3, sizeof(cl_mem), &gate[l].dS));
        CL_CHECK(clSetKernelArg(k_gu, 4, sizeof(cl_mem), &up[l].dB));
        CL_CHECK(clSetKernelArg(k_gu, 5, sizeof(cl_mem), &up[l].dS));
        CL_CHECK(clSetKernelArg(k_gu, 6, sizeof(cl_mem), &dH));
        CL_CHECK(clSetKernelArg(k_gu, 7, sizeof(int), &I));
        CL_CHECK(clSetKernelArg(k_gu, 8, sizeof(int), &H));
        CL_CHECK(clEnqueueNDRangeKernel(q, k_gu, 2, nullptr, gu_global, gu_local, 0, nullptr, ev ? &ev[0] : nullptr));
        if (fused) {
            CL_CHECK(clSetKernelArg(k_ds, 0, sizeof(cl_mem), &dIds));
            CL_CHECK(clSetKernelArg(k_ds, 1, sizeof(cl_mem), &dH));
            CL_CHECK(clSetKernelArg(k_ds, 2, sizeof(cl_mem), &dRW));
            CL_CHECK(clSetKernelArg(k_ds, 3, sizeof(cl_mem), &down[l].dB));
            CL_CHECK(clSetKernelArg(k_ds, 4, sizeof(cl_mem), &down[l].dS));
            CL_CHECK(clSetKernelArg(k_ds, 5, sizeof(cl_mem), &dO));
            CL_CHECK(clSetKernelArg(k_ds, 6, sizeof(int), &H));
            CL_CHECK(clSetKernelArg(k_ds, 7, sizeof(int), &I));
            CL_CHECK(clEnqueueNDRangeKernel(q, k_ds, 2, nullptr, ds_global, dn_local, 0, nullptr, ev ? &ev[1] : nullptr));
            return;
        }
        CL_CHECK(clSetKernelArg(k_dn, 0, sizeof(cl_mem), &dIds));
        CL_CHECK(clSetKernelArg(k_dn, 1, sizeof(cl_mem), &dH));
        CL_CHECK(clSetKernelArg(k_dn, 2, sizeof(cl_mem), &dRW));
        CL_CHECK(clSetKernelArg(k_dn, 3, sizeof(cl_mem), &down[l].dB));
        CL_CHECK(clSetKernelArg(k_dn, 4, sizeof(cl_mem), &down[l].dS));
        CL_CHECK(clSetKernelArg(k_dn, 5, sizeof(cl_mem), &dY));
        CL_CHECK(clSetKernelArg(k_dn, 6, sizeof(int), &H));
        CL_CHECK(clSetKernelArg(k_dn, 7, sizeof(int), &I));
        CL_CHECK(clEnqueueNDRangeKernel(q, k_dn, 2, nullptr, dn_global, dn_local, 0, nullptr, ev ? &ev[1] : nullptr));
        CL_CHECK(clSetKernelArg(k_rd, 0, sizeof(cl_mem), &dY));
        CL_CHECK(clSetKernelArg(k_rd, 1, sizeof(cl_mem), &dO));
        CL_CHECK(clSetKernelArg(k_rd, 2, sizeof(int), &H));
        CL_CHECK(clEnqueueNDRangeKernel(q, k_rd, 2, nullptr, rd_global, rd_local, 0, nullptr, ev ? &ev[2] : nullptr));
    };

    // correctness on layer 0
    run_layer(0, nullptr);
    std::vector<dt16> O((size_t)T * H);
    CL_CHECK(clEnqueueReadBuffer(q, dO, CL_TRUE, 0, O.size() * 2, O.data(), 0, nullptr, nullptr));
    double max_err = 0.0, max_ref = 0.0;
    for (int t = 0; t < T; ++t) {
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
                h[n] = tof(fromf((float)(sg / (1.0 + std::exp(-sg)) * su)));
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
    std::cout << "check: max_abs_err " << std::scientific << std::setprecision(3) << max_err << " (max |ref| "
              << max_ref << ", rel " << rel << ") " << (rel < 1e-2 ? "PASS" : "FAIL") << std::fixed << "\n";

    // timing: every iteration walks all layers once, after enough warm-up for the
    // GPU clocks to ramp (the host-side setup leaves it idle; 10 cold iterations read ~1.7x slow)
    for (int w = 0; w < 50; ++w)
        for (int l = 0; l < sets; ++l) run_layer(l, nullptr);
    CL_CHECK(clFinish(q));
    std::vector<double> t_gu, t_dn, t_rd;
    for (int it = 0; it < iters; ++it)
        for (int l = 0; l < sets; ++l) {
            cl_event ev[3];
            run_layer(l, ev);
            CL_CHECK(clWaitForEvents(n_ev, ev));
            double d[3] = {0.0, 0.0, 0.0};
            for (int k = 0; k < n_ev; ++k) {
                cl_ulong s0, s1;
                clGetEventProfilingInfo(ev[k], CL_PROFILING_COMMAND_START, sizeof s0, &s0, nullptr);
                clGetEventProfilingInfo(ev[k], CL_PROFILING_COMMAND_END, sizeof s1, &s1, nullptr);
                d[k] = (s1 - s0) * 1e-3;
                clReleaseEvent(ev[k]);
            }
            t_gu.push_back(d[0]);
            t_dn.push_back(d[1]);
            t_rd.push_back(d[2]);
        }
    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    // distinct experts touched per layer
    std::vector<int> uniq(ids);
    std::sort(uniq.begin(), uniq.end());
    const double ne = (double)(std::unique(uniq.begin(), uniq.end()) - uniq.begin());
    const double gu_bytes = ne * 2.0 * ((double)H / 16 * I * 4 + (double)H / kGS * I * 2);
    const double dn_bytes = ne * ((double)I / 16 * H * 4 + (double)I / kGS * H * 2);
    const double a = med(t_gu), b = med(t_dn), c = med(t_rd);
    const double gib = 1024.0 * 1024.0 * 1024.0;
    std::cout << std::setprecision(2) << "gate_up " << a << " us (" << gu_bytes / (a * 1e-6) / gib << " GiB/s)  down " << b
              << " us (" << dn_bytes / (b * 1e-6) / gib << " GiB/s)  reduce " << c << " us  | layer " << a + b + c
              << " us, " << (gu_bytes + dn_bytes) / ((a + b + c) * 1e-6) / gib << " GiB/s, x48 layers = "
              << 48 * (a + b + c) * 1e-3 << " ms\n";
    return 0;
}
