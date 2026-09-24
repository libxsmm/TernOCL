// Epilogue parity: the vLLM plugin's xetla int2 upcvt GEMM with fused post-ops
// (csrc/int2_fp16_upcvt_kernel.sycl, int2_upcvt_gemm_impl<..., POSTOP>) and the
// TernOCL OpenCL kernel run on the same USM inputs in one process; outputs are
// compared bit for bit and against the host fp32 gold.
//   parity --dtype fp16|bf16 --postop 1..4 [--m M --n N --k K] [--scale-hi s]
#include "int2_upcvt_xetla_kernels.hpp"  // truncated plugin source (build.sh)

#include <fstream>
#include <random>
#include <sstream>

#include "dt16.hpp"
#include "epilogue.hpp"

namespace syclex = sycl::ext::oneapi::experimental;
std::chrono::high_resolution_clock::time_point ref_time_point_ = std::chrono::high_resolution_clock::now();

static std::string slurp(const std::string &p) {
    std::ifstream f(p);
    if (!f) { std::cerr << "cannot open " << p << "\n"; std::exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

template <typename XT, int P>
sycl::event run_xetla(sycl::queue &q, bool mt, int M, int N, int K, uint16_t *A, int32_t *B,
        uint16_t *C, uint16_t *S, uint16_t *O, uint16_t *Bi) {
    // plugin tiers: decode (wg_n 32, LS 4), prefill M tile (32, 64, 8, 16, 128)
    auto *a = (XT *)A, *c = (XT *)C, *s = (XT *)S, *o = (XT *)O, *bi = (XT *)Bi;
    if (mt) {
        auto [na, nc] = upcvt_scratch_bytes<XT, 32, 64, 8, 16, 128, 1, 1, false>(M, N);
        auto &sc = get_scratch(q, na, nc);
        return int2_upcvt_gemm_impl<XT, 32, 64, 8, 16, 128, 1, 1, false, P>(
                q, M, N, K, a, B, c, s, sc.acc, sc.cnt, o, bi, N);
    }
    auto [na, nc] = upcvt_scratch_bytes<XT, 1, 32, 1, 16, 128, 1, 4, false>(M, N);
    auto &sc = get_scratch(q, na, nc);
    return int2_upcvt_gemm_impl<XT, 1, 32, 1, 16, 128, 1, 4, false, P>(
            q, M, N, K, a, B, c, s, sc.acc, sc.cnt, o, bi, N);
}

template <typename XT>
sycl::event run_xetla_p(int p, sycl::queue &q, bool mt, int M, int N, int K, uint16_t *A,
        int32_t *B, uint16_t *C, uint16_t *S, uint16_t *O, uint16_t *Bi) {
    switch (p) {
        case 1: return run_xetla<XT, 1>(q, mt, M, N, K, A, B, C, S, O, Bi);
        case 2: return run_xetla<XT, 2>(q, mt, M, N, K, A, B, C, S, O, Bi);
        case 3: return run_xetla<XT, 3>(q, mt, M, N, K, A, B, C, S, O, Bi);
        default: return run_xetla<XT, 4>(q, mt, M, N, K, A, B, C, S, O, Bi);
    }
}

struct Diff {
    size_t same = 0, le1 = 0, n = 0;
    uint32_t max_ulp = 0;
    double max_abs = 0;
    size_t worst = 0;
};
static Diff diff16(const uint16_t *x, const uint16_t *y, size_t n) {
    Diff d;
    d.n = n;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t u = x[i] > y[i] ? x[i] - y[i] : y[i] - x[i];
        const bool sign_flip_zero = ((x[i] | y[i]) & 0x7fff) == 0;  // +0 vs -0
        d.same += x[i] == y[i];
        d.le1 += u <= 1 || sign_flip_zero;
        const double a = std::fabs((double)tof(x[i]) - tof(y[i]));
        if (!sign_flip_zero && u > d.max_ulp) { d.max_ulp = u; d.worst = i; }
        d.max_abs = std::max(d.max_abs, a);
    }
    return d;
}

int main(int argc, char **argv) {
    int M = 1, N = 4096, K = 4096, postop = 4, time_iters = 0;
    float scale_hi = 0.08f;
    std::string cl_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&] { return std::string(argv[++i]); };
        if (a == "--m") M = std::stoi(nx());
        else if (a == "--n") N = std::stoi(nx());
        else if (a == "--k") K = std::stoi(nx());
        else if (a == "--postop") postop = std::stoi(nx());
        else if (a == "--dtype") dt_is_bf16() = nx() == "bf16";
        else if (a == "--scale-hi") scale_hi = std::stof(nx());
        else if (a == "--cl-dir") cl_dir = nx();
        else if (a == "--time") time_iters = std::stoi(nx());
        else { std::cerr << "usage: parity --dtype fp16|bf16 --postop 1..4 [--m --n --k] [--scale-hi s] --cl-dir d\n"; return 1; }
    }
    if (postop < 1 || postop > 4 || N % 64 || K % 512) { std::cerr << "postop 1..4, N % 64, K % 512\n"; return 1; }
    const bool mt = M > 1;
    Epilogue ep;
    ep.postop = postop;

    sycl::queue q(sycl::gpu_selector_v,
            sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const size_t sa = (size_t)M * K, sb = (size_t)K / 16 * N, ss = (size_t)K / 128 * N, sc = (size_t)M * N;
    std::mt19937 g(1234);
    std::uniform_real_distribution<float> ua(-5.f, 5.f), us(0.02f, scale_hi);
    std::vector<uint16_t> A(sa), S(ss), O, Bias16;
    std::vector<uint32_t> B(sb);
    std::vector<unsigned char> Bias;
    for (auto &x : A) x = fromf(ua(g));
    for (auto &x : B) { uint32_t r1 = g(), r2 = g(); x = (r1 & 0x55555555u) | (((r1 & r2) & 0x55555555u) << 1); }
    for (auto &x : S) x = fromf(us(g));
    fill_epilogue_inputs(ep, O, Bias, M, N, g);
    if (O.empty()) O.assign(sc, 0);
    Bias16.assign(N, 0);
    if (!Bias.empty()) std::memcpy(Bias16.data(), Bias.data(), N * 2);

    auto *dA = sycl::malloc_device<uint16_t>(sa, q);
    auto *dB = sycl::malloc_device<int32_t>(sb, q);
    auto *dS = sycl::malloc_device<uint16_t>(ss, q);
    auto *dO = sycl::malloc_device<uint16_t>(sc, q);
    auto *dBi = sycl::malloc_device<uint16_t>(N, q);
    auto *dCx = sycl::malloc_device<uint16_t>(sc, q);
    auto *dCo = sycl::malloc_device<uint16_t>(sc, q);
    q.memcpy(dA, A.data(), sa * 2); q.memcpy(dB, B.data(), sb * 4); q.memcpy(dS, S.data(), ss * 2);
    q.memcpy(dO, O.data(), sc * 2); q.memcpy(dBi, Bias16.data(), N * 2);
    q.memset(dCx, 0xff, sc * 2); q.memset(dCo, 0xff, sc * 2).wait();

    if (dt_is_bf16())
        run_xetla_p<gpu::xetla::bf16>(postop, q, mt, M, N, K, dA, dB, dCx, dS, dO, dBi).wait();
    else
        run_xetla_p<gpu::xetla::fp16>(postop, q, mt, M, N, K, dA, dB, dCx, dS, dO, dBi).wait();

    // TernOCL kernel through the SYCL OpenCL-C kernel compiler, same tiles as the driver defaults
    std::string src = slurp(cl_dir + "/int2_fp16_upcvt.cl");
    const std::string inc = "#include \"epilogue.clh\"";
    src.replace(src.find(inc), inc.size(), slurp(cl_dir + "/../common/epilogue.clh"));
    const int nsg = 2, ls = 4, mt_m = 64, mt_n = 16, wg_m = 2, wg_n = 4;
    std::string opts = std::string("-cl-std=CL3.0") + (dt_is_bf16() ? " -DBF16" : "") + " -DSGM=1 -DNSG_N="
            + std::to_string(nsg) + " -DLS=" + std::to_string(ls) + " -DU=2 -DPF=0 -DMT_M=" + std::to_string(mt_m)
            + " -DMT_N=" + std::to_string(mt_n) + " -DWG_M=" + std::to_string(wg_m) + " -DWG_N="
            + std::to_string(wg_n) + (mt ? " -cl-intel-256-GRF-per-thread" : "") + ep.opts();
    auto kb_src = syclex::create_kernel_bundle_from_source(q.get_context(), syclex::source_language::opencl, src);
    auto kb = syclex::build(kb_src, syclex::properties{syclex::build_options{opts}});
    sycl::kernel k = kb.ext_oneapi_get_kernel(mt ? "int2_fp16_upcvt_gemm_mt" : "int2_fp16_upcvt_gemm");
    size_t gl0, gl1, lo0;
    if (mt) {
        lo0 = 16 * wg_n * wg_m;
        gl0 = ((N + mt_n * wg_n - 1) / (mt_n * wg_n)) * lo0;
        gl1 = (M + mt_m * wg_m - 1) / (mt_m * wg_m);
    } else {
        lo0 = 16 * nsg * ls;
        gl0 = ((N + 16 * nsg - 1) / (16 * nsg)) * lo0;
        gl1 = M;
    }
    auto run_ocl = [&] {
        return q.submit([&](sycl::handler &h) {
            h.set_args(dA, dB, dS, dCo, dO, dBi, M, N, K);
            h.parallel_for(sycl::nd_range<2>({gl1, gl0}, {1, lo0}), k);
        });
    };
    run_ocl().wait();
    if (time_iters > 0) {  // same queue, same (cache-resident) buffers for both
        auto ns = [](sycl::event e) {
            return (double)(e.get_profiling_info<sycl::info::event_profiling::command_end>()
                    - e.get_profiling_info<sycl::info::event_profiling::command_start>());
        };
        auto xrun = [&] {
            return dt_is_bf16() ? run_xetla_p<gpu::xetla::bf16>(postop, q, mt, M, N, K, dA, dB, dCx, dS, dO, dBi)
                                : run_xetla_p<gpu::xetla::fp16>(postop, q, mt, M, N, K, dA, dB, dCx, dS, dO, dBi);
        };
        double tx = 0, to = 0;
        for (int rep = 0; rep < 2; ++rep) {  // back-to-back submits, one wait per loop; rep 0 = warm-up
            std::vector<sycl::event> ex, eo;
            for (int i = 0; i < time_iters; ++i) ex.push_back(xrun());
            q.wait();
            for (int i = 0; i < time_iters; ++i) eo.push_back(run_ocl());
            q.wait();
            if (rep) for (int i = 0; i < time_iters; ++i) { tx += ns(ex[i]); to += ns(eo[i]); }
        }
        std::printf("time (L0 events, %d iters): xetla %.3f us, ocl %.3f us\n", time_iters, tx / time_iters / 1e3,
                to / time_iters / 1e3);
    }

    std::vector<uint16_t> Cx(sc), Co(sc);
    q.memcpy(Cx.data(), dCx, sc * 2); q.memcpy(Co.data(), dCo, sc * 2).wait();

    // host gold: fp32 accumulator of the same inputs, then the epilogue
    std::vector<float> acc(sc);
#pragma omp parallel for
    for (long i = 0; i < (long)sc; ++i) {
        const int m = (int)(i / N), n = (int)(i % N);
        float s = 0;
        for (int kk = 0; kk < K; ++kk) {
            const uint32_t c = (B[(size_t)(kk / 16) * N + n] >> (2 * (kk % 16))) & 3u;
            if (c) s += tof(A[(size_t)m * K + kk]) * (c == 1 ? 1.f : -1.f) * tof(S[(size_t)(kk / 128) * N + n]);
        }
        acc[i] = s;
    }
    std::vector<unsigned char> Gb(sc * 2);
    epilogue_ref(ep, acc.data(), O.data(), Bias.data(), Gb.data(), M, N);
    const uint16_t *G = (const uint16_t *)Gb.data();

    size_t low = 0, low_x0 = 0, low_o0 = 0;  // acc <= -10: xetla_sigmoid clamps to 0
    for (size_t i = 0; i < sc; ++i)
        if (acc[i] <= -10.f) { ++low; low_x0 += (Cx[i] & 0x7fff) == 0; low_o0 += (Co[i] & 0x7fff) == 0; }
    auto pr = [&](const char *l, const Diff &d) {
        std::printf("  %-14s identical %6.2f%%  <=1ulp %6.2f%%  max ulp %u  max abs %.3g\n", l,
                100.0 * d.same / d.n, 100.0 * d.le1 / d.n, d.max_ulp, d.max_abs);
    };
    std::printf("%s postop=%d (%s) M=%d N=%d K=%d tile=%s  acc range sampled: %zu of %zu <= -10\n", dt_name(), postop,
            ep.name().c_str(), M, N, K, mt ? "mt" : "gemv", low, sc);
    Diff xo = diff16(Cx.data(), Co.data(), sc), xg = diff16(Cx.data(), G, sc), og = diff16(Co.data(), G, sc);
    std::printf("  %-14s identical %6.2f%%  <=1ulp %6.2f%%  max ulp %u (idx %zu: xetla %g ocl %g acc %g)\n",
            "xetla vs ocl", 100.0 * xo.same / sc, 100.0 * xo.le1 / sc, xo.max_ulp, xo.worst, tof(Cx[xo.worst]),
            tof(Co[xo.worst]), acc[xo.worst]);
    pr("xetla vs gold", xg);
    pr("ocl vs gold", og);
    if (postop == 1 || postop == 4)
        std::printf("  acc <= -10: xetla zero %zu/%zu, ocl zero %zu/%zu\n", low_x0, low, low_o0, low);
    for (auto *p : {(void *)dA, (void *)dB, (void *)dS, (void *)dO, (void *)dBi, (void *)dCx, (void *)dCo})
        sycl::free(p, q);
    return 0;
}
