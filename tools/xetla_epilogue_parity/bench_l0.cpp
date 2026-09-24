// Same-runtime GEMV benchmark: the plugin's xetla int2 upcvt kernel and the
// TernOCL OpenCL kernel (built through the SYCL OpenCL-C kernel compiler) in one
// SYCL / Level Zero queue, over the same rotating >= 2 GiB of distinct weight
// sets. Removes the OpenCL-vs-Level-Zero runtime difference of the two harnesses.
//   bench_l0 --n N --k K --xwgn 32 --xks 1 --xls 4 --owgn 32 --ols 4 --ou 1 [--dtype fp16|bf16]
//            [--iters 50] [--weights-gib 2] [--reps 3]
#include "int2_upcvt_xetla_kernels.hpp"  // truncated plugin source (build.sh)

#include <fstream>
#include <random>
#include <sstream>

#include "dt16.hpp"

namespace syclex = sycl::ext::oneapi::experimental;
std::chrono::high_resolution_clock::time_point ref_time_point_ = std::chrono::high_resolution_clock::now();

static std::string slurp(const std::string &p) {
    std::ifstream f(p);
    if (!f) { std::cerr << "cannot open " << p << "\n"; std::exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

template <typename XT, int WGN, int KS, int LS>
sycl::event xg(sycl::queue &q, int N, int K, uint16_t *A, int32_t *B, uint16_t *C, uint16_t *S) {
    auto [na, nc] = upcvt_scratch_bytes<XT, 1, WGN, 1, 16, 128, KS, LS, false>(1, N);
    auto &sc = get_scratch(q, na, nc);
    return int2_upcvt_gemm_impl<XT, 1, WGN, 1, 16, 128, KS, LS, false, 0>(
            q, 1, N, K, (XT *)A, B, (XT *)C, (XT *)S, sc.acc, sc.cnt);
}

// the plugin's GEMV tiers (csrc/int2_fp16_upcvt_kernel.sycl, both archs)
template <typename XT>
sycl::event xetla(int wgn, int ks, int ls, sycl::queue &q, int N, int K, uint16_t *A, int32_t *B,
        uint16_t *C, uint16_t *S) {
#define XT_(W, KS_, LS_) if (wgn == W && ks == KS_ && ls == LS_) return xg<XT, W, KS_, LS_>(q, N, K, A, B, C, S);
    XT_(32, 1, 1) XT_(32, 1, 2) XT_(32, 1, 4) XT_(32, 1, 8) XT_(64, 1, 1) XT_(64, 1, 2) XT_(64, 1, 4)
    XT_(128, 1, 1) XT_(128, 1, 2) XT_(256, 1, 1) XT_(32, 2, 4)
#undef XT_
    std::cerr << "xetla tile " << wgn << "," << ks << "," << ls << " not instantiated\n";
    std::exit(1);
}

static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv) {
    int N = 4096, K = 4096, xwgn = 32, xks = 1, xls = 4, owgn = 32, ols = 4, ou = 1, iters = 50, reps = 3;
    double wgib = 2.0;
    std::string cl_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&] { return std::string(argv[++i]); };
        if (a == "--n") N = std::stoi(nx());
        else if (a == "--k") K = std::stoi(nx());
        else if (a == "--xwgn") xwgn = std::stoi(nx());
        else if (a == "--xks") xks = std::stoi(nx());
        else if (a == "--xls") xls = std::stoi(nx());
        else if (a == "--owgn") owgn = std::stoi(nx());
        else if (a == "--ols") ols = std::stoi(nx());
        else if (a == "--ou") ou = std::stoi(nx());
        else if (a == "--iters") iters = std::stoi(nx());
        else if (a == "--reps") reps = std::stoi(nx());
        else if (a == "--weights-gib") wgib = std::stod(nx());
        else if (a == "--dtype") dt_is_bf16() = nx() == "bf16";
        else if (a == "--cl-dir") cl_dir = nx();
        else { std::cerr << "see header for usage\n"; return 1; }
    }
    sycl::queue q(sycl::gpu_selector_v,
            sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const size_t sa = K, sb = (size_t)K / 16 * N, ss = (size_t)K / 128 * N;
    const double wset = sb * 4.0 + ss * 2.0;
    const int sets = std::max(1, (int)std::ceil(wgib * (1 << 30) / wset));
    std::mt19937 g(7);
    std::uniform_real_distribution<float> ua(-5.f, 5.f), us(0.75f, 15.75f);
    std::vector<uint16_t> Ah(sa), Sh(ss);
    std::vector<uint32_t> Bh(sb);
    std::vector<uint16_t *> dA(sets), dS(sets), dC(sets), dCo(sets);
    std::vector<int32_t *> dB(sets);
    for (int s = 0; s < sets; ++s) {
        for (auto &x : Ah) x = fromf(ua(g));
        for (auto &x : Bh) { uint32_t r1 = g(), r2 = g(); x = (r1 & 0x55555555u) | (((r1 & r2) & 0x55555555u) << 1); }
        for (auto &x : Sh) x = fromf(us(g));
        dA[s] = sycl::malloc_device<uint16_t>(sa, q);
        dB[s] = sycl::malloc_device<int32_t>(sb, q);
        dS[s] = sycl::malloc_device<uint16_t>(ss, q);
        dC[s] = sycl::malloc_device<uint16_t>(N, q);
        dCo[s] = sycl::malloc_device<uint16_t>(N, q);
        q.memcpy(dA[s], Ah.data(), sa * 2);
        q.memcpy(dB[s], Bh.data(), sb * 4);
        q.memcpy(dS[s], Sh.data(), ss * 2);
    }
    uint16_t *dummy = sycl::malloc_device<uint16_t>(64, q);
    q.wait();

    std::string src = slurp(cl_dir + "/int2_fp16_upcvt.cl");
    const std::string inc = "#include \"epilogue.clh\"";
    src.replace(src.find(inc), inc.size(), slurp(cl_dir + "/../common/epilogue.clh"));
    const int nsg = owgn / 16;
    const std::string opts = std::string("-cl-std=CL3.0") + (dt_is_bf16() ? " -DBF16" : "") + " -DSGM=1 -DNSG_N="
            + std::to_string(nsg) + " -DLS=" + std::to_string(ols) + " -DU=" + std::to_string(ou) + " -DPF=0";
    auto kb_src = syclex::create_kernel_bundle_from_source(q.get_context(), syclex::source_language::opencl, src);
    auto kb = syclex::build(kb_src, syclex::properties{syclex::build_options{opts}});
    sycl::kernel k = kb.ext_oneapi_get_kernel("int2_fp16_upcvt_gemm");
    const size_t lo0 = 16 * nsg * ols, gl0 = ((N + owgn - 1) / owgn) * lo0;
    const int M = 1;
    auto ocl = [&](int s) {
        return q.submit([&](sycl::handler &h) {
            h.set_args(dA[s], dB[s], dS[s], dCo[s], dummy, dummy, M, N, K);
            h.parallel_for(sycl::nd_range<2>({1, gl0}, {1, lo0}), k);
        });
    };
    auto xet = [&](int s) {
        return dt_is_bf16() ? xetla<gpu::xetla::bf16>(xwgn, xks, xls, q, N, K, dA[s], dB[s], dC[s], dS[s])
                            : xetla<gpu::xetla::fp16>(xwgn, xks, xls, q, N, K, dA[s], dB[s], dC[s], dS[s]);
    };
    auto ns = [](const sycl::event &e) {
        return (double)(e.get_profiling_info<sycl::info::event_profiling::command_end>()
                - e.get_profiling_info<sycl::info::event_profiling::command_start>());
    };
    // wait: submit + wait per launch (what both harnesses do); stream: back-to-back, one wait
    auto run = [&](auto &&f, bool wait_each) {
        for (int s = 0; s < sets; ++s) f(s).wait();  // warm every set
        std::vector<sycl::event> ev;
        for (int i = 0; i < iters; ++i) {
            ev.push_back(f(i % sets));
            if (wait_each) ev.back().wait();
        }
        q.wait();
        double t = 0;
        for (auto &e : ev) t += ns(e);
        return t / iters / 1e3;
    };
    std::vector<double> xw, ow, xs, os;
    for (int r = 0; r < reps; ++r) {
        xw.push_back(run(xet, true)); ow.push_back(run(ocl, true));
        xs.push_back(run(xet, false)); os.push_back(run(ocl, false));
    }
    std::vector<uint16_t> c1(N), c2(N);
    q.memcpy(c1.data(), dC[0], N * 2); q.memcpy(c2.data(), dCo[0], N * 2).wait();
    size_t same = 0;
    for (int i = 0; i < N; ++i) same += c1[i] == c2[i];
    const double gb = (K / 4.0 * N + K / 128.0 * N * 2 + K * 2.0 + N * 2.0) / (1 << 30);
    std::printf("%s N=%d K=%d sets=%d (%.2f GiB weights) xetla(%d,%d,%d) ocl(wgn %d, ls %d, u %d) | "
                "wait: xetla %.3f us ocl %.3f us x%.2f | stream: xetla %.3f us ocl %.3f us x%.2f | "
                "GiB/s stream %.0f / %.0f | C identical %.1f%%\n",
            dt_name(), N, K, sets, wset * sets / (1 << 30), xwgn, xks, xls, owgn, ols, ou, med(xw), med(ow),
            med(xw) / med(ow), med(xs), med(os), med(xs) / med(os), gb / (med(xs) * 1e-6), gb / (med(os) * 1e-6),
            100.0 * same / N);
    for (int s = 0; s < sets; ++s) {
        sycl::free(dA[s], q); sycl::free(dB[s], q); sycl::free(dS[s], q); sycl::free(dC[s], q); sycl::free(dCo[s], q);
    }
    sycl::free(dummy, q);
    return 0;
}
