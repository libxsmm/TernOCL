#!/bin/bash
# Build the xetla <-> TernOCL epilogue parity tool against the plugin's own
# kernel source (csrc/int2_fp16_upcvt_kernel.sycl minus its torch bindings),
# with default device flags: the plugin's setup.py backend flags do not reach its
# device image (quoted -Xsycl-target-backend=), so production runs the defaults.
#   PLUGIN=/path/to/xetla_vllm_plugin bash build.sh      (needs oneAPI icpx >= 2026.0)
set -e
PLUGIN=${PLUGIN:?set PLUGIN to the xetla_vllm_plugin checkout}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $HERE/build
sed -e '/#include <c10\/xpu\/XPUStream.h>/d' -e '/#include <torch\/extension.h>/d' \
    -e '/^torch::Tensor int2_fp16_upcvt_gemm_run_torch/,$d' \
    $PLUGIN/csrc/int2_fp16_upcvt_kernel.sycl > $HERE/build/int2_upcvt_xetla_kernels.hpp
icpx -std=c++20 -O3 -fsycl -qopenmp -Wno-unusable-partial-specialization \
    -I$HERE/build -I$PLUGIN/csrc -I$PLUGIN/xetla/include -I$HERE/../../common \
    $HERE/parity.cpp -o $HERE/build/parity &
P1=$!
icpx -std=c++20 -O3 -fsycl -Wno-unusable-partial-specialization \
    -I$HERE/build -I$PLUGIN/csrc -I$PLUGIN/xetla/include -I$HERE/../../common \
    $HERE/bench_l0.cpp -o $HERE/build/bench_l0 &
P2=$!
wait $P1 && wait $P2
echo "built $HERE/build/{parity,bench_l0}"
