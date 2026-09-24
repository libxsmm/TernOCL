#!/bin/bash
# Build the xetla int2 x fp16/bf16 upcvt reference harness (copy of the
# plugin's int2_fp16_upcvt_dpas_fast_test driver with the same rotating-set
# rule as the OpenCL driver, explicit GEMV tiles --wgn/--ks/--ls and prefill
# tiles --mtile). Builds xetla_int2_upcvt_ref (fp16) and _bf16.
#   XETLA=/path/to/xetla bash build.sh        (needs oneAPI icpx >= 2026.0)
set -e
XETLA=${XETLA:?set XETLA to the xetla source tree (xetla_vllm_plugin/xetla)}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $HERE/build
F="-std=c++20 -O3 -fsycl -qopenmp -I$HERE/shim -I$XETLA/include -I$XETLA/tests \
   -I$XETLA/tests/utils -I$XETLA/tests/integration/gemm"
icpx $F $HERE/main.cpp -o $HERE/build/xetla_int2_upcvt_ref -lpthread -qopenmp &
icpx $F -DXETLA_REF_BF16 $HERE/main.cpp -o $HERE/build/xetla_int2_upcvt_ref_bf16 -lpthread -qopenmp &
wait
echo "built $HERE/build/xetla_int2_upcvt_ref{,_bf16}"
