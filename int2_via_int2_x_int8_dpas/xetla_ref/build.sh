#!/bin/bash
# Build the xetla int2 x int8 reference harness: the plugin's
# int2_fp16_dpas_fp16scales_fast_test with src/main.cpp replaced by the copy
# here (only the rotating-set rule differs, see the top of calculate_n_sets_runtime).
#   XETLA=/path/to/xetla bash build.sh        (needs oneAPI icpx >= 2026.0)
set -e
XETLA=${XETLA:?set XETLA to the xetla source tree (xetla_vllm_plugin/xetla)}
HERE=$(cd "$(dirname "$0")" && pwd)
T=$XETLA/int2_fp16_dpas_fp16scales_fast_test
mkdir -p $HERE/build
F="-std=c++20 -O3 -fsycl -qopenmp -I$XETLA/include -I$XETLA/tests -I$XETLA/tests/utils \
   -I$XETLA/tests/integration/gemm -I$T/src -I$HERE/shim"
cd $T
icpx $F -c src/precompiled_gemm.cpp -o $HERE/build/precompiled_gemm.o &
icpx $F -c src/variant_registry.cpp -o $HERE/build/variant_registry.o &
icpx $F -c $HERE/src/main.cpp -o $HERE/build/main.o &
wait
icpx $F $HERE/build/main.o $HERE/build/precompiled_gemm.o $HERE/build/variant_registry.o \
    -o $HERE/build/xetla_int2_int8_ref -lmkl_rt -lpthread -qopenmp
echo "built $HERE/build/xetla_int2_int8_ref"
