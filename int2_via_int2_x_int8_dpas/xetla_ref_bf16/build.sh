#!/bin/bash
# Build the xetla int2 x int8 bf16 reference harness: the plugin's
# int2_bf16_dpas_fast_test (bf16 A/C, fp32 scales) with src/main.cpp replaced
# by the copy here (only the rotating-set rule differs).
#   XETLA=/path/to/xetla bash build.sh        (needs oneAPI icpx >= 2026.0)
set -e
XETLA=${XETLA:?set XETLA to the xetla source tree (xetla_vllm_plugin/xetla)}
HERE=$(cd "$(dirname "$0")" && pwd)
T=$XETLA/int2_bf16_dpas_fast_test
mkdir -p $HERE/build
F="-std=c++20 -O3 -fsycl -qopenmp -I$XETLA/include -I$XETLA/tests -I$XETLA/tests/utils \
   -I$XETLA/tests/integration/gemm -I$T/src -I$HERE/../xetla_ref/shim"
cd $T
objs=""
for s in src/precompiled_*.cpp src/variant_registry.cpp; do
  o=$HERE/build/$(basename ${s%.cpp}).o; objs+=" $o"
  icpx $F -c $s -o $o &
  while (( $(jobs -r | wc -l) >= ${JOBS:-8} )); do sleep 2; done
done
icpx $F -c $HERE/src/main.cpp -o $HERE/build/main.o
wait
icpx $F $HERE/build/main.o $objs -o $HERE/build/xetla_int2_bf16_int8_ref -lmkl_rt -lpthread -qopenmp
echo "built $HERE/build/xetla_int2_bf16_int8_ref"
