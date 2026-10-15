#!/bin/bash
# sweep.sh [T]: gate_up, down and fused down_sum tiles for the Qwen3-30B-A3B expert shapes
source /swtools/intel-gpu/latest/intel_gpu_vars.sh >/dev/null 2>&1
source /swtools/intel/2026.0/oneapi-vars.sh --force >/dev/null 2>&1
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
T=${1:-1}
cd "$(dirname "$0")/build"
./int2_moe_ocl --t $T --dir . --iters 20 | tail -2
./int2_moe_ocl --t $T --dir . --iters 20 --fused | tail -2
for nsg in 1 2 4; do for ls in 2 4 8 16; do for u in 1 2 4; do
  echo "gu $nsg,$ls,$u $(./int2_moe_ocl --t $T --dir . --iters 20 --gu $nsg,$ls,$u 2>&1 | grep -oE 'gate_up [0-9.]+ us|FAIL' | tr '\n' ' ')"
done; done; done | sort -k4 -g | head -5
for f in "" --fused; do for nsg in 1 2 4; do for ls in 1 2 3 6; do for u in 1 2; do
  echo "dn$f $nsg,$ls,$u $(./int2_moe_ocl --t $T --dir . --iters 20 $f --dn $nsg,$ls,$u 2>&1 | grep -oE 'down [0-9.]+ us|FAIL' | tr '\n' ' ')"
done; done; done | sort -k4 -g | head -4; done
