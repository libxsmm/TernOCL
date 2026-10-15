#!/bin/bash
# sweep_prefill.sh T...: grouped-GEMM tiles (mt_m,mt_n,wg_m,wg_n) for the MoE prefill path
source /swtools/intel-gpu/latest/intel_gpu_vars.sh >/dev/null 2>&1
source /swtools/intel/2026.0/oneapi-vars.sh --force >/dev/null 2>&1
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
cd "$(dirname "$0")/build"
for T in "$@"; do
  for mm in 8 16 32 64; do for mn in 16 32; do for wm in 1 2 4; do for wn in 1 2 4 8; do
    [ $((wm * wn)) -gt 16 ] && continue
    r=$(./int2_moe_prefill_ocl --t $T --dir . --check 1 --iters 5 --tile $mm,$mn,$wm,$wn 2>&1 | grep -oE 'layer [0-9.]+ us|FAIL' | tr '\n' ' ')
    echo "T=$T tile $mm,$mn,$wm,$wn $r"
  done; done; done; done | sort -k5 -g | head -4
done
