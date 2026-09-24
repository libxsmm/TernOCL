#!/bin/bash
# Validation: OpenCL (both modes, GEMM + GEMV + ragged shapes, distinct sets)
# and the xetla reference at its GEMM/GEMV tiles.  bash validate.sh <dir>
cd "$1"; O=./build/int2_int8_dpas_ocl; X=./xetla_ref/build/xetla_int2_int8_ref
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
summ() { grep -a 'max ULP' | sed 's/.*max abs diff \([0-9.]*\).*max ULP diff \([0-9]*\).*pass rate \(.*\)/abs \1 ulp \2 pass \3/' | tr '\n' ' '; }
for dt in fp16 bf16; do for qm in 0 1; do
  for mkn in "1024 17408 5120" "1024 5120 14336" "1024 5120 34816" "1000 5120 4096" "77 128 48" \
             "129 256 272" "200 5120 16384" "1 5120 34816" "1 17408 5120" "1 5120 248320" "4 6144 5120"; do
    set -- $mkn
    o=$($O --m $1 --k $2 --n $3 --dtype $dt --qmode $qm --iters 3 --sets 2 --distinct-sets 2>&1)
    echo "ocl $dt qm=$qm M=$1 K=$2 N=$3 | $(grep -a 'scale_a' <<<"$o") | $(summ <<<"$o")| $(grep -a 'Validation summary\|error\|failed' <<<"$o" | head -2)"
  done
done; done
# xetla: plugin GEMM tile and a GEMV variant; 2 sets (host gold is slow)
for mkn in "1024 17408 5120 64 8 256 128 64" "1024 5120 14336 64 8 256 128 64" "1024 5120 34816 64 8 256 128 64" \
           "1 5120 34816 1 1 128 16 128" "1 17408 5120 1 1 128 16 128"; do
  set -- $mkn
  o=$($X --mat_m=$1 --mat_k=$2 --mat_n=$3 --scale_gs=$(($2 / 128)) --wg_m=$4 --sg_m=$5 --wg_n=$6 --sg_n=$7 --sg_k=$8 --iters=2 --sets=2 2>&1)
  echo "xetla M=$1 K=$2 N=$3 tile($4,$5,$6,$7,$8) | $(grep -ac '^PASSED' <<<"$o")/2 sets PASSED $(grep -a 'No matching\|FAIL' <<<"$o" | head -1)"
done
