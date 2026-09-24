#!/bin/bash
# Validation: OpenCL upcvt (GEMV + large-M GEMM + ragged shapes, distinct sets)
# and the xetla reference (GEMV tile + prefill tile).  bash validate.sh <dir>
cd "$1"; O=./build/int2_fp16_upcvt_ocl; X=./xetla_ref/build/xetla_int2_upcvt_ref
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
summ() { grep -a 'max ULP' | sed 's/.*max abs diff \([0-9.]*\).*max ULP diff \([0-9]*\).*pass rate \(.*\)/abs \1 ulp \2 pass \3/' | tr '\n' ' '; }
for dt in fp16 bf16; do
for mkn in "1024 17408 5120" "1024 5120 14336" "1024 5120 34816" "1000 5120 4096" "77 128 48" "129 256 272" \
           "200 5120 16384" "1 5120 34816" "1 17408 5120" "1 5120 248320" "1 6144 5120" "4 6144 5120"; do
  set -- $mkn
  o=$($O --m $1 --k $2 --n $3 --dtype $dt --iters 3 --sets 2 --distinct-sets 2>&1)
  echo "ocl $dt M=$1 K=$2 N=$3 | $(summ <<<"$o")| $(grep -a 'Validation summary\|error\|failed' <<<"$o" | head -2)"
done; done
for c in "1024 17408 5120 --mtile 3" "1024 5120 14336 --mtile 1" "1 5120 34816 --wgn 64 --ks 1 --ls 1" "1 17408 5120 --wgn 32 --ks 1 --ls 1"; do
  set -- $c
  o=$($X --m $1 --k $2 --n $3 ${@:4} --iters 2 --sets 2 --distinct-sets 2>&1)
  echo "xetla M=$1 K=$2 N=$3 ${*:4} | $(grep -a 'Validation summary' <<<"$o")"
done
