#!/bin/bash
# Validation: OpenCL int1 upcvt (GEMV + large-M GEMM + ragged shapes, distinct
# sets), fp16 and bf16.  bash validate.sh <dir>
cd "$1"; O=./build/int1_fp16_upcvt_ocl
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
summ() { grep -a 'max ULP' | sed 's/.*max abs diff \([0-9.]*\).*max ULP diff \([0-9]*\).*pass rate \(.*\)/abs \1 ulp \2 pass \3/' | tr '\n' ' '; }
for dt in fp16 bf16; do
for mkn in "1024 17408 5120" "1024 5120 14336" "1024 5120 34816" "1000 5120 4096" "77 128 48" "129 256 272" \
           "200 5120 16384" "1 5120 34816" "1 17408 5120" "1 5120 248320" "1 6144 5120" "4 6144 5120" \
           "3 256 48" "1024 5120 16384 --mt-m 64 --mt-n 16" "300 6144 5120 --mt-m 128 --mt-n 16 --wg-m 1 --wg-n 8" \
           "1 5120 34816 --shl" "4 6144 5120 --shl" "1024 5120 14336 --shl"; do
  set -- $mkn
  o=$($O --m $1 --k $2 --n $3 ${@:4} --dtype $dt --iters 3 --sets 2 --distinct-sets 2>&1)
  echo "ocl $dt M=$1 K=$2 N=$3 ${*:4} | $(summ <<<"$o")| $(grep -a 'Validation summary\|error\|failed' <<<"$o" | head -2)"
done; done
