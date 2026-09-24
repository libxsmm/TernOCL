#!/bin/bash
# Validation: OpenCL BITCOS GEMV and M-tiled GEMM (fp16 with each scale apply,
# bf16) over tiles, local K slices, zero densities, ragged M/N, epilogues and
# the Bonsai / paper shapes, distinct sets, against the host fp32 gold.  bash validate.sh <dir>
cd "$1"; O=./build/bitcos_fp16_upcvt_ocl
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
summ() { grep -a 'max ULP' | sed 's/.*max abs diff \([0-9.]*\).*max ULP diff \([0-9]*\).*pass rate \(.*\)/abs \1 ulp \2 pass \3/' | tr '\n' ' '; }
for ap in "" "--int-apply" "--simt-mul" "--dtype bf16"; do
for c in "1 4096 4096 --ls 4" "1 4096 4096 --ls 1 --wgn 256" "1 4096 6144 --ls 8 --wgn 64" \
         "1 12288 4096 --ls 8 --z 0.6" "1 17408 5120 --ls 4 --z 0.3" "1 16384 32768 --ls 1 --wgn 256" \
         "3 4096 2048 --ls 2" "8 1024 512 --ls 4 --z 0.1" "1 4096 1024 --ls 16 --wgn 32 --z 0.95" \
         "1 4096 4096 --postop 1" "1 4096 4096 --postop 3 --out-f32" \
         "64 4096 4096 --mt-m 32" "77 4096 2064 --mt-m 32 --z 0.05" "129 17408 5120 --mt-m 64" \
         "512 5120 14336 --mt-m 32 --mt-n 32 --wg-m 2 --wg-n 4" "9 6144 1024 --mt-m 16 --mt-n 64 --wg-m 2 --wg-n 2 --z 0.95" \
         "77 4096 2048 --mt-m 32 --postop 1" "77 4096 2048 --mt-m 32 --postop 3 --out-f32"; do
  set -- $c
  o=$($O --m $1 --k $2 --n $3 ${@:4} $ap --iters 2 --sets 2 --distinct-sets 2>&1)
  echo "ocl [${ap:-visa-hmul}] M=$1 K=$2 N=$3 ${*:4} | $(summ <<<"$o")| $(grep -a 'Validation summary\|error\|failed' <<<"$o" | head -2)"
done; done
