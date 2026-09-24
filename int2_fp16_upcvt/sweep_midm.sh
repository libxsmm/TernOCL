#!/bin/bash
# Mid-M (prompt-length) tile sweep for the upcvt kernel, one shape: best OCL
# config per M vs xetla's prefill tiles (plugin: WGM 8 for M<=8, 16 for M<=16,
# else 32 -> ref --mtile 3 / 2 / 1). >= 2 GiB rotating weights, device time.
#   bash sweep_midm.sh <K> <N> <out>   (cwd: int2_fp16_upcvt)
K=$1; N=$2; OUT=$3; : > $OUT
O=./build/int2_fp16_upcvt_ocl; X=./xetla_ref/build/xetla_int2_upcvt_ref
dt() { grep -a "Avg dev   time" | sed 's/.*: \([0-9.]*\) ms.*/\1/'; }
CFG=()
for mm in 16 32 64; do for mn in 16 32; do for wg in "1 8" "2 4" "4 2" "1 4" "2 2" "4 4"; do
  set -- $wg; CFG+=("--mt-m $mm --mt-n $mn --wg-m $1 --wg-n $2"); done; done; done
for wn in 16 32 64; do for ls in 1 2 4; do CFG+=("--sgm 8 --wgn $wn --ls $ls --u 1"); done; done
for M in 12 20 32 48; do
  bt=1e9; bc=""
  for c in "${CFG[@]}"; do
    t=$($O --m $M --k $K --n $N $c --iters 20 --no-validate 2>&1 | dt)
    [[ -n "$t" ]] && awk "BEGIN{exit !($t < $bt)}" && { bt=$t; bc=$c; }
  done
  xs=""; for mt in 1 2 3; do xs+=" mtile$mt=$($X --m $M --k $K --n $N --mtile $mt --iters 20 --no-validate 2>&1 | dt)"; done
  echo "K=$K N=$N M=$M best ocl [$bc] $bt ms | xetla$xs" | tee -a $OUT
done
