#!/bin/bash
# M >= 64 tile sweep for the upcvt M-tiled kernel, one shape, one M.
#   bash sweep_mt.sh <K> <N> <M> <out>   (cwd: int2_fp16_upcvt)
K=$1; N=$2; M=$3; OUT=$4
O=./build/int2_fp16_upcvt_ocl
dt() { grep -a "Avg dev   time" | sed 's/.*: \([0-9.]*\) ms.*/\1/'; }
bt=1e9; bc=""
for t in "32 16" "64 16" "128 16" "32 32" "64 32"; do for wg in "1 8" "2 4" "4 2" "2 2" "4 4" "1 4"; do
  set -- $t $wg; c="--mt-m $1 --mt-n $2 --wg-m $3 --wg-n $4"
  s=$($O --m $M --k $K --n $N $c --iters 5 --no-validate 2>&1 | dt)
  [[ -n "$s" ]] && awk "BEGIN{exit !($s < $bt)}" && { bt=$s; bc=$c; }
done; done
echo "K=$K N=$N M=$M best ocl [$bc] $bt ms" | tee -a $OUT
