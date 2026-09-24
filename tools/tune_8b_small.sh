#!/bin/bash
# Wider GEMV tile sweep for the small Bonsai-8B shapes (qkv, o_proj), then
# median-of-3 alternating re-measure of the best vs the plugin's xetla config.
cd "$(dirname "$0")/../int2_fp16_upcvt"
O=./build/int2_fp16_upcvt_ocl; X=./xetla_ref/build/xetla_int2_upcvt_ref
dt() { grep -a "Avg dev   time" | sed 's/.*: \([0-9.]*\) ms.*/\1/'; }
med() { printf "%s\n" "$@" | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
$O --m 1 --n 4096 --k 4096 --sets 1 --iters 1 | grep Device
for sh in "o_proj 4096 4096 --wgn 32 --ks 1 --ls 8" "qkv 4096 6144 --wgn 32 --ks 1 --ls 4"; do
  set -- $sh; name=$1 K=$2 N=$3; shift 3; xc="$*"
  best=""; bt=1e9
  for wn in 16 32 64; do for ls in 2 4 8 16; do for u in 1 2 4; do for pf in 0 1 2; do
    (( 16 * (wn / 16) * ls > 1024 )) && continue
    c="--wgn $wn --ls $ls --u $u --pf $pf"
    t=$($O --m 1 --k $K --n $N $c --iters 50 --no-validate 2>&1 | dt)
    echo "$name [$c] $t"
    [[ -n "$t" ]] && awk "BEGIN{exit !($t < $bt)}" && { bt=$t; best=$c; }
  done; done; done; done
  xs=(); os=()
  for r in 1 2 3; do
    xs+=($($X --m 1 --k $K --n $N $xc --iters 50 --no-validate 2>&1 | dt))
    os+=($($O --m 1 --k $K --n $N $best --iters 50 --no-validate 2>&1 | dt))
  done
  tx=$(med "${xs[@]}"); to=$(med "${os[@]}")
  echo "RESULT $name xetla($xc) $tx ms | ocl($best) $to ms | x$(awk "BEGIN{printf \"%.2f\", $tx/$to}") | reps x: ${xs[*]} o: ${os[*]}"
done
