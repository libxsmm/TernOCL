#!/bin/bash
# Zero-density shmoo of the paper's large GEMV (Fig. 13: N=32768, K=16384, M=1,
# every point tuned independently): xetla BITCOS (the paper's standalone
# bitcos_fp16_dpas_fp16scales_fast_test tuning binaries, production -D flags)
# vs this OpenCL kernel, plus the int2 references (xetla upcvt at the plugin's
# config and TernOCL int2_fp16_upcvt). Per z: sweep each kernel's tile list,
# then re-measure both winners alternately REPS times and take the median.
# >= 2 GiB of rotating distinct weights, device-event times.
#   ARCH=b70|lnl PACE=s bash zsweep.sh <out.csv>
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?out.csv}
XB=${XB:-/data/nfs_home/egeorgan/cpu_ternary_vllm/xetla/bitcos_fp16_dpas_fp16scales_fast_test/build}
O=$HERE/build/bitcos_fp16_upcvt_ocl
N=${N:-32768}; K=${K:-16384}; IT=${IT:-20}; REPS=${REPS:-3}
ZS=${ZS:-"0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95"}
[[ -z "${ARCH:-}" ]] && case $(hostname -s) in *lnl*) ARCH=lnl;; *) ARCH=b70;; esac
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
dt() { grep -a "Avg dev   time" | sed 's/.*: \([0-9.]*\) ms.*/\1/'; }
med() { printf "%s\n" "$@" | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
# actual bytes: bitmap + signs ((2-z) bits/weight) + offsets + scales + A + C
gbs() { awk -v z=$1 -v t=$2 -v n=$N -v k=$K 'BEGIN{printf "%.1f", (k*n*(2-z)/8 + 4*n + k/128*n*2 + 2*k + 2*n)/(t*1e-3)/2^30}'; }
# xetla tuning binaries (group of run_tuning_shmoo.sh), sg_n=16 sg_k=64
XC="1:32:1 1:32:2 1:32:4 1:32:8 1:32:16 1:64:1 1:64:2 1:64:4 1:64:8 2:128:1 2:128:2 2:128:4 11:128:8 2:256:1 2:256:2 11:256:4 2:512:1"
OC="32:1 32:2 32:4 32:8 32:16 64:1 64:2 64:4 64:8 64:16 128:1 128:2 128:4 128:8 256:1 256:2 256:4"
xrun() { IFS=: read g w l <<<"$1"; $XB/bitcos_tune_g$g --m 1 --n $N --k $K --z=$2 --iters $IT --min-footprint-gb 2.0 \
    --no-validate --wg-n $w --sg-n 16 --sg-k 64 --ks 1 --ls $l 2>&1 | dt; }
orun() { IFS=: read w l <<<"$1"; $O --m 1 --n $N --k $K --z $2 --wgn $w --ls $l --iters $IT --no-validate | dt; }
best() {  # best <fn> <z> <cfgs> -> "cfg time"
  local bc="" bt=1e9 c t
  for c in $3; do
    t=$($1 $c $2); echo "$ARCH,$1,$2,$c,$t" >> $OUT.sweep
    [[ -n "$t" ]] && awk "BEGIN{exit !($t < $bt)}" && { bt=$t; bc=$c; }
  done
  echo "$bc $bt"
}
echo "machine,z,xetla_cfg,xetla_ms,xetla_gibs,ocl_cfg,ocl_ms,ocl_gibs,speedup" > $OUT
: > $OUT.sweep
for z in $ZS; do
  read xc _ <<<"$(best xrun $z "$XC")"
  read oc _ <<<"$(best orun $z "$OC")"
  xs=(); os=()
  for r in $(seq $REPS); do
    sleep ${PACE:-0}; xs+=($(xrun $xc $z))
    sleep ${PACE:-0}; os+=($(orun $oc $z))
  done
  tx=$(med "${xs[@]}"); to=$(med "${os[@]}")
  echo "$ARCH,$z,wgn${xc#*:},$tx,$(gbs $z $tx),wgn$oc,$to,$(gbs $z $to),$(awk "BEGIN{printf \"%.3f\", $tx/$to}")" | tee -a $OUT
done
# int2 references at the same shape (2 bits/weight)
I2=$HERE/../int2_fp16_upcvt
case $ARCH in lnl) xi="--wgn 128 --ks 1 --ls 2";; *) xi="--wgn 128 --ks 1 --ls 1";; esac
xi=${XI:-$xi}
xis=(); ois=()
for r in $(seq $REPS); do
  sleep ${PACE:-0}; xis+=($($I2/xetla_ref/build/xetla_int2_upcvt_ref --m 1 --n $N --k $K $xi --iters $IT --no-validate | dt))
  sleep ${PACE:-0}; ois+=($($I2/build/int2_fp16_upcvt_ocl --m 1 --n $N --k $K --iters $IT --no-validate | dt))
done
echo "# int2 xetla($xi) $(med "${xis[@]}") ms | int2 TernOCL (default tile) $(med "${ois[@]}") ms" | tee -a $OUT
