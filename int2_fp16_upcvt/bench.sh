#!/bin/bash
# xetla runs the vLLM plugin's tuned config per arch; OpenCL is tuned per shape
# (best of the OT list), both with >= 2 GiB of rotating distinct weights
# (same set rule), device-event times, dumps off. PACE=s sleeps before each
# re-measure run (LNL: shared-memory bandwidth drifts under sustained load).
#   ARCH=b70|lnl DT=fp16|bf16 M=1024 bash bench.sh <dir> <out.txt>   (M=1 -> GEMV tiles)
DT=${DT:-fp16}
cd "$1"; OUT=$2; O="./build/int2_fp16_upcvt_ocl --dtype $DT"
X=./xetla_ref/build/xetla_int2_upcvt_ref; [[ $DT == bf16 ]] && X+=_bf16
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
[[ -z "$ARCH" ]] && case $(hostname -s) in *lnl*) ARCH=lnl;; *) ARCH=b70;; esac
M=${M:-1024}; IT=${IT:-20}; REPS=${REPS:-3}
SHAPES=${SHAPES:-"8B.qkv:4096:6144 8B.o_proj:4096:4096 8B.gate_up:4096:24576 8B.down:12288:4096 8B.lm_head:4096:151680 27B.gate_up:5120:34816 27B.down:17408:5120 27B.qkvz:5120:16384 27B.out_proj:6144:5120 27B.qkv:5120:14336 27B.lm_head:5120:248320"}
dt() { grep -a "Avg dev   time" | sed 's/.*: \([0-9.]*\) ms.*/\1/'; }
if [[ $M -gt 1 ]]; then
  OT=""; for t in "32 16" "64 16" "96 16" "128 16" "32 32" "64 32"; do for wg in "2 4" "4 4" "2 2" "1 8" "4 2" "8 1"; do
    set -- $t $wg; OT+="--mt-m $1 --mt-n $2 --wg-m $3 --wg-n $4|"; done; done
else
  OT=""; for wn in 16 32 64; do for ls in 1 2 4 6 8; do for u in 1 2; do OT+="--wgn $wn --ls $ls --u $u|"; done; done; done
fi
OT=${OT%|}
fl() { awk -v m=$M -v n=$1 -v k=$2 -v t=$3 'BEGIN{printf "%.1f", 2*m*n*k/t/1e9}'; }
gb() { awk -v m=$M -v n=$1 -v k=$2 -v t=$3 'BEGIN{printf "%.1f", (m*k*2+k*n/4+m*n*2+(k/128)*n*2)/t*1e3/2^30}'; }
med() { printf "%s\n" "$@" | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
best() {  # best <binary> <cfg list> <label> -> "cfg|time"
  local bin=$1 bc="" bt=1e9 c t; IFS='|' read -ra cs <<<"$2"
  for c in "${cs[@]}"; do
    t=$($bin --m $M --k $K --n $N $c --iters $IT --no-validate 2>&1 | dt)
    echo "$name $3 [$c] $t" >> $OUT.sweep
    [[ -n "$t" ]] && awk "BEGIN{exit !($t < $bt)}" && { bt=$t; bc=$c; }
  done
  echo "$bc|$bt"
}
for sh in $SHAPES; do IFS=: read name K N <<<"$sh"
  # xetla: the vLLM plugin's tuned dispatch (csrc/int2_fp16_upcvt_kernel.sycl)
  if [[ $M -gt 1 ]]; then xc="--mtile 1"
  else case "$ARCH:$K:$N" in
    b70:4096:6144) xc="--wgn 32 --ks 1 --ls 4";;    lnl:4096:6144) xc="--wgn 32 --ks 1 --ls 1";;
    b70:4096:4096) xc="--wgn 32 --ks 1 --ls 8";;    lnl:4096:4096) xc="--wgn 32 --ks 1 --ls 2";;
    b70:4096:24576) xc="--wgn 32 --ks 1 --ls 4";;   lnl:4096:24576) xc="--wgn 256 --ks 1 --ls 1";;
    b70:12288:4096) xc="--wgn 32 --ks 1 --ls 8";;   lnl:12288:4096) xc="--wgn 32 --ks 1 --ls 2";;
    b70:4096:151680) xc="--wgn 32 --ks 1 --ls 4";;  lnl:4096:151680) xc="--wgn 128 --ks 1 --ls 2";;
    b70:5120:34816) xc="--wgn 32 --ks 1 --ls 8";;   lnl:5120:34816) xc="--wgn 64 --ks 1 --ls 1";;
    b70:17408:5120) xc="--wgn 32 --ks 1 --ls 4";;   lnl:17408:5120) xc="--wgn 32 --ks 1 --ls 1";;
    b70:5120:16384) xc="--wgn 32 --ks 1 --ls 2";;   lnl:5120:16384) xc="--wgn 32 --ks 1 --ls 1";;
    b70:6144:5120) xc="--wgn 32 --ks 1 --ls 4";;    lnl:6144:5120) xc="--wgn 32 --ks 1 --ls 1";;
    b70:5120:14336) xc="--wgn 32 --ks 1 --ls 2";;   lnl:5120:14336) xc="--wgn 32 --ks 1 --ls 4";;
    b70:5120:248320) xc="--wgn 32 --ks 1 --ls 4";;  lnl:5120:248320) xc="--wgn 64 --ks 1 --ls 1";;
    *) echo "no plugin config for $ARCH:$K:$N"; exit 1;;
  esac; fi
  IFS='|' read oc _ <<<"$(best "$O" "$OT" ocl)"
  xs=(); os=()
  for r in $(seq $REPS); do
    sleep ${PACE:-0}
    xs+=($($X --m $M --k $K --n $N $xc --iters $IT --no-validate 2>&1 | dt))
    sleep ${PACE:-0}
    os+=($($O --m $M --k $K --n $N $oc --iters $IT --no-validate 2>&1 | dt))
  done
  tx=$(med "${xs[@]}"); to=$(med "${os[@]}")
  printf "%-12s %s M=%-4s K=%-5s N=%-6s | xetla(%s) %s ms %s TFLOPS %s GiB/s | ocl(%s) %s ms %s TFLOPS %s GiB/s x%.2f\n" \
    $name $DT $M $K $N "$xc" $tx $(fl $N $K $tx) $(gb $N $K $tx) "$oc" $to $(fl $N $K $to) $(gb $N $K $to) $(awk "BEGIN{print $tx/$to}") | tee -a $OUT
  echo "   reps xetla: ${xs[*]} | ocl: ${os[*]}" | tee -a $OUT
done
