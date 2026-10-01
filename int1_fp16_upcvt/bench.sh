#!/bin/bash
# int1_fp16_upcvt benchmark: tile sweep per shape (best of the OT list), then the
# winner re-measured REPS times, median reported. >= 2 GiB of rotating distinct
# weights, device-event times, IGC dumps off.
#   ARCH=b70|lnl DT=fp16|bf16 M=1|1024 [X=--shl] bash bench.sh <dir> <out.txt>   (M=1 -> GEMV tiles)
DT=${DT:-fp16}
cd "$1"; OUT=$2; O="./build/int1_fp16_upcvt_ocl --dtype $DT ${X:-}"
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
M=${M:-1024}; IT=${IT:-20}; REPS=${REPS:-3}
SHAPES=${SHAPES:-"27B.gate_up:5120:34816 27B.down:17408:5120 27B.qkvz:5120:16384 27B.out_proj:6144:5120 27B.qkv:5120:14336 27B.lm_head:5120:248320"}
dt() { grep -a "Avg dev   time" | sed 's/.*: \([0-9.]*\) ms.*/\1/'; }
if [[ -n "$OT" ]]; then :
elif [[ $M -gt 1 ]]; then
  OT=""; for t in "32 16" "64 16" "128 16" "32 32" "64 32"; do for wg in "2 4" "4 4" "4 2" "1 8" "2 2"; do
    set -- $t $wg; OT+="--mt-m $1 --mt-n $2 --wg-m $3 --wg-n $4|"; done; done
else
  OT=""; for wn in 16 32 64; do for ls in 1 2 4 8; do for u in 1 2 4; do OT+="--wgn $wn --ls $ls --u $u|"; done; done; done
fi
OT=${OT%|}
fl() { awk -v m=$M -v n=$1 -v k=$2 -v t=$3 'BEGIN{printf "%.1f", 2*m*n*k/t/1e9}'; }
gb() { awk -v m=$M -v n=$1 -v k=$2 -v t=$3 'BEGIN{printf "%.1f", (m*k*2+k*n/8+m*n*2+(k/128)*n*2)/t*1e3/2^30}'; }
med() { printf "%s\n" "$@" | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
best() {  # best <binary> <cfg list> -> "cfg|time"
  local bin=$1 bc="" bt=1e9 c t; IFS='|' read -ra cs <<<"$2"
  for c in "${cs[@]}"; do
    t=$($bin --m $M --k $K --n $N $c --iters $IT --no-validate 2>&1 | dt)
    echo "$name [$c] $t" >> $OUT.sweep
    [[ -n "$t" ]] && awk "BEGIN{exit !($t < $bt)}" && { bt=$t; bc=$c; }
  done
  echo "$bc|$bt"
}
for sh in $SHAPES; do IFS=: read name K N <<<"$sh"
  IFS='|' read oc _ <<<"$(best "$O" "$OT")"
  os=()
  for r in $(seq $REPS); do
    sleep ${PACE:-0}
    os+=($($O --m $M --k $K --n $N $oc --iters $IT --no-validate 2>&1 | dt))
  done
  to=$(med "${os[@]}")
  printf "%-12s %s M=%-4s K=%-5s N=%-6s | ocl(%s) %s ms %s TFLOPS %s GiB/s\n" \
    $name $DT $M $K $N "$oc" $to $(fl $N $K $to) $(gb $N $K $to) | tee -a $OUT
  echo "   reps: ${os[*]}" | tee -a $OUT
done
