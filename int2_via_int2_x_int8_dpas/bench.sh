#!/bin/bash
# Best-of-xetla vs best-of-OpenCL, both with >= 2 GiB of rotating distinct
# weights, device-event times, dumps off. PACE=s sleeps before each re-measure run
# (LNL: shared-memory bandwidth drifts under sustained load).
#   DT=fp16|bf16 M=1024 bash bench.sh <dir> <out.txt>     (M=1 -> GEMV tiles)
DT=${DT:-fp16}
cd "$1"; OUT=$2; O="./build/int2_int8_dpas_ocl --dtype $DT"
X=./xetla_ref/build/xetla_int2_int8_ref; [[ $DT == bf16 ]] && X=./xetla_ref_bf16/build/xetla_int2_bf16_int8_ref
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
M=${M:-1024}; IT=${IT:-20}; REPS=${REPS:-3}; SHAPES=${SHAPES:-"8B.qkv:4096:6144 8B.o_proj:4096:4096 8B.gate_up:4096:24576 8B.down:12288:4096 8B.lm_head:4096:151680 27B.gate_up:5120:34816 27B.down:17408:5120 27B.qkvz:5120:16384 27B.out_proj:6144:5120 27B.qkv:5120:14336 27B.lm_head:5120:248320"}
ot() { grep -a "Avg dev   $1" | sed 's/.*: \([0-9.]*\) ms.*/\1/'; }     # ms
xt() { awk '/device GEMM time per set/{g=$(NF-1)} /Average device time per set/{t=$(NF-1)} END{print g, t}'; }
if [[ $M -gt 1 ]]; then
  XT="64,8,128,128,32 32,8,256,128,64 256,8,160,160,32 32,8,160,160,32 64,8,160,160,32 64,8,256,128,128 64,8,256,128,64"
  OT="--mt-m 8 --mt-n 128 --wg-m 8 --wg-n 2|--mt-m 8 --mt-n 128 --wg-m 4 --wg-n 2|--mt-m 8 --mt-n 128 --wg-m 4 --wg-n 4|--mt-m 8 --mt-n 128 --wg-m 16 --wg-n 1|--mt-m 8 --mt-n 128 --wg-m 2 --wg-n 4|--mt-m 16 --mt-n 64 --wg-m 4 --wg-n 2|--mt-m 16 --mt-n 64 --wg-m 8 --wg-n 2|--mt-m 8 --mt-n 64 --wg-m 8 --wg-n 2|--mt-m 32 --mt-n 32 --wg-m 4 --wg-n 2"
else
  # each harness's precompiled GEMV variants (the bf16 one has no sg_k=64, other wg_n)
  XT=""; if [[ $DT == bf16 ]]; then
    for wn in 32 64 80 128; do for sn in 16 32; do for sk in 32 128 256; do XT+=" 1,1,$wn,$sn,$sk"; done; done; done
  else
    for wn in 64 128 256; do for sn in 16 32; do for sk in 32 64 128; do XT+=" 1,1,$wn,$sn,$sk"; done; done; done
  fi
  OT=""; for nsg in 1 2 4; do for ls in 1 2 4 8; do for u in 1 2; do OT+="--nsg $nsg --ls $ls --u $u|"; done; done; done; OT=${OT%|}
fi
fl() { awk -v m=$M -v n=$1 -v k=$2 -v t=$3 'BEGIN{printf "%.1f", 2*m*n*k/t/1e9}'; }   # TFLOPS/TOPS
gb() { awk -v m=$M -v n=$1 -v k=$2 -v t=$3 'BEGIN{printf "%.1f", (m*k*2+k*n/4+m*n*2+(k/128)*(n+m)*2)/t*1e3/2^30}'; }  # GiB/s
for sh in $SHAPES; do IFS=: read name K N <<<"$sh"
  # 1) pick best xetla tile and best OCL tile per mode by total device time
  bx=""; bxt=1e9
  for t in $XT; do IFS=, read wm sm wn sn sk <<<"$t"
    read g tt < <($X --mat_m=$M --mat_k=$K --mat_n=$N --scale_gs=$((K/128)) --wg_m=$wm --sg_m=$sm --wg_n=$wn --sg_n=$sn --sg_k=$sk --iters=$IT --validation=0 2>&1 | xt)
    echo "$name xetla $t gemm=$g total=$tt" >> $OUT.sweep
    [[ -n "$tt" ]] && awk "BEGIN{exit !($tt < $bxt)}" && { bxt=$tt; bx=$t; }
  done
  declare -A bo bot; for qm in 0 1; do bo[$qm]=""; bot[$qm]=1e9
    IFS='|' read -ra cfgs <<<"$OT"
    for c in "${cfgs[@]}"; do
      tt=$($O --m $M --k $K --n $N --qmode $qm $c --iters $IT --no-validate 2>&1 | ot time)
      echo "$name ocl qm=$qm [$c] total=$tt best=${bot[$qm]}" >> $OUT.sweep
      [[ -n "$tt" ]] && awk "BEGIN{exit !($tt < ${bot[$qm]})}" && { bot[$qm]=$tt; bo[$qm]="$c"; }
    done
  done
  # 2) re-measure the three winners alternately REPS times, report the median
  xs=(); o0=(); o1=()
  IFS=, read wm sm wn sn sk <<<"$bx"
  for r in $(seq $REPS); do
    sleep ${PACE:-0}
    read g tt < <($X --mat_m=$M --mat_k=$K --mat_n=$N --scale_gs=$((K/128)) --wg_m=$wm --sg_m=$sm --wg_n=$wn --sg_n=$sn --sg_k=$sk --iters=$IT --validation=0 2>&1 | xt); xs+=($tt)
    sleep ${PACE:-0}
    o0+=($($O --m $M --k $K --n $N --qmode 0 ${bo[0]} --iters $IT --no-validate 2>&1 | ot time))
    sleep ${PACE:-0}
    o1+=($($O --m $M --k $K --n $N --qmode 1 ${bo[1]} --iters $IT --no-validate 2>&1 | ot time))
  done
  med() { printf "%s\n" "$@" | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
  tx=$(med "${xs[@]}"); t0=$(med "${o0[@]}"); t1=$(med "${o1[@]}")
  line=$(printf "%-12s %s M=%-4s K=%-5s N=%-6s | xetla(%s) %s ms %s TOPS %s GiB/s | ocl-vISA(%s) %s ms %s TOPS %s GiB/s x%.2f | ocl-upfront(%s) %s ms %s TOPS %s GiB/s x%.2f" \
    $name $DT $M $K $N "$bx" $tx $(fl $N $K $tx) $(gb $N $K $tx) "${bo[1]}" $t1 $(fl $N $K $t1) $(gb $N $K $t1) $(awk "BEGIN{print $tx/$t1}") \
    "${bo[0]}" $t0 $(fl $N $K $t0) $(gb $N $K $t0) $(awk "BEGIN{print $tx/$t0}"))
  echo "$line" | tee -a $OUT
  echo "   reps xetla: ${xs[*]} | vISA: ${o1[*]} | upfront: ${o0[*]}" | tee -a $OUT
done
