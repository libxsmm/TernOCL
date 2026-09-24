#!/bin/bash
# Bonsai 27B / Bonsai 2 27B (Qwen3.5-27B) decode GEMV shapes at one zero density:
# zsweep.sh per shape (xetla BITCOS vs OpenCL BITCOS, each tuned), with the int2
# baselines at the vLLM plugin's per-arch int2 config for that shape.
#   ARCH=b70|lnl Z=0.40 PACE=s bash shapes27b.sh <out.csv>
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?out.csv}; Z=${Z:-0.40}
[[ -z "${ARCH:-}" ]] && case $(hostname -s) in *lnl*) ARCH=lnl;; *) ARCH=b70;; esac
# name:K:N:plugin int2 (wgn,ls) b70:lnl
SH="gate_up:5120:34816:32,8:64,1 down:17408:5120:32,4:32,1 qkvz:5120:16384:32,2:32,1 out_proj:6144:5120:32,4:32,1 qkv:5120:14336:32,2:32,4 lm_head:5120:248320:32,4:64,1"
: > $OUT
for s in $SH; do IFS=: read nm K N ib il <<<"$s"
  c=$ib; [[ $ARCH == lnl ]] && c=$il; IFS=, read w l <<<"$c"
  N=$N K=$K ZS=$Z ARCH=$ARCH XI="--wgn $w --ks 1 --ls $l" bash $HERE/zsweep.sh $OUT.$nm > /dev/null
  { [[ -s $OUT ]] || head -1 $OUT.$nm | sed 's/^/shape,K,N,/'; tail -n +2 $OUT.$nm | sed "s/^/$nm,$K,$N,/"; } >> $OUT
  cat $OUT.$nm.sweep >> $OUT.sweep; rm -f $OUT.$nm $OUT.$nm.sweep
done
cat $OUT
