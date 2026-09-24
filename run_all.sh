#!/bin/bash
# Validate and benchmark TernOCL against xetla on the current GPU node.
#   bash run_all.sh [validate] [epilogues] [bench] [VARIANT=int2_fp16_upcvt|int2_via_int2_x_int8_dpas|all]
#                   [DTYPES="fp16 bf16"] [MS="1 1024"]
# Results go to <variant>/results/{val,bench}_<arch>_<dtype>_m<M>.txt (+ .sweep
# with every tile tried). On LNL set PACE=15 (shared-memory bandwidth drifts
# under sustained load). Needs oneAPI + intel-gpu env and built binaries (README).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ARCH=${ARCH:-$(case $(hostname -s) in *lnl*) echo lnl;; *) echo b70;; esac)}
VARIANT=${VARIANT:-all}; DTYPES=${DTYPES:-fp16 bf16}; MS=${MS:-1 1024}
[[ $VARIANT == all ]] && VARIANT="int2_fp16_upcvt int2_via_int2_x_int8_dpas"
DO=${*:-validate epilogues bench}
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
for v in $VARIANT; do
  d=$HERE/$v; mkdir -p $d/results
  if [[ " $DO " == *" validate "* ]]; then
    bash $d/validate.sh $d > $d/results/val_$ARCH.txt 2>&1
    echo "$v validate [$ARCH]: $(grep -c '2/2 sets PASSED' $d/results/val_$ARCH.txt) passed," \
         "$(grep -c 'Validation summary' $d/results/val_$ARCH.txt) OpenCL cases"
  fi
  if [[ " $DO " == *" epilogues "* ]]; then
    bash $HERE/validate_epilogues.sh $v > $d/results/val_epi_$ARCH.txt 2>&1 || true
    echo "$v epilogues [$ARCH]: $(tail -1 $d/results/val_epi_$ARCH.txt)"
  fi
  if [[ " $DO " == *" bench "* ]]; then
    for dt in $DTYPES; do for m in $MS; do
      out=$d/results/bench_${ARCH}_${dt}_m$m.txt; rm -f $out $out.sweep
      it=20; [[ $m -gt 1 ]] && it=10; [[ $v == int2_fp16_upcvt && $m == 1 ]] && it=50
      ARCH=$ARCH DT=$dt M=$m IT=$it bash $d/bench.sh $d $out > /dev/null
      echo "$v bench [$ARCH $dt M=$m] -> $out"
    done; done
  fi
done
