#!/bin/bash
# Validate the fused epilogues of both variants against the host fp32 gold:
# POSTOP 0..4 (none, silu(acc)*other, acc+other, acc+bias, sigmoid) with DT and
# fp32 outputs, fp16 + bf16, GEMV (M=1, ragged M=3) and large-M (ragged M=77).
#   BUILD=build bash validate_epilogues.sh [variant ...]
R=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-build}
VARIANTS=${*:-int2_fp16_upcvt int2_via_int2_x_int8_dpas}
pass=0; fail=0
run() {  # label, command...
    local label=$1; shift
    local out; out=$("$@" 2>&1)
    if grep -q "sets PASSED" <<<"$out" && ! grep -qE "FAILED|failed" <<<"$out"; then
        pass=$((pass + 1)); printf "PASS  %s\n" "$label"
    else
        fail=$((fail + 1)); printf "FAIL  %s\n" "$label"; grep -E "validation|FAILED|failed|idx" <<<"$out" | head -5
    fi
}
for v in $VARIANTS; do
    bin=$(ls "$R/$v/$BUILD"/*_ocl)
    qmodes=""; [[ $v == int2_via_int2_x_int8_dpas ]] && qmodes="0 1"
    for dt in fp16 bf16; do
        for p in 0 1 2 3 4; do
            for f32 in "" --out-f32; do
                for shape in "1 4096 5120" "3 2048 4096" "77 4096 2048"; do
                    set -- $shape
                    for q in ${qmodes:-x}; do
                        qa=(); [[ $q != x ]] && qa=(--qmode "$q")
                        run "$v $dt postop=$p ${f32:-dt-out} M=$1 N=$2 K=$3 ${qa[*]}" \
                            "$bin" --m "$1" --n "$2" --k "$3" --dtype "$dt" --postop "$p" $f32 \
                            "${qa[@]}" --sets 2 --iters 2
                    done
                done
            done
        done
    done
done
echo "epilogue validation: $pass passed, $fail failed"
[[ $fail == 0 ]]
