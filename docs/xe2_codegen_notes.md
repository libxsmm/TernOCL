# Codegen notes for Xe2 OpenCL

Findings from porting the XeTLA int2 kernels. They apply to XeTLA as well.

* **B loads:** load the packed B tile as 8 single-row 2D block reads, not one
  8-row read. The first DPAS then waits for 64 B instead of 512 B: 5-9% on
  decode GEMV, and parity on the smallest (4096x4096) shape.
* **fp16 accumulator spills:** in the large-M fp16 kernel, IGC folds
  `convert_half8` into the accumulator chain and spills at MT_M*MT_N >= 2048.
  An empty `asm volatile("" : "+rw"(v))` before the store stops it.
* **Divides:** under `-cl-fp32-correctly-rounded-divide-sqrt`, `1/x` becomes an
  IEEE divide sequence. Use `native_recip` (`math.inv`).
* **int8 saturation:** `convert_char_sat` costs 5 `sel` per element.
  `convert_char(clamp(f, -128, 127))` gives one `mov.sat`.
* **Sigmoid clamp:** IGC miscompiles `x <= -10 ? 0 : s` in the fp16 large-M
  silu store (all outputs 0). Multiply by `convert_float(x > -10)` instead.
* **Guarded loads:** guarded per-lane `sub_group_block_read` gets serialized.
  Use 2D block reads (pitch % 16 B, width >= 64 B).
* **Low-bit decode = predicated selects:** a VNNI2 DPAS B register, read as a
  SIMD32 16-bit operand, is channel j = (column j/2, row 2c + (j & 1)).
  Building it with `(P) sel (32) :uw -s2, +s2` (plus `(~Pz) mov 0` for ternary)
  costs one instruction per register once the predicate exists. The
  predicate comes from a scalar `setp` when the weights are stored in that
  channel order (int1), or from a SIMD32 `and.nz` with source region
  `<2;2,0>` (each lane's half-word to its two channels) against a `<0;2,1>`
  mask pair when they are not (XeTLA int2 layout). IGC turns any OpenCL C
  formulation into shifts and masks, so this needs inline vISA; `and` +
  `cmp.ne 0` in vISA folds into one flag-writing `and`. The int2 decode drops
  from ~10 to 4 instructions per register: x1.14-1.18 on decode GEMV, which
  was ALU-bound (the int2 GEMV with the decode removed reaches 556 GiB/s).
