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
