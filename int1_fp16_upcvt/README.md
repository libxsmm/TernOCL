# int1_fp16_upcvt

1-bit (binary) weight GEMM/GEMV with fp16/bf16 activations, the 1-bit sibling
of [int2_fp16_upcvt](../int2_fp16_upcvt/):

    C[m,n] = DT( epi( sum_k A[m,k] * sgn(B[k,n]) * S[k/128, n] ) ),   bit 0 -> +1, bit 1 -> -1

| tensor | type | layout |
| --- | --- | --- |
| A | DT | `[M, K]` row-major |
| B | uint32 | `[K/32, N]`: word `(kp, n)` holds the bits of K rows `32kp..32kp+31` of column n, VNNI2 bit order (below) |
| S | DT | `[K/128, N]` |
| C | DT, or fp32 with `--out-f32` | `[M, N]` |

DT is fp16, or bf16 with `--dtype bf16` (`-DBF16`). N must be a multiple of
16, K a multiple of 128; any M works. Weights take 1 bit + 16/128 bits of
scale = 1.125 bits per weight.

## Layout and dequantization ([int1_fp16_upcvt.cl](int1_fp16_upcvt.cl))

The B operand of the f16/bf16 DPAS is VNNI2: dword `c` of lane `n` holds K
rows `2c` (low half) and `2c+1` (high half) of column `n`. B uses the same
pairing at bit granularity: **row `32kp + 2i` is bit `i`, row `32kp + 2i + 1`
is bit `16 + i`** (i = 0..15). The two bits of pair `i` are then exactly 16
apart, so one shift puts both on the DT sign bits (15 and 31), and one `bfn`
merges them into the hoisted scale pair `s2 = s | s << 16`:

    b[i] = s2 ^ ((w << (15 - i)) & 0x80008000)     // +s where the bit is 0, -s where it is 1

That is 2 ALU instructions (`shl` + `bfn`) per DPAS dword (1 for `i = 15`),
no shuffles, no SLM, no moves. One B word feeds two K16 DPAS. The same code
serves bf16, whose sign is also bit 15. The scale row is read in the
column-per-lane order of the B tile (lane `n` gets `S[g, n]`), so it needs no
rearrangement either.

### `int1_fp16_upcvt_gemm` (decode, M small)

Same structure as the int2 GEMV: a SIMD16 sub-group owns 16 columns and `SGM`
rows (DPAS repeat count), walks its K slice in steps of 128 (4 B words, one
scale, 8 DPAS), `LS` sub-groups split K and reduce through SLM.
Knobs: `SGM`, `NSG_N` (`--wgn`), `LS`, `U` (k-steps whose loads are issued
before any compute), `BR` (B rows per 2D block read: 1, 2 or 4; 1 = one
64 B message per row, measured equal within noise).

### `int1_fp16_upcvt_gemm_mt` (prefill, any M)

A sub-group computes an `MT_M x MT_N` tile; a work-group is `WG_M x WG_N`
sub-groups. Per 128-K step: one 4-row 2D read of B and one 2D scale read per
16-column block. Per B word (K32): `MT_M/AR` A reads of `AR x 32` (`AR` = 32
by default: one 2 KiB `16b_32r16x2c` message per 32 rows), then each
16-column block is dequantized once per K16 and reused for all `MT_M/8` DPAS.
2D block I/O zero-fills out-of-range reads (A rows, B/S columns) and clips
writes, so ragged tiles need no masks. 256 GRF by default (`--grf128`).

### Generated code (Xe2, IGC 26.35)

Checked with `ocloc -device bmg` and the IGC shader dumps:

* GEMV inner loop: per B word (2 DPAS) exactly 15 `shl` + 16 `bfn` +
  2 `dpas.8x1`; B as 1-row `load_block2d.d32` messages, A as one
  `load.d32x64t` per 128 K. Scale pair: 3 instructions per 128 K. No `mov`s.
* M-tiled loop (`MT_M = MT_N = 32`): 64 `dpas.8x8`, 128 `bfn`, 122 `shl`,
  2 B + 2 scale + 4 A 2D loads, 2 scalar `mov`s per 128 K. No spills, no
  A/B register moves (the 2D-read GRF layout is the DPAS operand layout).

The compiler emits the intended sequence, so no inline vISA is needed.

## Driver ([main.cpp](main.cpp))

```bash
make CXX=g++
./build/int1_fp16_upcvt_ocl --m 1 --k 17408 --n 5120                    # GEMV, validated
./build/int1_fp16_upcvt_ocl --m 1024 --k 5120 --n 34816 --dtype bf16     # large-M kernel
./build/int1_fp16_upcvt_ocl --m 1 --k 5120 --n 34816 --postop 1          # fused SwiGLU gate
```

Same methodology and flags as the int2 driver: host fp32 gold, then the
epilogue, then the output cast; `xetla_buff_cmp` pass rule; `>= 2 GiB` of
rotating distinct weight sets (B + S) with one warm-up pass; device-event
timing. GEMV tile flags: `--sgm --wgn|--nsg --ls --u --br`; large-M:
`--mt-m --mt-n --wg-m --wg-n --grf128` (and `--opts -DAR=8|16|32`).
M >= 64 or `--mt-m` selects the large-M kernel.

[validate.sh](validate.sh) covers GEMV, large-M and ragged shapes with
distinct sets for fp16 and bf16; `validate_epilogues.sh int1_fp16_upcvt`
covers POSTOP 0-4 with DT and fp32 outputs. [bench.sh](bench.sh) sweeps the
tiles per shape and reports the median of 3 re-measures of the winner. There
is no XeTLA counterpart.
