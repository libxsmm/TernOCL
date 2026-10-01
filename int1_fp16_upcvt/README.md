# int1_fp16_upcvt

1-bit (binary) weight GEMM/GEMV with fp16/bf16 activations, the 1-bit sibling
of [int2_fp16_upcvt](../int2_fp16_upcvt/):

    C[m,n] = DT( epi( sum_k A[m,k] * sgn(B[k,n]) * S[k/128, n] ) ),   bit 0 -> +1, bit 1 -> -1

| tensor | type | layout |
| --- | --- | --- |
| A | DT | `[M, K]` row-major |
| B | uint32 | `[K/32, N]` select masks: in K32 row `kp` and 16-column block `n0`, word `(kp, n0 + p)` holds row pair `p` of all 16 columns (below) |
| S | DT | `[K/128, N]` |
| C | DT, or fp32 with `--out-f32` | `[M, N]` |

DT is fp16, or bf16 with `--dtype bf16` (`-DBF16`). N must be a multiple of
16, K a multiple of 128; any M works. Weights take 1 bit + 16/128 bits of
scale = 1.125 bits per weight.

## Layout and dequantization ([int1_fp16_upcvt.cl](int1_fp16_upcvt.cl))

The B operand of the f16/bf16 DPAS is VNNI2: dword `c` of lane `n` holds K
rows `2c` (low half) and `2c+1` (high half) of column `n`. Register `c` of
the operand, read as 32 16-bit channels, is therefore channel
`j` = (column `j/2`, row `2c + (j & 1)`). B stores exactly that order: **mask
word `p` of a 16-column block has bit `j` = weight of (row `32kp + 2p + (j & 1)`,
column `n0 + j/2`)**, so the 1-bit weights map 1:1 onto the operand's
16-bit channels. Each DPAS register is one predicated select between the
hoisted `-s` and `+s` pairs (`s2 = s * 0x10001`, `n2 = s2 ^ 0x80008000`, once
per scale group), with the weight word as the predicate:

    setp P <- w(lane c)                       // the 32 weight bits are the predicate
    (P) sel (32) B[c]:uw  n2, s2              // -s where the bit is 1, +s where it is 0

OpenCL C cannot express a predicate from data bits (IGC turns it into
shifts), so this is inline vISA. The same code serves bf16, whose sign is
also bit 15. Size and 2D shape of B are those of a plain `[K/32, N]` uint32
array, so the loads are unchanged; a sub-group's lane `p` holds the mask of
row pair `p`, and the predicate comes from a scalar region of that register.

`--shl` (`-DSHL`) keeps the earlier path with a per-column VNNI2 bit order
(word `(kp, n)`: row `32kp + 2i` at bit `i`, row `32kp + 2i + 1` at bit `16 + i`),
where `s2 ^ ((w << (15 - i)) & 0x80008000)` builds dword `i` with one `shl` and
one `bfn` in plain C. The driver packs the gold weights into either layout.

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
16-column block. A comes in `MT_M/AR` reads of `AR x AK` (`AR` = 32, `AK` = 32 by
default: one 2 KiB `16b_32r16x2c` message per 32 rows and B word; `AK` = 16
for `MT_M > 64`, whose 32-K A tile would spill), then each
16-column block is dequantized once per K16 and reused for all `MT_M/8` DPAS.
2D block I/O zero-fills out-of-range reads (A rows, B/S columns) and clips
writes, so ragged tiles need no masks. 256 GRF by default (`--grf128`).

### Generated code (Xe2, IGC 26.35)

Checked with `ocloc -device bmg` and the IGC shader dumps (select path):

* Per DPAS register: one scalar flag write from the weight word and one
  SIMD32 `sel` on `:uw`, IGC rotating `f0..f3`:

      (W)      mov (1|M0)  f1.0<1>:ud  r4.11<0;1,0>:d
      (W&f0.0) sel (32|M0) r131.0<1>:uw  r49.0<1;1,0>:uw  r9.0<1;1,0>:uw   // -s, +s

* GEMV main loop (256 K): 16 `dpas.8x1`, 128 flag writes + 128 `sel`;
  B as 1-row `load_block2d.d32` messages, A as one `load.d32x64t` per 128 K.
* M-tiled loop (`MT_M = MT_N = 32`, 128 K): 64 `dpas.8x8`, 128 flag writes +
  128 `sel`, 2 `xor` (the hoisted `-s`), 2 B + 2 scale + 4 A 2D loads. No
  spills, no A/B register moves (the 2D-read GRF layout is the DPAS operand
  layout).

The `--shl` path has 15 `shl` + 16 `bfn` per B word in the same loops.

## Results (Arc Pro B70, fp16, Bonsai 27B shapes)

[bench.sh](bench.sh): best tile per shape, median of 3, >= 2 GiB of rotating
weights, device-event times. GiB/s counts A + B + S + C; B70 peak is 608 GB/s
(566 GiB/s). int2 = [int2_fp16_upcvt](../int2_fp16_upcvt/) OpenCL with its
select decode (same B70, same methodology).

Decode GEMV, M = 1 (`sel` and `--shl` are equal within 0.5%, the kernel is
DRAM-bound):

| shape | K x N | tile (wgn, ls, u) | time (us) | GiB/s | % peak | vs int2 |
| --- | --- | --- | --- | --- | --- | --- |
| gate_up | 5120 x 34816 | 32, 4, 2 | 45.3 | 517 | 91% | x1.92 |
| down | 17408 x 5120 | 16, 4, 2 | 23.3 | 502 | 89% | x1.83 |
| in_proj_qkvz | 5120 x 16384 | 32, 2, 2 | 21.9 | 504 | 89% | x1.83 |
| out_proj | 6144 x 5120 | 64, 4, 2 | 10.3 | 404 | 71% | x1.67 |
| qkv | 5120 x 14336 | 32, 2, 2 | 19.5 | 495 | 87% | x1.82 |
| lm_head | 5120 x 248320 | 32, 1, 2 | 302.2 | 553 | 98% | x1.89 |

Prefill, M = 1024:

| shape | tile (mt, wg) | `--shl` (ms) | `sel` (ms) | `sel` TFLOPS | sel vs shl | vs int2 |
| --- | --- | --- | --- | --- | --- | --- |
| gate_up | 64x32, 4x4 | 2.834 | 2.795 | 130.6 | +1.4% | x1.14 |
| down | 32x32, 1x8 | 1.359 | 1.343 | 135.9 | +1.2% | x1.25 |
| in_proj_qkvz | 32x32, 2x2 | 1.291 | 1.255 | 136.9 | +2.9% | x1.13 |
| out_proj | 32x32, 1x8 | 0.491 | 0.486 | 132.4 | +1.0% | x1.18 |
| qkv | 32x32, 2x2 | 1.110 | 1.102 | 136.4 | +0.7% | x1.11 |
| lm_head | 64x32, 4x2 | 21.37 | 20.92 | 124.5 | +2.1% | x1.13 |

With the dequantization replaced by a single `xor` (not a valid kernel, a
ceiling), the same M-tiled loop reaches about 148 TFLOPS on qkv.

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
`--mt-m --mt-n --wg-m --wg-n --grf128` (and `--opts -DAR=8|16|32`); `--shl`
for the shift path. M >= 64 or `--mt-m` selects the large-M kernel.

[validate.sh](validate.sh) covers GEMV, large-M and ragged shapes with
distinct sets for fp16 and bf16; `validate_epilogues.sh int1_fp16_upcvt`
covers POSTOP 0-4 with DT and fp32 outputs. [bench.sh](bench.sh) sweeps the
tiles per shape and reports the median of 3 re-measures of the winner. There
is no XeTLA counterpart.
