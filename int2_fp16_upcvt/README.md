# int2_fp16_upcvt

OpenCL port of the XeTLA int2 weight-only GEMM with fp16/bf16 activations
(`xetla/include/experimental/group/gemm/impl/int2_fp16_upcvt_xmx_xe.hpp`,
which vLLM runs through `csrc/int2_fp16_upcvt_kernel.sycl`).

    C[m,n] = DT( epi( sum_k A[m,k] * code(B[k,n]) * S[k/128, n] ) )

| tensor | type | layout |
| --- | --- | --- |
| A | DT | `[M, K]` row-major |
| B | uint32 | `[K/16, N]`: word `(kp, n)` holds the 2-bit codes of K rows `16kp..16kp+15` of column n |
| S | DT | `[K/128, N]` |
| C | DT, or fp32 with `--out-f32` | `[M, N]` |

DT is fp16, or bf16 with `--dtype bf16` (`-DBF16`, XeTLA's `XT` template
parameter). Codes are `{0, 1, 3}` = `{0, +1, -1}`, the same as the plugin
sidecars. N must be a multiple of 16, K a multiple of 128; any M works.

## Kernels ([int2_fp16_upcvt.cl](int2_fp16_upcvt.cl))

**Dequantization** (both kernels): predicated selects on the unchanged
XeTLA weight layout (inline vISA). Register `c` of the DPAS B operand, read
as 32 16-bit channels, is channel `j` = (column `j/2`, row `2c + (j & 1)`). Both
codes of row pair `c` sit in one half-word of lane `j/2`'s B word, so a SIMD32
`and.nz` whose source region `<2;2,0>` hands each lane's half-word to its two
channels, against a hoisted mask pair read as `<0;2,1>` (even / odd row),
yields the sign and the nonzero predicate of all 32 channels at once:

    (W)       and (32|M0) (ne)f1.0 null:uw  w.h<2;2,0>:uw  mk.sign<0;2,1>:uw
    (W)       and (32|M0) (ne)f2.0 null:uw  w.h<2;2,0>:uw  mk.nz<0;2,1>:uw
    (W&f1.0)  sel (32|M0) B[c]:uw  -s2  +s2
    (W&~f2.0) mov (32|M0) B[c]:hf  0

4 instructions per register (the `and` + `cmp.ne` pairs of the vISA fold into
flag-writing `and`s), against about 10 for XeTLA's integer upconvert,
`(scale2 ^ sign) & magnitude` per nibble, which `-DINT_DQ` (`--int-dq`) keeps.
The output bits are the same (`+s`, `s ^ 0x8000`, `0`): results are
bit-identical to the XeTLA plugin kernel, see
[tools/xetla_epilogue_parity](../tools/xetla_epilogue_parity/). No shuffle,
no SLM, and the same for bf16 (sign = bit 15).

### `int2_fp16_upcvt_gemm` (decode, M small)

* **Work split:** a SIMD16 sub-group owns 16 columns (one per lane) and `SGM`
  rows, the DPAS repeat count (1, 2, 4 or 8). It walks its K slice in steps
  of 128 (one scale group). `LS` sub-groups split K and reduce through SLM;
  any LS works, not just powers of 2.
* **Per step:** B as 8 single-row 2D block reads (see below), one scale row,
  `SGM` A rows, then 8 DPAS.
* **B load form:** single-row reads let the first DPAS start once row 0 has
  arrived. Against one 8-row 2D read this is 5-9% faster (`-DB_BLK8` restores
  the 8-row read).
* **Knobs:** `SGM`, `NSG_N` (a work-group covers `16*NSG_N` columns), `LS`, `U`
  (k-steps whose loads are issued before any compute) and `PF` (B prefetch
  distance, 0 = off).

### `int2_fp16_upcvt_gemm_mt` (prefill, any M)

* **Tile:** a sub-group computes an `MT_M x MT_N` tile; a work-group is
  `WG_M x WG_N` sub-groups.
* **Reuse:** per K16, each 16-column B block is dequantized once and reused
  for all `MT_M/8` DPAS row blocks.
* **Memory:** A, B and C go through 2D block I/O, which zero-fills
  out-of-range reads and clips writes, so ragged M and N tiles need no masks.
* **Loads:** B is one 8-row 2D read per 16-column block; `-DMT_B_ROWS` gives
  single-row reads (experimental).
* **Registers:** 256 GRF by default (`--grf128` for 128).
* **fp16 spill fix:** an empty `asm volatile` on each accumulator before the
  store stops IGC from folding the fp16 conversion into the K loop, which
  spilled at `MT_M*MT_N >= 2048` (fp16 only).

## Driver ([main.cpp](main.cpp))

```bash
make CXX=g++
./build/int2_fp16_upcvt_ocl --m 1 --k 17408 --n 5120                     # GEMV, validated
./build/int2_fp16_upcvt_ocl --m 1024 --k 5120 --n 34816 --dtype bf16      # large-M kernel
./build/int2_fp16_upcvt_ocl --m 1 --k 5120 --n 34816 --postop 1           # fused SwiGLU gate
```

* **Inputs and gold:** the same input generation as the XeTLA harness; host
  gold in fp32, then the epilogue, then the output cast.
* **Pass rule:** `xetla_buff_cmp` (fp16: ulp 64 / abs 8, bf16: 32 / 16,
  rel 1e-3). fp32 output: `|d| <= 0.05 || rel <= 2e-4`.
* **Sets:** with shared inputs, every set must reproduce set 0 bit for bit.

Flags:

| group | flags |
| --- | --- |
| problem | `--m --n --k --dtype fp16\|bf16` |
| timing and validation | `--iters --no-validate --distinct-sets`, `--sets N` or `--weights-gib G` (default 2) |
| GEMV tile | `--sgm --wgn\|--nsg --ls --u --pf` |
| large-M tile | `--mt-m --mt-n --wg-m --wg-n --grf128` |
| epilogue | `--postop 0..4 --out-f32` |
| debug | `--cl <file> --opts "-D..." --build-log`, `--int-dq` (XeTLA's integer decode) |

M >= 64 or `--mt-m` selects the large-M kernel. The default tiles are the B70
27B table in `default_tiles()`; [bench.sh](bench.sh) sweeps the tiles per shape.

## XeTLA reference ([xetla_ref/](xetla_ref/))

A copy of the plugin's `int2_fp16_upcvt_dpas_fast_test` driver, with the same
rotating-set rule as the OpenCL driver. It takes explicit GEMV tiles
(`--wgn --ks --ls`, the plugin's dispatch table) and prefill tiles (`--mtile 1..7`;
1 is the plugin's default for M > 16). [bench.sh](bench.sh) runs it at the
plugin's tuned per-arch config for every shape.

## Results (Arc Pro B70, fp16, median of 3, >= 2 GiB rotating weights)

Decode GEMV, M = 1, with the plugin's tuned XeTLA config per shape (pcl-arl01,
one full run). "before" = the previously published OpenCL times with XeTLA's
integer decode (still available as `--int-dq`).

| shape | K x N | XeTLA (us) | OpenCL (us) | GiB/s | speed-up | before |
| --- | --- | --- | --- | --- | --- | --- |
| 8B qkv | 4096 x 6144 | 16.72 | 14.39 | 434 | x1.16 | 16.51 |
| 8B o_proj | 4096 x 4096 | 11.78 | 10.35 | 403 | x1.14 | 11.70 |
| 8B gate_up | 4096 x 24576 | 58.39 | 47.23 | 528 | x1.24 | 55.36 |
| 8B down | 12288 x 4096 | 29.96 | 25.19 | 496 | x1.19 | 28.95 |
| 8B lm_head | 4096 x 151680 | 358.1 | 278.9 | 552 | x1.28 | 348.0 |
| 27B gate_up | 5120 x 34816 | 109.7 | 86.81 | 509 | x1.26 | 103.7 |
| 27B down | 17408 x 5120 | 54.27 | 42.76 | 517 | x1.27 | 51.82 |
| 27B in_proj_qkvz | 5120 x 16384 | 47.93 | 40.05 | 519 | x1.20 | 47.11 |
| 27B out_proj | 6144 x 5120 | 21.48 | 17.16 | 455 | x1.25 | 20.88 |
| 27B qkv | 5120 x 14336 | 42.41 | 35.54 | 512 | x1.19 | 42.14 |
| 27B lm_head | 5120 x 248320 | 731.9 | 572.5 | 550 | x1.28 | 693.7 |

B70 peak is 608 GB/s (566 GiB/s); with the decode removed altogether the
same kernel reads at up to 556 GiB/s, so the large shapes are now at the
DRAM limit.

Prefill, M = 1024. XeTLA runs the plugin's M-tiled prefill kernel
(`--mtile 1`); OpenCL uses the best `gemm_mt` tile per shape.

| shape | XeTLA (ms) | OpenCL (ms) | OpenCL TFLOPS | speed-up | before (ms) |
| --- | --- | --- | --- | --- | --- |
| 8B qkv | 2.200 | 0.421 | 122 | x5.22 | 0.460 |
| 8B o_proj | 1.475 | 0.294 | 117 | x5.02 | 0.322 |
| 8B gate_up | 9.059 | 1.739 | 119 | x5.21 | 1.911 |
| 8B down | 4.396 | 0.835 | 124 | x5.27 | 0.917 |
| 8B lm_head | 57.42 | 11.34 | 112 | x5.06 | 12.48 |
| 27B gate_up | 16.39 | 3.173 | 115 | x5.17 | 3.475 |
| 27B down | 7.915 | 1.673 | 109 | x4.73 | 1.833 |
| 27B in_proj_qkvz | 7.496 | 1.415 | 121 | x5.30 | 1.553 |
| 27B out_proj | 2.745 | 0.574 | 112 | x4.78 | 0.634 |
| 27B qkv | 6.455 | 1.220 | 123 | x5.29 | 1.338 |
| 27B lm_head | 118.6 | 23.56 | 111 | x5.03 | 25.71 |

bf16 (27B shapes, B70): GEMV x1.20-1.29 (455-555 GiB/s), GEMM x4.74-5.33
(111-125 TFLOPS); with the integer decode it was x1.01-1.07 and x4.35-4.88.
