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

**Dequantization** (both kernels, as in XeTLA, integer ops only). Each nibble
(two K codes) becomes one VNNI dword `(scale2 ^ sign) & mask` holding both
weights as 0 or +/-scale. One B word is exactly the `int8` B operand of
`intel_sub_group_{f16_f16,bf16_bf16}_matrix_mad_k16`, so there is no shuffle
and no SLM. This works for bf16 too: the sign is bit 15 and 0 is all-zero in
both formats.

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
| debug | `--cl <file> --opts "-D..." --build-log` |

M >= 64 or `--mt-m` selects the large-M kernel. The default tiles are the B70
27B table in `default_tiles()`; [bench.sh](bench.sh) sweeps the tiles per shape.

## XeTLA reference ([xetla_ref/](xetla_ref/))

A copy of the plugin's `int2_fp16_upcvt_dpas_fast_test` driver, with the same
rotating-set rule as the OpenCL driver. It takes explicit GEMV tiles
(`--wgn --ks --ls`, the plugin's dispatch table) and prefill tiles (`--mtile 1..7`;
1 is the plugin's default for M > 16). [bench.sh](bench.sh) runs it at the
plugin's tuned per-arch config for every shape.

## Results (Arc Pro B70, fp16, median of 3, >= 2 GiB rotating weights)

Decode GEMV, M = 1, with the plugin's tuned XeTLA config per shape. The 8B
shapes, 27B down and 27B out_proj were measured on pcl-arl01 with the
single-row B loads; the other 27B shapes are from the full run on pcl-zen4
with the earlier 8-row B read.

| shape | K x N | XeTLA (us) | OpenCL (us) | speed-up |
| --- | --- | --- | --- | --- |
| 8B qkv | 4096 x 6144 | 16.70 | 16.51 | x1.01 |
| 8B o_proj | 4096 x 4096 | 11.78 | 11.70 | x1.01 |
| 8B gate_up | 4096 x 24576 | 58.38 | 55.36 | x1.05 |
| 8B down | 12288 x 4096 | 29.91 | 28.95 | x1.03 |
| 8B lm_head | 4096 x 151680 | 358.0 | 348.0 | x1.03 |
| 27B gate_up | 5120 x 34816 | 109.7 | 103.7 | x1.06 |
| 27B down | 17408 x 5120 | 54.38 | 51.82 | x1.05 |
| 27B in_proj_qkvz | 5120 x 16384 | 47.93 | 47.11 | x1.02 |
| 27B out_proj | 6144 x 5120 | 21.49 | 20.88 | x1.03 |
| 27B qkv | 5120 x 14336 | 42.47 | 42.14 | x1.01 |
| 27B lm_head | 5120 x 248320 | 733.2 | 693.7 | x1.06 |

Prefill, M = 1024. XeTLA runs the plugin's M-tiled prefill kernel
(`--mtile 1`); OpenCL uses the best `gemm_mt` tile per shape.

| shape | XeTLA (ms) | OpenCL (ms) | OpenCL TFLOPS | speed-up |
| --- | --- | --- | --- | --- |
| 8B qkv | 2.224 | 0.460 | 112 | x4.84 |
| 8B o_proj | 1.487 | 0.322 | 107 | x4.62 |
| 8B gate_up | 9.232 | 1.911 | 108 | x4.83 |
| 8B down | 4.490 | 0.917 | 112 | x4.90 |
| 8B lm_head | 58.22 | 12.48 | 102 | x4.67 |
| 27B gate_up | 16.83 | 3.475 | 105 | x4.84 |
| 27B down | 8.100 | 1.833 | 100 | x4.42 |
| 27B in_proj_qkvz | 7.587 | 1.553 | 111 | x4.89 |
| 27B out_proj | 2.796 | 0.634 | 102 | x4.41 |
| 27B qkv | 6.553 | 1.338 | 112 | x4.90 |
| 27B lm_head | 120.2 | 25.71 | 101 | x4.68 |

bf16 (27B shapes, B70): GEMV x1.01-1.07, GEMM x4.35-4.88 (100-113 TFLOPS).
