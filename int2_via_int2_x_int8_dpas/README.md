# int2_via_int2_x_int8_dpas

OpenCL port of the XeTLA int2 x int8 DPAS GEMM
(`int2_fp16_dpas_xmx_xe.hpp`, `int2_bf16_dpas_xmx_xe.hpp`; test drivers
`int2_fp16_dpas_fp16scales_fast_test`, `int2_bf16_dpas_fast_test`).
Activations are quantized to int8 per row and 128-group. The weights are
never unpacked: the DPAS consumes the 2-bit codes natively (s8 x s2,
`intel_sub_group_i8_i2_matrix_mad_k32`).

    q[m,k]  = sat_int8(A[m,k] * SA[g,m]),   SA[g,m] = DT(127 / absmax_g(A[m,:]))
    C[m,n]  = DT( epi( sum_g float(sum_{k in g} q[m,k] * code(B[k,n])) * (SB[g,n] / SA[g,m]) ) )

| tensor | type | layout |
| --- | --- | --- |
| A | DT | `[M, K]` row-major |
| B | uint32 | `[K/16, N]`: 2-bit two's-complement codes of K rows `16kp..16kp+15` of column n |
| SB | DT | `[K/128, N]` |
| SA | DT | `[K/128, LDSA(M)]`, `LDSA(M) = (M + 31) & ~31` (written by the pre-kernel) |
| C | DT, or fp32 with `--out-f32` | `[M, N]` |

DT is fp16, or bf16 with `--dtype bf16` (`-DBF16`). N must be a multiple of
16, K a multiple of 128; any M works.

## A quantization (`--qmode`)

| qmode | scheme | kernel code |
| --- | --- | --- |
| 1 | XeTLA's: a pre-kernel writes SA only, and the GEMM quantizes each DT A tile on the fly (`external_scale_a_calc=1`) | inline vISA (`quant8x32`: SIMD32 `mov hf->f`, `mul`, `mov.sat` to `:b`), the same instruction sequence as XeTLA's tile op |
| 0 | upfront: the pre-kernel writes SA and int8 A, and the GEMM reads int8 A | pure OpenCL |

Both paths are validated bit-exact against the same host gold
(`sat_int8(trunc(A * SA))`, int32 dot per group, fp32 rescale). Reported times
always include the pre-kernel.

## Kernels ([int2_int8_dpas.cl](int2_int8_dpas.cl))

* **`quant_a`:** per (row, 128-group) absmax, then SA, plus the int8 A for
  qmode 0.
* **`int2_int8_gemv` (decode):**
  * Work split: SIMD16 sub-groups over 16 columns, `SGM` rows, local
    k-slicing `LS` with SLM reduction.
  * Knobs: `SGM`, `NSG_N`, `LS`, `U`.
* **`int2_int8_gemm_mt` (prefill):**
  * Tile: a sub-group owns an `MT_M x MT_N` tile; a work-group is
    `WG_M x WG_N` sub-groups.
  * Reuse: per 128-group, B and SB are loaded once and reused for all 8-row
    blocks. Each 8 x 128 A block is loaded (and for qmode 1 quantized) once
    and reused for all 16-column blocks, and the dequant is amortized over the
    8-row DPAS.
  * Memory: 2D block I/O throughout, so ragged tiles are zero-filled and
    clipped.
  * Registers: 256 GRF; default tile 8x128, WG 8x2.

## Driver ([main.cpp](main.cpp))

```bash
make CXX=g++
./build/int2_int8_dpas_ocl --m 1 --k 5120 --n 34816 --qmode 1
./build/int2_int8_dpas_ocl --m 1024 --k 17408 --n 5120 --qmode 0 --dtype bf16
./build/int2_int8_dpas_ocl --m 1 --k 5120 --n 14336 --postop 3 --out-f32
```

* **Checks:** the host gold above, then the epilogue; also a check of the
  device SA against the host SA.
* **Pass rule:** as in the upcvt driver (`xetla_buff_cmp`: fp16 ulp 64 / abs 8,
  bf16 32 / 16).

Flags:

| group | flags |
| --- | --- |
| problem | `--m --n --k --dtype fp16\|bf16 --qmode 0\|1` |
| timing and validation | `--iters --no-validate --distinct-sets`, `--sets N` or `--weights-gib G` |
| GEMV tile | `--sgm --wgn\|--nsg --ls --u` |
| large-M tile | `--mt-m --mt-n --wg-m --wg-n` |
| epilogue | `--postop 0..4 --out-f32` |
| debug | `--cl <file> --opts "-D..." --build-log` |

## XeTLA references

* **[xetla_ref/](xetla_ref/) (fp16):** the plugin's
  `int2_fp16_dpas_fp16scales_fast_test`, with only `main.cpp` replaced (same
  rotating-set rule as the OpenCL driver).
* **[xetla_ref_bf16/](xetla_ref_bf16/) (bf16):** `int2_bf16_dpas_fast_test`,
  treated the same way.
* **How [bench.sh](bench.sh) runs them:** it takes the best of each harness's
  precompiled tiles: 7 GEMM tiles for M = 1024, and all GEMV variants for
  M = 1. XeTLA's time includes its absmax pre-kernel.

## Results (Arc Pro B70, median of 3, >= 2 GiB rotating weights)

Speed-up is XeTLA time / OpenCL time, including A quantization on both sides.

bf16, Bonsai 27B shapes:

| shape | M = 1 XeTLA (us) | vISA | upfront | M = 1024 XeTLA (ms) | vISA | upfront |
| --- | --- | --- | --- | --- | --- | --- |
| gate_up | 101.3 | x1.22 | x1.22 | 3.017 | x1.55 | x1.75 |
| down | 100.0 | x2.33 | x2.31 | 1.408 | x1.27 | x1.55 |
| in_proj_qkvz | 57.4 | x1.41 | x1.41 | 1.218 | x1.34 | x1.55 |
| out_proj | 36.7 | x2.02 | x1.99 | 0.451 | x1.21 | x1.44 |
| qkv | 52.9 | x1.47 | x1.46 | 1.040 | x1.33 | x1.59 |
| lm_head | 560.1 | x0.99 | x0.99 | 22.35 | x1.45 | x1.79 |

fp16, M = 1, Bonsai 8B shapes:

| shape | XeTLA (us) | vISA | upfront |
| --- | --- | --- | --- |
| qkv | 43.2 | x2.85 | x2.80 |
| o_proj | 42.9 | x3.72 | x3.68 |
| gate_up | 65.4 | x1.35 | x1.35 |

The GEMV wins come mostly from XeTLA's slow absmax pre-kernel at M = 1. lm_head
is bandwidth-bound and on par. At M = 1024, upfront int8 A beats in-GEMM
quantization, because each A tile is quantized once instead of once per
column block.
