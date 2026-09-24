# bitcos_fp16_upcvt

OpenCL port of the XeTLA BITCOS ternary weight-only GEMV with fp16/bf16
activations (`xetla/include/experimental/group/gemm/impl/bitcos_fp16_upcvt_xmx_xe.hpp`
as the vLLM plugin builds it: `BITCOS_FP16_LUT`, `BITCOS_SIGN_GATHER4/3`).
BITCOS (BITmap + COmpacted Signs, [arXiv:2609.16338](https://arxiv.org/abs/2609.16338))
stores a presence bit per weight and a sign bit per non-zero, so a zero density
`z` costs `2 - z` bits per weight.

    C[m,n] = DT( epi( sum_k A[m,k] * code(k,n) * S[k/128, n] ) ),  code in {-1, 0, +1}

| tensor | type | layout |
| --- | --- | --- |
| A | DT | `[M, K]` row-major |
| B | uint32 | one buffer, three planes: bitmap `[K/32][N]` (bit c of word `(kp, n)` = weight `kp*32+c` of column n is non-zero), offsets `[N]` (first sign word of column n), signs (one bit per non-zero, column-major in k, 1 = -1), plus pad words |
| SR | uint32 | slice ranks `[LS-1][N]`: non-zeros of column n above each local-K-slice boundary `s*K/LS` |
| S | DT | `[K/128, N]` |
| C | DT, or fp32 with `--out-f32` | `[M, N]` |

This is the layout of `xetla_vllm_plugin.pack_bitcos` (host port in
[../common/bitcos.hpp](../common/bitcos.hpp)). N must be a multiple of 16 and
K a multiple of `64 * LS`; M = 1..8 (decode).

## Kernel ([bitcos_fp16_upcvt.cl](bitcos_fp16_upcvt.cl))

Per lane (one output column) and 64-row K step, the same messages as XeTLA
(and Fig. 6 of the paper):

* **Bitmap:** one 2D block read gives the two 32-row bitmap words of 16 columns.
* **Sign window:** the column's rank (its cursor into the sign stream) advances
  by one `popcount` per 32-row block. A single `vload3` at word `rank/32`
  (`load.ugm.d32x3`) covers the windows of both blocks, because a block
  consumes at most 32 sign bits. The window is muxed and aligned with `bfn`
  and shifts.
* **Lookup:** each block is 8 four-row groups. The presence nibble and the
  next four sign bits index a 256-entry SLM table of four codes in VNNI order
  (`load.slm.d32x2`, a 4-bit `pdep`), and the window shifts by
  `popcount(nibble)`.
* **Scale apply:**
  * fp16 (default): XeTLA's two SIMD32 `hf` multiplies on the looked-up dwords,
    as inline vISA (`hmul2`).
  * `--simt-mul`: the same in plain OpenCL (`half4 * half`). IGC de- and
    re-interleaves the four codes around four SIMD16 multiplies: 369 instead of
    213 loop instructions, 15% slower.
  * `--int-apply` (always used for bf16, which has no 16-bit multiply): the
    table holds `0 / 0x7fff / 0xffff` and one AND with `scale | 0x8000` gives
    `0 / +s / -s`. The B tile is bit-identical to the multiply.
* **Compute:** two DPAS of k = 16 per block, fp32 accumulate.
* **Work split:** `SGM` rows per sub-group, `NSG_N` sub-groups along N
  (`--wgn = 16*NSG_N`), `LS` sub-groups splitting K. Slice `s` enters the sign
  stream at `SR[s-1]`, and the slices reduce through SLM.

Inner loop per 64 K rows on BMG (offline `ocloc` ISA):

| scale apply | instructions | `mov` | SIMD32 `hf` mul |
| --- | --- | --- | --- |
| XeTLA (plugin build) | 244 | 19 | 32 |
| OpenCL, inline-vISA `hmul2` (default) | 213 | 5 | 32 |
| OpenCL, `--int-apply` | 211 | 5 | 0 |
| OpenCL, `--simt-mul` | 369 | 132 | 0 |

## Driver ([main.cpp](main.cpp))

```bash
make CXX=g++
./build/bitcos_fp16_upcvt_ocl --m 1 --n 32768 --k 16384 --z 0.40 --wgn 256 --ls 1
```

Weights are random ternary at zero density `--z`, packed as `pack_bitcos`.
Methodology is the same as the int2 drivers: `--weights-gib` of rotating
distinct sets (default 2 GiB), device-event timing, and the `xetla_buff_cmp`
pass rule against an fp32 host gold. `--distinct-sets` validates every set
against its own gold. Epilogues: `--postop 0..4` and `--out-f32`, see the top
README. [validate.sh](validate.sh) covers every scale apply, bf16, LS 1-16,
z 0.1-0.95, M = 1..8 and the epilogues.

## Zero-density sweep ([zsweep.sh](zsweep.sh))

This is the paper's GEMV sweep (Fig. 13): N = 32768, K = 16384, M = 1, and z
from 0.05 to 0.95, with every point tuned independently.

* **XeTLA reference:** the paper's own harness,
  `bitcos_fp16_dpas_fp16scales_fast_test` (xetla branch `exp/bitcos-lut8-bitmask`,
  `make tuning`, production `-D` flags). Tiles: sg_n = 16, sg_k = 64 over
  wg_n x LS.
* **OpenCL:** wg_n x LS, the default fp16 apply.
* **Measurement:** per z, both winners are re-measured alternately, median of
  3, with >= 2 GiB of rotating weights.
* **int2 baselines:** the XeTLA and TernOCL int2 upcvt kernels at the same
  shape (2 bits/weight).

```bash
XB=/path/to/xetla/bitcos_fp16_dpas_fp16scales_fast_test/build bash zsweep.sh results/zsweep_b70.csv
PACE=15 XB=... bash zsweep.sh results/zsweep_lnl.csv                  # Lunar Lake
```

[shapes27b.sh](shapes27b.sh) runs the same per-shape procedure on the Bonsai 27B /
Bonsai 2 27B decode shapes at one z. Its int2 baselines use the vLLM plugin's
per-arch int2 config for each shape.

## Results (Arc Pro B70, fp16 activations)

Speed-up is XeTLA BITCOS time / OpenCL BITCOS time. Lunar Lake is not tuned
yet: at LS > 1 XeTLA gains from local K slicing there and this kernel does not.

**Bonsai 27B / Bonsai 2 27B decode GEMV shapes at z = 0.40** (M = 1, each
kernel tuned per shape, tile = wg_n:LS):

| shape | K x N | XeTLA BITCOS | OpenCL BITCOS | speed-up | int2 XeTLA / TernOCL |
| --- | --- | --- | --- | --- | --- |
| mlp.gate_up_proj | 5120 x 34816 | 0.102 ms (32:8) | 0.102 ms (32:8) | x1.00 | 0.110 / 0.108 ms |
| mlp.down_proj | 17408 x 5120 | 0.057 ms (32:4) | 0.052 ms (32:4) | x1.09 | 0.054 / 0.059 ms |
| linear_attn.in_proj_qkvz | 5120 x 16384 | 0.051 ms (32:8) | 0.049 ms (32:4) | x1.05 | 0.048 / 0.046 ms |
| linear_attn.out_proj | 6144 x 5120 | 0.022 ms (32:4) | 0.021 ms (32:4) | x1.05 | 0.022 / 0.020 ms |
| self_attn.qkv_proj | 5120 x 14336 | 0.047 ms (64:2) | 0.044 ms (64:2) | x1.07 | 0.042 / 0.041 ms |

**Paper GEMV sweep** (32768 x 16384; the `--int-apply` scale path, whose B
tile is bit-identical to the default and which is 1-3% slower here):

| z | XeTLA BITCOS | OpenCL BITCOS | speed-up |
| --- | --- | --- | --- |
| 0.05 | 0.278 ms | 0.266 ms | x1.04 |
| 0.10 | 0.294 ms | 0.273 ms | x1.08 |
| 0.20 | 0.296 ms | 0.272 ms | x1.09 |
| 0.30 | 0.290 ms | 0.270 ms | x1.07 |
| 0.40 | 0.278 ms | 0.267 ms | x1.04 |
| 0.50 | 0.270 ms | 0.265 ms | x1.02 |
| 0.60 | 0.265 ms | 0.263 ms | x1.01 |
| 0.70 | 0.257 ms | 0.263 ms | x0.98 |
| 0.80 | 0.248 ms | 0.250 ms | x0.99 |
| 0.90 | 0.231 ms | 0.230 ms | x1.01 |
| 0.95 | 0.216 ms | 0.210 ms | x1.03 |

int2 at the same shape: XeTLA 0.294 ms, TernOCL 0.287 ms. With the default
(inline-vISA) apply, the first sweep points are x1.06 (z = 0.05), x1.08
(z = 0.10) and x1.10 (z = 0.15).
