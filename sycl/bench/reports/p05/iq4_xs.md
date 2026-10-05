# P0 #5 — IQ4_XS (ggml_type 23) pre-unpack feasibility

Feasibility of porting the Q6_K pre-unpack win (decode a signed-byte block once, then a no-bit-unpack
load + dp4a kernel; see Q6UBlock / `native_q6k_preunpack_kernel` / `WideQ32Q6U`) to IQ4_XS.
Transcribed from `include/strata/artifact/dequant.hpp` and
`sycl/src/kernels/cuda/native_mmvq.dp.cpp`; line numbers cite those files as read.

## 1. IQ4XSBlock layout and the native decode kernel

`IQ4XSBlock` is an on-disk 136-byte super-block holding 256 elements.

- `include/strata/artifact/dequant.hpp`: “IQ4_XS: 256 elements from a 136-byte super-block: `fp16 d`,
  `uint16 scales_h`, `scales_l[4]`, `qs[128]`” (lines 259-263); `dequantize_iq4_xs` reads `d` at
  `block+0` (265), `scales_h` at `block+2` (266), `scales_l` at `block+4` (267), `qs` at `block+8`
  (268).
- `sycl/src/kernels/cuda/native_mmvq.dp.cpp`: the SYCL block struct mirrors this exactly
  (lines 78-83): `struct IQ4XSBlock { sycl::half d; uint16_t scales_h; uint8_t scales_l[4];
  uint8_t qs[128]; }`. `static_assert`s pin it: `sizeof == 136`, `alignof == 2`, `scales_h` at
  offset 2, `scales_l` at 4, `qs` at 8 (lines 117-119).

Which kernel decodes it — the small/multi path is **`native_iq4_xs_mmvq_kernel`** (template on
`SmallK`, QK=256 / QI=32 / VDR=4, lines 525-527), dispatched as `<true>` (small K) and `<false>`
(lines 2395-2402) from `native_iq4_xs_mmvq` (2333). Each lane runs `iq4_xs_q8_dot` (498), which
decodes the 4-bit nibbles through the 16-entry nonlinear IQ4_NL codebook: the codebook lives as the
device table `iq4nl_values = kIq4nlTable` (lines 474-480) and is gathered by the two-stage
byte-permute helper `iq4_table_lookup` (481-495). The `ncols > 1` route goes through
`IQ4XSTraits` → `launch_multi` (1094, 2343). A separate wide kernel path exists as
**`WideIQ4XS`** (1641), selected first by `try_wide<WideIQ4XS>` (2341); its `load` (1645) does the
nibble→codebook expansion with `iq4_lut4` (1630), which packs the same 16 codebook values into four
register constants `t0..t3` (1631-1632).

Note: the requested name “SmallIQ4XS” does **not** exist in the source — `SmallIQ4XS` has zero
hits; the small-K flavor is the `SmallK=true` instantiation of `native_iq4_xs_mmvq_kernel`
(2395). `WideIQ4XS` is real (1641). Both paths rely on the 16-entry IQ4_NL codebook
(`kIq4nlTable`, 480 / `iq4_lut4` constants, 1631-1632).

## 2. Why pre-unpack may NOT help: the i-quants are LUT-bound

The Q6_K `~2x` win (exp 27/28) came from eliminating an ALU-pipe 6-bit unpack that was the
dominant in-kernel cost. IQ4_XS is different: experiments **04** and **11** measured the i-quants
as **LUT-bound / ALU–LUT-bound, not memory-bound** (`docs/sycl-experiments/04` and `/11`;
`README.sycl.md`). The decode cost is the per-nibble codebook gather (`iq4_table_lookup`,
481-495), not the bit unpacking. A bit-repack pre-unpack that turns nibbles into signed bytes
beforehand would not remove that gather — it would still have to materialize the signed codebook
value per element — so the mechanism behind the Q6_K speedup does not transfer.

## 3. Viability: block fields and per-subblock operations

**Block fields** (`IQ4XSBlock`, `native_mmvq.dp.cpp` 78-83): `sycl::half d` (one block scale),
`uint16_t scales_h` (16-bit field holding 2 high bits of each of 8 per-subblock scales),
`uint8_t scales_l[4]` (16 bits of the 4 low bits of those 8 scales), `uint8_t qs[128]` (256
nibbles, 8 subblocks × 16 bytes). The 8 subblocks share `d`; each subblock has its own 6-bit
scale.

**Per-subblock ops** (32-element subblock, from `iq4_xs_q8_dot`, lines 498-521):
- **unpack/LUT**: for `j` in 0..3 load one `int` of `qs` and expand its 4 nibbles with
  `iq4_table_lookup` → `sycl::int2 v` (509-513); i.e. a nibble code → codebook-value byte gather.
- **dot**: two `dp4a` per `j` (8 total) of `v` against the two `int` loads of the Q81 activation
  (`x[iqs/4].qs[j]` / `[j+4]`), accumulated into `sumi` (510-515).
- **scale**: the 6-bit signed scale is assembled `ls = (scales_l[iqs/8] >> (iqs&0x04) & 0x0f) |
  ((scales_h >> (iqs/2) & 3) << 4)`, then `sumi *= (ls - 32)` in the integer domain (517-519).
- **sum/final**: `d = (float)w->d * x[iqs/4].ds[0]` (half scales multiplied, 520) and
  `return d * sumi` (521).

The `WideIQ4XS` path merges LUT-expansion of `lo`/`hi` nibbles once in `load` (1647-1657) and the
final `sumi` halves through `r.d = b->d * (ls-32)` + `dp4a4` in `apply` (1658-1660). In both paths
the nibble→codebook gather is intrinsic to every element and cannot be folded into a one-time
byte pre-unpack the way Q6_K's 6-bit shifts/ORs could.

## 4. Verdict

Not worth pre-unpacking IQ4_XS: it is already LUT-bound (exp 04/11), so removing the bit unpack
does not remove the per-element codebook gather that dominates the cost — unlike Q6_K's ALU-bound
6-bit unpack.