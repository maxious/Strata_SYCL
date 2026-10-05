# Experiment 31 - the i-quant dequant's speed half: IQ4_NL is codebook-select-bound (P0 #3, measured null)

## Question

README P0 #3's remaining half: faster Strata-side dequant of the i-quants feeding the accepted dequant+oneMKL FP16
prompt path. exp 11 said `iq_dequant_f16` reads only 12-30% of bandwidth (ALU/LUT-bound, not write-bound), the i-quants
being the headroom (Q2_0 is already bandwidth-bound at ~472 GB/s). This experiment measures the exact ceiling and tests
the README's named lever ("fewer table lookups per value").

## Baseline (dequant_bench, 2560x1280 gate/up, two runs each, this box)

| type | GB/s |
|---|---|
| IQ4_NL (20) | 75 |
| IQ4_XS (23) | 76 |
| IQ3_XXS (18) | 87 |
| IQ2_S (22) | 183 |
| IQ2_XS (17) | 246 |
| Q2_0 (42) | 472 |

The i-quants IQ4_NL / IQ4_XS / IQ3_XXS sit ~4-6x below Q2_0 and the ~608 peak. Not write-bound (the write is ~6.55 MB
over ~0.088 ms).

## Bottleneck (reading dq_iq4_nl, iq_kernels.dp.cpp)

`kvalues_iq4nl` is a compile-time `int8[16]` codebook (`{-127,-104,...,104}` - not an arithmetic grid, so no
bit-arithmetic shortcut), and the value output is `d * codebook[nibble]`. The per-value cost is a 16-way register
select + int->fp convert + fp mul + fp16 re-round; 32 lanes per 256-block, 8 values/lane, high workgroup count (a
parallelism/occupancy issue can be ruled out). The compiler already keeps the constexpr codebook in registers (a
select-tree, ~4-5 compares/value) - NOT a global-memory gather (the i-quants' "LUT-bound" label is a register
select-tree).

## The test: hoist `d*codebook` into a per-block register table

Rewrite `dq_iq4_nl` to precompute `half d*codebook[16]` once per block and select from it (per-value cost = select +
copy, amortizing the convert+mul across the block).

Result (dequant_bench): **checksum bit-identical (3fe10f4c7bc3b479, unchanged) but GB/s unchanged** - 72-77, the same
~75 baseline. The 16-way select dominates; removing the convert+mul does not move the pipe. A `iq4nl_lut4`-style
4-way-bucket magic (same technique the MMVQ path already uses) is ~comparable in compares to a 16-entry select-tree
(~4-5), so it offers no real headroom either. Reverted; the original stands.

## Verdict

**P0 #3's dequant-speed half is measured-bounded: the i-quant dequant is codebook-select-bound, and neither removing
the per-value convert/mul (register-table hoist: bit-identical, null) nor a wider-chunk / magic-bucket rewrite offers
real headroom.** This confirms exp 04/11's LUT-bound finding at the source level and exp 11's "a faster dequant would
be a Strata-side kernel project" - the project is correctly telling people to pick Q2_0 (bandwidth-bound at 472), not
the i-quants. Q2_0 stays the prompt dequant pick; the i-quant dequant half of P0 #3 is parked as resist-optimization.

## Traps worth knowing

- The i-quant "LUT-bound" label is a register **select-tree** over a compile-time constexpr codebook, not a memory
  gather, so "put the LUT in registers" is already true and buys nothing.
- The fp16 output must be bit-identical for any rewrite (dequant_bench checksum oracle); the hoist kept it identical,
  confirming the arithmetic order is preserved.
- A null bit-identical change was reverted - only measured wins stay.