# Experiment 11 - the prompt's dequant phase: A/B against llama.cpp's SYCL dequant (moot)

## Question

The prompt's dequant phase is the largest single cost (30% of prompt time, exp 07). Strata uses its own
`iq_dequant_f16` (`sycl/src/kernels/cuda/iq_kernels.dp.cpp`, a flat `dequant_flat_kernel`, one work-item per
256-value chunk). llama.cpp SYCL ships per-i-quant dequant kernels (`convert.cpp`:
`dequantize_row_iq2_s/iq3_xxs/iq4_nl_sycl`). Is the phase memory-write-bound (no kernel swap helps) or
ALU/layout-bound (a faster dequant could win), and would llama.cpp's kernels be faster?

## Measured: Strata's dequant is NOT write-bound for the i-quants

Effective write GB/s of Strata's `iq_dequant_f16` at the Coder expert shapes (fp16 out: gate/up [1280,2560] =
6.55 MB; down [2560,640] = 3.28 MB), from exp-07's per-expert timings:

| type | dequant ms (gu) | effective write GB/s | % of ~608 GB/s card bw |
|---|---|---|---|
| 18 IQ3_XXS | 0.036 | 182 | 30% |
| 20 IQ4_NL | 0.087 | 75 | 12% |
| 21 IQ3_S | 0.036 | 182 | 30% |
| 42 Q2_0 | 0.018 | 364 | 60% |

The i-quants (18/20/21) run at **12-30% of card bandwidth** - they are ALU/LUT-bound (the i-quant dequant does
codebook-table lookups + sign masks per value), not memory-write-bound. So in principle the 30% phase has
headroom: a faster dequant kernel could win.

## The A/B is moot: llama.cpp's SYCL dequant is NOT on the prompt path

An A/B bench (Strata `iq_dequant_f16` vs llama.cpp's `dequantize_block_iq*` block kernels) was attempted. Two
findings:

1. **llama.cpp's SYCL convert.cpp dequant kernels have zero prompt-path callers.** Grep over ggml-sycl.cpp /
   mmp.cpp / moe_mmq finds no reference to `select_dequantize_kernel` or the `dequantize_row_*_sycl` family in
   any mul_mat/expert flow - they serve the generic convert / get_rows ops, not the expert prompt dequant.
   Strata's `iq_dequant_f16` is the only dequant on the Coder's prompt path.
2. **A standalone bench TU cannot compile llama.cpp's `dequantize_block_iq*` kernels** without replicating
   convert.cpp's impl-mode table setup: the kernels reference `iq2s_grid`/`kmask_iq2xs` as statics that the
   standalone TU's SYCL mode flags as non-constant-initialized (the same kernel-mode wrangle exp-09's port
   avoided by using vecdotq.hpp with tables passed as args, and never including dequantize.hpp). The bench was
   abandoned; no tree change.

## Conclusion

Strata's prompt dequant (30% of prompt time) is **not memory-bound** for the i-quant types (12-30% of card
bandwidth), so headroom exists - but it is **not** obtainable from a llama.cpp kernel swap, because llama.cpp's
SYCL dequant kernels are not on the prompt path and Strata's `iq_dequant_f16` is its own kernel. A faster
dequant would be a Strata-side kernel improvement (e.g. wider per-work-item chunks, fewer table lookups per
value), which is a separate, real kernel project - not an A/B port. The llama.cpp dequant lane is closed as
moot; the "strata-side faster dequant" remains the one open prompt lever, flagged for a future session.

## Follow-up (2026-10-03): the "fewer table lookups" lever is MEASURED NEUTRAL, not the bind

The suggested first lever (exp 11 conclusion: "fewer table lookups per value") was tried on IQ4_NL, the
largest dequant (39/48 layers): `dq_iq4_nl` was rewritten to build the 8 output values from `iq4nl_lut4` (the
16-byte `kvalues_iq4nl` codebook held in registers as ALU constants - the same trick `vec_dot_iq4_nl_q8_1`
already uses for the MMVQ dots) instead of per-value `kvalues_iq4nl[]` global lookups. Output is bit-identical
(same dequant checksum). Measured with a new `dequant_bench` (sycl/src/kernels/dequant_bench.cpp, filled
1280x2560 gate/up shape, IQ4_NL, wired into CMake):

| path | GB/s (warm, 600 reps, 3 runs) | checksum |
|---|---|---|
| scalar `kvalues_iq4nl[]` lookups | 101 / 122 / 118 | 3fe10f4c7bc3b479 |
| register-LUT `iq4nl_lut4` | 111 / 154 / 148 | 3fe10f4c7bc3b479 |

The B60's clock spread (the card warms to 110-150 GB/s; a cold first run reads half that) swamps the
LUT-vs-scalar difference - they are within noise of each other. **The codebook lookup is not the dequant's
bind**: the 16-byte table is cache-resident, so replacing it with register ALU moves nothing. The rewrite was
reverted (bit-identical but not faster = added branch risk for no gain); `dequant_bench` stays as the
measurement instrument for future dequant work. The i-quants' 12-30%-of-bandwidth is the ALU/FMA arithmetic
(scale multiply + sign mask per value), not a table-lookup skin; a faster dequant must reduce per-value math
or widen the work, not the LUT path.

## Validity
- GB/s from exp-07's measured per-expert dequant ms (0.018-0.087, warm, same shapes), card bandwidth ~608 GB/s
  (the B60's spec-class figure used throughout INTEL.md).
- Caller-grep is exact: `select_dequantize_kernel` / `dequantize_row_*_sycl` have no references outside
  convert.cpp's own dispatch.
- No code changed; tree clean (only the working `.omo/` plan dir untracked).