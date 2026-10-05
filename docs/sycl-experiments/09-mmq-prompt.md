# Experiment 09 - prompt-batched i-quant MMQ: parity clean, but the matvec shape loses to GEMM (parked)

## Question

README.sycl.md's former P0: replace the dequant+oneMKL FP16 prompt path (58.6% of prompt time, exp 07) with a
batched i-quant matmul. The original "wrap native-SYCL MMQ" plan was disproven in discovery (llama.cpp disables
SYCL MMQ, `supports_mmq`->false; no SYCL i-quant prompt matmul exists). Per the pivot, port llama.cpp's SYCL
**decode** i-quant matvec kernels (mmvq.cpp) into a batched prompt matmul behind `strata::prefill::mmq`.

## What was built (committed work, opt-in)

`sycl/src/prefill/moe_mmq.dp.cpp` rewritten: the CUDA includes (`mmq.cuh`/`common.cuh`/`quantize.cuh`) are gone;
now SYCL-only — `common.hpp` (presets + ggml-common SYCL block types), `quantize.hpp`
(`quantize_row_q8_1_sycl`), `vecdotq.hpp` (the i-quant `vec_dot_*_q8_1` dots). Seven i-quant batched matvec
kernels ported (`iq2_xxs/xs/s`, `iq3_xxs/s`, `iq4_nl/xs`) with a `mul_mat_vec_q_launch` batched head over the
Product's token bounds; `Context::run` switches over Q2_0 + the i-quants; `quantize` uses
`quantize_row_q8_1_sycl`; gather/swiglu/iota unchanged. CMake wires it into `strata_prefill` under
`STRATA_NATIVE_EXPERTS` (ggml-sycl include path + `GGML_COMMON_DECL_SYCL`).

**`built()` is env-gated (opt-in):** `STRATA_PREFILL_MMQ=1` enables it, default off — the FP16 path stays
default because the port is a measured regression at prefill (below).

## Measurements (B60, Coder IQ1_M, warm)

1. **Build (criterion 1): PASS.** `ninja strata` 0 errors; the TU's include closure is pure SYCL
   (`sycl.hpp`/`common.hpp`/`quantize.hpp`/`vecdotq.hpp`), no `cuda_*.h` reachable.
2. **API honesty (criterion 2): PASS.** `supported()` covers the Coder's types (18 IQ3_XXS, 22 IQ2_S, 20 IQ4_NL,
   42 Q2_0); `built()` is opt-in.
3. **Parity (criterion 3): PASS.** MMQ-off vs on (2 runs), 1,280-token prompt: token sequences identical
   (`271 5328 3165 279 1414 3294 1608 5976 6511 1954 13 271 248068 198 760 1156`).
4. **Perf (criterion 4): FAIL for MMQ itself - measured regression; default path unchanged at baseline.**
   At 1,280 tokens: MMQ-on reads in **17,540 ms (~73 tok/s)** vs MMQ-off **2,858 ms (~448 tok/s)**; the default
   FP16 path runs **571.7 tok/s** (baseline, exp 07, restored). `STRATA_PREFILL_TIMING` with MMQ-on: `gemm
   gate/up 8,271 ms (47.7%) + gemm down 7,603 ms (43.8%) = 91.5%` of the prompt. The dequant dropped to 2.8%
   (was 30%) - the port runs correctly, but the matvec math is ~30x slower than oneMKL GEMM at prompt scale.
5. **Regression (criterion 5): PASS.** xmx_gemm_bench unchanged (0.24x fused-XMX), elementwise_parity 0 failures,
   gr_parity 0 failures; iq_parity needs fixtures (pre-existing skip).

## Why it loses (the honest result)

The ported kernels are **decode-shaped matvecs**: one work-item per (token-row, output-row), each doing i-quant
codebook lookups + dp4a on a single row. For a 1-token decode that is the right shape; for a 1,280-token prefill
it is ~30x worse than a GEMM-shaped tiled reduction with contiguous F16 loads (oneMKL XMX), because the
codebook-LUT path (IQ2_S/IQ3_XXS/IQ4_NL) cannot amortize across rows the way the F16 dequant does. This is
precisely why llama.cpp's SYCL backend keeps MMQ off (accuracy gating aside, the SYCL matvec path is for decode)
and uses dequant+GEMM for prompt processing.

## Conclusion

The i-quant MMQ port is **correct, parity-exact, buildable, and committed — but a measured ~6x regression at
prompt scale**, so the FP16 dequant+oneMKL path remains the default (571.7 tok/s) and the port is **opt-in**
(`STRATA_PREFILL_MMQ=1`). The README "P0 - PRIMARY GOAL" MMQ item is closed as a measured-regression result:
the i-quant prompt GEMM has no SYCL implementation in llama.cpp *because the matvec shape does not win at prefill*;
the real prompt lever (from exp 07) stays the 30% dequant phase, which would need a GEMM-shaped INT8 path, not a
matvec port. INTEL.md/README updated accordingly. `STRATA_PREFILL_MMQ=0` (or unset) = the fast default;
`=1` = the (slow, parity-exact) MMQ experiment.