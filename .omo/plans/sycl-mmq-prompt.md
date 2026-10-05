# Plan: prompt-batched i-quant matmul in Strata SYCL (mmvq/dmmv port) — EXECUTED + MEASURED

Status: **executed, measured, parked-opt-in (negative perf result)**. Committed `9959fe5` on branch `b70`.

## Outcome vs plan
- Discovery corrected the plan: llama.cpp disables SYCL MMQ (`supports_mmq`->false); no SYCL i-quant prompt
  matmul exists; i-quant kernels live in the CUDA mmq.cuh + SYCL decode mmvq.cpp.
- Implemented: SYCL decode `mul_mat_vec_q_iq*_q8_1` kernels ported into a batched prompt matmul behind
  `strata::prefill::mmq` (SYCL-only includes; `quantize_row_q8_1_sycl`; `vecdotq.hpp` dots; CMake under
  STRATA_NATIVE_EXPERTS + GGML_COMMON_DECL_SYCL); `built()` env-gated (opt-in).
- Verified: 0-error build; no cuda headers reachable; parity-exact token streams on vs off; xmx_gemm_bench +
  elementwise/gr parity green.
- Measured: MMQ-on 73 tok/s vs FP16 default 571.7 tok/s at 1,280 tokens (gemm phases 91.5% of GPU time) — the
  matvec shape's per-token i-quant codebook dots lose to oneMKL GEMM; same reason llama.cpp keeps SYCL MMQ off.
- Decision per the plan's own gate rule ("default on measured-fastest"): kept opt-in
  (`STRATA_PREFILL_MMQ=1`, default off); FP16 dequant+oneMKL remains default. Report:
  docs/sycl-experiments/09-mmq-prompt.md.

## Success criteria (from the goal) — final disposition
1. Build 0 errors + no cuda headers: PASS (ninja strata exit 0; rg cuda_* over the include closure empty).
2. API honesty (built/supported/fits): PASS (supported covers 18/22/20/42; built() env-gated).
3. Parity on vs off: PASS (token sequences identical, 2 runs).
4. Perf >= 571 tok/s: FAILED for MMQ itself (73 tok/s, 6x regression); the default path runs at 571.7 tok/s.
   Per the gate rule the slow path is opt-in and the fast default is kept.
5. Regression (bench + parity): PASS.

## Verification gate
HEAVY — reviewer spawn records the audit of goal/criteria/code/evidence.