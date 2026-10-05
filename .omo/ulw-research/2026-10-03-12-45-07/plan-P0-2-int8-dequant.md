# P0-2 plan — GEMM-shaped INT8 prompt dequant (Strata-side kernel project)

Verdict: **ACTIONABLE BUT RE-SCOPED** — the research confirms there is **no llama.cpp SYCL port to make directly** (the lane verdict: CONFIRMED no i-quant prompt GEMM exists in llama.cpp SYCL); the win is entirely a **Strata-side faster dequant kernel** feeding the already-accepted dequant + oneMKL FP16 path (571.7 tok/s baseline).

## Research verdict (grounded in primary source, verified at lines)
- `ggml_sycl_supports_mmq()` returns **false unconditionally** ("accuracy issues in MMQ", ggml-sycl.cpp:4084-4088).
- reorder-MMVQ covers only Q1_0..Q6_K (no i-quants) — ggml-sycl.cpp:4123-4137; vecdot.hpp specializations same set.
- mmq.cpp has **zero** i-quant MMQ kernels (only q4_0..q5_K). i-quant mmvq kernels are single-column **decode** only (mmvq.cpp:335-760).
- The GEMM-shaped int8 i-quant path is **CUDA-only**: `ggml-cuda/mmq-load-tiles.cuh` (iq1_s:1036 .. iq4_nl:1495) + mmq.cu dispatch + NVIDIA tensor-core `mul_mat_q_process_tile` (mmq.cuh:792-839). Not portable to SYCL/XMX.
- llama.cpp's SYCL i-quant dequant kernels (convert.cpp:342-527 launchers; dequantize.hpp:1410-1640 block kernels, 32 WI/8 values, 1 LUT + sign mask) exist but serve **convert/get_rows** (dmmv.cpp:2224, fattn-common.hpp:965, matmul fallback ggml-sycl.cpp:3036/3050) — **not** the expert prompt path. Strata's `iq_dequant_f16` (iq_kernels.dp.cpp:2298) is the sole prompt-path dequant.

Why it's ALU/LUT-bound: exp-11 measured i-quant dequant at 12-30% of card BW (IQ3_XXS/IQ4_NL/IQ3_S) → not memory-write-bound, so a faster kernel can win.

## Strata-side levers (own kernel work, no llama.cpp port)
1. **Halve LUT pressure** — use existing `get_int_from_table_16` (iq_kernels.dp.cpp:64) in `dq_iq4_nl`/`dq_iq4_xs`: 8 scalar codebook lookups → 2 wide lookups behind byte-perm (pattern from CUDA mmq-load-tiles.cuh:1528). Highest-value, immediately actionable.
2. **Register-block the dequant output** into tiles (extend the pattern already in `xmx_gemm_iq`, iq_kernels.dp.cpp:2542) so fp16 output doesn't round-trip global memory before oneMKL.
3. **Wider per-work-item chunks** — 16 WI × 16 values vs current 32 × 8; halves WG count, amortizes the codebook lookup.
4. **Amortize per-sub-block scale** across `il` iterations (iq2_xs/iq3_s).

## Files to touch →
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp` (dequant kernels + `iq_dequant_f16`).

## Acceptance / success criteria
- C1: prompt tok/s **up** at 2,184 and 8,000 tokens (`python3 sycl/setup_intel.py --model IQ3_S` / stride benches), versus the 571.7 tok/s baseline.
- C2: **identical output** — byte-exact fp16 dequant (stride parity / gguf-py comparison) with every input type.
- C3: `iq_parity` green; log a row in INTEL.md's speed table in the existing style.

## Note (from exp 07/09/10/11 + this run)
- MMQ / oneMKL-INT8-with-scale-correction routes already parked as slower/regressed; do NOT reopen except with new evidence. The dequant is the lever, confirmed again this run.

## Risk
Wide LUT reads may add register pressure / lower occupancy; the A/B must be done end-to-end (dequant+GEMM fused time), not kernel-isolated, because the dequant feeds oneMKL.