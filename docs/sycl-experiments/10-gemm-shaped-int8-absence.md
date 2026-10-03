# Experiment 10 - the GEMM-shaped INT8 prompt path: research verdict (does not exist in SYCL)

## Question

README's next lever (from exp 09): "a GEMM-shaped INT8 dequant path, not a matvec port" for the prompt's 30%
dequant + 28.5% GEMM phases. The expectation was that llama.cpp ships one (the exp-09 matvec port was the wrong
shape). This experiment researches llama.cpp top to bottom for a GEMM-shaped i-quant batched prompt matmul, and
if one exists in SYCL, ports it.

## What the research found (dispatch-level evidence)

1. **SYCL MMQ is disabled for i-quants.** `ggml_sycl_supports_mmq()` returns false unconditionally
   (ggml-sycl.cpp:4084, "TODO: accuracy issues in MMQ"). The `use_mul_mat_q` GEMM path (ggml-sycl.cpp:4865-4873)
   never applies to `GGML_TYPE_IQ*`.
2. **SYCL reorder-MMVQ covers only non-i-quants.** `ggml_sycl_supports_reorder_mmvq` (ggml-sycl.cpp:4123) =
   Q1_0/Q4_0/Q8_0/Q2_K..Q6_K; the `reorder_vec_dot_q_sycl<>` specializations (vecdotq.hpp) are the same set.
   No i-quant reorder path exists.
3. **The i-quant batched prompt in SYCL runs `mul_mat_vec_q`** (the multi-column/plain matvec kernels - the
   shape exp-09 measured ~6x slower) or dequant+GEMM. There is **no i-quant `ncols>` batched/GEMM kernel in
   ggml-sycl** - a full-tree grep for i-quant mul_mat/ncols/batched variants returns nothing.
4. **The GEMM-shaped int8 path is CUDA-only:** `ggml_cuda_mul_mat_q` (ggml-cuda.cu:1924/1992) + the i-quant
   tile loaders in `mmq-load-tiles.cuh` (1,768 lines; `ggml_cuda_mmq_load_tiles_iq2_s/iq3_xxs/iq4_nl/xs`),
   feeding the NVIDIA-tensor-core `mul_mat_q_process_tile` GEMM (TURING_MMA/AMD_MFMA paths).

## Why a CUDA port would not win on the B60 (measured)

The B60's XMX int8 (dp4a/joint_matrix) measured **0.24-0.37x of oneMKL FP16 GEMM** (`xmx_gemm_bench`, exp 02/03:
fused quantized GEMM 0.24x; oneMKL INT8 GEMM would need per-block scale correction that oneMKL's exposed overload
cannot carry). The CUDA tile GEMM is optimized for NVIDIA's tensor cores; ported to SYCL it would run the same
dp4a path the matvec port already measured 6x slower at prompt scale (exp 09). Every evidence point says a
~1,700-line CUDA tile-GEMM port would reproduce exp-09's regression, not beat the 571 tok/s baseline.

## Verdict

**llama.cpp does not ship a GEMM-shaped i-quant batched prompt matmul for SYCL - by design** (it keeps MMQ off
and routes i-quant prompt through dequant+GEMM). The "good GEMM-shaped INT8 dequant" exists only in CUDA and is
tensor-core-tuned; the B60 does not have that hardware. Strata's dequant+oneMKL FP16 prompt path (571.7 tok/s,
exp 07) is the measured frontier on this box. The ~30% dequant phase remains the theoretical lever, but a win
there needs a GEMM-shaped INT8 path that neither llama.cpp nor oneMKL's exposed API provides for i-quants.

## Decision (user-confirmed)

Accept the frontier: keep dequant+oneMKL FP16 as the prompt path. No code change in this experiment. The README's
"GEMM-shaped INT8" follow-on from exp 09 is closed with this measured research verdict.