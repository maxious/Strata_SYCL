# Wave 0 — P0-2 INT8/dequant return (complete, st_01a101cd)

## VERDICT: CONFIRMED — llama.cpp SYCL has no prompt-path i-quant mechanism worth porting as-is
1. supports_mmq false unconditional (ggml-sycl.cpp:4084-4088 'accuracy issues in MMQ'); use_mul_mat_q always false (4865); IQ2_XXS excluded even if (4871).
2. reorder-MMVQ only Q1_0/Q4_0/Q8_0/Q2_K..Q6_K, no i-quants (4123-4137); vecdot.hpp same set.
3. mmq.cpp zero i-quant MMQ kernels (only q4_0..q5_K, 1357-1706). i-quant mmvq (mmvq.cpp:335-760) single-column decode only (ne1=1).
4. GEMM-shaped int8 i-quant path CUDA-only: ggml-cuda/mmq-load-tiles.cuh iq1_s(1036)..iq4_nl(1495), mmq.cu dispatch 48-71/135, TURING_MMA/AMD_MFMA (mmq.cuh 792-839).

## SYCL i-quant dequant kernels enumerated (no prompt-path callers)
- convert.cpp dequantize_row_iq1_s(342)..iq4_nl(511) launchers; dequantize.hpp dequantize_block_iq* block kernels (1410-1640, 32 work-items / 8 values each, 1 codebook LUT per thread + sign mask); inline dequantize_iq* for get_rows (483-680).
- Callers: convert.cpp:656/746 (ggml_get_to_fp16/fp32) <- dmmv.cpp:2224 (src1->fp16 for DMMV), fattn-common.hpp:965 (KV), ggml-sycl.cpp:3036/3050 (matmul fallback FP16 GEMM). NOT on Strata's expert prompt dequant path.
- Strata iq_dequant_f16 (iq_kernels.dp.cpp:2298) is the sole prompt-path dequant.

## Strata-side levers (NOT llama.cpp ports — own kernel work, confirms P0-2 is Strata-side)
1. get_int_from_table_16 (iq_kernels.dp.cpp:64) in dq_iq4_nl/iq4_xs: 8 scalar LUT -> 2 calls (halve LUT pressure). CUDA mmq-load-tiles.cuh:1528 pattern.
2. Register-block dequant output into tiles (already in xmx_gemm_iq 2542; extend to other paths).
3. Wider per-work-item chunks: 16 WI x 16 values vs 32 x 8. Halves WG count.
4. Amortize per-sub-block scale across il iterations (iq2_xs/iq3_s).

## Acceptance: prompt tok/s up at 2,184 & 8,000 tokens, bitwise-identical fp16 dequant. Baseline 30% prompt time (exp11).
## EXPAND
- LEAD: lever 1 (get_int_from_table_16 for iq4_nl/iq4_xs) immediately actionable with code locations.
- DEADEND: no llama.cpp SYCL port exists for P0-2; it is a Strata-side kernel project (confirms previous exp 09/10/11).
