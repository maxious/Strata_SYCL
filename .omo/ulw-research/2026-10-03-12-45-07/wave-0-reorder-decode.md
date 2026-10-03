# Wave 0 — P0-L1 reorder+ESIMD return (complete, st_01a101cc)

## Reorder mechanism
- Gating: should_reorder_tensor (ggml-sycl.cpp:4749): g_ggml_sycl_enable_optimize && opt_feature.reorder && op==MUL_MAT && src1->ne[1]<=8; arch probe opt_feature.reorder from intel_gpu (l213). lazy one-shot via opt_for_reorder (l4758) -> reorder_qw, flips extra->optimized_feature.reorder.
- Types: Q4_0,Q8_0,Q2_K..Q6_K dense (4729-4743); MoE Q4_K/Q5_K/Q6_K (4717-4723, moe variants at 4373/4418/4467 keep expert slices self-contained stride nb[2]).
- Per-type SOA layouts listed (esimd.hpp comments) for Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K.
- Activation lockstep: quantize_and_reorder_q8_1_soa (quantize.hpp:63) on MMVQ path.

## ESIMD kernels
- Compile gate dmmv.cpp:8-13 (#if __INTEL_LLVM_COMPILER -> include esimd.hpp, define GGML_SYCL_DMMV_HAS_ESIMD).
- API: namespace sycl::ext::intel::esimd — simd<T,N>, block_load<T,N>, simd::select, convert, reduce, [[intel::sycl_explicit_simd]].
- Generic dequantize_mul_mat_vec_reorder_esimd<T> dmmv.cpp:1875: work-group owns row-pair, block_load<float,256> activations, traits::mac_pair (4WG x 2-row). per-type *_sycl_reorder_esimd l1923-2007; explicit Q8_0 path q8_0_mac_stripe (2034) WG scaled by nblk_row (2119-2131).
- esimd.hpp per-type traits Q2_K..Q6_K incl mac_pair, splat_lo_hi, unpack_scale_min_k4.

## Call path = decode (confirmed)
- ggml_sycl_mul_mat l4825 ladder; reorder-ESIMD branch l4882-4893 (needs enable_esimd + supports_reorder_esimd); then DMMV opt_for_reorder -> ggml_sycl_op_dequantize_mul_mat_vec (dmmv.cpp:2188); MMVQ -> op_mul_mat_vec_q with reorder if optimized.
- XMX = prompt (batched ne[1]>8, mmq.cpp:2962); ESIMD = decode (ne[1]==1..8). supports_reorder_esimd: Q2_K..Q6_K+Q8_0; mmvq additionally Q1_0/Q4_0.
- envs: GGML_SYCL_ENABLE_ESIMD=1, ENABLE_OPT=1, PRIORITIZE_DMMV=0.

## Port map (Strata files -> llama.cpp funcs -> parity)
- native_mmvq.dp.cpp <- dmmv.cpp ESIMD kernels (Q2..Q6 + Q8_0) -> s2_gemv_q8k_parity + NEW esimd_kq_parity
- NEW reorder_weights.dp.cpp <- reorder_qw_q* (4251-4654) -> s_gemv_parity
- NEW quantize_act_reorder.dp.cpp <- quantize_and_reorder_q8_1_soa (quantize.hpp:63) -> quantize_act_parity
- native_mmvq.dp.cpp <- mul_mat_vec_q_reorder/_ncols (mmvq.cpp:28,81) Q1_0/Q4_0 -> iq_parity + iq_multi_parity
- shared_expert.dp.cpp <- ggml_sycl_mul_mat_vec_q_glu_reorder (mmvq.cpp:3249)+fused GLU (ggml-sycl.cpp:4967) -> native_expert_parity
- s2_expert_grouped.dp.cpp <- ggml_sycl_mul_mat_vec_q_id_reorder (mmvq.cpp:2994) -> native_expert_parity

## EXPAND leads (from lane)
- LEAD1 quantize_and_reorder_q8_1_soa act-side: Strata has no SOA-variant parity test; porting ESIMD w/o it silently wrong. => add quantize_act_parity SOA case.
- LEAD2 MoE expert reorder reorder_qw_q{4,5,6}_k_moe per-expert SOA -> s2_expert_grouped.dp.cpp could unlock ESIMD for MoE decode (the Coder's dominant workload).
- DEAD END IQ4_XS load16_a2 (exp04) 2-9% slower; LUT/ALU-bound not load-bound.
