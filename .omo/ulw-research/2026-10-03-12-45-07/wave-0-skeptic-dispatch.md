# Wave 0 — skeptic dispatch read (lead, from source)
source: /home/maxious/ComfyUI/koboldcpp/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp
- ggml_sycl_supports_mmq() returns false unconditionally (l4084): confirms exp-10, NO SYCL i-quant GEMM/MMQ. refutes any P0-2 'port llama.cpp's prompt GEMM' route.
- supports_reorder_mmvq (Q1_0,Q4_0,Q8_0,Q2_K..Q6_K, l4123): no i-quants — confirms exp-10. L1/P0-4 dense reorder is the mechanism.
- supports_reorder_esimd (QS_K..Q6_K + Q8_0, l4140 under #GGML_SYCL_DMMV_HAS_ESIMD): the ESIMD reorder covers dense decode types + Q8_0 — connects P0-L1 and P0-4.
- should_reorder_tensor (l4749): needs opt reorder + dst->op==MUL_MAT + src1->ne[1]<=8 (+ ne[2]/ne[3]==1) → single/small-batch DECODE path only. reorder is a decode-path optimization, not prompt. Confirms README's 'ESIMD is decode'.
- dispatch ladder (l4876-4925): reorder+ESIMD branch requires !prioritize_dmmv AND should_reorder_tensor AND supports_reorder_mmvq; else DMMV dequantize_mul_mat_vec or use_mul_mat_vec_q(quantize_and_reorder_q8_1_soa when optimized). => reorder/ESIMD is REAL on the decode path, gated by GGML_SYCL_ENABLE_OPT env + device opt_feature.reorder.
- SCEPTIC CAVEAT: the whole mechanism is OPT-IN in llama.cpp too (env GGML_SYCL_ENABLE_OPT / enable_esimd). Strata must also gate it; default-fastest stays non-reorder. Consistent w/ Strata's opt-in flag philosophy.
