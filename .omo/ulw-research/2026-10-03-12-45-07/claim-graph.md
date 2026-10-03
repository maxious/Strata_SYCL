# Claim graph

## verified-claims digest
(high-risk non-code claims that cleared Phase 4b — none yet)

## Claims
| claim_id | statement | type | risk | scope | intent | supporting obs | contradicting | convergence | primary source | status |
|---|---|---|---|---|---|---|---|---|---|---|
| C-P0L1-1 | llama.cpp decode reorder_qw + ESIMD on decode path, gated by opt + arch probe | code | normal | P0-L1 | IT1 | ggml-sycl.cpp:4749,4758,4882-4893; dmmv.cpp:1875,1923-2137; esimd.hpp | none | confirmed | source | supported |
| C-P0L1-2 | Strata port = SOA reorder + ESIMD mac_pair + mul_mat_vec_q_reorder + fused GLU/MoE; parity s2_gemv_q8k/iq/native_expert + NEW esimd_kq_parity | port-map | normal | P0-L1 | IT1 | lane f:l refs | none | confirmed | Strata sources | supported |
| C-P0-2-1 | supports_mmq false; no SYCL i-quant GEMM/MMQ; i-quant path CUDA-only tensor-core | code | normal | P0-2 | IT2 | ggml-sycl.cpp:4084,4123,4871; mmq.cpp zero iq; mmq-load-tiles.cuh 1036-1550; mmq.cu 48-71 | — | confirmed (lane verified all 4) | source | supported |
| C-P0-2-2 | llama.cpp SYCL i-quant dequant kernels have no prompt-path callers (conv/fattn/matmul fallback only); Strata iq_dequant_f16 sole prompt dequant | code | normal | P0-2 | IT2 | convert.cpp:342-527,656,746; dequantize.hpp 1410-1640; iq_kernels.dp.cpp:2298 | — | confirmed | source | supported |
| C-P0-2-3 | P0-2 is a STRATA-SIDE kernel project (not llama.cpp port): 4 levers (get_int_from_table_16, register-block, wider chunks, amortize scale) | judgment | med | P0-2 | IT2 | lane levers f:l | — | verdict | Iq_kernels.dp.cpp | supported |
| C-P0-3-1 | direct dev2dev P2P inapplicable to Strata (separate contexts); host-staged forced | code | high | P0-3 | IT3 | generate.cpp:2668,4816; ggml-sycl.cpp:7189; grep empty | README premise | confirmed | both | refuted-as-gain |
| C-P0-3-2 | hand-off not decode bottleneck (~34us); gain from compute distribution | measured | high | P0-3 | IT3 | exp03 | — | confirmed | exp03 | supported |
| C-P0-3-3 | RECOMMEND P0-3 re-scope/drop | judgment | med | P0-3 | IT3 | C-3-1+C-3-2 | — | verdict | — | supported |
