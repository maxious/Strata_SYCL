# Sources ledger
Session: .omo/ulw-research/2026-10-03-12-45-07 · llm.cpp checkout HEAD cb7934c52 (branch master)

| # | source | what it contains | reliability | accessed |
|---|---|---|---|---|
| 1 | ~/ComfyUI/koboldcpp/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp | dispatch: supports_mmq(4084), reorder gating(4749-4654 reorder_qw), mul_mat ladder(4825-4925), host-pinned(1626), dev2dev(822-828), comm_exchange host staging(7063/7189) | primary | 2026-10-03 |
| 2 | .../ggml-sycl/dmmv.cpp | ESIMD decode kernels: generic 1876, Q8_0 2012/2046, launchers 1923-2137, dispatch 2188/2205-2370 | primary | 2026-10-03 |
| 3 | .../ggml-sycl/esimd.hpp | esimd_reorder_q_traits Q2_K(74)..Q6_K(586); mac_pair, splat_lo_hi, unpack_scale_min_k4 | primary | 2026-10-03 |
| 4 | .../ggml-sycl/vecdotq.hpp | reorder_vec_dot_q8_0_wide(419), Q2_K(481), Q5_K(673); vec_dot_iq* 1434-1760 | primary | 2026-10-03 |
| 5 | .../ggml-sycl/mmvq.cpp / quanitze.hpp | mul_mat_vec_q_reorder(28/81), MoE reorder(2994/3249); quantize_and_reorder_q8_1_soa(63) | primary | 2026-10-03 |
| 6 | .../ggml-sycl/common.hpp, quants.hpp, presets.hpp | DEV2DEV_MEMCPY(125), opt_feature.reorder, vdr_mmvq, PEER_MAX_BATCH_SIZE=128 | primary | 2026-10-03 |
| 7 | .../ggml-cuda/mmq-load-tiles.cuh + mmq.cu | CUDA-only i-quant GEMM (proves SYCL absence) | primary | 2026-10-03 |
| 8 | /home/maxious/Strata_SYCL/sycl/src/program/generate.cpp | separate-context-per-stage(2655-2680), host-pinned hand-off(4816, 4801-4824) | primary (own repo) | 2026-10-03 |
| 9 | Strata sycl/src/kernels/cuda/iq_kernels.dp.cpp | iq_dequant_f16(2298), dq_iq*(1615-1732), get_int_from_table_16(64), xmx_gemm_iq(2542) | primary (own repo) | 2026-10-03 |
| 10 | Strata native_mmvq.dp.cpp / native_mmvq.hpp | Wide kernels; supported codes (no Q2_K) | primary (own repo) | 2026-10-03 |
| 11 | Strata docs/sycl-experiments/03,04,07,09,10,11 | prior P2P/alignment/bottleneck/dequant measurements | repo log | 2026-10-03 |
| 12 | Strata docs/INTEL.md, README.sycl.md | prior trap log + P0 list | repo log | 2026-10-03 |

Distinct primary sources: llama.cpp checkout (1) + Strata repo (2). Unique source domains: 2 (both local repos).