# P0-L1 plan — decode matvec weight reorder + ESIMD (llama.cpp port)

Verdict: **ACTIONABLE** — the single biggest open decode lever, on-by-default in llama.cpp, clean port map.

## Mechanism (what we port)
llama.cpp's decode matvec trick is a **one-shot SoA weight reorder at (lazy) load time**, then an **ESIMD mac_pair kernel** (and a non-ESIMD `mul_mat_vec_q_reorder`) that streams the contiguous reordered bytes. XMX is NOT involved — this is the memory-rate decode win the README section 2 foretold.

## llama.cpp reference code (all verified at exact lines, HEAD cb7934c52)
- Reorder step — `reorder_qw` dispatcher (ggml-sycl.cpp:4707), per-type `reorder_qw_q{4_0,8_0,2_k,3_k,4_k,5_k,6_k}` (ggml-sycl.cpp:4251-4654). SoA layouts documented in esimd.hpp comments. Triggered lazily by `opt_for_reorder` (ggml-sycl.cpp:4758), gated by `should_reorder_tensor` (4749): `enable_optimize && opt_feature.reorder (arch probe: ext_oneapi_arch_is(intel_gpu), l213) && op==MUL_MAT && src1->ne[1] <= 8`. Activation side reordered in lockstep: `quantize_and_reorder_q8_1_soa` (quantize.hpp:63).
- ESIMD kernel — generic `dequantize_mul_mat_vec_reorder_esimd<T>` (dmmv.cpp:1876): work-group owns a row-pair, `block_load<float,256>` activations, `traits::mac_pair`; per-type `*_sycl_reorder_esimd` (dmmv.cpp:1923-2007); explicit Q8_0 `q8_0_mac_stripe` (dmmv.cpp:2012)/`dequantize_mul_mat_vec_q8_0_reorder_esimd` (2046). Traits in esimd.hpp (`esimd_reorder_q_traits<Q2_K..Q6_K>`:74-586, `mac_pair`, `splat_lo_hi`, `unpack_scale_min_k4`).
- API surface to port: `sycl::ext::intel::esimd` → `simd<T,N>`, `block_load<T,N>`, `simd::select`, `convert`, `reduce`, `[[intel::sycl_explicit_simd]]`. Compile gate: `__INTEL_LLVM_COMPILER` + `<sycl/ext/intel/esimd.hpp>` (dmmv.cpp:8-13).

## Call-path proof (skeptic-verified)
The reorder-ESIMD branch is in the single/small-batch **decode** ladder of `ggml_sycl_mul_mat` (ggml-sycl.cpp:4882-4893), not prompt (XMX/MMQ is prompt, `ne[1] > 8`, mmq.cpp:2962). Env `GGML_SYCL_ENABLE_ESIMD`/`GGML_SYCL_ENABLE_OPT` default to **1** — it's the shipped default decode path, not dead/opt-in-by-env.

## Strata files to touch → port → acceptance parity
1. `sycl/src/kernels/cuda/native_mmvq.dp.cpp` ← dmmv.cpp ESIMD kernels (Q2_K..Q6_K + Q8_0) → `s2_gemv_q8k_parity` + NEW `esimd_kq_parity`.
2. NEW `sycl/src/kernels/cuda/reorder_weights.dp.cpp` ← `reorder_qw_q*` (load-time one-shot SoA) → `s_gemv_parity` still passes after layout change.
3. NEW `sycl/src/kernels/cuda/quantize_act_reorder.dp.cpp` ← `quantize_and_reorder_q8_1_soa` → `quantize_act_parity`.
4. `native_mmvq.dp.cpp` ← `mul_mat_vec_q_reorder`/`_ncols` (mmvq.cpp:28,81) for Q1_0/Q4_0 (no ESIMD trait) → `iq_parity` + `iq_multi_parity`.
5. `shared_expert.dp.cpp` ← `ggml_sycl_mul_mat_vec_q_glu_reorder` (mmvq.cpp:3249) + fused GLU (ggml-sycl.cpp:4967) → `native_expert_parity`.
6. `s2_expert_grouped.dp.cpp` ← `ggml_sycl_mul_mat_vec_q_id_reorder` (mmvq.cpp:2994, MoE per-expert SoA) → `native_expert_parity`.

NEW test needed: `esimd_kq_parity` — the ESIMD reorder output vs existing `s_gemv` scalar reference on the same reordered weights; **and** a quantize-act SoA parity case (leads 1-2 from the lane; without the act-SoA parity, a layout mismatch silently breaks the decode).

## Acceptance / success criteria
- C1: `esimd_kq_parity` green (byte-exact/tol vs `s_gemv` reference); `s2_gemv_q8k_parity`, `iq_parity`, `iq_multi_parity`, `native_expert_parity` green.
- C2: decode tok/s on the Coder (B70): the misaligned IQ4_XS/IQ4_NL/Q8_0 rows move to ≥400 GB/s aligned-class; INTEL.md "still misaligned" list updated.
- C3: gate behind `STRATA_*` env (default off until measured faster); keep default on current-fastest.

## Risk
ESIMD is Intel-only (compile gate `__INTEL_LLVM_COMPILER`). Verify the Strata icpx/oneAPI toolchain accepts `<sycl/ext/intel/esimd.hpp>` (B60 build) before investing in the full infra. The non-ESIMD `mul_mat_vec_q_reorder` is the safe fallback that needs no ESIMD.