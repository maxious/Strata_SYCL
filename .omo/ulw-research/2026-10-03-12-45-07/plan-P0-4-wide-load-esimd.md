# P0-4 plan — Q8_0/Q8_1 wide-load + Q2_K/Q5_K reordered ESIMD (decode follow-on)

Verdict: **PARTIAL / RE-PRIORITIZED** — most of P0-4's mechanisms are either **already present in Strata's wide MMVQ kernels** or **duplicate of P0-L1's ESIMD infra**, and the one true type gap (Q2_K) has **no deployed-model user**. Only one cheap A/B is clearly worth it on its own.

## Mechanisms identified (distinct from P0-L1; verified at lines)
1. **Q8_0/Q8_1 wide-load MMVQ + DMMV ESIMD (#29186):** `reorder_vec_dot_q8_0_wide` (vecdotq.hpp:419-446) — single `sycl::int4` reinterpret_cast of 4 consecutive dwords + 4 dp4a (gated `GGML_SYCL_MMVQ_WIDE`, default 1); ESIMD `q8_0_mac_stripe<NBLK>` (dmmv.cpp:2012) + `dequantize_mul_mat_vec_q8_0_reorder_esimd` (dmmv.cpp:2046) using `block_load<int8_t,32*NBLK>`, WG scales with `nblk_row`.
2. **Q2_K reorder ESIMD (#27490):** `reorder_vec_dot_q_sycl<Q2_K>` (vecdotq.hpp:481-511) SoA [qs/scales/dm]; `esimd_reorder_q_traits<Q2_K>` (esimd.hpp:74-157) mac_pair `block_load<uint8_t,64>`/`<16>`, 8 chunks of 32.
3. **Q5_K ESIMD (#26376):** `reorder_vec_dot_q_sycl<Q5_K>` (vecdotq.hpp:673-730); `esimd_reorder_q_traits<Q5_K>` (esimd.hpp:374-485) with `extract_bit_to_pos4`.

Distinction from L1: L1 = the reorder infrastructure + generic ESIMD skeleton + dispatch; P0-4 = the *specific* new kernels (wide Q8_0, Q2_K/Q5_K traits). The ESIMD DMMV list covers Q2_K..Q6_K+Q8_0 only (no i-quants).

## Port map — where Strata already stands
Strata **already has** wide MMVQ for the relevant types (`native_mmvq.dp.cpp`): `Wide32Q8` (1710), `Wide32IQ4NL` (1730), `WideQ4K` (1569), `WideQ5K` (1612), `WideIQ4XS` (1628), Q6_K wide `load16_a2`, Q3_K generic MMVQ, `s_gemv` Q8_0 act. So:
- **Q8_0 wide already done** (via `load16_a2`); llama.cpp's `sycl::int4` reinterpret_cast is the only variant difference.
- Q4_K/Q5_K/Q6_K reorder already covered by Strata wide MMVQ.
- **Q2_K is the only genuinely missing dense decode type** (native_mmvq.hpp has Q2_0 but no Q2_K). **BUT** no Strata model uses Q2_K (absent from all READMEs/INTEL, Coder is IQ1_M) → latent gap, not an active bottleneck.

## Re-prioritized scope
1. **(Keep) Q8_0 `int4` wide-load A/B** — swap `Wide32Q8::load` (`load16_a2`) for llama.cpp's `sycl::int4` reinterpret_cast (vecdotq.hpp:432-433) and bench on the B60. Cheap, one kernel, might beat two-aligned-loads+shift.
2. **(Defer) ESIMD DMMV infra** — Strata has zero ESIMD code; it only pays once P0-L1 lands (the same infra). Do NOT build it standalone for P0-4; fold into P0-L1.
3. **(Defer-pending-model) Q2_K reorder MMVQ port** — vecdotq.hpp:481-511 + esimd.hpp:74-157. Only if a Q2_K model enters the test matrix; today no user.

## Acceptance / success criteria
- C1 (A/B): Q8_0 int4 wide-load ≥ current `Wide32Q8` throughput at B70 decode shapes, with `s2_gemv_q8_parity`/`iq_parity` green; if slower, keep `load16_a2` (document in INTEL.md).
- C2: no regression — Q8_0/IQ4_NL stay ≥ 230 GB/s, IQ4_XS stays LUT-path (do NOT touch — `load16_a2` is 2-9% slower, exp-04 confirmed), Q4_K/Q5_K ≥ 230-280, Q6_K ≥ 400.

## Risk
ESIMD DMMV is Intel-only (`__INTEL_LLVM_COMPILER` + `<sycl/ext/intel/esimd.hpp>`). Q2_K deferral is correct today (no model user) — re-open the moment the IQ3_S/model matrix adds a Q2_K shard.