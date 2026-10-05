# Ultrawork Notepad — llama.cpp SYCL → P0 lane plans
Started: 2026-10-03T12:45Z

## Plan
1. Bootstrap: session dir + brief + goal (DONE)
2. Scope solo: read the 6 exp docs + INTEL.md P0-relevant bits (partly done)
3. Fan out team (4 axis owners + skeptic) via task category=deep-low/ultrabrain
4. Saturation waves; expand leads; debate contested claims
5. Verify contested claims by reading/running the checkout
6. Synthesize per-lane plan + synthesis; deliver; teardown

## Success criteria + QA scenarios
(from goal)

## Now
Bootstrap + scope

## Todo
- [x] bootstrap session dir + brief + goal
- [ ] read exp docs 00/03/04/07/09/10/11 + INTEL.md P0 flavor
- [ ] emit brief.md, register goal
- [ ] fan out team 6 members
- [ ] collect wave returns, journal, expand
- [ ] debate contested claims
- [ ] verify by running code
- [ ] write per-lane plans + SYNTHESIS.md
- [ ] teardown (cancel all lanes)

## Findings
- llama.cpp checkout CONFIRMED at ~/ComfyUI/koboldcpp/llama.cpp, HEAD cb7934c52, branch master, ggml/src/ggml-sycl present with all reference files (dmmv.hpp, mmq.cpp, esimd.hpp implied, fattn-*).
- Session dir: .omo/ulw-research/2026-10-03-12-45-07

## Update [wave 0.5]
- P2P lane DONE (st_01a101ce): full host-pinned + dev2dev mapping. KEY: P0-3 premise refuted as a *gain* — direct P2P can't apply (separate contexts = silent no-op, ggml-sycl.cpp:7189; confirmed no peer_access in Strata grep), hand-off is already host-pinned via malloc_host (generate.cpp:4816) and is not the bottleneck (34us/256KiB, exp03:40-46). 1.4x decode gain (21.5-25.4 -> 35.5) is from layer-split compute distribution, already shipped. => RECOMMEND P0-3 re-scope to drop/park the "pipe through native dev2dev" goal; verdict recorded (C-P0-3-1 refuted-as-gain, C-P0-3-3 recommend drop).
- Remaining lanes running: reorder-decode (st_01a101cc), int8-dequant (st_01a101cd), wide-load (st_01a101cf).

## Update [wave 1]
- P0-L1 lane DONE (st_01a101cc): full reorder+ESIMD port map. reorder_qw(SOA, load-time lazy) + esimd mac_pair + mul_mat_vec_q_reorder + fused GLU/MoE; map to native_mmvq.dp.cpp / new reorder_weights.dp.cpp / new quantize_act_reorder.dp.cpp / shared_expert / s2_expert_grouped. C-P0L1-1/-2 supported. NEW esimd_kq_parity needed + LEAD1 (act SOA parity) + LEAD2 (MoE per-expert SOA).
- 2/4 lanes done (P0-L1, P0-3). Remaining: int8-dequant (st_01a101cd), wide-load (st_01a101cf).

## Update [wave 2]
- P0-2 lane DONE (st_01a101cd): VERDICT CONFIRMED — llama.cpp SYCL has no prompt-path i-quant mechanism worth porting (supports_mmq false 4084; mmq.cpp zero iq; i-quant GEMM CUDA-only). BUT P0-2 is a Strata-side kernel project: 4 levers (get_int_from_table_16 halve LUT, register-block output, wider chunks 16x16, amortize scale). C-P0-2-1/2/3 supported.
- 3/4 lanes done (P0-L1, P0-2, P0-3). Remaining: wide-load (st_01a101cf).
- Note: P0-2 does NOT depend on a llama.cpp port — it's Strata kernel optimization. Distinct from P0-L1 (which IS a llama.cpp ESIMD port).

## Update [wave 3]
- P0-4 lane DONE (st_01a101cf): 4/4 lanes complete. Q2_K = only missing dense decode type (port reorder MMVQ vecdotq.hpp:481-511, esimd.hpp:74-157). Q8_0/Q4_K/Q5_K/Q6_K/IQ4_NL already have Strata wide MMVQ (native_mmvq.dp.cpp). ESIMD DMMV infra absent in Strata (Intel-only).
