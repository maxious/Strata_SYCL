# P0 #5: pre-unpacked (byte-layout) decode for the remaining dense K-quants

Goal UUID: from create_goal (strata-sycl). Phases:

## Phase 1 - analysis (mass-ulw run `p05-analysis`)
One report node per dense type -> sycl/bench/reports/p05/<type>.md; each node reads dequant.hpp +
native_mmvq.dp.cpp Q*KTraits/q*_q8_dot_impl, plus counts the type's tensors in the real GGUF shards
(~/ComfyUI/koboldcpp, 15 files) via the repo's gguf reader. Nodes: q8_0, q5_k, q4_k, q3_k, q2_0, iq4_xs,
+ a `usage` node that totals dense tensors by type across the shards.

## Phase 2 - decision (lead)
Merge reports; choose USED + FEASIBLE + likely-WIN subset. Non-viable -> park in README log.

## Phase 3 - implement (serialized on shared native_mmvq.dp.cpp)
Per chosen type: native_q*K_preunpack (generalize Q6UBlock to {d0,d1[,m0,m1];int8 q[32]}), routed
no-bit-unpack decode (extend Wide32-style kernel with m*sum(a) for min-offset types), native_dense
default-on + env opt-out, parity ctest ~1e-7 vs native oracle, measure bench.

## Phase 4 - verify + record
strata_engine builds; default no-env path unchanged; README/INTEL/memory updated; one atomic commit /
increment.

## Key constraints
- Parallel write-scope: implementation lanes share native_mmvq.dp.cpp -> SERIALIZE or merge; reports are
  disjoint (own file each).
- Measure-first: Q6_K won because its 6-bit unpack was the pipe cost and it's a clean centered format;
  the min-offset K-quants (d*q - m form) need m*sum(a) in decode - verify feasibility + usage before code.
