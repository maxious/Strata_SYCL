# Implementation plan — mass-ulw P0 lanes (B60), from research

Session env verified 2026-10-03: icpx 2026.1.1 + 2x Arc Pro B60 (LZ JIT); build-b60 ninja; **ESIMD probe PASSED** (block_load/half + explicit_simd → SUM=1 on "Intel Arc Pro B60"). P0-L1 ESIMD gate cleared.

## Topology (phased runs per mass-ulw — one run per phase, synthesize between)

Phase 1 (parallel, disjoint write scopes, both independent of ESIMD):
- **P0-2 dequant levers** — iq_kernels.dp.cpp ONLY (4 levers; get_int_from_table_16 first). File-disjoint from P0-4.
- **P0-4 Q8_0 int4 A/B** — native_mmvq.dp.cpp Wide32Q8::load swap → bench/verdict. File-disjoint from P0-2.
  → one run, 2 fan-out lanes + their own verify; decades the loop and settles int4-vs-load16_a2 for P0-L1's reorder design.

Phase 2 (THE port, sequential, needs Phase-1 verdict + baseline):
- **P0-L1 reorder+ESIMD** — ESIMD infra + reorder_qw_* + quantize_act_soa + mul_mat_vec_q_reorder + fused GLU/MoE + new esimd_kq_parity. Cross-file but centered on native_mmvq.dp.cpp (NOT parallel with P0-4 — serialize).

Phase 3 (regression + docs):
- Full parity suite re-run on B60; INTEL.md + README.sycl.md updates in repo style.
- P0-3: doc-only (research refuted-as-gain).

## Write-scope matrix (disjoint to parallelize; else chain)
- iq_kernels.dp.cpp → P0-2 only.
- native_mmvq.dp.cpp (+ reorder_weights.dp.cpp, quantize_act_reorder.dp.cpp, new parity) → P0-4 phase 1, P0-L1 phase 2 (serialized: same file family).
- docs → phase 3 only.

## Node prompt contract
Each node: TASK / DELIVERABLE / SCOPE (exact paths, hard boundary) / VERIFY (exact ninja target, parity command, bench invocation, binary observable) / STOP WHEN. Route: P0-2 = deep-low (kernel-tuning judgment), P0-4 A/B = quick→unspecified-low, P0-L1 = deep-high (cross-file kernel + parity design). Verification wave = the parity/bench build+run node depending on producers.

Baseline to capture BEFORE any change (Phase 0, done in-session): current parity suite green + decode/prompt baseline numbers (for fair before/after).

## BASELINE (captured in-session on B60, pre-change)
- Env: icpx 2026.1.1, 2x Arc Pro B60 LZ:0/1, build-b60 ninja. ESIMD probe PASSED (SUM=1).
- Parity green pre-change: iq_multi_parity (0 fail, bitwise equal incl IQ1_M/IQ4_XS/Q2_0/IQ4_NL), s_gemv_q8k_parity (0 fail, rel<5e-9), s2_gemv_q8_parity (0 fail, worst rel 3.7e-5), quantize_act_parity (0 fail). iq_parity NO fixture (needs gguf-py module, infra dep). native_expert_parity needs a GGUF shard arg (Qwen3.8-Flash-Next-IQ1_M-00001 present).
- Bench baseline (mmvq_sg_bench): Q8_0 Wide 2560x10240 cols2 427.4 GB/s, cols4 314.9, cols6 282.7; 6144x2560 cols2 691; 2560x2560 cols2 621.6; Q6_K 140-300; Q5_K 108-216.
- Model shards: Qwen3.8-Flash-Next-IQ1_M / IQ3_S (2-shard), gemma3 Q4_K_M, Qwen3.8-27B-Q3_K_M.


## DIRECT-IMPLEMENTATION AUDIT RESULTS (cross-checked against real code)
1. **mass-ulw routing is DOWN**: every dag category incl. 'quick' fails 'No available model' (deterministic x5 dag + x2 task earlier). Only subagent_type=explore (READ-ONLY) routes. No writable worker can be spawned => cannot parallelize implementation via dag.
2. **P0-3 refuted-as-gain** (already known): drop. no code.
3. **P0-4 int4 A/B is layout-invalid standalone**: Wide32Q8 runs on Strata's AOS 34-byte Q80Block (2-byte aligned, qs@2); a raw sycl::int4 reinterpret_cast would misalign. llama.cpp's int4 wide only valid on its SoA REORDERED layout. => P0-4 FOLDS INTO P0-L1 (needs the weight reorder first). Not an independent quick lane.
4. **P0-2 lever 1 (get_int_from_table_16) inapplicable to fp16 path**: that helper (lines 390/412/1232) packs int8 bytes for the dp4a *vec_dot accumulators*. dq_iq4_nl/iq4_xs (1726/1767) write fp16 via d*kvalues_iq4nl[...] — a scalar LUT lookup per value that cannot be expressed as int8-packed-table extraction bit-identically. The research lane over-applied the CUDA mmq (int8/GEMM) trick to the fp16 dequant. P0-2 safe levers remaining: #3 wider chunks (16 WI x 16) + #4 amortize per-sub-block scale.
5. Env FULLY FEASIBLE: icpx 2026.1.1, 2x B60 LZ, ESIMD probe PASSED (only P0-L1's flagged gate, now cleared). Baseline parity green + benches captured.
# reorder_esimd_bench - P0-L1 Q8_0 vertical slice, B60 (2026-10-03)

VERIFIED: reorder(SoA)+ESIMD decode == native AOS output within float rounding (rel 6.5e-8..4.2e-7). ESIMD WINS at multi-column shapes:
  cols1: 0.89-0.99x (wash)
  cols2: 1.03-1.27x | cols4: 1.33-1.73x | cols6: 1.49-2.13x
  6144x2560 cols6: AOS 370 GB/s -> ESIMD 788 GB/s (2.13x)
  ESIMD ~ flat 400-420 GB/s at 2560x{10240/12288}, 786-790 at 6144x2560 (cols2/4/6)
Source: sycl/src/kernels/reorder_esimd_bench.cpp (+ CMake target, links strata_kernels)
Reference ported: llama.cpp dmmv.cpp q8_0_mac_stripe / dequantize_mul_mat_vec_q8_0_reorder_esimd + reorder_qw_q8_0.


## Ledger (end of turn)
- P0-L1 Q8_0 reorder+ESIMD SLICE: PROVEN + COMMITTED (2 commits).
  - e07752f: bench slice — reorder(SoA)+ESIMD decode == AOS output within float rounding (rel 6.5e-8..4.2e-7); wins multi-column decode 1.03-2.13x (6144x2560 cols6 370->788 GB/s); cols1 wash 0.89-0.99x. Parity gates green.
  - 3c7d62a: --selftest regression gate (esimd_kq_parity), ctest 100% pass.
- REMAINING: production dispatch wiring (per-tensor SoA buffer + STRATA opt-in + wire into shared_expert/ple decode). Large integration, next.
- P0-2 dequant levers: not yet started (audit: lever1 get_int_from_table_16 inapplicable to fp16 path; safe subset = wider chunks + amortize scale).
- P0-4: folded into P0-L1 (int4 only valid on reordered SoA). P0-3: research refuted-as-gain, no code.
- ENV facts: mass-ulw category router DOWN (empty registry, not config-fixable, reverted omo.json); direct implementation path.


## Full parity suite (B60, after my 2 commits)
All pass when invoked correctly (my first batch passed wrong args): bf16_gemv, cvec, dequant_s2, elementwise, gdn, gr, iq_multi, kv_hybrid, kv_q4, kv_q8, kv_stream, native_expert(with shard), native_grouped, qsa, qsa_prompt_attn, quantize_act, rope, router_top10, s2_gemv, s2_gemv_q8, sampler, s_gemv, s_gemv_q8k, shared_expert = PASS.
- PRE-EXISTING failure: s2_expert_grouped_parity rc=1 "4 failures" (moe_grouped_s2 DIFFERS). PROVEN unrelated: my 2 commits changed only docs/12-*.md + CMakeLists(+additive target) + new reorder_esimd_bench.cpp; no kernel .cpp touched. => pre-existing, tracked, not caused.
- Fixture-gated (known infra): iq_parity (needs gguf-py fixtures), ple_parity (table file missing, partial skip).
- MY new gate: reorder_esimd_bench --selftest (esimd_kq_parity) = ctest PASS.


## C4 (P0-4) verdict - committed reorder_esimd_bench
The int4/reorder wide-load (llama.cpp vecdotq.hpp:419-446, one wide 4-dword load fed dp4a on a reordered SoA layout = the alignment precondition int4 needs) IS the qs-loading in my reorder+ESIMD path. Measured B60: >= AOS Wide32Q8 at cols 2/4/6 (1.03-2.13x), not slower; cols1 wash. => C4 satisfied by the P0-L1 slice. Q2_K stays deferred (no model user).

## Real-surface tok/s (C2/C3) - NOT RUN (honest)
strata binary runs under sourced oneAPI, but a real generate needs --pack (absent) + --ple-gguf + 2 shards + native flags. Building the pack is a heavy setup stage not in this checkout. Cannot produce a real decode/prompt tok/s number without it.
P0-2: flat 32-WI dequant already amortizes per-thread scale; get_int_from_table_16 inapplicable to fp16 path; wider-chunk lever needs the pack harness to measure.


## REAL-SURFACE HARNESS: HARD INFRA BLOCK (verified 2nd attempt)
strata_pack.py build RUNS (reads 1223 tensors, builds experts.bin 31.64 GiB) but pack_layer.py:82 asserts t.type_name == "Q2_0"; present models are IQ1_M/IQ3_S (Flash-Next), Q3_K_M (27B), gemma Q4_K_M - NO Q2_0 shard exists, so a real `strata generate` decode/prompt tok-s cannot be built from any model here. C2/C3 model-tok-s evidence unproducible without a Q2_0 Flash-Next shard.
BENCH-LEVEL C2 IS satisfied: Q8_0 reorder+ESIMD moved misaligned cols 4/6 row from 229-415 GB/s (AOS) to 397-424 GB/s (aligned-class), and 6144x2560 cols6 370->788 GB/s (mmvq-class bench).


## CRITICAL CORRECTION (2026-10-03)
My committed P0-L1 '2.13x multi-column win' was a BENCH BUG: the ESIMD DMMV kernel is single-column, and the bench ran it ONCE against the AOS path's nc columns. Honest per-column measurement (commit 9b4ad3b): single-col wash (0.88-1.14x), multi-col 0.25-0.63x SLOWER (ESIMD re-launched per column loses AOS column-vectorization). Port is CORRECT (--selftest/esimd_kq_parity green) but NOT wired into production; parked as not-a-win. Docs corrected.
