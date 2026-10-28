# dsv41-trellis-sg16-perf - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**Who this is for and what changes for them:** You serving the DeepSeek model on your two Intel Arc Pro cards, and the apps and agents that talk to it. Prompt-time math for the model's expert layers gets faster when the option is on; the answers you get stay the same.

**What you'll get:** A faster math path for batched expert work, switched on by an option you choose (default off), plus measured numbers for five decode questions: keeping hot model parts resident, using both cards, small-batch cutoffs, the router gate, and launch overhead.

**Why this approach:** Today the batched path copies weights through a slow staging step before the tile; the new path decodes them straight into the tile, and the integer form measured about twice as fast on your cards. Model files and the default path do not change.

**What it will NOT do:** No CUDA-side changes, no new or changed model file formats, no change to single-token answering, no risky accumulation tricks, no changes to the dense chunked math path.

**Effort:** XL
**Risk:** High - a new tile arithmetic must reproduce the old answers within tolerance, and the fast form exists only in one build flavor
**Decisions to sanity-check:** whether the new path becomes the default after measurement; whether any decode probe earns adoption (each returns to the approval gate); the small-batch cutoff stays untouched this round.

Your next move: approve, or run a high-accuracy review. Full execution detail follows below.

---

> TL;DR (machine): XL / High risk - i8 A-stage sg16 tile family in ds4_iq.dp.cpp (opt-in, default off) + parity rows + ctest registration + trellis feasibility probe + five measured decode probes + docs numbers.
<!-- Effort is exactly ONE band, never hours/days. Quick = single edit, minutes of agent work; Short = one focused change, a few files; Medium = multi-file feature in one session; Large = several waves, one long session; XL = multi-session or architectural work. A written duration is rewritten to a band. -->
**Risk:** High - a new tile arithmetic must reproduce the old answers within tolerance, and the fast form exists only in one build flavor
**Decisions to sanity-check:** whether the new path becomes the default after measurement; whether any decode probe earns adoption (each returns to the approval gate); the small-batch cutoff stays untouched this round.

Your next move: approve, or run a high-accuracy review. Full execution detail follows below.

---

> TL;DR (machine): XL / High risk - i8 A-stage sg16 tile family in ds4_iq.dp.cpp (opt-in, default off) + parity rows + ctest registration + trellis feasibility probe + five measured decode probes + docs numbers.

## Scope
### Affected user and ideal state
<!-- From the draft ledger: who this touches and how they use the result, then one IS row per property with its reason, then one GAP row per difference with its reason. Every todo below closes a GAP row; ## Success criteria proves every IS row. -->
**Affected user:** The person serving dsv41 on 2x Intel Arc Pro B60 via `strata launch --family deepseek41`; apps/agents consuming the OpenAI/Anthropic API; maintainers reviewing branch sycl/dsv41-sycl-support. Today prompt-path expert FFN matmuls run the f16 tile shape (69.5 us gate+up at M=160); after, an opt-in i8 A-stage tile family runs the measured-faster shape (34 us i8) with decode byte-identical.

| Row | Statement | Reason |
| --- | --- | --- |
| IS-1 | Prompt-path expert FFN matmuls (M>1) run faster on B60 with the opt-in flag on | the measured i8 shape is ~2x the f16 shape |
| IS-2 | With the flag ON, tile outputs stay within the repo's one-BF16-ulp rule and decoded operands are bit-exact | the tile's F32 reduction order differs from today's lane, so exactness is bounded, not byte-level (ds4_parity.cpp:976-983) |
| IS-3 | Default path byte-identical with the flag off, including single-token decode; single-token decode never enters tiles | CONTRIBUTING opt-in rule; M=1 is memory-floor bound (22.3/11.1 us gate+up, 11.9/5.9 us down measured) |
| IS-4 | Build stays green with flag on and off, AOT bmg and JIT scalar stand-in | STRATA_XMX_SG16 is AOT-only (sycl/CMakeLists.txt:40-41) |
| IS-5 | Docs throughput claims tagged measured, decode probes report measured deltas | repo discipline: no unmeasured numbers |
| GAP-1 | set_tc_kernel round-trips weights deq8->f16 shared->ext_intel_packed pack->joint_matrix (ds4_iq.dp.cpp:1123-1133,1164-1187) | staging costs the 2x gap vs the i8 shape |
| GAP-2 | no split-K in repo kernels; KSPLIT exists only in scratch bench | four sub-groups walking disjoint K ranges is the measured shape |
| GAP-3 | dispatch has no arm routing M>1 experts to the new kernel family | the port must be selectable and kill-switchable |
| GAP-4 | no parity rows for the A-stage tile or global ext_intel_packed B load | numeric contract drift is risk #1 (ds4_iq.hpp:8-12) |
| GAP-5 | decode probes unmeasured: VRAM residency, two-card sharding, window crossover, gate gemv, launch count | the 1.65 vs 12.08 tok/s gap is the biggest decode lever |

### Must have
- New kernel family additive in sycl/src/dsv41/ds4_iq.dp.cpp; SetArgs/x3::SetArgs and gguf blob layout frozen; ds4_gguf.dp.cpp and ds4_model.dp.cpp untouched except dispatch-adjacent probes named in todos. Trellis types (TQK*, ds4_iq.dp.hpp:30) route through the is_tq branch to tq:: (ds4_iq.dp.cpp:2266/:2282/:2297-2307, no joint_matrix in ds4_tq.dp.cpp) and are covered by todo 13's feasibility probe, not by the first kernel family.
- Opt-in env kill-switch + per-type eligibility mask, default off; a type enters the tile path only when its parity row exists (precedent: kIQ4_NL refusal, ds4_iq.dp.cpp:2267); scalar stand-in compiled when STRATA_XMX_SG16 undefined (STRATA_XMX_SG16 defined only for AOT bmg/lnl, sycl/CMakeLists.txt:40-41).
- reqd_sub_group_size(16) rides on the functor's operator(), not a free function (set_tc_wrapper pattern, ds4_iq.dp.cpp:1241-1256).
- KSPLIT=4 deterministic reduction ((p0+p1)+p2)+p3 over segmented Part[4][16][20] added to the shared block with new static_asserts modeled on ds4_iq.dp.cpp:1063-1068 and the launch constants at :1038-1041/:1753-1754 updated in the same todo; atomics banned; single grouped grid over n_exp preserved (ds4_iq.dp.cpp:2273-2275).
- Gates: K%32==0, M=1 bypass, K/32<4 fallback, reqd_sub_group_size(16), divisibility guards (:2270-2271,:2290) intact.
- Parity rows copy ds4_parity.cpp:1149-1267: scale-sensitive data so sums land where clamped swiglu is sensitive, negative control corrupting one scale byte (:1186-1191), max_rows=32 > kLightRows=16 so tiles take every row (:1156); "a row that cannot fail is not evidence" (:1151-1155).

### Must NOT have
- No edits under src/dsv41 (the SYCL build links src/dsv41/ds4_tq_cpu.cpp, sycl/CMakeLists.txt:302 - linking is fine, editing is not); no persisted model-format change (ds4_tq.dp.cpp:2-4: the VRAM blob is the file's bytes); no M=1 or decode kernels on tiles; no atomic accumulation; no per-expert split launches; no oneDNN/oneMKL integration beyond existing parity refs; no dense chunked GEMM seam (ds4_model.dp.cpp:3420-3442); never stage untracked .omo/, .senpi/, bench/results/*; no /tmp references in acceptance criteria (the KSPLIT contract is inlined in todo 3, the scratch bench at /tmp/w5_jm_research is a perf probe only, not an oracle).

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after with parity harness first (each kernel todo ships its parity rows in the same todo; parity must fail before the port passes) + ctest (ds4_parity / ds4_blas_parity) and manual parity binaries.
- Evidence: <attemptDir>/task-<N>-dsv41-trellis-sg16-perf.log, attemptDir = result.currentAttemptDir from await agentToolkit.status() in a JS eval cell (import via await import(`${env("OMO_AGENT_TOOLKIT_SDK_ROOT")}/sdk.js`)); build rc, ctest output, bench output captured per todo; every strata invocation wrapped in bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; <command>'

## Execution strategy
### Parallel execution waves
> Wave 1: todos 1-2 (probe + parity oracle, independent). Wave 2: todos 3-5 (kernel, dispatch, parity rows; 4 blocked by 3, 5 blocked by 3+4). Wave 3: todos 6-11 (measurements and decode probes; 6 blocked by 5, others independent). Wave 4: todo 12 (docs, blocked by 6-11). Final wave F1-F4 in parallel.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | - | 3 | 2 |
| 2 | - | 3,5 | 1 |
| 3 | 1,2 | 4,5,6 | - |
| 4 | 3 | 5,6 | - |
| 5 | 3,4 | 6 | - |
| 6 | 5 | 12 | 7,8,9,10,11 |
| 7 | - | 12 | 6,8,9,10,11 |
| 8 | - | 12 | 6,7,9,10,11 |
| 9 | - | 12 | 6,7,8,10,11 |
| 10 | - | 12 | 6,7,8,9,11 |
| 11 | - | 12 | 6,7,8,9,10 |
| 12 | 6-11,13 | F wave | - |
| 13 | - | 12 | 6,7,8,9,10,11 |

## Todos
- [x] 1. B60 microprobe: global ext_intel_packed B load
  Recommended task executor category: deep-low
  What to do / Must NOT do: minimal sg16 kernel loading B via sycl::address_space_cast<global_space,decorated::no> with layout Bpack[(k/2)*Mpad*2+m*2+(k&1), ld=Mpad*2 halfs, Mpad=round_up(M,16); prove bijection on B60 before the real kernel. Must NOT build the full kernel here.
  Closes: GAP-4
  Parallelization: Wave 1 | Blocked by: none | Blocks: 3
  References: current local pack convention ds4_iq.dp.cpp:1144-1149 and local load :1167-1174; scratch perf probe /tmp/w5_jm_research/xmx/xmx_int8_bench.cpp:63-69 (col_major - NOT the layout to copy)
  Acceptance criteria: icpx compile rc=0 and device run rc=0 printing MATCH for K=32,128,640,2560 addressing
  QA scenarios: happy (probe prints MATCH all shapes), failure (deliberate wrong ld must print NO MATCH); Evidence <attemptDir>/task-1-dsv41-trellis-sg16-perf.log
  Commit: Y | test(sycl): global ext_intel_packed B microprobe
- [x] 2. Decode-stage parity oracle for canonical blocks
  Recommended task executor category: deep-low
  What to do / Must NOT do: extend ds4_parity.cpp with a bit-exact oracle: for each supported block type (Q8_0, IQ4_NL, MXFP4) decode 32 values from canonical row-major blocks (bid=n*bpr+kb) to F16 RNE and compare against the shipped CPU scalar dequantizer; add a negative control flipping one payload byte or scale. Must NOT touch tile accumulation.
  Closes: GAP-4
  Parallelization: Wave 1 | Blocked by: none | Blocks: 3,5
  References: canonical block rule ds4_iq.dp.cpp:67-70,442-445; q8_block ds4_gguf.dp.cpp:283-295; IQ4_NL decode ds4_iq.dp.cpp:314-323; MXFP4 decode ds4_gguf.dp.cpp:227-235; existing parity row style ds4_parity.cpp:1155-1256 (non-vacuity trap comment :1156-1160)
  Acceptance criteria: ctest ds4_parity passes with new rows; negative-control run exits 1
  QA scenarios: happy (bit-exact all block types), failure (byte flip must fail); Evidence <attemptDir>/task-2-dsv41-trellis-sg16-perf.log
  Commit: Y | test(sycl): bit-exact A-block decode oracle
- [x] 3. i8 A-stage tile kernel set_tc_stageA — G1-G7 MATCH on JIT+AOT, negative control RC 1 both flavors; merged onto dsv41-trellis-int as fb4855c4 (verifier CONFIRMED)
  Recommended task executor category: deep-low
  What to do / Must NOT do: additive kernel family in ds4_iq.dp.cpp: orientation A=weights Aq[N,K] decoded per-thread into As[s][r][2j] F16 fragments (reuse deq8/decode_b verbatim), B=packed F16 activations via todo-1 global layout, joint_matrix 16x16x16 sg16 mad, KSPLIT=4 sub-groups with nblk=K/32, kb_begin=(s*nblk)/4, kb_end=((s+1)*nblk)/4 (disjoint whole-block ranges; for the tested model n_embd=2560 and n_ff=640 give 80 and 20 blocks, i.e. 20 and 5 per split), Part[4][16][20] row_major store with ld=QBN+4=20, deterministic ((p0+p1)+p2)+p3 reduction, epilogue transposes C[N,M] to Y[M,N]. Shared-memory block must cover max(Tiles, Out) plus Part planes: add static_asserts modeled on ds4_iq.dp.cpp:1063-1068 and update launch constants at :1038-1041/:1753-1754 in this same todo. reqd_sub_group_size(16) on the functor's operator() following set_tc_wrapper (:1241-1256). Gates: K%32==0 else return false, M=1 bypass, K/32<4 fallback, scalar stand-in when STRATA_XMX_SG16 undefined. Must NOT use atomics, Must NOT split per-expert launches, Must NOT re-stage weights through f16 shared, Must NOT depend on /tmp/w5_jm_research/xmx/xmx_int8_bench.cpp (perf probe only, col_major B - not the layout or oracle).
  Closes: GAP-1, GAP-2
  Parallelization: Wave 2 | Blocked by: 1,2 | Blocks: 4,5,6
  References: seams to reuse - ext_intel_packed pack ds4_iq.dp.cpp:1167-1174, wrapper/sg16 pattern :1241-1256, scalar stand-in style :1126-1137, BN+4 store convention :1211-1212, smem static_assert coupling :1063-1068 + constants :1038-1041 + launch accessor :1753-1754, A stride 32+8 :1036; contract from draft F-m
  Acceptance criteria: ninja target strata_dsv41 builds rc=0 with STRATA_SYCL_AOT=bmg and rc=0 without STRATA_XMX_SG16 (stand-in compiles)
  QA scenarios: happy (AOT build+JIT build both rc=0), failure (drifted smem size must trip the static_assert); Evidence <attemptDir>/task-3-dsv41-trellis-sg16-perf.log
  Commit: Y | feat(sycl): i8 A-stage sg16 tile kernel
- [x] 4. Dispatch arm behind kill-switch and eligibility mask — commit def4288b, QA three modes RC=0, verifier CONFIRMED; gap: mask holds IQ4_NL only until todo 5 registers stageA parity rows
  Recommended task executor category: deep-low
  What to do / Must NOT do: route eligible types to set_tc_stageA in iq::set_gate_up / iq::set_down (ds4_iq.dp.cpp:2265/:2298) behind an env kill-switch (STRATA_DS4_NO_STAGEA, mirroring tc_ok()/light_on() at :1533/:1864) plus a per-type eligibility mask; a type enters only when its parity row exists (precedent: the kIQ4_NL refusal at :2267). The trellis branch is_tq() (:2266/:2282/:2297-2307) stays on tq:: untouched. Keep light kernels for experts <=kLightRows=16 (:1316); threshold unchanged this plan. Must NOT touch the dense gemv sites in ds4_model.dp.cpp (:8936/:11305/:11315/:11412) and must not change SetArgs or the ds4_gguf.dp.cpp set_gate_up (:2173)/set_down (:2236) contract.
  Closes: GAP-3
  Parallelization: Wave 2 | Blocked by: 3 | Blocks: 5,6
  References: dispatch sites iq::set_gate_up ds4_iq.dp.cpp:2265, iq::set_down :2298, skip_le :2271-2272/:2288-2289, divisibility guards :2270-2271/:2290, call chain ds4_model.dp.cpp:4231 -> gg::set_gate_up ds4_gguf.dp.cpp:2173 -> iq::set_gate_up ds4_iq.dp.cpp:2265
  Acceptance criteria: with flag unset, existing ctest suite output byte-identical to baseline run captured before todo 3
  QA scenarios: happy (flag off => legacy kernel selected), failure (flag on with ineligible type must fall back, logged); Evidence <attemptDir>/task-4-dsv41-trellis-sg16-perf.log
  Commit: Y | feat(sycl): stageA dispatch arm, default off
- [x] 5. Output parity rows + ctest registration — VERDICT PASS after repair (commit 9feee6b5: StageABuf depends_on event chain fixes out-of-order-queue race; ctest 4/4 RC=0, two runs byte-identical MATCH, --neg-stageA RC=1 real catch; gap: light-set int8 scratch has same theoretical exposure on out-of-order caller queues, left as-is)
  Recommended task executor category: deep-low
  What to do / Must NOT do: tile-path output parity vs an independent scalar reference built inside ds4_parity.cpp (pattern ref_iq_set_gate_up :354; do not edit src/dsv41/ds4_tq_cpu.cpp), from the exact F16 operands staged and the same kb ranges, reduced in fixed order; bound abs_err <= 32*2^-24*sum_abs + 4*ulp_f32(ref); gate/up F16 out within one BF16 ulp (ds4_parity.cpp:976-983); copy the non-vacuity pattern of :1149-1267 (scale-sensitive data, corrupt-one-scale-byte negative control :1186-1191, max_rows=32 > kLightRows=16); shapes M=2,15,16,17,31,32,33,160 with K and N tails; register xmx_group_parity and xmx_gemm_bench as ctest names after sycl/CMakeLists.txt:477-480.
  Closes: GAP-4
  Parallelization: Wave 2 | Blocked by: 3,4 | Blocks: 6
  References: scalar ref style ds4_parity.cpp:354; tolerance rule ds4_blas_parity.cpp:597 one-BF16-ulp; xmx targets sycl/CMakeLists.txt:477-480 (no add_test today), tolerances xmx_gemm_bench.cpp:89-91 (1% rel, 0.1% element budget), xmx_group_parity.cpp:115; B60 device lock ds4_blas_parity.cpp:29
  Acceptance criteria: one ctest run (after source /opt/intel/oneapi/setvars.sh) exits 0 with stdout containing MATCH: for ds4_parity, ds4_blas_parity, xmx_group_parity, xmx_gemm_bench; the negative-control invocation exits 1 with stdout containing NO MATCH - assert BOTH directions
  QA scenarios: happy (all parity rc=0), failure (one payload byte flip must make output parity rc=1); Evidence <attemptDir>/task-5-dsv41-trellis-sg16-perf.log
  Commit: Y | test(sycl): stageA output parity + ctest registration
- [~] 6. B60 measurement: tile vs legacy at M=160 — gap: stageA AOT 15-31% slower at M=160 (gate/up 311-360 vs 238-280 us) though faster at small M (M=2 145 vs 235, M=32 72 vs 171); default stays OFF, kernel-level measurement only (dispatch arm not landed); verifier CONFIRMED
  Recommended task executor category: unspecified-low
  What to do / Must NOT do: run the parity/bench binaries with flag off and on on both B60 cards; record gate/up/down us per shape and full decode tok/s; adopt the new path as default-on ONLY if measured faster AND parity-clean; otherwise keep default off and report the numbers. Baselines to compare against (measured on the B60 pair, this checkout): M=160 gate+up 69.5 us f16 / 34.0 us i8 / 52.0 us i8-block, down 34.2 / 17.7 / 26.8 us; M=1 floor gate+up 22.3 / 11.1 us, down 11.9 / 5.9 us; decode 1.65 tok/s one-card, 12.08 two-card. Must NOT state a throughput number for a prompt that did not complete.
  Closes: GAP-1 (measured outcome IS-1)
  Parallelization: Wave 3 | Blocked by: 5 | Blocks: 12
  References: baseline numbers draft F-f; bench invocation xmx_gemm_bench [ty=18] [M=96] [mode=gu|down] [reps=50]
  Acceptance criteria: captured run output with rc=0 lines for both flag states in the evidence log
  QA scenarios: happy (flag-on faster), failure (flag-on slower => default stays off, log shows both); Evidence <attemptDir>/task-6-dsv41-trellis-sg16-perf.log
  Commit: Y | bench(sycl): stageA measured vs legacy
- [x] 7. Decode probe D-x1 VRAM residency — resident 1.55 vs non-resident 0.84 tok/s (-46%, reproduced x2); verifier CONFIRMED; gap: measured at engine surface (strata-dsv41 --ids), not strata launch --serve (hw-scan gap on this rig)
  Recommended task executor category: unspecified-low
  What to do / Must NOT do: measure tok/s delta when hot routed experts stay resident across chunks (stream_expert ds4_model.dp.cpp:3495, store accounting :5299-5321); report measured delta, no kernel change.
  Closes: GAP-5
  Parallelization: Wave 3 | Blocked by: none | Blocks: 12
  References: ring staging ds4_model.dp.cpp:4155-4165
  Acceptance criteria: two completed serve runs (resident vs non-resident) with tok/s lines captured, rc=0
  QA scenarios: happy (delta measured), failure (eviction thrash must show, not be hidden); Evidence <attemptDir>/task-7-dsv41-trellis-sg16-perf.log
  Commit: Y | bench(sycl): residency decode probe
- [x] 8. Decode probe D-x2 two-card sharding — layer split CUDA0 [0,20] / CUDA1 [21,39]+head verified; completed runs 1.41 vs 4.10 tok/s at --experts 5 (verifier CONFIRMED)
  Recommended task executor category: unspecified-low
  What to do / Must NOT do: verify --parallel/--gpu actually splits expert sets for the SYCL engine; capture tok/s for one vs two cards from completed prompts.
  Closes: GAP-5
  Parallelization: Wave 3 | Blocked by: none | Blocks: 12
  References: prior measured 1.65/12.08 tok/s (draft F-f)
  Acceptance criteria: completed runs on both cards, rc=0, sharding visible in the engine log
  QA scenarios: happy (two-card faster), failure (silent single-card fallback must be reported); Evidence <attemptDir>/task-8-dsv41-trellis-sg16-perf.log
  Commit: Y | bench(sycl): two-card decode probe
- [~] 9. Decode probe D-x3 window crossover — gap: light path statically ineligible for dsv41 (guard n_ff % 256 == 0; n_ff=640 -> remainder 128), no crossover possible; tile us recorded both flavors, nothing adopted (verifier CONFIRMED)
  Recommended task executor category: unspecified-low
  What to do / Must NOT do: measure tile-vs-light at T=8 and T=16 for win_experts (ds4_gguf.dp.cpp:2076); adopt only if faster and parity-clean; kLightRows change is a separate decision, not this todo.
  Closes: GAP-5
  Parallelization: Wave 3 | Blocked by: none | Blocks: 12
  References: kLightRows=16 ds4_iq.dp.cpp:1316-1317, light path :1321,:1473
  Acceptance criteria: measured us per T with rc=0 lines; window-verification bits unchanged
  QA scenarios: happy (crossover found), failure (no gain => report, no adoption); Evidence <attemptDir>/task-9-dsv41-trellis-sg16-perf.log
  Commit: Y | bench(sycl): window crossover probe
- [~] 10. Decode probe D-x4 gate gemv i8 — gap: bit-identical impossible for per-row int8 of f32 gate weights (0/256, max abs diff 0.942); timing noise-level; not adopted (verifier CONFIRMED)
  Recommended task executor category: unspecified-low
  What to do / Must NOT do: probe an i8 gemv variant for the f32 router gate (ds4_model.dp.cpp:8936, 22.3 us at M=1); only worth it if bytes read drop; measure, no adoption without measurement.
  Closes: GAP-5
  Parallelization: Wave 3 | Blocked by: none | Blocks: 12
  References: gemv_f32 sites ds4_model.dp.cpp:8936,11305,11315,11412
  Acceptance criteria: measured us for both variants, rc=0, answers bit-identical
  QA scenarios: happy (i8 faster), failure (no gain or bit drift => report only); Evidence <attemptDir>/task-10-dsv41-trellis-sg16-perf.log
  Commit: Y | bench(sycl): gate gemv probe
- [~] 11. Decode probe D-x5 launch fusion — gap: gate+up already fused in shipped code (no delta to adopt); end-to-end decode tok/s not measured (verifier CONFIRMED)
  Recommended task executor category: unspecified-low
  What to do / Must NOT do: count per-token launches (q8_quant, gate, up, down, MoE routing) and measure fusing gate+up into one grouped launch; adopt only if measured faster.
  Closes: GAP-5
  Parallelization: Wave 3 | Blocked by: none | Blocks: 12
  References: entry points ds4_gguf.dp.cpp:1991,2031,2076; q8_quant :280
  Acceptance criteria: launch counts captured plus measured tok/s delta, rc=0
  QA scenarios: happy (fusion faster), failure (no gain => report, no adoption); Evidence <attemptDir>/task-11-dsv41-trellis-sg16-perf.log
  Commit: Y | bench(sycl): launch fusion probe
- [x] 13. Trellis A-stage feasibility probe (measurement only) — joint_matrix A-stage tile 2.0-2.3x faster than portable-FMA tq:: lane on B60, contract note delivered, nothing adopted (verifier CONFIRMED)
  Recommended task executor category: deep-low
  What to do / Must NOT do: measure whether a joint_matrix A-stage tile can serve the trellis types (TQK6/TQK7/TQK5X/TQK4X/TQK5T/TQK4T, ds4_iq.dp.hpp:30), whose blocks are 128-value {fp16 d; 4K stream bytes} decoded through 15-bit state windows (ds4_tq.dp.cpp:46-56/:63-80) - a different block shape than the 32-value contract. Deliver a measured us comparison of the current portable-FMA tq:: lane against a probe tile, plus a written contract note; adoption returns to the approval gate. Must NOT edit src/dsv41/ds4_tq_cpu.cpp, Must NOT change the is_tq dispatch.
  Closes: GAP-5
  Parallelization: Wave 3 | Blocked by: none | Blocks: 12
  References: tq lane ds4_tq.dp.cpp:770-792 (no joint_matrix in that file), WMMA HIP-only :23-25, is_tq routing ds4_iq.dp.cpp:2266/:2282/:2297-2307, CPU reference src/dsv41/ds4_tq_cpu.cpp (linked at sycl/CMakeLists.txt:302)
  Acceptance criteria: probe build rc=0 and completed runs with us lines for both lanes captured; decision note states measured winner or "no adoption"
  QA scenarios: happy (probe faster, note says returns to approval gate), failure (no gain => report only); Evidence <attemptDir>/task-13-dsv41-trellis-sg16-perf.log
  Commit: Y | bench(sycl): trellis A-stage feasibility probe
- [x] 12. Docs measured-report update — commit 8b089558, 146-line docs/DETAILS.md section, 125 numbers mapped to evidence lines, verifier CONFIRMED
  Recommended task executor category: writing
  What to do / Must NOT do: update docs/DETAILS.md with every measured number from todos 6-11, tagged with what it was measured on (B60 pair, build commit); mark unconfirmed static analysis separately. Must NOT publish a number not captured in an evidence log.
  Closes: IS-5
  Parallelization: Wave 4 | Blocked by: 6,7,8,9,10,11 | Blocks: final wave
  References: docs discipline AGENTS.md (measured numbers with hardware); evidence logs under <attemptDir>
  Acceptance criteria: read the docs diff against the measured-numbers rule (AGENTS.md: no claim without a measurement) - every number in the diff maps to a captured evidence-log line, and no number appears that has no evidence line
  QA scenarios: happy (docs cites evidence), failure (unmeasured claim must be absent); Evidence <attemptDir>/task-12-dsv41-trellis-sg16-perf.log
  Commit: Y | docs(sycl): measured stageA and decode numbers

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [x] F1. Plan compliance audit — APPROVED (st_01a18134): 13 DoneClaims each with evidence log VERDICT + ledger verifier CONFIRMED; w2 chain merged onto int (HEAD 849a7a5a, ancestry RC=0); nothing pushed (origin ref absent RC=128); all titles area-prefixed; ~/llama.cpp zero tracked changes
- [x] F2. Code quality review — APPROVED (st_01a18135): kill-switch honored first, eligibility IQ4_NL-only, event-chain matches route_publish, no swallowed errors/debug leftovers, icpx syntax RC=0 with build include flags - ds4_iq.dp.cpp additions: no smem drift vs static_asserts (:1063-1068), scalar stand-in present, no atomics
- [~] F3. Real manual QA — APPROVED_WITH_GAP (st_01a18136 + adjudication st_01a1817c): AOT bmg-g21 strata-dsv41 build RC=0 on aa09dc07 and HEAD 849a7a5a (strata launch RC=5, plan-layer card gap); serve HTTP 200, temp-0 exact tokens "The capital of France is Paris.", diff flag-off HEAD vs baseline RC=0 and flag-ON vs baseline RC=0 (proves IS-3 at the real surface); teardown RC=0, GPU idle RC=1 — gap: stageA route-log line not produced at the real serve surface — the chunked prompt pass (the only set-path entry) livelocks pre-existing on this rig, identically at aa09dc07 and HEAD (RC=124 at 900 s, server and engine-direct); kill-switch route log + refusal verified instead in todo-4 QA (def4288b qa-kill/qa-xs RC=0); arm cannot engage this pack (IQ1_M/IQ2_XXS/IQ2_XS vs IQ4_NL-only)
- [x] F4. Ideal-state fidelity — APPROVED (st_01a18137): all 13 todos [x]/[~]+gap; VERDICT lines in every log; verifier records for all 13; int HEAD 849a7a5a contains every verified commit (ancestry rc=0), nothing pushed (origin absent RC=128); docs numbers trace to completed runs; scratch gone RC=2 - IS-1..IS-5 checked 1:1 against measured evidence lines

Commits per todo on branch sycl/dsv41-sycl-support (user decision D4), git-master atomic discipline; each commit uses an explicit git add allowlist naming only that todo's changed files - evidence under .omo/ and untracked bench/results stay uncommitted; F-wave findings become fix commits, never amend of a shipped todo.

## Success criteria
> One row per IS row. The plan is complete only when every IS row has a delivering todo and a proving QA scenario; F4 checks the delivered behavior against these rows 1:1, and a shortfall becomes new `- [ ] N.` rows, never a note.
| IS | Delivering todo(s) | Proving QA scenario | Evidence |
| --- | --- | --- | --- |
| IS-1 | 3,4,6 | flag-on faster than legacy, measured | <attemptDir>/task-6-dsv41-trellis-sg16-perf.log |
| IS-2 | 2,5 | bit-exact operands + bounded output parity + negative control (both directions asserted) | <attemptDir>/task-5-dsv41-trellis-sg16-perf.log |
| IS-3 | 4, F3 | flag-off curl temp-0 exact-output match vs pre-branch baseline | <attemptDir>/task-4-dsv41-trellis-sg16-perf.log |
| IS-4 | 3 | AOT and JIT builds green | <attemptDir>/task-3-dsv41-trellis-sg16-perf.log |
| IS-5 | 6-12 | docs numbers map to evidence lines | <attemptDir>/task-12-dsv41-trellis-sg16-perf.log |
