---
slug: dsv41-trellis-sg16-perf
status: drafting
intent: clear
review_required: true
plan_path: .omo/plans/dsv41-trellis-sg16-perf.md
plan_sha256: null
review_round_id: null
review_round_limit: 5
pending-action: write and review .omo/plans/dsv41-trellis-sg16-perf.md
review:
  plan_reviewer:
    status: pending
    workspace_root: null
    runtime_home: null
    target: .omo/plans/dsv41-trellis-sg16-perf.md
    round_id: null
    plan_sha256: null
    launch_id: null
    session: null
    result: null
approach: Option B first (architect lane): new i8-activation A-stage tile family in ds4_iq.dp.cpp reusing ext_intel_packed pack + sg16 pattern, exact integer dots + per-32 f32 scales, KSPLIT reduction, env kill-switch + per-type eligibility mask, scalar stand-in for non-AOT; Option A f16 sibling second; SoA repack (C) shelved until measured
---

# Draft: dsv41-trellis-sg16-perf

## Affected user and ideal state
Users: (1) the person serving dsv41 on 2x Intel Arc Pro B60 (strata launch, TUI + API); (2) apps/agents consuming the OpenAI/Anthropic-compatible API; (3) maintainers reviewing the fork branch sycl/dsv41-sycl-support.
IS-1 | prompt throughput rises for batched expert FFN matmuls, decode throughput unchanged | measured floor says M=1 is memory-bound, tiles cannot beat gemv there
IS-2 | answers identical to today for every prompt, tile path parity-checked vs scalar dequant | quant-block decode inside the tile must reproduce dequant.hpp results
IS-3 | default path byte-identical; the tile path is opt-in | CONTRIBUTING.md rule: new feature opt-in, default path byte-identical to last release
IS-4 | build stays green on both B60s with the flag on and off | IGC/runtime version gating is real (libsycl .so.8 fails, .so.9 passes)
IS-5 | docs claims tagged measured/static, no unmeasured throughput numbers | repo docs discipline (AGENTS.md, measured-report commits)
GAP-1 | batched expert matmuls dequantize whole blocks to float registers (mx_block/q8_block ds4_gguf.dp.cpp:227/284) then multiply scalar; no tile A stage consumes packed blocks, no SoA reorder layout (grep reorder_qw/SoA: 0 hits in ds4_gguf/ds4_iq) | tile path would load blocks directly | todo: i8 A-stage tile family (Option B)
GAP-2 | no split-K (KSPLIT) in repo kernels; FG_KSPLIT=4 exists only in the scratch bench harness | repo grep shows KSPLIT only in /tmp/w5_jm_research/xmx/xmx_int8_bench.cpp | todo: KSPLIT reduction in tile kernel
GAP-3 | dispatch does not route batched expert matmuls to the sg16 tile kernels with block decode | ds4_model.dp.cpp gemv call sites at :8936/:11198/:11305/:11315/:11412; tile kernels gated by compile-time STRATA_XMX_SG16 | todo: dispatch switch behind flag
GAP-4 | no parity harness for the new A stage against scalar reference | existing parity targets cover current kernels only | todo: parity bench extension

## Components (topology ledger)
C1 SoA weight repack | shelved until decode cost isolated after A/B measured | deferred | architect lane F-l
C2 i8-activation A-stage tile family (Option B) | int DPAS tiles, per-32 f32 scale epilogue, KSPLIT reduction | active | ds4_iq.dp.cpp seams :1164-1188,:1245-1264
C3 KSPLIT=4 split-K accumulation + reduction | partial-sum correctness, no atomics race | active | scratch /tmp/w5_jm_research/xmx/xmx_int8_bench.cpp contract
C4 dispatch switch for M>1 behind opt-in flag | decode M=1 stays gemv byte-identical | active | ds4_model.dp.cpp call sites
C6 decode-path exploration (opt-in probes, measurement-gated) | raise tok/s without changing answer bits | active | research ledger M=1 floor numbers, ds4_model:8936/3495/5299-5321
C5 parity + bench verification on B60 | bit-exact/tolerance vs scalar dequant, M=1 and M=160 | active | sycl/CMakeLists.txt parity targets, xmx bench harness

## Open assumptions (announced defaults)
A1 opt-in flag | STRATA_XMX_SG16 stays compile-time ON for SYCL builds; new A-stage path gated by a separate runtime env or cmake option, default OFF | repo evidence: CONTRIBUTING.md opt-in rule; reversible yes
A2 SoA repack in-memory at load | model files unchanged, repack cached in-process | persisted layout change would touch distribution - owner-decision avoided by default; reversible yes
A3 KSPLIT=4 adopted | matches the measured bench harness config | reversible yes
A4 test strategy tests-after with parity harness first | kernel parity must fail before the port passes; agent-executed QA always included | reversible yes

## Findings (cited - path:lines)
F-a sycl/CMakeLists.txt:41 add_compile_definitions(STRATA_XMX_SG16=1); :43 STRATA_NO_DG2_XMX=1; :50 STRATA_SYCL_I8_ESIMD AUTO/ON/OFF
F-b ds4_iq.dp.cpp:1167-1174 joint_matrix a(row_major)/b(ext_intel_packed) load+mad, :1211-1212 store BN+4 row_major
F-c ds4_gguf.dp.cpp:1120 mx_set_tc_kernel, :1475-1480 STRATA_XMX_SG16 dispatch
F-d ds4_model.dp.cpp gemv_f32 gate routing :8936/:11305/:11315/:11412, gemv_f16_rows i_wproj :11198
F-e KSPLIT appears only in the scratch bench, not in sycl/src (grep rc shown this session)
F-f measured kernel numbers (ulw-research ledger): M=160 gate+up 69.5/34.0/52.0 us f16/i8/i8-block, down 34.2/17.7/26.8; M=1 22.3/11.1 memory floor; decode 1.65/12.08 tok/s one/two card
F-k dispatch map (lane st_01a17c47, verified): entry points moe_gate_up/moe_down/win_experts (M=1..16, ds4_gguf:1991/2031/2076) and set_gate_up/set_down (M>>1, ds4_gguf:2169/2237, called from ds4_model:4231). mx_experts() per-layer split (ds4_gguf.hpp:44). M=1: q8_quant pre-quantizes activations (:280), weights decoded inline per 32 (mx_block :207 / decode_fast_b ds4_iq:753). M>1: light kernels <=16 rows (kLightRows=16 ds4_iq:1316), remainder via tc_ok() (:1433, runtime major>=7) to *_set_tc_kernel else scalar *_set_kernel. Today's tc path decodes weights into F16 SHARED per k-step (ds4_gguf:1195-1205, deq8 ds4_iq:1050) then joint_matrix 8x16x16 sg16 - no packed-block A-stage, no SoA, no KSPLIT. STRATA_XMX_SG16 compile-time only (CMake:41, bmg/lnl AOT); STRATA_SYCL_I8_ESIMD gates separate prefill/esimd_moe chain (esimd_moe.hpp:43 choose_mode) outside ds4 dispatch.
F-m ultrabrain contract (lane st_01a17c4b): orientation A=weights Aq[N,K], B=packed F16 activations B[K,M], C[N,M], epilogue transposes to Y[M,N]; 16x16x16 F16 sg16 tile, WG=4 sg=64 items; A reads canonical blocks bid=n*bpr+kb (SoA optional later); Bpack[(k/2)*Mpad*2+m*2+(k&1), Mpad=round_up(M,16); global ext_intel_packed load via address_space_cast needs a B60 microprobe first; KSPLIT=4 block-aligned kb_begin/end, Part[4][16][20] row_major store, deterministic ((p0+p1)+p2)+p3, atomics banned; gates: K%32==0, M=1 bypass, M tail zero-fill+mask, K/32<4 fallback, reqd_sub_group_size(16); parity: bit-exact decoded A/B operands + bounded output err + negative control + shapes M=2..160,K tails,N tails; legacy M=1 byte-identical when flag off.
F-l architect advisory (lane st_01a17c4a): today's set_tc_kernel already has grouped launch + dequant-in-kernel; gap is smem round trip deq8->f16 wg/wu (ds4_iq:1123-1133) -> ext_intel_packed pw/pu (:1164-1187) -> jm 16x16x16 sg16 (:1102). Option B (i8 activations per-32 f32 scales, int DPAS int32 acc, epilogue scale) matches measured i8 34us vs f16 69.5us and reuses light path's exact-dot contract (light_q8_kernel :1321, sub_dot :307). SoA has no hook: weights re-stream per chunk (ds4_model:4155-4165), C2 = second VRAM copy, C1/C3 non-starters. Risks: numeric contract drift (ds4_iq.hpp:8-12), AOT-only SG16 (CMake:40-41) needs scalar stand-in (:1126-1137), smem static_asserts (:1063-1068), kLightRows threshold changes bits, grouped single-grid launch (:2275-2276) must survive, divisibility guards (:2270-2271,:2290). Scope: ds4_iq + ds4_parity only; SetArgs frozen; decode M=1 stays gemv.
F-j expert FFN call chain (architect lane, running): ds4_model.dp.cpp:4231 -> ds4_gguf.dp.cpp:2173 -> ds4_iq.dp.cpp:2265; sg16 tiles ds4_iq.dp.cpp:1102-1212
F-i QA wiring: ds4_blas_parity ctest --selftest, one-BF16-ulp rule (bad==0 pass, :597), device lock to Arc Pro B60 (:29), negative controls --corrupt*; xmx_gemm_bench manual, 1% rel tol, 0.1% elements may fail (:89-91), no add_test; xmx_group_parity manual, max|err|<=0.2%*max|ref| strict finite (:115), 6 format cases {18,21,22,23}/{20,42}, no add_test; ds4_parity BF16-bit-exact-or-one-ulp. tests/terminal has no SYCL coverage.
F-h PR #29245 contract (snapshots S4/S10, commit d81d0d8f): SoA streams fg_soa<T> offsets S4:515-567 (Q8_0 [qs][d]; Q4_K [qs][scales][dm]; Q5_K [qs][qh][scales][dm]; Q6_K [ql][qh][scales][d]); A stage one lane per row decodes FG_BK=32 values per k-step, scales folded into registers (S4:514); FG_SUPERBLOCK kb/(QK_K/32) sub-block indexing (S4:577-582); FG_KSPLIT=4 sub-groups walk disjoint K ranges (kb_begin/end S4:880-890), partials in SLM tile_c[FG_KSPLIT*SG_ROWS*BN] reduced then written (S4:945-957); WG_SIZE=FG_KSPLIT*SG=64; tile combo f16 8x16x16 sg16 (S4:220-229 case 0: SG_ROWS=16, RPL=1, MT=2, BN=32, NT=2); FG_MAX_N=64 (S10:7)
F-g dirty_worktree risk: untracked .omo/, .senpi/, bench/results/* - keep out of scope

## Decisions (with rationale)
D1 intent clear: outcome known from completed ulw-research (SYNTHESIS.md implementation plan 1-4); review_required true (default-on)
D2 stance one-by-one (cold start, CLEAR)
D3 work location: repo branch sycl/dsv41-sycl-support directly (user answer 2026-10-27, fork Location)
D4 commit per todo on the branch (user answer 2026-10-27, fork Commits)
D5 approval: user answered 'Approve' 2026-10-27 (Option B tile port + decode probes D-x1..D-x5)

## Scope IN
sycl/src/dsv41 (ds4_gguf, ds4_iq, ds4_gemv dispatch via ds4_model), sycl/CMakeLists.txt parity/bench targets, docs measured-report update; branch sycl/dsv41-sycl-support.

## Scope OUT (Must NOT have)
CUDA-side changes (src/dsv41), persisted model-format changes (C1/C3), decode-path changes (M=1 stays gemv/mv), oneDNN/oneMKL integration, dense chunked GEMM seam, changes to untracked bench/results or .omo/.senpi artifacts.

## Decode exploration (new, measurement-gated probes)
D-x1 VRAM residency: keep hot routed experts resident across chunks (stream_expert ds4_model:3495, store accounting :5299-5321) so decode stops re-streaming weights per token; measure tok/s delta one-card.
D-x2 two-card decode: verify --parallel/--gpu sharding actually splits expert sets for dsv41 SYCL; 12.08 vs 1.65 tok/s already measured - plan a config probe, not a kernel change.
D-x3 window path crossover: win_experts batches T<=16 (ds4_gguf:2076); measure tile-vs-light crossover at T=8/16 on B60; only adopt if measured faster and parity-clean.
D-x4 gate gemv: gemv_f32 router gate at ds4_model:8936 costs 22.3us at M=1; probe i8 gemv variant for gate weights (bandwidth win only if bytes drop).
D-x5 launch overhead: count per-token launches (q8_quant + gate + up + down + MoE routing) and probe fusing gate+up into one grouped launch; measure before adopting.
All decode items stay byte-identical on answers; none move M=1 onto tiles (ultrabrain gate: M=1 bypass).

## Open questions
Q-owner-1 work location: implement directly on /home/maxious/Strata branch vs scratch checkout (standing agreement scoped to QA work says scratch; implementation location is the user's call).
Q-owner-2 commit policy during execution: commits per todo vs no commits until user asks (boundary: never commit unless requested).

## Approval gate
status: awaiting-approval
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->

## Review receipts
- R1 plan-consultant (st_01a17cc0, completed): 9 findings applied - dispatch seam retargeted to iq::set_gate_up/is_tq chain (not dense gemv sites); IS-2 split into bit-exact operands + one-BF16-ulp output bound with byte-identical guaranteed flag-OFF only; reqd_sub_group_size(16) on functor operator() (set_tc_wrapper :1241-1256); filenames ds4_iq.dp.hpp/ds4_gguf.dp.hpp; symbol-name citations for moe_gate_up/moe_down/win_experts; KSPLIT contract inlined (no /tmp dependency); parity rows copy ds4_parity.cpp:1149-1267 non-vacuity + negative control; trellis types (128-value blocks, no joint_matrix in ds4_tq.dp.cpp) added as measurement-only todo 13; SLM Part planes covered by new static_asserts + launch constants; effort raised XL.
- R2 plan-reviewer: SKIPPED - user Stop held task admission twice (spawn aborted, then 'Task admission is held by your Stop'). Structural self-check passed instead: 13 todo rows, 13 executor lines, 4 F rows, 0 template placeholders, all GAP rows closed, every IS row mapped to a delivering todo. Reviewer round can be re-run on request.
- Gate: approved by user 'Approve' 2026-10-27; execution is /ulw-execute by the user only.
