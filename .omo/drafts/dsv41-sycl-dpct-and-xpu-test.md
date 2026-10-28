---
slug: dsv41-sycl-dpct-and-xpu-test
status: approved
intent: clear
review_required: true
plan_path: .omo/plans/dsv41-sycl-dpct-and-xpu-test.md
plan_sha256: 17e8e2759e8fe85f6182328edb588f8a1e73c9b00a53b6702ecffead1ccb454f
review_round_id: round-1
review_round_limit: 5
pending-action: record the round-1 reviewer verdict in review.plan_reviewer.result
approval:
  source: the user's ulw-execute invocation (no-plan bootstrap - the phrase itself is the approval)
  authorized: writing this plan only; execution is a separate ulw-execute phase
  note: the earlier Approval question returned "Unanswered"; the ulw-execute invocation is the approval receipt
review:
  plan_reviewer:
    status: pending
    workspace_root: null
    runtime_home: null
    target: .omo/plans/dsv41-sycl-dpct-and-xpu-test.md
    round_id: round-1
    plan_sha256: 17e8e2759e8fe85f6182328edb588f8a1e73c9b00a53b6702ecffead1ccb454f
    launch_id: st_01a17718
    session: st_01a17718
    target_note: plan-reviewer is plan-gated and refused in this session because ulw-execute was already
      invoked, so the high-accuracy round runs as omo-native-gate-reviewer on the same contract
    result: VERDICT PASS - no blocking issues, notes only; structural 11 impl rows / 5 F rows / 11 category
      lines / 0 indented rows, coverage of all eight areas, decision completeness (what to do, what must not,
      file:line refs, un-piped *_RC= observables, QA scenario, evidence path, no TBD), constraint fidelity
      (throughput only from a completed prompt, one card first, no product-code edit claimed, no .omo/ or
      bench/results commit, default byte-identical and Intel opt-in), and mechanism risk (sycl::atomic_ref on
      host USM forbidden, every spin bounded, no blanket DPCT1065 barrier conversion). Two advisory notes were
      resolved by plan edits (todo 3 control must be run and measured; todo 8 malformed ids file must be
      genuinely malformed). Report: .omo/evidence/dsv41-sycl-dpct-and-xpu-test-gate-review.md
    verdict: PASS
    round: 1 of limit 5
approach: rectify the DPCT comment TODOs that are real defects in the SYCL dsv41 port (one commit per
  class, each proven by a run), fix the two correctness gaps that block an honest quality report (host->GPU
  doorbell delivery, CPU-lane expert routing), then run the pack on one B60 and on both B60s with the encoded
  ids for "what is the capital of france" and report tok/s plus the decoded answer.
---

# Draft: dsv41-sycl-dpct-and-xpu-test

## Affected user and ideal state
- The person with 2x Arc Pro B60: runs `strata launch` / the app on Intel cards and wants a real answer, not a hang.
- The agent that drives `--serve` (serve/server.py's StrataEngine): speaks INFO/READY/GEN/T/DONE and needs clean exit.
- A maintainer reading the port: needs DPCT notes either fixed or explained, not left as silent warnings.
- IS-1 | a prompt on one card returns a correct answer and the process exits 0 | the run must be usable, not wedged
- IS-2 | measured tok/s is reported only from a completed prompt, with the card and flags named | no claim without a measurement
- IS-3 | the CPU-lane experts actually contribute when the pack's VRAM budget needs them | the engine's design is hybrid
- IS-4 | both cards work when asked (`--gpus 0,1`) or the port says plainly that they do not yet | multi-card is a shipped path
- IS-5 | the default (non-SYCL) build and ctest -R ds4 stay green | the default path must stay byte-identical
- IS-6 | every DPCT note in the port is either rectified or recorded as accepted with a reason | a reader can tell which
- GAP-1 | today the answer is eight identical ids 129276, not "Paris" | sampling/head or routing is wrong | todo TBD
- GAP-2 | today CPU lane contributes 0 of 1680 experts | host->GPU store is not observed on this card | todo TBD
- GAP-3 | today the process aborts at exit (RUN_RC=134) unless both statics leak | exit-order frees through a dead queue | fixed in 18a27e02, verify RUN_RC=0
- GAP-4 | today DPCT notes for 1101/1110/1049/1013/1093/1118/1065/1010 are untouched | unreviewed advisories | todo TBD
- GAP-5 | today multi-card paths (all-reduce/xfer flags) are compiled but never exercised | unproven | todo TBD

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->

## Open assumptions (announced defaults)
- card count | one B60 first, both second as a separate increment | stored preference: new code on one card first; multi-card paths are unexercised | reversible
- text prompt | feed ids encoded by the pack tokenizer (the exe has no text flag; `--ids` and `--serve` only) | src/dsv41/ds4_main.cpp:5-12,87-92 | reversible
- DPCT scope | rectify classes that are real defects; document the rest as accepted with a reason | most notes are advisories | reversible
- quality bar | the decoded answer must name Paris, plus a short coherence check on a second prompt | the ask is quality, not only speed | reversible
- throughput | reported only from a completed prompt, with card, flags, and prompt/decode ms quoted | standing agreement | reversible

## Findings (cited - path:lines)
- sycl/CMakeLists.txt:297-306 - STRATA_SYCL_DSV41_EXE builds strata-dsv41 from src/dsv41/ds4_main.cpp,
  ds4_packinfo.cpp, ds4_gguf_cpu.cpp, ds4_iq_cpu.cpp, ds4_tq_cpu.cpp, ds4_gg_cpu.cpp plus the .dp twins.
- src/dsv41/ds4_main.cpp:5-12,87-92 - flags: --pack --gpus --max-context --reserve-mib --vram-experts
  --cpu-threads --expert-profile --ssd-cache --ssd-policy --no-expert-cache --verbose --tp --threads
  --prefill-chunk --dspark --dspark-block --dspark-conf --experts --vision --mmap-experts, and
  (--ids I,J [--images FILE] [--n N] [--dump-logits FILE] | --serve). No text-prompt flag.
- DPCT tally in sycl/src/dsv41 + vendor: 1065 152, 1093 107, 1124 101, 1118 83, 1101 59, 1110 44, 1013 38,
  1078 28, 1010 21, 1024 17, 1049 15, 1048 13, 1009 13, 1119 10, 1026 8, 1017 8, 1001 7, 1106 6, 1000 6,
  1098 5, 1083 5, 1027 5, 1025 5, 1008 4, 1007 2, 1006 2, 1113 1, 1019 1, 1002 1.
- Zero async_memcpy / memcpy_async sites in dsv41, so DPCT1124 has no event to wait on (checked by grep).
- sycl/src/dsv41/ds4_model.dp.cpp:1339-1340 - destroy_event before destroy_queue; :3035,:3053 polled ring
  allocation; :1343,:2854 host_free_polled.
- sycl/src/dsv41/ds4_iq.dp.cpp:577 and ds4_kernels.dp.cpp:1889 - leaked codebook statics (commit 18a27e02).
- run 14: EXE_RC=0, prompt 1 tokens in 62 ms, decode 8 tokens in 385 ms, CPU lane 0 of 1680 experts,
  ids 129276 x8, RUN_RC=134 (before the kernels-side leak reached the build).
- Lane finding (st_01a176b1): the GPU does not observe host stores even with sys_load + zeMemAllocHost
  polled (200k reads, NEVER SAW); GPU->host direction works. ext_oneapi_empty() at ds4_model.dp.cpp:8640
  can never be true, so DS4_FLOW_PUMP_WAIT=0 is a no-op there.

## Decisions (with rationale)
- One commit per DPCT class, each followed by a run, so a regression is attributable.
- Quality test goes through encoded ids plus a decode back to text, because the exe has no text flag.
- Host->GPU delivery is treated as an open defect with candidate mechanisms, not assumed solved by sys_load.

## Scope IN
- DPCT rectification in sycl/src/dsv41 (+ vendor twins) and sycl/tools/fixups.py rules.
- Correctness: host->GPU answer delivery, CPU-lane expert routing, clean exit.
- Runs on one card and both cards; default build + ctest -R ds4 stay green.

## Scope OUT (Must NOT have)
- No throughput claim from a run that did not complete a prompt.
- No new engine features, no default changes, no .omo/ or bench/results commits.
- No ggml_sycl build; the sysman memory query stays mirrored with attribution.

## Owner answers
- Q1 ANSWERED (user verbatim: "yes chase correctness for the ulw-plan"): correctness work is IN scope. The plan must drive
  the model to answer "Paris" for "what is the capital of france", not report the degenerate output as-is.

## Lane findings folded in (verified against the repo)
- st_01a176fd (DPCT classification): every remaining advisory class is ADVISORY, 0 ACTIONABLE. DPCT1000/1001/1009/1010
  error stubs are the deliberate SYCL pattern (23 sites in ds4_model.dp.cpp, 4 in ds4_gguf.dp.cpp); DPCT1013 rounding
  matches CUDA round-to-nearest-even; DPCT1065 full barrier kept because some barriers separate global writes;
  DPCT1078 acq_rel matches __threadfence()/__threadfence_system() intent; DPCT1049/1110/1118 are launch/register/convergence
  notes, not correctness; DPCT1124 sites have no async_memcpy left; DPCT1098/1048 are handled by fixups rules 7d/7e.
  So DPCT rectification = finish the residual sites and drop the stale advisories, not rewrite kernels.
- st_01a176fe (interface): strata-dsv41 has NO text-prompt path. --ids/@file, --serve GEN, --score/--profile-run/--spec-eval
  all take ids; Model exposes feed/step/draft only; zero tokenizer references in src/dsv41. The pack carries
  tokenizer/{tokenizer.json,vocab.json,merges.txt,token_type.json,chat_template.jinja}, used by the Python server.
  Quality test therefore = encode with an HF tokenizer from that dir, feed --ids, decode the ids back.
  Flags pinned: --temperature ds4_main.cpp:1254, --seed :1255, --n :1257, --window :1258, --dump-logits :1259,
  --verbose :1248, --gpus :1211, --cpu-threads :1216, --experts :1232.
- st_01a176ff (degenerate output): ranked causes - (1) HIGH host->GPU doorbell unobserved, so the GPU spin hits the bound
  and reads the memset zeros (ds4_model.dp.cpp:3062-3063), the CPU contribution is absent, hidden state collapses;
  (2) MEDIUM allocation mismatch (plain sycl::malloc_host vs an uncached-bias host alloc); (3) LOW the experimental
  uncached hint may be unimplemented on Battlemage. Discriminating experiment: print the spin count and the value read
  in wait_seq_kernel (ds4_kernels.dp.cpp:2179-2186), sweep STRATA_SPIN_MAX, read DS4_PROF stage timings, and confirm
  serve_lane receives requests (ds4_model.dp.cpp:3068-3071, cpu_experts += nslots).
- Verified nuance: sycl/include/strata/sycl_doorbell.hpp:48-50 records the uncached float4 hint working (2.4 us per round
  trip, 0 of 200 waits hitting the bound), while the revived lane measured sys_load on a polled ze allocation as NEVER SAW
  over 200k reads. Both are measured facts on this card; the plan must settle which read path the CPU-lane response uses.
- st_01a17700 (gates): default build+ctest needs no GPU; SYCL build needs setvars.sh + icpx + -DSTRATA_SYCL_DSV41=ON;
  ctest --test-dir build-sycl -R ds4 selects ds4_parity (sycl/CMakeLists.txt:277) and ds4_blas_parity (:291) and needs
  the GPU; pytest tests/terminal and apps/host npm test need neither.

- st_01a17701 (architect, design A recommended): 774 raw dpct markers across 13 files. Only two classes need code
  rectification now - DPCT1098 (every host-mapped payload read must go through ld_fresh/sys_load_mapped; the plain-load
  site at ds4_kernels.dp.cpp:1857 reads device-resident E and needs only a note) and DPCT1093 (per-device queue creation
  must be on the owning thread; ds4_model.dp.cpp:2068-2077, threads mode "a thread a GPU" at :831). DPCT1007 (occupancy,
  hardcoded n=2 at exl3.dp.cpp:1951-1953) is worth a measured A/B via STRATA_GLM_EXL3_DECODE=legacy|new, not a rewrite.
  Everything else (~730 markers) is one documentation commit replacing no-op dpct blocks with accepted notes, per the
  branch rule "converted code keeps DPCT comments until reviewed" (docs/SYCL_DSV41_PLAN.md). Blanket DPCT1065 barrier
  conversion is a regression hazard: barriers at ds4_kernels.dp.cpp:2232/2237/2263 fence global host-mapped traffic.
  Ordering: Inc0 baseline build+ctest -R ds4 (no commit); Inc1 documentation pass (byte-identical parity rows);
  Inc2 DPCT1098 audit + one-card smoke; Inc3 DPCT1093 audit + --gpus 0,1 smoke; Inc4 first measured one-B60 run, >=3
  repeats, greedy, compared token-by-token against the CPU-only reference; Inc5 two-card run against the same-day
  single-card run; Inc6 optional measured A/Bs.
  Blast radius: sycl/ enters only under STRATA_ENABLE_SYCL=ON (:14/:20), STRATA_SYCL_DSV41 and _EXE default OFF
  (sycl/CMakeLists.txt:257-258, :297-298); zero STRATA_SYCL_DSV41 references in src/ or include/. Risk map:
  ds4_kernels.dp.hpp signatures (:312-348) rebuild the whole port; the parity harnesses include only
  ds4_iq/gguf/numerics/launch/gemv/softmax (ds4_parity.cpp:24-31), so the doorbell, xfer and all-reduce paths have NO
  ctest gate - increments 2-3 must pair with engine smokes, not ctest. Do not edit the upstream CPU TUs
  (ds4_main.cpp, *_cpu.cpp) or ds4_parity.cpp.
  Multi-card: nm==1 skips every allreduce (:10887, :10985, :8592, :8765, :8805); tp=1 hands layers through a pinned host
  bounce buffer (:499-504, xfer_put/get, hand_xfer :10449); tp=2 all-reduces d.ao and d.moe_out with the two-member
  ar_sum path (:4325-4345) or ar_rs (:2305-2350). Every wait is bounded because an unbounded spin resets the GT on xe
  (sycl_doorbell.hpp:118-121, measured twice); the decode spin bound is 1,000,000 uncached reads (~2.4 s, under the 5 s
  driver timeout).
  Evidence bar: hardware+software block, build identity (commit + flags + AOT target), pack identity, greedy single
  request >=3 repeats with warm/cold split and loading excluded, tok/s from the engine's own stderr lines
  (ds4_main.cpp:922-923, 1380, 1445), CPU-lane share, xe reset watch (must be none), per-card memory via xpu-smi;
  quality = same-day ds4_parity + ds4_blas_parity MATCH rows WITH the negative controls exiting non-zero
  (--bad-scale/--corrupt/--corrupt-scale/--corrupt-ldx), plus the real prompt compared token-by-token against the
  CPU-only reference (ds4_gg_bench, STRATA_BUILD_DSV41=ON). Four tags on every claim: (measured on this PC) /
  (static reading) / (prior art) / (not measured yet).
  Unverified by the lane: strata-dsv41 link state (no binary in sycl/build-test) - contradicted by this session's
  EXE_RC=0 builds, so the plan cites our own build instead; which exl3 family the B60 takes by default.

## Approval gate
status: drafting
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
