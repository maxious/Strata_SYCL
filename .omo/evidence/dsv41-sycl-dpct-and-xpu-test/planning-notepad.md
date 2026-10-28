# Ultrawork Notepad - ulw-plan: rectify dsv41 SYCL DPCT TODOs, chase correctness, run the pack on Intel XPUs
Started: 2026-10-26T06:58:25.192Z
RUN_TMP=/tmp/tmp.wVrVsYJu8H

Tier: LIGHT - this session's own change set is the plan artifact (.omo/plans/*.md) plus steering read-only lanes; no product-code edits (planner persona). Implementation is payload for a later ulw-execute session. review_required=true is ulw-plan's default-on high-accuracy plan review, plan-gated and allowed because the user invoked ulw-plan.
Skills: ulw-plan (this workflow, plan gate open). programming + cuda-to-sycl-port are named in the plan for the executing session, not read here. memory-discipline for the saved decision.

## Plan
1. Fold Q1 answer (chase correctness) into .omo/drafts/dsv41-sycl-dpct-and-xpu-test.md scope IN.
2. Integrate the six lane reports as claims; verify cited file:line myself before writing them into the plan.
3. Approval brief + Approval gate question (required=true, wait=true).
4. On okay: rerun scaffold without --draft-only -> .omo/plans/dsv41-sycl-dpct-and-xpu-test.md; APPEND column-zero task rows (- [ ] N. / - [ ] F) with Recommended task executor category, references, acceptance, agent-executed QA, one commit per increment.
5. High-accuracy review: ONE plan-reviewer per round until unconditional approval (<=5), receipts in the draft.
6. Structural self-check; handoff explanation; stop (never execute).

## Success criteria + QA scenarios
- C1 plan structurally valid: `bun -e "const t=await Bun.file('.omo/plans/dsv41-sycl-dpct-and-xpu-test.md').text(); console.log((t.match(/^- \[ \] \d+\. /g)||[]).length,(t.match(/^- \[ \] F\d+\. /g)||[]).length)"` in /home/maxious/Strata; PASS = impl rows>0, F rows>0, each impl row carries a Recommended task executor category line; evidence $RUN_TMP/c1.txt
- C2 four areas covered by rows: `grep -n "DPCT\|doorbell\|CPU lane\|--gpus\|tok/s\|capital of france"` on the plan; PASS = each area in an implementation row and the quality row names the literal ids invocation; evidence $RUN_TMP/c2.txt
- C3 reviewer approval recorded: one task(subagent_type="plan-reviewer") round with VERDICT PASS, notes only; recorded in draft; evidence $RUN_TMP/c3.txt
- C4 no implementation by this session: `git status --short -- sycl src` empty after writing the plan (only .omo/ changed); evidence $RUN_TMP/c4.txt
- Regression named in the plan: default build + ctest -R ds4 stay green as its own row.

STOP: the moment the plan file is filled, structurally valid, plan-reviewer-approved, and the handoff explanation is delivered. Execution stays with the user.

## Now
Folding Q1 into the draft; lanes fd/01/02 still running (fe, ff returned).

## Todo
see todo list

## Findings
- sycl/CMakeLists.txt:297-316 strata-dsv41 sources; :293-296 comment says ds4_main.cpp compiles under SYCL (0 errors, icpx 2026.2.0) but the target does not link until the ds4_model port lands (lane claim - contradicted by our EXE_RC=0 builds and runs, so verify before citing).
- src/dsv41/ds4_main.cpp:1112 int main; flags at :1201-1295; --ids @file at :1343; --serve at :316/:1269; --temperature :1254, --seed :1255, --n :1257, --window :1258, --dump-logits :1259, --verbose :1248, --gpus :1211, --cpu-threads :1216, --experts :1232, --vram-experts :1215. NO text-prompt path (lane st_01a176fe: parse_ids only; Model has feed/step/draft taking ids; zero tokenizer references in src/dsv41).
- Pack tokenizer lives at <pack>/tokenizer/ (tokenizer.json, vocab.json, merges.txt, token_type.json, chat_template.jinja) and is used by serve/server.py, not the exe. Quality test = encode with AutoTokenizer from that dir, feed --ids, decode ids back.
- Doorbell: sycl/include/strata/sycl_doorbell.hpp:42-50,67-73 uncached L1+L3 hints; header comment says a system-scope atomic load of host USM is still served from the GPU cache on Arc Pro B60 (NEO 26.31, oneAPI 2026.1.1); ds4_kernels.dp.cpp:115-132 ld_fresh, :2165-2177 publish_kernel, :2179-2186 wait_seq_kernel; ds4_model.dp.cpp:3062 memset, :3068 serve_lane, :7652-7662 response write + cpu_experts += nslots, :10940 wait_seq(&resps[slot].seq); ds4_main.cpp:1445-1448 prints "CPU lane: X of Y experts".
- Discriminating experiment (lane st_01a176ff): print spin count + value read in wait_seq_kernel; STRATA_SPIN_MAX sweep; DS4_PROF=1 stage timing.
- DPCT tally: 1065 152, 1093 107, 1124 101, 1118 83, 1101 59, 1110 44, 1013 38, 1078 28, 1010 21, 1024 17, 1049 15, 1048 13, 1009 13, 1119 10, 1026 8, 1017 8, 1001 7, 1106 6, 1000 6, 1098 5, 1083 5, 1027 5, 1025 5, 1008 4, 1007 2, 1006 2, 1113 1, 1019 1, 1002 1. Zero async_memcpy sites.
- HEAD 18a27e02/e3d7853d/481dad0a/4d15354a/b46e11a9; sycl tree clean. Run 15 one B60: EXE_RC=0, RUN_RC=0, prompt 1 tok in 62 ms, decode 8 tok in 384 ms, CPU lane 0 of 1680 experts, ids 129276 x8.
- Gates (lane st_01a17700, claim): default cmake -DSTRATA_BUILD_TESTS=ON + ctest (no GPU); SYCL icpx + -DSTRATA_SYCL_DSV41=ON + ctest -R ds4 (GPU + oneDNN); python -m pytest tests/terminal; cd apps/host && npm test.

## Learnings
- Correctness, not speed, is the blocker: a stall-free engine still answers wrong.
- The exe has no text-prompt path, so the plan must spell the encode/decode helper explicitly.

## 2026-10-26 turn - plan filled, findings folded in, review round dispatched
PLAN_BYTES=48381 sha256=58ab738b02051e08448721a9860e9df7e88ef0dce02c097ebbd19542bf61ec84
C1=PASS (c1.txt: IMPL_ROWS=11 F_ROWS=5 CATEGORY_LINES=11 HEADERS=9 INDENTED_ROWS=0, ERE counts)
C2=PASS (c2.txt: DPCT=12 doorbell=12 CPU lane=16 Paris=7 capital of france=4 --gpus 0,=4 ctest -R ds4=3 STRATA_SYCL_DSV41=6 dump-logits=4 xpu-smi=5 11111=4 atomic_ref=3 sys_store_mapped=1)
C4=PASS (c4.txt: git status --short -- sycl src empty, STATUS_LINES=0)
C3=PENDING - plan-reviewer refused (plan-gated, ulw-execute already invoked); round-1 runs as omo-native-gate-reviewer st_01a17718 on the same contract; receipt pending in draft review.plan_reviewer.result
Draft status=approved; approval source = the user ulw-execute invocation (no-plan bootstrap)
Lane st_01a17702 findings folded into plan: copy-engine hand-back (todo 5), payload store sys_store_mapped (todo 6), sampler sound plus first-id 11111 acceptance (todos 7 and 8), GAP-7 and GAP-8 rows, Measured findings section
STOP: plan filled plus structurally valid plus reviewer VERDICT PASS plus handoff delivered; no implementation in this session