# Gate review — todo 8: one-card quality and throughput run (dsv41-sycl-dpct-and-xpu-test)

Date: 2026-10-26
Reviewer: omo-native-gate-reviewer (st_01a17885), final gate; independent, no edits made, report artifact only.
Plan audited: `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md`
Lane under review: st_01a17850 (todo 8 escalation lane); commit `fed551b7e8f4fe932d86196d01a743b1b41726e7` in worktree `/home/maxious/Strata-wt/dsv41-dpct-w8`.

## recommendation

**APPROVE** — todo 8 checkbox lands **[~]** (not [x]).

The code fix is correct, faithfully matches the CUDA original, is scoped to one `sycl/` file, and the
user's actual desired outcome (IS-5/IS-6 substance) is satisfied and independently reproduced. The todo is
[~], not [x], only because two of its *literal* acceptance criteria are mis-specified in the plan and are
therefore unmet by any correct run; the plan must amend them (an F1/F4 action), not the code.

## blockers

none.

No criterion that reflects the user's actual desired outcome fails. The two literal criteria below are
mis-specified (see "criteria to amend"); enforcing them would reject a correct answer, so they are recorded
as plan defects, not blockers of this delivery.

## todo 8 checkbox gap line (for the [~] mark)

todo 8's literal FIRST_ID criterion (first emitted id 11111 = " Paris") is not met — the correct greedy
answer starts with 671 ("The") and ends with EOS (id 1, token type 3, which also trips the literal
CONTROL=no-type-3 criterion). Both criteria are mis-derived: the plan's own todo-7 gap records 11111 as
"the SYCL's own logit for 11111, not a reference value", and the pack's `token_type.json` classifies id 1
(eos) as type 3, so "no emitted id has token type 3" can never pass for a terminating answer. IS-5's
substance — decoded text contains Paris, no vision control token (129276 gone), ≥3 byte-identical repeats —
is satisfied. Last evidence: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-8-dsv41-sycl-dpct-and-xpu-test.log`
(G3 `FIRST_ID_RC=1` → 671; G4 `CONTROL_RC=0` → id 1 type 3).

## originalIntent

The user (owner of this PC, 2x Intel Arc Pro B60) wants Strata's DeepSeek dsv41 engine to answer a real
question on the Intel card: "what is the capital of france" must return an answer containing "Paris" (not
the repeated vision-control token `129276`), with real throughput numbers from a completed prompt and the
host CPU lane contributing experts. Todo 8 is the one-card quality/throughput run that closes IS-5 and IS-6
after the todos 5/6/7 correctness work.

## desiredOutcome

A one-card greedy run of the chat-template prompt "What is the capital of France?", ≥3 byte-identical
repeats, whose decoded answer contains "Paris", carries no vision control token, quotes prompt/decode tok/s
from a completed prompt with build/pack/card/date identity and the CPU-lane share, with the first emitted
id 11111 and no type-3 id per the plan's (mistaken) literal criteria.

## userOutcomeReview

**Satisfied.** The fix in `sub_dot()` is the real root cause, the answer is correct, and it reproduces.

- **Fix correctness (verified against CUDA):** CUDA `src/dsv41/ds4_iq.cu:372` computes
  `v = __fsub_rn(v, __fadd_rn(__fmul_rn(s.m0, (float)sl), __fmul_rn(s.m1, (float)sh)))`
  = `v - (m0*sl + m1*sh)`. The SYCL port now computes `v = v - (s.m0 * (float)sl + s.m1 * (float)sh);`
  (`sycl/src/dsv41/ds4_iq.dp.cpp:423`). Semantic match confirmed. The decode_b comment in **both** files
  states the K-quant weight is `d sc q - dmin m` (CUDA `ds4_iq.cu:294`, SYCL `ds4_iq.dp.cpp:330`), and
  `m0/m1` are both set to `dmin * (c>>4)` (CUDA `:330-331`, SYCL `:367-368`), so both halves must be
  subtracted — the pre-fix `v - m0*sl + m1*sh` was the flipped sign the lane claims.
- **Answer (independently reproduced):** decoded ids `671 6102 294 8760 344 2619 51119 42499 1` from
  `vocab.json` are `The Ġcapital Ġof ĠFrance Ġis Ġ** Paris **. <｜end▁of▁sentence｜>` = "The capital of France is **Paris**." + EOS
  (Ġ = space). Contains "Paris"; no `<｜box｜>` (129276). First id 671 = "The", not the plan's 11111 = "ĠParis".
- **Repeats:** ids are identical across run1–run4 (`671 6102 294 8760 344 2619 51119 42499 1`), 3 repeats with
  own `REPEAT2/3/4_RC=0` — byte-identical, as claimed.
- **Throughput (from completed prompts):** run1 `prompt: 10 tokens in 414 ms (24.13 tok/s)`, `decode: 9 tokens
  in 4017 ms (1.99 tok/s)`; run3 prompt 281 ms (35.61 tok/s). Quoted from a completed prompt; build identity,
  pack path, card, date are in the log.
- **CPU lane:** non-zero in every run (`211/156/158/152 of 1920` in the four quality runs; `619 of 1680` raw;
  `1481 of 1680` in `fix_run.log`), vs the `0 of 1680` baseline. IS-4 is met.
- **One card:** the engine's own `ds4: CUDA0 layers [0, 39] + head: weights 5.86 GB, experts 14.86 GB ...` line
  appears with no `CUDA1`, and card 1 is idle in the xpu-smi listing. (See gap G3 on xpu-smi memory.used.)

## checked artifact paths

- Plan: `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (todo 8, IS-5, IS-6, todo-7 gap text)
- Evidence log: `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-8-dsv41-sycl-dpct-and-xpu-test.log`
- Commit diff: `git show fed551b7e8f4fe932d86196d01a743b1b41726e7` (worktree `/home/maxious/Strata-wt/dsv41-dpct-w8`)
- Fix file: `/home/maxious/Strata-wt/dsv41-dpct-w8/sycl/src/dsv41/ds4_iq.dp.cpp:420-424`
- CUDA reference: `/home/maxious/Strata/src/dsv41/ds4_iq.cu:353-373`, `:294`, `:330-331`
- Worktree state: `git status --porcelain` = 0 lines (clean); HEAD `fed551b7`, parent `a698bb84` (confirmed)
- Run artifacts: `/tmp/dpct_run/w8/{run1..run4,fix_run,run_raw,run_xpu,neg,neg2}.log`, `decoded1.txt`
- Tokenizer: `/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/tokenizer/{vocab.json,token_type.json}`
- Parity/ctest: `/tmp/dpct_w8_build/strata-dsv41` (exists, 37897584 bytes), `/tmp/dpct_w8_build/Testing/Temporary/LastTest.log` (ds4_blas_parity OK)

## exact evidence gaps (notes and plan-amendment actions — none block this delivery)

- **G1 (criteria to amend, FIRST_ID).** Plan todo-8 acceptance and IS-5 require first id 11111 (" Paris",
  type 1). Observed first id is 671 ("The"). The plan's own todo-7 gap text already records 11111 as "the
  SYCL's own logit ... not a reference value", so the expectation was never authoritative; the correct
  answer necessarily starts with "The". **Amend:** IS-5/todo-8 to expect 671 (or drop the specific first-id
  expectation). Evidence: evidence log G3 (`671`, `FIRST_ID_RC=1`).
- **G2 (criteria to amend, CONTROL).** Plan requires "no emitted id has token type 3" (`CONTROL_RC=1`).
  Observed `CONTROL_RC=0` because the emitted set ends with id 1, the pack's eos token, which
  `token_type.json` classifies as type 3 (verified: id 0 → 3, id 1 → 3, id 129276 → 3, id 671 → 1,
  id 11111 → 1). Excluding EOS, no emitted id is type 3 and 129276 is gone. **Amend:** to "no type-3 id
  other than EOS" / "no vision control token (e.g. 129276)". Evidence: evidence log G4; `token_type.json`.
- **G3 (note, xpu-smi).** `xpu-smi` `memory.used` reports 255 MiB for card 0 both idle and during decode —
  it does not reflect Level Zero allocations on this driver. The one-card "loaded" receipt is therefore the
  engine's own `CUDA0 ... 5.86 GB + 14.86 GB ... 597 MiB free` line plus the absence of any `CUDA1`, not
  xpu-smi. The lane discloses this; not a defect, but the plan's "xpu-smi shows one card loaded" is only
  partially evidenced by xpu-smi. Evidence: evidence log G7 and section 8.
- **G4 (note, prose precision).** The lane's VERDICT and the handoff summary say "CPU lane 1481 of 1680
  experts (0 before)". That number is from `fix_run.log` (`--n 8`, decode line `CPU lane: 1481 of 1680`), not
  the four quality runs, which quote `211/156/158/152 of 1920`. Non-zero CPU lane is confirmed in all runs,
  so no outcome is affected, but the summary number should not be read as the chat-template-run figure.
  Evidence: `fix_run.log:32` vs evidence log lines 114/141/146/151.
- **G5 (note, parity coverage).** `ds4_parity`/`ds4_blas_parity` cover `iq::dequant_rows`, gemv_f16/f32 and
  softmax — not the K-quant GEMV (`iq::mv -> mv_kernel -> sub_dot`). They pass before and after the fix, so
  `CTEST_RC=0` is not evidence for this change (the lane states this correctly). The fix is instead verified
  by static CUDA comparison + the end-to-end answer. The K-quant GEMV needs its own parity row; that is a
  plan finding, not this commit's defect. Evidence: `LastTest.log`; lane evidence section 5.
- **G6 (note, no separate code-review/QA-matrix artifact).** No independent code-review report or manual-QA
  matrix file was provided for todo 8; the lane's evidence log doubles as the code review (root cause, CUDA-vs-
  SYCL line comparison, module-wide `m0/m1` sweep, duplicate-symbol resolution, parity-coverage statement) and
  QA record. The `remove-ai-slops` overfit/slop and `programming` passes were not run by the lane, so I ran
  them directly — see below.

## skill-perspective check (remove-ai-slops overfit/slop pass + programming criteria)

Run directly over the diff, the tests, and production code; report coverage does not replace this check.

- **Diff is a 1-line semantic fix + 2 comment lines** in `sub_dot()`. No test was added, changed, or deleted.
  No excessive/useless tests, no deletion-only tests, no test that merely verifies a requested removal, no
  tautological or implementation-mirroring test. **Slop: none.**
- **No production extraction/parsing/normalization** was introduced; the change is a correctness fix at the
  single accumulation site, not a refactor. **Scope drift: none.**
- **Maintenance burden:** none added (minimal, commented, scoped to `sycl/`). The comment correctly states the
  reason in-line with todo 2's accepted-note convention.
- **False-confidence risk (documented, not hidden):** the parity harness does not cover this path, so
  `ctest 2/2` cannot catch a regression here. The lane explicitly flags this as a coverage gap rather than
  letting "parity green" imply GEMV coverage — acceptable, but it is why G5 must be closed by the plan with a
  K-quant GEMV parity row before the branch is relied upon.
- **Cross-backend safety:** the change is inside the SYCL port only; CUDA (`src/`), HIP, and the default build
  are untouched, matching the plan's "every edit lands in `sycl/` or `docs/`" guardrail. `git show --stat`
  lists only `sycl/src/dsv41/ds4_iq.dp.cpp`.

## decision rationale (APPROVE + [~])

The lane correctly found and fixed the real defect (the K-quant min-term sign in the GEMV), the fix is a
faithful match to the CUDA reference, the commit is cleanly scoped, and the user's actual outcome — a correct
"Paris" answer with no vision control token, byte-identical over 4 greedy runs, real throughput, non-zero CPU
lane, one card — is reproduced from the artifacts. The only unmet items are two plan criteria that are
provably mis-specified (11111 came from the defective run; EOS is legitimately type 3), so no *valid* success
criterion fails. The todo is therefore [~] with the gap line above, and the plan must amend IS-5/todo-8's
FIRST_ID and CONTROL criteria under F1/F4.
