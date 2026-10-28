# Gate review — todo 9 and todo 4 (two-card DEVICE_LOST close, commit ddc3c9e7)

Date: 2026-10-26 (host clock 2026-10-27)
Reviewer: omo-native-gate-reviewer (independent; no edits made; report artifact only)
Plan: /home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md
Fix under review: ddc3c9e7 "[Engine] dsv41 SYCL: avoid cross-context events in two-card hand-offs"
  (cherry-picked from lane commit 41fefad1af77d41ef8a556a986b19fdf2da0ecaf; one file
  sycl/src/dsv41/ds4_model.dp.cpp, 19 insertions / 48 deletions).

## recommendation

APPROVE — per checkbox:

- **todo 9** (line 190, `[ ]`): APPROVE -> mark `[x]`.
- **todo 4** (line 134, `[~]`): APPROVE -> the DEVICE_LOST gap is closed by ddc3c9e7; the
  ordinal mapping was already verified and still holds, and the two-card acceptance is now met.
  Mark `[x]` (the `[~]` gap text is stale).

## blockers

None. Every stated acceptance criterion and audit item reproduced clean. The four items below
are NOTES, not blockers, because each is either satisfied by the evidence as recorded or is not
tied to a stated criterion.

## verdicts per checkbox

### todo 9 — APPROVE

All 12 gate RCs named in the audit are present in
`.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-9-dsv41-sycl-dpct-and-xpu-test.log`, each
un-piped and adjacent to its command: RUN_RC=0, PROMPT_RC=0, PARIS_RC=0, XPU_TWO_RC=0,
RESET_RC=1, RESET_JOURNAL_RC=1, SAME_BUILD_RC=0, SYCL_CONFIGURE_RC=0, SYCL_BUILD_RC=0,
SYCL_CTEST_RC=0, COMMIT_RC=0, DIFF_SCOPE_RC=0. The log contains the completed `prompt:` line
and the decoded Paris answer, and both ds4 parity tests are named and pass ("100% tests passed,
0 tests failed out of 2 (ds4_parity, ds4_blas_parity)").

Every throughput figure sits beside a completed `prompt:` line in the same run block, verified
in the on-disk logs `/tmp/dpct_run/w9/one_final.log` (35.17 / 2.48), `two_final.log`
(28.29 / 3.46), `two_xpu_receipt.log` (36.48 / 15.30). No figure appears without one.

SAME_BUILD_RC=0 means what it says: the one-card control and the two-card runs all used
`/tmp/dpct_w9_build/strata-dsv41`, built once at lane commit
`41fefad1af77d41ef8a556a986b19fdf2da0ecaf` with SHA-256
`53fedc2e83828c91bf9e3b9334c514476387db44cf075b623769cc89b00e65ca`. The sha256 and commit
receipts were recorded before the one-card control (01:50) and the two-card runs followed
(01:51/01:54) with no intervening rebuild; the binary hash file is still present on disk.

Guardrails hold (verified against the repo, not lane prose):
`git log 18a27e02..HEAD` (oldest-first) is exactly ca0f6f94, b18f6c40, 2e5c53b9, 433868e5,
a698bb84, 39b128ad, 28bd2aaa, ddc3c9e7. `git status --porcelain` shows zero tracked
modifications (`.omo/` and `bench/results/` entries are untracked `??`, none committed).
`git diff 18a27e02..HEAD` touches only `sycl/` and `docs/`. ddc3c9e7's diff contains only the
cross-context event removal plus host-side `wait()` — no barrier-scope conversion, no occupancy
rewrite, no `ggml_sycl` build (build target was `strata-dsv41`). SPIN_RUN_RC=1 with
SPIN_FAIL_LOUD_RC=0 and the loud "the card never saw an answer" lines is present.

Cleanup receipts use `node:fs` `existsSync === false` (BUILD_GONE_RC=0 and the
`/tmp/dpct_run/w9/cleanup_receipt.txt`), not ls/test/stat exit codes.

### todo 4 — APPROVE (gap closed)

Ordinal mapping still holds: todo 4 log "gap close" section RUN_G1_RC=0 and RUN_G1B_RC=0 show
`--gpus 1` -> `ds4: CUDA1 layers [0, 39] + head` with a completed prompt, and zero
`Ryzen`/`opencl`/`not supported` markers; `--gpus 0,1` shows two GPU device lines (CUDA0 + CUDA1).

Two-card acceptance now met: the todo 9 log's final two-card run exits 0 (RUN_RC=0) with a
completed `prompt:` line, decodes "The capital of France is **Paris**.<｜end▁of▁sentence｜>",
shows both cards loaded (`xpu-smi ps` PID on device 0 and device 1, ~23.1 GiB each), and has
zero engine/kernel reset matches (RESET_RC=1, RESET_JOURNAL_RC=1). The earlier wedge
(`UR_RESULT_ERROR_DEVICE_LOST`, level_zero error 20, at `ds4_model.dp.cpp:11213`) is gone.

Fix is the reason: the pre-fix baseline with the same invocation still fails
(`BASELINE_NO_WARM_RC=124`, `DEVICE_LOST` present, journal shows 8870 reset matches on card 1),
and the patched smoke + final runs complete. The ddc3c9e7 diff is patch-id-identical to the
lane commit 41fefad1 that produced this evidence (`git patch-id --stable` =
`2d65fe2dbc89373f1e60a4074b15daa0259b531d` for both), so the lane-worktree evidence applies to
the merged commit.

## originalIntent

(From the plan and the two checkboxes.) todo 4 audits DPCT1093 (queue created on the thread
that owns its device) and, after the ordinal-mapping gap was closed in 39b128ad, owed a
two-card smoke that exits 0 and prints a prompt. todo 9 runs the same prompt/flags/build on
`--gpus 0,1` and reports throughput only from completed prompts against a same-day, same-build
single-card run, while watching for the hand-off failure and a GPU reset.

## desiredOutcome

The owner's two Intel Arc Pro B60 cards answer "what is the capital of france" as Paris, with
real throughput numbers from completed prompts, a two-card run that no longer wedges with
DEVICE_LOST, and a same-build single-card baseline for comparison — with the Intel path staying
opt-in and the defaults untouched.

## userOutcomeReview

The shipped artifact delivers that outcome. The committed fix removes the cross-context Level
Zero event dependency (three `sync_barrier` + `ext_oneapi_submit_barrier` sites and the three
`event_ptr` fields/allocations), replacing each with a host-side `d.st->wait()` before the
destination copies are queued. The two-card run now exits 0, prints a completed prompt, and
decodes to Paris with both cards loaded and no reset. Throughput is reported only from
completed prompts, and the one-card control (35.17 / 2.48) is the same binary, same day, as the
two-card runs (28.29 / 3.46 and 36.48 / 15.30); the log honestly reports both two-card repeats
instead of selecting one scaling factor. The guardrails hold: no default changed, no
barrier-scope conversion, no occupancy rewrite, no `ggml_sycl` build, no `.omo/`/`bench/results`
commit, and the spin-limit control fails loudly.

## Checked artifact paths

- /home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md (todos 4 and 9, guardrails, IS-7)
- /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-9-dsv41-sycl-dpct-and-xpu-test.log (full)
- /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-4-dsv41-sycl-dpct-and-xpu-test.log (full; "todo 4 gap close" section)
- /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test-todo4-gate-review.md (REJECT on the pre-fix wedge)
- /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md (skill-perspective coverage)
- git: `git log --reverse 18a27e02..HEAD`, `git diff 18a27e02..HEAD --stat/--name-only`,
  `git show ddc3c9e7`, `git show 41fefad1`, `git patch-id --stable` for both, `git status --porcelain`
- /tmp/dpct_run/w9/{build_commit.txt,build_binary.sha256,one_final.log,two_final.log,
  two_xpu_receipt.log,patch_smoke.log,prefix_no_warm.log,spin_fail.log,cleanup_receipt.txt,
  two_xpu_ps_unpiped.txt,two_xpu_decoded.txt,baseline_journal.txt}
- /home/maxious/Strata/sycl/src/dsv41/ds4_model.dp.cpp (ddc3c9e7 diff; :11213 context)
- /home/maxious/.omo/binary-runtime/5.1.48/plugin/skills/remove-ai-slops/SKILL.md (loaded)
- /home/maxious/.omo/binary-runtime/5.1.48/plugin/skills/programming/SKILL.md (loaded)

## Evidence gaps (exact)

- No dedicated code-review report covers ddc3c9e7 specifically; the F2 review's skill-perspective
  section predates it (it covers 18a27e02..28bd2aaa). I applied the remove-ai-slops / programming
  criteria to ddc3c9e7 directly (see below). NOTE, not a blocker — no success criterion requires
  a named reviewer file.
- The SHA-256/commit receipts are recorded once before the one-card control rather than re-hashed
  after the two-card runs. NOTE — the two-card runs used the same un-rebuilt binary file and the
  commit receipt is re-checked by SAME_BUILD_RC=0; the binary hash file is still on disk.
- The todo 9 evidence was produced on the lane worktree at 41fefad1, not at the integration-branch
  hash ddc3c9e7. NOTE — `git patch-id --stable` proves the two commits are patch-identical, so the
  evidence applies to the merged commit.

## Skill-perspective check (remove-ai-slops / programming)

Applied directly over the ddc3c9e7 diff (19+/48-), tests, and production code.

- **Tests.** No test code was added, removed, or changed; the lane only ran the pre-existing
  `ds4_parity`/`ds4_blas_parity` harnesses. No excessive, tautological, implementation-mirroring,
  deletion-only, or removal-verifying tests; no prose pins. No overfit/slop test class applies.
- **Production extraction / parsing / normalization.** The diff removes three obsolete
  `event_ptr` fields, their allocation sites, and the cross-context
  `sync_barrier`/`ext_oneapi_submit_barrier` calls, replacing each with a host-side `d.st->wait()`.
  This is a root-cause fix (the receiving card's exec queue was reset by a source-context event
  dependency), not a wrapper, speculative abstraction, or normalization. The F2 review already
  records the same conclusion for the broader range.
- **Maintenance burden / false confidence.** The `wait()` is a standard synchronization primitive,
  no new state is introduced, and the removed objects were dead after the reorder. No default or
  interface widened. The only reporting-risk found is the honest "two-card repeats varied" note,
  which avoids overclaiming a scaling factor — good, not slop.
- **Scope drift.** One file under `sycl/`; nothing outside the plan's boundaries. No barrier-scope
  conversion, occupancy rewrite, or `ggml_sycl` build.

## Verdict

Both checkboxes pass. todo 9 is APPROVE -> `[x]`; todo 4 is APPROVE -> `[x]` with the DEVICE_LOST
gap closed by ddc3c9e7.
