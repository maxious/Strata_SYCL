# Gate Review: dsv41-sycl-dpct-and-xpu-test — todo 10 (default-build regression gate)

- **recommendation:** APPROVE
- **blockers:** none
- **reviewer:** omo-native-gate-reviewer (st_01a17766), final gate for todo 10
- **date:** 2026-10-26
- **plan audited:** `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` todo 10 (acceptance text read first)
- **evidence audited:** `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-10-dsv41-sycl-dpct-and-xpu-test.log` (139 lines, full read)

> Report path note: the fallback report path from the task is
> `.omo/evidence/<goal>-gate-review.md`, which is
> `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-gate-review.md` and already holds the todo-1
> gate review (st_01a1774e). To avoid clobbering that separate artifact, this todo-10 review is
> written to `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-todo10-gate-review.md`.

## originalIntent

The user wants the default (no-SYCL) build proven untouched by the dsv41 SYCL port: the Intel
path stays behind its opt-in switch, the default path builds and its full ctest remains green,
and no product file changed. Todo 10 is a gate, not a change (plan: `Commit: N | gate only`,
closes IS-8).

## desiredOutcome

A gate evidence log that (1) works from a clean worktree at the integration base `18a27e02`,
(2) configures/builds the default tree with `-DSTRATA_BUILD_TESTS=ON` and runs the full
`ctest`, (3) proves the OFF configure/build lists no dsv41 source while the ON configure lists
it, (4) proves `base..HEAD` is an empty diff, and (5) leaves cleanup receipts showing the
build dirs and worktrees are gone. It must not claim the default ctest is fully green if it is
not.

## userOutcomeReview

The shipped log satisfies the outcome. The default build is green (`DEFAULT_BUILD_RC=0`,
161/161 targets). The default full `ctest` has exactly one failure, test #55 `isa_guard_check`,
and the lane proves it is pre-existing by reproducing the identical failure name and violation
on a clean `origin/main` control tree (`CTL_HEAD=4b9e05a2`, `CTL_CTEST_RC=8`). The OFF compile
list has zero `ds4_`/`dsv41/` references and the ON compile list has 75 `ds4_` references, so
the grep controls can fail in both directions. `base..HEAD` is empty (0 files) and no violating
path exists. Cleanup receipts are present and I independently confirmed the build dirs and both
worktrees are gone. The lane explicitly states the default ctest gate is NOT fully green.

## Gate results

### Gate 1 — Plan reread — PASS
- `WT_RC=0` and `WORKTREE_HEAD=18a27e02fd5d3b63dddf9a5a1863abaf5581f769` — log lines 6–8;
  the worktree was added at `18a27e02`.
- Independently reproduced against the repo (not the log): `git rev-parse HEAD` = `18a27e02...`
  (base commit object exists, `git cat-file -t` = commit), and `git rev-parse origin/main` =
  `4b9e05a2...` matching the control `CTL_HEAD`.

### Gate 2 — Automated build/test — PASS
- `DEFAULT_CONFIGURE_RC=0` — log line 10.
- `DEFAULT_BUILD_RC=0` — log line 12.
- `DEFAULT_CTEST_RC=8` — log line 31; the expected single pre-existing failure.
- Single failure = test #55 `isa_guard_check`: `/tmp/dpct_w10_failed.txt` (exists on disk) is
  `111:55/57 Test #55: isa_guard_check ...***Failed` and `131: 55 - isa_guard_check (Failed)`;
  `FAILED_NAMES_RC=0` (log line 34).
- Pre-existing on the origin/main control: `CTL_WT_RC=0` (line 41), `CTL_BUILD_RC=0` (line 46),
  `CTL_CTEST_RC=8` (line 48), `CTL_FAILED_NAMES_RC=0` (line 51), `CTL_HEAD=4b9e05a2a806...`
  (line 42). `/tmp/dpct_w10_ctl_failed.txt` (exists on disk) names the same test #55
  `isa_guard_check`; `diff` of the two failed-name files is empty (`FAILDIFF_RC=0`, line 58).
- The lane did NOT claim a green default ctest: log lines 60–62 "The default ctest gate is NOT
  green" and "NOT fully green - it is green except this pre-existing isa_guard_check".
- Cause explained from the test's own assertion output, not asserted: `/tmp/dpct_w10_ctestout.log`
  line 113 prints `VIOLATION strata::kernels::cpu::iq512x_rows(...) (777 wide instructions,
  first vmovq): wide-ISA code in an unlisted public function [iq_avx512.cpp.o]`. I read
  `tools/check_isa_guard.py` and confirmed `judge()` returns exactly
  `"wide-ISA code in an unlisted public function"` for a non-anonymous, non-weak function not in
  `WIDE_PUBLIC`, and that `WIDE_PUBLIC` contains `iq512_rows`/`iq512_gu_rows` but not
  `iq512x_rows` (`grep -c "\"iq512_rows\""` = 1, `grep -c "\"iq512x_rows\""` = 0; also captured
  in `/tmp/dpct_w10_allow_x.txt` = 0 and `/tmp/dpct_w10_allow_y.txt` = 1).

### Gate 3 — Positive/negative controls — PASS
- `DSV41_OFF_GREP_RC=1` (count=0) on the OFF configure output — log line 16.
- OFF compile list also count 0: `OFF_OBJGREP_RC=1` (count=0) and `OFF_DSV41_RC=1` (count=0)
  against `build.ninja` — log lines 18, 20.
- `DSV41_ON_GREP_RC=0` (count=75) on the ON compile list `build.ninja` — log line 30.
- Counts are shown and able to fail: the ON count (75) differs from the OFF count (0) using the
  same `ds4_` pattern; independently confirmed via the on-disk grep outputs
  `/tmp/dpct_w10_off_objgrep.txt` = 0, `/tmp/dpct_w10_off_dsv41grep.txt` = 0,
  `/tmp/dpct_w10_on_objgrep.txt` = 75.

### Gate 4 — Scope — PASS
- `SCOPE_RC=0` (base..HEAD diff empty: 0 files) — log line 24.
- `SCOPE_VIOL_RC=1` (violating paths: 0) — log line 26.
- The lane changed no product files: the diff is empty, and on-disk
  `/tmp/dpct_w10_scope.txt` / `/tmp/dpct_w10_scope_viol.txt` are empty.

### Gate 5 — Cleanup receipts — PASS
- `CLEANUP_RC=0` for `/tmp/dpct_w10_build` — log lines 79 and 110.
- `CTL_CLEANUP_RC=0` for `/tmp/dpct_w10_ctl_build` — log lines 74 and 112.
- `WT_RM_RC=0` and `CTL_WT_RM_RC=0` for the two worktrees — log lines 113–116.
- `LS_RC=2` — log line 118; `ls` cannot access the build/ctl/on dirs (exit 2).
- Verified on disk: `/tmp/dpct_w10_build`, `/tmp/dpct_w10_ctl_build`, `/tmp/dpct_w10_on`,
  `/home/maxious/Strata-wt/dsv41-dpct-w10`, and
  `/home/maxious/Strata-wt/dsv41-dpct-w10-ctl` are all absent; `git worktree list` no longer
  contains the two dsv41-dpct-w10 worktrees.

## *_RC line placement / exit-code integrity

Every `*_RC=` line sits directly below its own command with no pipe in the chain. The gate
commands use only output redirection (`> file`) or `-c` with redirection, never a shell pipe, so
each reported RC is the command's own un-piped exit code. `grep -c` exit codes are interpreted
correctly (1 = zero matches for the OFF gates, 0 = non-zero matches for the ON gate). The one
`bash -c "...; ctest ..."` ON-configure wrapper is a config command whose `ON_CONFIGURE_RC=0`
is reported separately and is not used to carry a measured result; the measured grep gates use
their own un-piped `grep -c` commands.

## Skill-perspective check (remove-ai-slops / programming)

`remove-ai-slops` and `programming` were loaded from
`/home/maxious/.omo/binary-runtime/5.1.48/plugin/skills/`. Their overfit/slop criteria were
applied directly over the diff, tests, and production code:

- There is no diff to slop-review: `git diff --name-only 18a27e02 HEAD` is empty (0 files), so
  there is no production extraction, parsing, or normalization to flag.
- No tests were added, removed, or modified by this lane: the lane only ran the pre-existing
  `ctest` suite and grep controls. Therefore there are no excessive, tautological,
  implementation-mirroring, deletion-only, or removal-verifying tests.
- The `grep -c` controls are genuine two-sided controls (OFF count 0 with exit 1; ON count 75
  with exit 0), not prose pins or tautological assertions; they measure machine-consumed
  compile-list membership, which is the actual criterion.
- No code review report or manual QA matrix was supplied in this task's inputs, so there is no
  code review report to check for skill-perspective coverage. This is noted as an input gap, not
  a blocker: the changed-files set is empty, so a code review would have nothing to review.

## checked artifact paths

- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (todo 10 acceptance text, full read)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-10-dsv41-sycl-dpct-and-xpu-test.log` (full read, 139 lines)
- `/tmp/dpct_w10_failed.txt` (exists; names test #55 isa_guard_check)
- `/tmp/dpct_w10_ctl_failed.txt` (exists; names the same test)
- `/tmp/dpct_w10_faildiff.txt` (empty; the two failed-name files are identical)
- `/tmp/dpct_w10_ctestout.log` (98% passed, 1 failed out of 57; line 113 is the VIOLATION for iq512x_rows)
- `/tmp/dpct_w10_ctl_ctestout.log` (same 98%/1-of-57 summary on the control)
- `/tmp/dpct_w10_isa.txt` (isa_guard_check run alone on the control: 0% passed, 1 failed; same VIOLATION)
- `/tmp/dpct_w10_off_objgrep.txt` = 0, `/tmp/dpct_w10_off_dsv41grep.txt` = 0, `/tmp/dpct_w10_on_objgrep.txt` = 75 (grep counts reproduced)
- `/tmp/dpct_w10_allow_x.txt` = 0, `/tmp/dpct_w10_allow_y.txt` = 1 (allow-list grep reproduced)
- `/home/maxious/Strata/tools/check_isa_guard.py` (read: WIDE_PUBLIC allow-list and judge() return string)
- `/home/maxious/Strata` git state: `HEAD` = 18a27e02, `origin/main` = 4b9e05a2, `git worktree list` (the two dsv41-dpct-w10 worktrees absent)
- Disk absence checks: `/tmp/dpct_w10_build`, `/tmp/dpct_w10_ctl_build`, `/tmp/dpct_w10_on`,
  `/home/maxious/Strata-wt/dsv41-dpct-w10`, `/home/maxious/Strata-wt/dsv41-dpct-w10-ctl` (all gone)

## exact evidence gaps

None against the stated criteria. The only artifacts a criterion requires but that are no longer
on disk are the build trees themselves, and their absence is exactly what Gate 5 requires; the
receipts and the retained log/grep outputs carry the evidence.

## Advisory notes (non-blocking)

1. The plan's todo-10 acceptance text writes `CTEST_RC=0` on the default build, but the
   lane's measured result is `DEFAULT_CTEST_RC=8` (one pre-existing failure). The task's gate
   contract for todo 10 explicitly expects `DEFAULT_CTEST_RC=8` and a proven pre-existing
   `isa_guard_check`, and the lane satisfies that contract and discloses non-greenness. The
   plan-text `CTEST_RC=0` is therefore a plan-wording lag, not a blocker for this gate.
2. `DSV41_ON_CFGGREP_RC=1` (count 0) on the ON configure *output* is an extra control that
   shows the ON configure output does not itself expand to `ds4_` tokens; the positive control
   that matters (`DSV41_ON_GREP_RC=0`, count 75) is on the ON *compile list* `build.ninja`, as
   the task requires. No gap.
3. No code review report or manual QA matrix was among the inputs for this task. Since todo 10
   changes zero files, there is no diff to review; noted, not blocking.
