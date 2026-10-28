# Gate Review: dsv41-sycl-dpct-and-xpu-test — todo 2 (DPCT note triage)

- **recommendation:** APPROVE
- **blockers:** none
- **reviewer:** omo-native-gate-reviewer (st_01a1777c), final gate for todo 2
- **date:** 2026-10-26
- **target artifact:** `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-2-dsv41-sycl-dpct-and-xpu-test.log`
- **plan todo audited:** `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` todo 2
- **worktree audited:** `/home/maxious/Strata-wt/dsv41-dpct-w2` (branch `dsv41-dpct-w2`)
- **report path note:** the todo-1 gate review already occupies
  `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-gate-review.md`; this todo-2 review is written
  to the per-todo path `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-todo2-gate-review.md`
  (same convention as the existing `-todo10-gate-review.md`), to avoid clobbering todo 1.

## originalIntent

The plan's todo 2 closes GAP-1 / IS-1: the residual dpct advisory comment blocks in
`sycl/src/dsv41` must stop looking like unreviewed unfinished work. Every advisory must be
either rectified or carry a one-line accepted note naming its class and why it was left,
using the pre-settled classification, with no code change, no barrier-scope change, no fence
change, and no launch-shape change. The plan's commit line is "record why each was accepted".

## desiredOutcome

A reviewer of the branch can audit the port without triaging 774+ raw dpct blocks themselves:
each DPCT class gets a classification (RESOLVED / ACTIONABLE / ACCEPTED_AS_IS /
NOT_APPLICABLE) and a substantive reason, the ACTIONABLE classes are handed to the right
later todos, the count is measured and self-consistent, and the SYCL dsv41 target still
builds and passes `ctest -R ds4`. The commit must be `[Engine]`-prefixed, single-file under
`sycl/`, off base `18a27e02`, with no `.omo/` or `bench/results` content, and the lane's
`/tmp` build/scratch trees cleaned up with un-piped receipts.

## userOutcomeReview

The shipped artifact satisfies the outcome. The commit `ca0f6f94` on branch `dsv41-dpct-w2`
(parent `18a27e02`) adds exactly one file, `sycl/docs/sycl-dsv41-dpct-triage.md`, whose table
covers 30 DPCT classes / 776 notes with a reason per row, and the "How the numbers come from"
section transparently documents the raw-grep-to-note refinement. I independently reproduced
the note count (776 `DPCT<num>` occurrences, per-class sums matching the table exactly), the
corrected citation (`ds4_gguf.dp.cpp:241-242`, the real measured `sycl::exp` record, not the
stale `:236`), the commit scope, the `[Engine]` title, and the on-disk cleanup of
`/tmp/dpct_w2_build` and `/tmp/dpct_w2`. The SYCL configure/build/ctest rows are all 0 with
2/2 ds4 tests and the summary quoted verbatim. The five NOTES below are naming, mechanism,
and alternative-design observations, not failures of any success criterion.

## Gates

1. **Plan reread / worktree — PASS.** Worktree exists at `/home/maxious/Strata-wt/dsv41-dpct-w2`,
   branch `dsv41-dpct-w2`; independently reproduced: `HEAD=ca0f6f94599ea5e5c82740dc8c72fb02235769ed`,
   `PARENT=18a27e02fd5d3b63dddf9a5a1863abaf5581f769` (`git log --pretty=%P -n1 HEAD`), and
   `git worktree list` shows the path with that branch. Log line: `WT_ADD_RC=0` after
   `git worktree add /home/maxious/Strata-wt/dsv41-dpct-w2 -b dsv41-dpct-w2 18a27e02fd5d3b63dddf9a5a1863abaf5581f769`.

2. **Automated build/test — PASS.** Evidence log carries `SYCL_CONFIGURE_RC=0`,
   `SYCL_BUILD_RC=0`, `SYCL_CTEST_RC=0`. Test names quoted: `Start 1: ds4_parity`,
   `Start 2: ds4_blas_parity`, `1/2 Test #1: ds4_parity ... Passed`,
   `2/2 Test #2: ds4_blas_parity ... Passed`. Summary quoted verbatim:
   `100% tests passed, 0 tests failed out of 2`.

3. **Table completeness + can-fail control + recount — PASS.** `ENUM_SUM_RC=0` and
   `ENUM_SUM_NEG_RC=1` (corrupt control fails as required). Counts reconciled: 30 ids =
   4 + 2 + 15 + 9; 776 notes = 25 + 112 + 563 + 76. Independent recount:
   `DPCT_RECOUNT_RC=0`; `grep -rn "DPCT" sycl/src/dsv41` = **1046 lines**, and the note
   count (distinct `DPCT<num>` occurrences) = **776**, matching the table exactly
   (per-class tally reproduced: DPCT1065=156 … DPCT1002=1). See NOTE N-1 and N-2.

4. **Citation fidelity — PASS.** `RESOLVED_CITE_RC=0` and final `CITED_LINES_RC=0` in the log.
   Reading the worktree file (`CITE_READ_RC=0`), lines 241-242 of `sycl/src/dsv41/ds4_gguf.dp.cpp`
   are the real measured record ("exp, not native::exp … Measured on the B60 (ds4_parity
   probe, 2026-10-25) … sycl::exp matches expf bit-exactly … native::exp is up to 8 F32 ulps
   off"), while line 236 is the DPCT1013 advisory block, confirming the disclosed stale cite.
   `TABLE_CITE_RC=0`: the committed table's DPCT1013 row cites `:241-242`; the only "236" in
   the table is `:2236` (a barrier cite), not the stale `:236`.

5. **Scope and commit — PASS.** `COMMIT_RC=0`; HEAD `ca0f6f94599ea5e5c82740dc8c72fb02235769ed`
   on `dsv41-dpct-w2`; independently reproduced `SCOPE_RECHECK_RC=0` with
   `git diff --name-only 18a27e02 HEAD` listing exactly one file:
   `sycl/docs/sycl-dsv41-dpct-triage.md`. Title `[Engine] triage the DPCT conversion notes in
   the SYCL dsv41 port` starts with `[Engine]`. No `.omo/` or `bench/results` path committed
   (grep count 0).

6. **Cleanup receipts — PASS.** `CLEANUP_RC=0`; `LS_BUILD_RC=2` and `LS_TMP_RC=2` in the log.
   Independently verified on disk: `ls -d /tmp/dpct_w2_build` and `ls -d /tmp/dpct_w2` both
   exit 2 ("No such file or directory").

7. **Pipe audit — PASS.** `PIPE_AUDIT_RC=0`. Spot-check: each `*_RC=` line follows its own
   un-piped command (configure/build/ctest redirect to log files, not through a pipe).

8. **Notes state a reason — PASS.** `NOTES_REASON_RC=0` (my check): all 30 reason cells give a
   substantive reason; no bare TODO/TBD. The `todo 3` / `todo 4` strings are owner-todo
   pointers on the two ACTIONABLE rows, not placeholders.

## Checked artifact paths

- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (todo 2 acceptance text)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-2-dsv41-sycl-dpct-and-xpu-test.log`
- `/home/maxious/Strata-wt/dsv41-dpct-w2/sycl/docs/sycl-dsv41-dpct-triage.md` (committed table)
- `/home/maxious/Strata-wt/dsv41-dpct-w2/sycl/src/dsv41/ds4_gguf.dp.cpp:228-249` (cite site)
- `/home/maxious/Strata-wt/dsv41-dpct-w2/` git metadata (HEAD, parent, branch, diff, commit message)
- `/tmp/dpct_w2_build` and `/tmp/dpct_w2` (cleanup, both absent)

## Evidence gaps / NOTES (none is a blocker)

- **N-1 — RC naming.** The literal `DPCT_ENUM_RC=0` does not exist in the log. The lane's
  pre-worktree probe produced `DPCT_ENUM_RC=1` (and `GIT_STATUS_RC=1`, `GIT_HEAD_RC=1`,
  `NOTES_WC_RC=1`) because it ran inside a worktree it had not yet created; the successful
  re-run is recorded as `DPCT_ENUM2_RC=0` / `NOTES_WC2_RC=0`. The substantive criterion
  (a successful DPCT enumeration feeding the table) is met and independently re-verified.
- **N-2 — Recount mechanism.** The instruction's literal command `grep -rn "DPCT"
  sycl/src/dsv41` yields 1046 LINES, not 776. The "note count" (776) is the number of
  `DPCT<num>` occurrences, which I tallied independently as exactly 776 with per-class sums
  matching the table. The table itself documents the 1046 → 776 → 270 split.
- **N-3 — Scope/method divergence (alternative design).** The plan's todo-2 "What to do" text
  said "replace each no-op dpct comment block … with one line"; the lane instead delivered a
  single docs triage table and left the source blocks in place. This is consistent with the
  plan's commit title ("record why each was accepted") and with the gate-5 expected artifact
  (the docs file), and it delivers IS-1's reviewer-facing outcome at per-class granularity
  rather than per-block in-source granularity. Not a failed criterion.
- **N-4 — Ephemeral verifiers.** `verify_cites.py`, `enum_check.py`, `stats.py`, `analyze.py`
  lived in `/tmp/dpct_w2` and were removed at cleanup, so `RESOLVED_CITE` / `ENUM_SUM` checks
  are not re-runnable from the committed tree. The key numbers (776, per-class sums, the
  :241-242 cite) were independently re-derived by this review.
- **N-5 — `DIFF_PARITY_RC` not recorded.** The plan's todo-2 acceptance list included
  `DIFF_PARITY_RC=1` (parity rows byte-identical to todo 1). It is absent from the log. With
  zero source changes the parity is trivially identical, and `SYCL_CTEST_RC=0` (2/2) confirms
  no behavioural regression, so this is a moot-but-unrecorded criterion, not a failure.

## Slop / programming-criteria check

- **Tests:** no tests added or deleted; the lane ran the existing `ds4_parity` /
  `ds4_blas_parity` harnesses. No deletion-only, tautological, or implementation-mirroring
  test was introduced. `ENUM_SUM_NEG_RC=1` is a legitimate can-fail control (corrupt table
  must fail), not a test that merely verifies a removal.
- **Production code:** none changed; the commit is documentation only. No production
  extraction, parsing, or normalization was committed (the analysis scripts stayed in `/tmp`
  and were removed).
- **Maintenance burden / false confidence:** low. The table labels its class decisions
  "unconfirmed static analysis" and hands the two ACTIONABLE classes to todos 3 and 4. The
  DPCT1072 row counts two prose `(DPCT1072)` mentions alongside true advisory blocks, which
  is disclosed by the 1046/776/270 breakdown. No scope drift beyond the N-3 alternative
  design, which the gate instructions themselves describe as the expected artifact.
