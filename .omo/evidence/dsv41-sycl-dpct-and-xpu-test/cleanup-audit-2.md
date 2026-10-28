# F5 cleanup audit (re-audit) - dsv41-sycl-dpct-and-xpu-test

Auditor: independent F5 re-audit child `st_01a1796f` (deep-low; no executor lane audited its own work).
Date: 2026-10-26 UTC (2026-10-27 AEDT). Branch `sycl/dsv41-sycl-support`, HEAD `aa09dc0707f20d492b595bc42fc0f6fcfa421f1d`, pushed tip `fork/sycl/dsv41-sycl-support` equal.
Plan: `.omo/plans/dsv41-sycl-dpct-and-xpu-test.md`. Prior audit: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/cleanup-audit.md` (VERDICT FAIL on rows 18/F2 artifacts; both rows are `[~]` rows naming gaps now).
Scope: re-audit F5 cleanup at the HEAD above. Nothing deleted; no product file edited; no monitor used.
Receipt format: every RC is printed un-piped directly above the command that produced it (no pipeline between the command and its exit code).

## Check 1 - lane worktrees and scratch trees: PASS

```
WORKTREE_LIST_RC=0
cd /home/maxious/Strata && git worktree list
```

Output, exactly one line: `/home/maxious/Strata aa09dc07 [sycl/dsv41-sycl-support]`. No lane worktree survives.

The `node` CLI is absent on this host, recorded once as expected:

```
NODE_EXISTS_RC=127
node -e "require('node:fs').existsSync('/tmp')"
```

Output: `/bin/bash: line 1: node: command not found`. The path checks were therefore run in the JS kernel with `node:fs` (JS kernel result lines, no shell RC applies):

```
existsSync('/home/maxious/Strata-wt/dpct-notes') === false -> true
existsSync('/home/maxious/Strata-wt/dsv41-dpct-w6c') === false -> true
existsSync('/home/maxious/Strata-wt/dsv41-dpct-w9') === false -> true
existsSync('/home/maxious/Strata-wt/row14') === false -> true
existsSync('/home/maxious/Strata-wt/row15') === false -> true
existsSync('/home/maxious/Strata-wt/row11') === false -> true
existsSync('/home/maxious/Strata-wt/row11fix') === false -> true
existsSync('/home/maxious/Strata-wt/row11tags') === false -> true
existsSync('/home/maxious/Strata-wt/row11fix2') === false -> true
existsSync('/tmp/f1_wt') === false -> true
```

All 10 paths are absent. Check 1: confirmed.

## Check 2 - scratch build dirs absent: PASS

```
LS_R15V_RC=2
ls /tmp/r15v_build
LS_R16V5_RC=2
ls /tmp/r16v5_build
LS_R16V6_RC=2
ls /tmp/r16v6_build
LS_R18V_RC=2
ls /tmp/r18v_build
LS_F1_DEF_RC=2
ls /tmp/f1_def_build
LS_F1_SYCL_RC=2
ls /tmp/f1_sycl_build
LS_F1_DEF_TESTS_RC=2
ls /tmp/f1_def_tests
LS_R14GATE_RC=2
ls /dev/shm/r14gate_build
LS_F3TC_RC=2
ls /tmp/f3tc_build
```

Each `ls` printed `ls: cannot access '<path>': No such file or directory` and exited 2. All 9 scratch dirs are gone. Check 2: confirmed.

## Check 3 - no process holding a card: PASS

```
PGREP_DSV41_RC=1
pgrep -c strata-dsv41
PGREP_STRATA_RC=1
pgrep -c strata
```

Both printed count `0` and exited 1 (no match). No `strata`/`strata-dsv41` process is holding a card. Check 3: confirmed.

## Check 4 - no live watcher: PASS

```
PGREP_TAIL_RC=0
pgrep -af "tail -n 0 -F"
```

Output, exactly one line: `3448908 /bin/bash -c pgrep -af "tail -n 0 -F" echo PGREP_TAIL_RC=$?`. Read in full, that line is this probe's own wrapper shell (its command text carries the pattern); it is not a `tail` process. No live watcher survives. Check 4: confirmed.

## Check 5 - push receipt and repository state: PASS

```
GIT_HEAD_RC=0
git rev-parse HEAD
GIT_FORK_TIP_RC=0
git rev-parse fork/sycl/dsv41-sycl-support
```

Both printed `aa09dc0707f20d492b595bc42fc0f6fcfa421f1d`, equal to the dispatched HEAD.

```
GIT_STATUS_RC=0
git status --porcelain
```

Output, 11 lines, every one untracked: `?? .omo/` and `?? bench/results/2026-10-22-gyro-coder-carveout-p2p/`, `?? bench/results/2026-10-22-gyro-coder-carveout/`, `?? bench/results/2026-10-22-gyro-coder-context/`, `?? bench/results/2026-10-22-gyro-coder-single-3072-carveout-gated/`, `?? bench/results/2026-10-22-gyro-coder-single-3072-carveout/`, `?? bench/results/2026-10-22-gyro-m-dual-3072-carveout-gated/`, `?? bench/results/2026-10-22-gyro-m-dual-3072-carveout/`, `?? bench/results/2026-10-22-gyro-m-dual-3072-host/`, `?? bench/results/2026-10-22-gyro-sycl-2xArcProB60/`, `?? bench/results/2026-10-23-023-regression/`. Zero tracked changes; only the expected untracked `.omo/` and `bench/results` roots.

```
GIT_DIFF_QUIET_RC=0
git diff --quiet
GIT_DIFF_CACHED_QUIET_RC=0
git diff --cached --quiet
```

Both confirm zero tracked modifications, staged and unstaged. Check 5: confirmed.

## Check 6 - evidence completeness for the 13 `[x]` rows: PASS

Method (per the dispatch correction recorded in ledger L449): for each `[x]` row, name the artifact under `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/` and the verdict evidence - either a `VERDICT confirmed` / `VERDICT: PASS` / explicit `APPROVE` line anywhere in the artifact, or the ledger line quoting that reviewer verdict. File-tail position is not decisive. A lane `DONECLAIM` is never counted as a verdict; every verdict below comes from a gate reviewer other than the executor lane.

Plan inventory at HEAD (JS regex over the plan text): `1=[x] 2=[x] 3=[x] 4=[x] 5=[x] 6=[x] 7=[x] 8=[~] 9=[x] 10=[~] 11=[ ] 12=[x] 13=[x] 14=[x] 15=[x] 16=[x] 17=[~] 18=[~] 19=[~] F1=[ ] F2=[~] F3=[~] F4=[~] F5=[ ]`. The `[x]` inventory is 13 rows: 1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16. Rows 11, F1 and F5 are open and reported as pending below, not counted. Rows 18 and F2 are `[~]` rows naming their gaps with last evidence `row18-gate-review.md` and `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md`, so the `[x]`-row artifact rule no longer applies to them.

Artifact receipts (each RC un-piped directly above its command):

```
ART_task_1_dsv41_sycl_dpct_and_xpu_test_log_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-1-dsv41-sycl-dpct-and-xpu-test.log
ART_task_2_dsv41_sycl_dpct_and_xpu_test_log_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-2-dsv41-sycl-dpct-and-xpu-test.log
ART_task_3_dsv41_sycl_dpct_and_xpu_test_log_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-3-dsv41-sycl-dpct-and-xpu-test.log
ART_task_4_dsv41_sycl_dpct_and_xpu_test_log_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-4-dsv41-sycl-dpct-and-xpu-test.log
ART_task_5_dsv41_sycl_dpct_and_xpu_test_log_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-5-dsv41-sycl-dpct-and-xpu-test.log
ART_task_6_dsv41_sycl_dpct_and_xpu_test_log_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-6-dsv41-sycl-dpct-and-xpu-test.log
ART_task_7_dsv41_sycl_dpct_and_xpu_test_log_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-7-dsv41-sycl-dpct-and-xpu-test.log
ART_task_9_dsv41_sycl_dpct_and_xpu_test_log_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-9-dsv41-sycl-dpct-and-xpu-test.log
ART_todo5_gate_review_md_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo5-gate-review.md
ART_todo6b_gate_review_md_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo6b-gate-review.md
ART_todo6c_gate_review_md_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo6c-gate-review.md
ART_todo7b_gate_review_md_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo7b-gate-review.md
ART_todo9_gate_review_md_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo9-gate-review.md
ART_row13_gate_review_md_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row13-gate-review.md
ART_row14_gate_review_md_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row14-gate-review.md
ART_row15_gate_review_md_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row15-gate-review.md
ART_row16d_gate_review_md_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row16d-gate-review.md
ART_row18_gate_review_md_RC=0
ls /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row18-gate-review.md
```

Row-by-row naming (artifact under the evidence dir; verdict evidence beside it):

| Row | Named artifact | Verdict evidence (gate reviewer, never a lane DONECLAIM) | Result |
| --- | --- | --- | --- |
| 1 | `task-1-dsv41-sycl-dpct-and-xpu-test.log` | ledger L256 `"verdict":"confirmed"` (gate reviewer `st_01a1774e`) and L277 quoting `gate review st_01a1774e VERDICT confirmed (APPROVE, no blockers)` | confirmed |
| 2 | `task-2-dsv41-sycl-dpct-and-xpu-test.log` | ledger L278 quoting `gate review st_01a1777c VERDICT confirmed, 8 gates PASS` | confirmed |
| 3 | `task-3-dsv41-sycl-dpct-and-xpu-test.log` | ledger L288 quoting `gate reviewer VERDICT confirmed, 5 gates PASS` (review artifact `dsv41-sycl-dpct-and-xpu-test-todo3-gate-review.md`) | confirmed |
| 4 | `task-4-dsv41-sycl-dpct-and-xpu-test.log` | `todo9-gate-review.md:15`/`:60` `todo 4 - APPROVE (gap closed)`; ledger L373 quoting `gate reviewer st_01a178b9 APPROVE both checkboxes`, L374 `todo 4 -> [x] ... after gate reviewer st_01a178b9 APPROVE` | confirmed |
| 5 | `task-5-dsv41-sycl-dpct-and-xpu-test.log` + `todo5-gate-review.md` | `todo5-gate-review.md:3-4` `APPROVE` / `VERDICT: VERDICT PASS`; ledger L310 `gate_reviewer: st_01a177e4 VERDICT PASS` | confirmed |
| 6 | `task-6-dsv41-sycl-dpct-and-xpu-test.log` + `todo6b-gate-review.md`, `todo6c-gate-review.md` | `todo6b-gate-review.md:3` and `todo6c-gate-review.md:3` `APPROVE`; ledger L355 `gate review st_01a1788a APPROVE - todo 6 lands [x]`, L367 `gate reviewer st_01a178ae APPROVE ... todo 6 [x] correct on 28bd2aaa` | confirmed |
| 7 | `task-7-dsv41-sycl-dpct-and-xpu-test.log` + `todo7b-gate-review.md` | `todo7b-gate-review.md:235` `VERDICT confirmed: APPROVE`; ledger L398 `gate-verdict` quoting reviewer `st_01a178f8` re-running the merged build on `ce6218ee` | confirmed |
| 9 | `task-9-dsv41-sycl-dpct-and-xpu-test.log` + `todo9-gate-review.md` | `todo9-gate-review.md:12` `APPROVE - per checkbox`, `:27` `todo 9 - APPROVE`; ledger L373 quoting `gate reviewer st_01a178b9 APPROVE both checkboxes` | confirmed |
| 12 | resolves to todo 9's artifacts (row text: "Same work as todo 9's open gap; tracked there") - both named above exist in the dir | ledger L374 `row 12 -> [x] (tracked by todo 9)` under `st_01a178b9`'s APPROVE | confirmed |
| 13 | `row13-gate-review.md` | `row13-gate-review.md:3` `recommendation: APPROVE`; ledger L378 `gate reviewer st_01a178c0 APPROVE row 13 against primary evidence` | confirmed |
| 14 | `row14-gate-review.md` | `row14-gate-review.md:11` `APPROVE`; ledger L401 `gate-verdict` quoting reviewer `st_01a178f6` reproducing the gates on `ce6218ee` | confirmed |
| 15 | `row15-gate-review.md` | `row15-gate-review.md:114` (its final line) `VERDICT confirmed`; ledger L423 quoting the row-15 gate verdict `VERDICT confirmed` before `row 15 marked [x]` | confirmed |
| 16 | `row16d-gate-review.md` | `row16d-gate-review.md:11` `APPROVE`, `:24` `VERDICT confirmed` (reviewer `omo-native-gate-reviewer (st_01a1791a)`); ledger L420 `row-16 reviewer returned APPROVE with all four executed gates` | confirmed |

All 13 `[x]` rows name an artifact under `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/` and carry an independent gate verdict of `VERDICT confirmed` / `VERDICT: PASS` / explicit `APPROVE`, in the artifact or quoted in the ledger. No row rests on a lane `DONECLAIM` alone; no row is unverified.

Pending rows (not counted, per dispatch): row 11 `[ ]` - the docs commit `cd0b1c4e` landed and the branch is pushed, but `todo11-gate-review.md` and `todo11-gate-review-2.md` both end in a FAIL verdict and a third re-gate is in flight (ledger L449 names `st_01a1796c`), so row 11 is reported pending; F1 `[ ]` - `f1-plan-compliance.md` ends in a FAIL verdict over the then-`[x]` rows 18/F2, and the F1 re-audit is in flight (`st_01a17962`), reported pending; F5 `[ ]` - this very audit was in progress at check time and closes with this file, reported pending.

`[~]` rows carry named gaps and last evidence in the plan (8: `task-8-...log` + `todo8-gate-review.md`; 10 and 17: `regression-gates.md`; 18: `row18-gate-review.md`; 19: `dsv41-sycl-dpct-and-xpu-test-manual-qa.md`; F2: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md`; F3: `dsv41-sycl-dpct-and-xpu-test-manual-qa-two-card.md`; F4: `f4-fidelity.md`). Per the dispatch, rows 18 and F2 are `[~]` rows naming gaps with last evidence `row18-gate-review.md` and `dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md`, so the `[x]`-row artifact rule no longer applies to them.

Check 6: confirmed.

## Check 7 - ledger integrity: PASS

```
LEDGER_WC_RC=0
wc -l .omo/ulw-execute/ledger.jsonl
```

Output: `449 /home/maxious/Strata/.omo/ulw-execute/ledger.jsonl`. The JS kernel loop then parsed every line with `JSON.parse` (the orchestrator appends to this file live, so the count is a point-in-time receipt):

```
LEDGER_JSON_LOOP: non-empty lines=449 parse-failures=0
```

Every one of the 449 lines parses as JSON. Check 7: confirmed.

## Outside the enumerated checks: run-root temp dir

```
LS_DPCT_RUN_RC=0
ls /tmp/dpct_run
```

`/tmp/dpct_run` still exists. It is reported as **pending orchestrator cleanup**, not as a failure: it is the live ledger artifact dir (the ledger names `/tmp/dpct_run/*.txt` artifacts) and the orchestrator's remaining lanes (todo 11 re-gate, F1 re-audit) are still running at this audit's close. The dispatch directed exactly this treatment.

## Summary before the verdict

Check 1: one worktree (`/home/maxious/Strata` only), 10 named scratch trees absent per `node:fs`, `NODE_EXISTS_RC=127` recorded once. Check 2: all 9 scratch dirs `ls` exit 2. Check 3: `pgrep -c` exit 1, count 0, for both names. Check 4: the only `tail -n 0 -F` match is the probe's own wrapper shell. Check 5: HEAD, fork tip and the dispatched SHA are one commit, and the tree carries zero tracked changes. Check 6: all 13 `[x]` rows name an in-dir artifact and an independent gate verdict; rows 11, F1 and F5 are pending and uncounted; rows 18 and F2 are `[~]` gap rows outside the rule. Check 7: 449 ledger lines, 0 JSON parse failures. `/tmp/dpct_run` is pending orchestrator cleanup as directed. Every check of the re-audit dispatch is confirmed.

VERDICT confirmed
