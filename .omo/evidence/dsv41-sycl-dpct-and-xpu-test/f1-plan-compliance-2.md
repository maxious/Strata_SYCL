# F1 plan-compliance re-audit - dsv41-sycl-dpct-and-xpu-test

Auditor: independent F1 child lane `st_01a17962` (deep-low; no executor lane reviewed its own work here).
Date: 2026-10-26/27. Branch `sycl/dsv41-sycl-support`, audited HEAD `aa09dc0707f20d492b595bc42fc0f6fcfa421f1d`, base `18a27e02`, 15 commits.
Plan: `.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (plan state final for this audit: 24 checkbox rows).
This supersedes `f1-plan-compliance.md` (VERDICT FAIL at `d337362b`), whose only open findings were rows 18 and F2; those are now `[~]` gap rows citing `row18-gate-review.md` and are re-audited below.

Mid-audit note: the audit began at `a53335fd`; the report-reword cherry-pick landed as `aa09dc07` and was pushed to `fork` while this audit ran. Every range-dependent receipt was re-run at `aa09dc07` and the values below are the re-run values. First-pass receipts at `a53335fd` are kept in the appendix so every RC this audit produced is on record.

Convention: each `*_RC=` line sits directly above the command whose exit code it is. Every RC is the un-piped exit code of that command (`rc=$?` immediately after it), never a pipe's exit code.

## Check 0 - bounded one-shot wait for the row-18 gate review

GATE_WAIT_RC=0
timeout 900 bash -c 'until [ -s /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row18-gate-review.md ]; do sleep 5; done; printf GATE_READY\n'

The gate printed its ready sentinel (the file was already non-empty when the wait started). The review is complete: 297 lines, and its final line reads `VERDICT FAIL`. Row-18 receipts:

ROW18_WC_RC=0
wc -l /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row18-gate-review.md

ROW18_TAIL_RC=0
tail -n 1 /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row18-gate-review.md

`row18-gate-review.md` is cited as rows 18/F2's last evidence (their gap text names it). Its verdict is the independent verdict of record for those two rows: the AOT half of the F2 closure is confirmed (`AOT_CONFIGURE_RC=0`, `AOT_BUILD_RC=0`, [260/260], zero FAILED), while the literal byte-identity claim is false at HEAD (`DIFF_RC=1`: 169 of 175 baseline rows byte-identical, six set-tile tallies reclassified exact <-> within-one-BF16-ulp under AOT codegen with identical outside counts, and commit `4e003d34` added the GEMV rows after todo 1's baseline).

## Check 1 - checkbox state, JS line-by-line recount

RECOUNT_RC=0
bun /tmp/f1_recount.js

Method: `/tmp/f1_recount.js` walks the plan line by line, collects every `- [.]` row under `## Todos` and `## Final verification wave`, and per row reports the checkbox state, the number of `gap:` clauses, and whether each gap clause names the row's last evidence (`Last evidence` text or an `.omo/evidence/` path). It reports only what the file says.

Counts (final plan state):

| Quantity | Value |
| --- | --- |
| rows in scope | 24 |
| `[x]` | 15 (rows 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, F5) |
| `[~]` | 8 (rows 8, 10, 17, 18, 19, F2, F3, F4) |
| open `[ ]` | 1 (F1 - this audit's own row, flipped on this verdict) |
| rows carrying a `gap:` clause | 9 (all 8 `[~]` rows plus row 7 `[x]`) |
| gap clauses not naming last evidence | none (`GAP_CLAUSE_NOT_NAMING_EVIDENCE (none)`) |
| rows with no `gap:` clause | 1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16, F5 (all `[x]`) |

Every `[~]` row names its gap and its last evidence (the plan's F1 criterion: "each `[~]` names its gap and its last evidence"); the gap clause is the `[~]` state's requirement, so the `[x]` rows listed above carry none (row 7 additionally documents its residual gap). Every row is `[x]` or `[~]`-with-gap, with only F1 itself open, which is this audit row. Check 1 result: PASS.

## Check 2 - Must NOT have guardrails over 18a27e02..HEAD

All receipts below ran at `aa09dc07`.

GITLOG_NAMES_RC=0
git log --format= --name-only 18a27e02..HEAD > /tmp/f1_names.txt

GGML_SYCL_PATHS_RC=1
grep -c -E 'ggml[-_]?sycl' /tmp/f1_names.txt

(grep prints 0 and exits 1 on zero matches - no commit touches `ggml_sycl` sources, so no `ggml_sycl` build is in the range.)

SCOPE_PATHS_RC=1
grep -v -c -E '^(sycl/|docs/|$)' /tmp/f1_names.txt

(count 0: every committed path is under `sycl/` or `docs/`.)

OMO_BENCH_PATHS_RC=1
grep -c -E '^(\.omo/|bench/results)' /tmp/f1_names.txt

(count 0: no committed `.omo/` or `bench/results` path.)

SUBJECTS_FILE_RC=0
git log --format=%s 18a27e02..HEAD > /tmp/f1_subjects.txt

SUBJECT_PREFIX_RC=1
grep -v -c -E '^(\[Engine\]|\[Docs\])' /tmp/f1_subjects.txt

(count 0: all 15 subjects carry `[Engine]` (11) or `[Docs]` (4).)

SYCL_DIFF_RC=0
git diff 18a27e02..HEAD -- sycl/ > /tmp/f1_sycl.diff

Barrier statements, added side extracted and removed side extracted, then compared:

BARRIER_ADD_EXTRACT_RC=0
awk '/^[+]/ && /barrier[[:space:]]*\(/ && /;[[:space:]]*$/ {sub(/^\+/,""); print}' /tmp/f1_sycl.diff > /tmp/f1_bar_added.txt

BARRIER_DEL_EXTRACT_RC=0
awk '/^[-]/ && /barrier[[:space:]]*\(/ && /;[[:space:]]*$/ {sub(/^-/,""); print}' /tmp/f1_sycl.diff > /tmp/f1_bar_removed.txt

BARRIER_STMT_DIFF_RC=1
diff /tmp/f1_bar_added.txt /tmp/f1_bar_removed.txt

The statement lists differ only by (a) movement - commit `83417a6f` moved the hyper-connect/sparse-attn kernels from `ds4_kernels.dp.cpp` to `ds4_attn_hc.dp.cpp`, so their unchanged calls appear on both sides of the file split - and (b) the todo-9 two-card hand-off rewrite (`ddc3c9e7`, `ext_oneapi_submit_barrier`/`dpct::sync_barrier` form changed), which is that todo's sanctioned measured fix, not a barrier-scope change. No barrier statement changed scope: every `sycl::group_barrier(...)` line is text-identical on both sides and keeps `get_sub_group()`; every `nd_item::barrier()` call remains the bare `item_ct1.barrier()` / `it.barrier()` form with no `fence_space` argument in code (`fence_space` occurs only inside DPCT comment bodies and the accepted audit notes). The named DPCT1065 sites keep their full barriers, exactly as todo 2's notes record.

OCCUPANCY_CHANGED_RC=1
grep -c -E '^[+-].*(occupancy|reqd_work_group_size)' /tmp/f1_sycl.diff

(count 0: no changed line touches occupancy or work-group-size logic.)

N2_CHANGED_RC=0
grep -c -E '^[+-].*(n = 2)' /tmp/f1_sycl.diff

SCOPE_TOKEN_LINES_RC=0
grep -n -E '^[+-].*(fence_space|group_barrier|get_sub_group|get_group|n = 2)' /tmp/f1_sycl.diff

The three `n = 2` diff hits are the DPCT1007 triage-table row, the DPCT1007 accepted audit note, and one kernel-index expression unrelated to occupancy; the DPCT1007 occupancy site itself (`exl3.dp.cpp:1951-1953`, hardcoded `n = 2`) is unchanged code. No occupancy rewrite.

Check 2 result: PASS - no `ggml_sycl` build or source, no barrier-scope conversion, no occupancy rewrite, zero committed `.omo/`/`bench/results` paths, all subjects prefixed, all paths `sycl/` or `docs/`.

## Check 3 - implementation todos' commits

COMMIT_STATS_RC=0
git log --format='%h|%s' --stat 18a27e02..HEAD

Mapping from the commit subjects (15 commits at `aa09dc07`):

| Commit | Subject | Maps to |
| --- | --- | --- |
| `ca0f6f94` | `[Engine] triage the DPCT conversion notes in the SYCL dsv41 port` | todo 2 (residual dpct advisories -> accepted notes; `sycl/docs/sycl-dsv41-dpct-triage.md`) |
| `b18f6c40` | `[Engine] fix the wrong DPCT1098 device read sites in the SYCL dsv41 port` | todo 3 |
| `2e5c53b9` | `[Engine] audit DPCT1093 queue binding and record the queue-creation table` | todo 4 |
| `433868e5` | `[Engine] settle the host->GPU signalling path and keep every wait bounded` | todo 5 |
| `a698bb84` | `[Engine] publish the routing ids the host reads` | todo 6 |
| `39b128ad` | `[Engine] bind --gpus ordinals to real GPU devices` | todo 4 follow-up (ordinal->device binding; the "ordinal fix" todo 9's gate cites) |
| `28bd2aaa` | `[Engine] subtract the K-quant min term in the GEMV` | todo 7 (the repeated-token root-cause fix; row 14's min-term family) |
| `ddc3c9e7` | `[Engine] avoid cross-context events in two-card hand-offs` | todo 9 (two-card DEVICE_LOST fix; row 12's chase) |
| `4e003d34` | `[Engine] cover the K-quant GEMV dot in the parity harness` | row 14 |
| `ce6218ee` | `[Engine] record an accepted audit note at every remaining dpct advisory block` | row 16 |
| `83417a6f` | `[Engine] give the hyper-connect and sparse-attn kernel copies one home` | row 15 |
| `d337362b` | `[Docs] dsv41 SYCL: measured report for the ported engine` | todo 8's measured record / todo 11's report |
| `74110114` | `[Docs] dsv41 SYCL: correct the pack identity and the test claim in the measured report` | todo 11 |
| `a53335fd` | `[Docs] dsv41 SYCL: tag every claim as measured, static, prior art, or not measured yet` | todo 11 |
| `aa09dc07` | `[Docs] dsv41 SYCL: split the test claim into passed, failed, and not run` | todo 11 (report reword, pushed tip) |

Todos 1 and 10 are `Commit: N` measurement/gate rows by the plan's own commit strategy ("Todos 1 and 10 are measurements and gates, so they commit nothing"), so they map to their evidence logs (`task-1-...log`, `task-10-...log`), not to commits. Todo 8's planned commit says "evidence log only if no code change"; no todo-8 code change was needed because the defect's fix landed as `28bd2aaa`, and the measured record landed as `d337362b` plus `task-8-...log`. Every commit-bearing todo (2-7, 9, 11) has its commit; every commit is `[Engine]`/`[Docs]` prefixed and touches only `sycl/` or `docs/`. Check 3 result: PASS.

## Check 4 - independent verdicts for the [x] rows

Rule applied (per the dispatch correction recorded in ledger L449): a row is verified by a `VERDICT confirmed` / `VERDICT: PASS` / explicit `APPROVE` line anywhere in its artifact, or by the ledger checkpoint line quoting that reviewer verdict; a lane `DONECLAIM` is never a verdict. File-tail position is not decisive. Verdict lines below were pulled by `grep -Hn` from the artifacts and from `.omo/ulw-execute/ledger.jsonl`.

ARTIFACT_VERDICT_COUNTS_RC=0
grep -c -E 'VERDICT confirmed|VERDICT: PASS|APPROVE' /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/*.md

SIBLING_VERDICT_COUNTS_RC=0
grep -c -E 'VERDICT confirmed|VERDICT: PASS|APPROVE' /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test*.md

ARTIFACT_VERDICT_LINES_RC=0
grep -Hn -E 'VERDICT confirmed|VERDICT: PASS|APPROVE' <row13 row14 row15 row16d todo5 todo6b todo6c todo7b todo9 gate-review files>

SIBLING_VERDICT_LINES_RC=0
grep -Hn -E 'VERDICT confirmed|VERDICT: PASS|APPROVE' <todo1 todo2 todo3 todo4 sibling gate-review files>

LEDGER_VERDICT_LINES_RC=0
awk '/VERDICT confirmed|VERDICT.: PASS|APPROVE|REJECT/ {print NR ": " substr($0,1,300)}' /home/maxious/Strata/.omo/ulw-execute/ledger.jsonl

ROW11_F5_VERDICT_LINES_RC=0
grep -Hn -E 'VERDICT confirmed|VERDICT: PASS|APPROVE|REJECT' /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo11-gate-review-3.md /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/cleanup-audit-2.md

Per-row verdict evidence for all 15 `[x]` rows:

| Row | Artifact under `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/` | Independent reviewer verdict (in-file or ledger-quoted) |
| --- | --- | --- |
| 1 | `task-1-dsv41-sycl-dpct-and-xpu-test.log` | ledger L256 `"verdict":"confirmed"` and L277 quoting `gate review st_01a1774e VERDICT confirmed (APPROVE, no blockers)`; sibling `dsv41-sycl-dpct-and-xpu-test-gate-review.md:3` `recommendation: APPROVE` |
| 2 | `task-2-dsv41-sycl-dpct-and-xpu-test.log` | ledger L278 quoting `gate review st_01a1777c VERDICT confirmed, 8 gates PASS`; sibling `...-todo2-gate-review.md:3` `recommendation: APPROVE` |
| 3 | `task-3-dsv41-sycl-dpct-and-xpu-test.log` | ledger L288 quoting `gate reviewer VERDICT confirmed, 5 gates PASS`; sibling `...-todo3-gate-review.md:9/:94` `APPROVE - all five gates PASS` |
| 4 | `task-4-dsv41-sycl-dpct-and-xpu-test.log` | `todo9-gate-review.md:15/:60` `todo 4 - APPROVE (gap closed)`; ledger L373 `gate reviewer st_01a178b9 APPROVE both checkboxes` |
| 5 | `task-5-...log` + `todo5-gate-review.md` | `todo5-gate-review.md:3` `Recommendation: APPROVE`; ledger L310 `gate_reviewer st_01a177e4 VERDICT PASS` |
| 6 | `task-6-...log` + `todo6b-gate-review.md`, `todo6c-gate-review.md` | `todo6b-gate-review.md:3`, `todo6c-gate-review.md:3` `recommendation: APPROVE`; ledger L355 `gate review st_01a1788a APPROVE - todo 6 lands [x]`, L367 `gate reviewer st_01a178ae APPROVE` |
| 7 | `task-7-...log` + `todo7b-gate-review.md` | `todo7b-gate-review.md:235` `VERDICT confirmed: APPROVE`; ledger L398 `gate-verdict` reviewer `st_01a178f8` |
| 9 | `task-9-...log` + `todo9-gate-review.md` | `todo9-gate-review.md:12/:27` `APPROVE`; ledger L373 `gate reviewer st_01a178b9 APPROVE both checkboxes` |
| 11 | `todo11-gate-review-3.md` | `todo11-gate-review-3.md:413` `VERDICT confirmed` (final line), at HEAD `aa09dc07` |
| 12 | resolves to todo 9's artifacts ("tracked there") | ledger L374 `row 12 -> [x] (tracked by todo 9)` under `st_01a178b9`'s APPROVE |
| 13 | `row13-gate-review.md` | `row13-gate-review.md:3` `recommendation: APPROVE`; ledger L378 `gate reviewer st_01a178c0 APPROVE row 13` |
| 14 | `row14-gate-review.md` | `row14-gate-review.md:11` `APPROVE`; ledger L401 `gate-verdict` reviewer `st_01a178f6` |
| 15 | `row15-gate-review.md` | `row15-gate-review.md:114` `VERDICT confirmed` (also its final line); ledger L423 |
| 16 | `row16d-gate-review.md` | `row16d-gate-review.md:11` `APPROVE`, `:24` `VERDICT confirmed`; ledger L420 |
| F5 | `cleanup-audit-2.md` | `cleanup-audit-2.md:215` `VERDICT confirmed` (final line) |

No row rests on a lane `DONECLAIM` alone. Rows 18 and F2 are `[~]`, not `[x]`: their gap clauses name last evidence `row18-gate-review.md` (`VERDICT FAIL`, 297 lines) and the F2 receipts plus `dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md` (REJECT), so no confirmed verdict is required of them - they are compliant gap rows recording exactly what the independent review found. Cross-check: `cleanup-audit-2.md`'s per-row verdict table (its lines 165-179) agrees with the independent greps above. Check 4 result: PASS.

## Check 5 - gates characterization in the plan's rows

Source of record: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates.md` (final line `VERDICT confirmed`). Its recorded facts and where the plan rows state them:

1. `DEF_CTEST_RC=8` over 57 tests with the sole failure `55 - isa_guard_check` on `iq512x_rows` (violation names `iq512x_rows`, `iq_avx512.cpp.o`), a pre-existing `tools/check_isa_guard.py` allow-list gap outside this plan's `sycl/`+`docs/` scope (row 17). Stated honestly and verbatim by rows 10 and 17 and by F3's gap text (including the origin/main `4b9e05a2` control with `FAILDIFF_RC=0`).
2. `SYCL_CTEST_RC=0` naming `ds4_parity` and `ds4_blas_parity` (2/2, 100% tests passed). Stated honestly by rows 14, 15 and 16 (`SYCL_CTEST_RC=0` naming both suites with their pass times).
3. `SYCL_CTEST_RC_LITERAL=1` because `build-sycl-dsv41/sycl` does not exist under `cmake -S sycl` (regression-gates.md section 3: `SYCL_TESTDIR_LS_RC=2`, tests register at the build root; `row18-gate-review.md` Gate 4 records the same as `LITERAL_CTEST_RC=1` / `LS_SYCL_SUBDIR_RC=2`). This audit re-checked it live:

SYCL_SUBDIR_LS_RC=2
ls -d /home/maxious/Strata/build-sycl-dsv41/sycl

The literal-path cause is fully characterized in the evidence record. The plan rows quote the `<build>/sycl` acceptance form (row 13, F2's plan-wording note, verification-strategy line 67); that form is the correct target for the root-configured executor trees - `/tmp/dpct_run/f2_configure3.txt` shows one configure emitting both the root `CMakeLists.txt:19` `STRATA_ENABLE_SYCL` warning and `sycl/CMakeLists.txt:199`, so tests register under `<build>/sycl` there - while the plan's literal `cmake -S sycl` gate registers them at the build root. No plan row mis-states a gate outcome (no row claims the literal command passes and no row hides `DEF_CTEST_RC=8`); the path-form wording is frame-mixed and is recorded here as a wording caveat, matching the note `row13-gate-review.md` itself carries. Check 5 result: PASS on the recorded outcomes, with the path-form wording caveat recorded.

## Check 6 - push receipt

PUSH_HEAD_RC=0
git rev-parse HEAD

PUSH_FORK_RC=0
git rev-parse fork/sycl/dsv41-sycl-support

Both print `aa09dc0707f20d492b595bc42fc0f6fcfa421f1d` (re-checked again at final state with the same two commands and equal values). The pushed tip's own receipt:

NEW_COMMIT_RC=0
git show --name-only --format='%H%n%s' aa09dc0707f20d492b595bc42fc0f6fcfa421f1d

Subject `[Docs] dsv41 SYCL: split the test claim into passed, failed, and not run`; the only path is `sycl/docs/sycl-dsv41-measured-report.md`. Check 6 result: PASS - `git rev-parse HEAD` equals `git rev-parse fork/sycl/dsv41-sycl-support`.

## Summary

Check 1 (recount): 24 rows - 15 `[x]`, 8 `[~]` each naming its gap and last evidence, 1 open (F1 itself). PASS.
Check 2 (guardrails): no `ggml_sycl` build, no barrier-scope conversion, no occupancy rewrite, zero committed `.omo/`/`bench/results` paths, every subject `[Engine]`/`[Docs]`, every path `sycl/` or `docs/`. PASS.
Check 3 (commits): every implementation commit maps to its todo/row from its subject; todos 1 and 10 are `Commit: N` rows by the plan's own strategy; all prefixes and scopes hold. PASS.
Check 4 (independent verdicts): every `[x]` row carries a gate-reviewer verdict in its artifact or quoted in the ledger; rows 18/F2 are compliant `[~]` gap rows whose last evidence is `row18-gate-review.md` (`VERDICT FAIL` on the stale byte-identity claim, recorded as their gap). PASS.
Check 5 (gates characterization): the plan rows state `DEF_CTEST_RC=8` with the sole out-of-scope `isa_guard_check`/`iq512x_rows` failure and `SYCL_CTEST_RC=0` naming `ds4_parity` and `ds4_blas_parity` exactly as `regression-gates.md` records; the `SYCL_CTEST_RC_LITERAL=1` path cause is recorded in the evidence, with the plan's `<build>/sycl` acceptance form noted above as a non-blocking wording caveat. PASS (caveat recorded).
Check 6 (push receipt): HEAD `aa09dc07` equals the `fork/sycl/dsv41-sycl-support` tip. PASS.

Known gaps carried by `[~]` rows (out of this audit's scope, named for the record): row 17's upstream `iq512x_rows` allow-list fix; rows 18/F2's AOT-vs-JIT byte-identity wording; row 19's unanswered `src/dsv41/ds4_main.cpp` scope question; F3's loud-spin control exiting 0 instead of 1 and its A770 label; F4's IS-1/IS-5/IS-7/IS-8 partials.

## Appendix - first-pass receipts at the starting HEAD a53335fd (superseded by the mid-audit HEAD move; range checks re-run at aa09dc07 above)

START_HEAD_RC=0
git rev-parse HEAD

START_BRANCH_RC=0
git rev-parse --abbrev-ref HEAD

START_REMOTES_RC=0
git remote -v

START_FORK_REF_RC=0
git rev-parse fork/sycl/dsv41-sycl-support

START_HEAD_MATCH_RC=0
git rev-parse a53335fd598e5b9e976d830faa23badc1192a924

START_EVIDENCE_LS_RC=0
ls -la /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/

START_SIBLINGS_LS_RC=0
ls -la /home/maxious/Strata/.omo/evidence/

START_LASTLINES_RC=0
for f in /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/*.md /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/*.log; do awk -v n="${f##*/}" 'NF{l=$0} END{print n "\t" l}' "$f"; done

START_GITLOG_RC=0
git log --format='%H %s' 18a27e02..HEAD

START_NAMES_RC=0
git log --format= --name-only 18a27e02..HEAD > /tmp/f1_names.txt

START_GGML_RC=1
grep -c -E 'ggml[-_]?sycl' /tmp/f1_names.txt

START_SCOPE_RC=1
grep -v -c -E '^(sycl/|docs/|$)' /tmp/f1_names.txt

START_OMO_RC=1
grep -c -E '^(\.omo/|bench/results)' /tmp/f1_names.txt

START_SUBJECTS_RC=0
git log --format=%s 18a27e02..HEAD > /tmp/f1_subjects.txt

START_SUBJECT_PREFIX_RC=1
grep -v -c -E '^(\[Engine\]|\[Docs\])' /tmp/f1_subjects.txt

START_BUN_RC=0
command -v bun

START_NODE_RC=1
command -v node

START_TAILS_RC=0
for f in <row13 row14 row15 row16d todo5 todo6b todo6c todo7b todo9 gate-review files>; do echo "=== ${f##*/} ==="; tail -n 12 "$f"; done

START_SYCL_DIFF_RC=0
git diff 18a27e02..HEAD -- sycl/ > /tmp/f1_sycl.diff

START_BARRIER_COUNT_RC=0
grep -c -E '^[+-].*[Bb]arrier' /tmp/f1_sycl.diff

START_BARRIER_LINES_RC=0
grep -n -E '^[+-].*[Bb]arrier' /tmp/f1_sycl.diff

START_OCC_COUNT_RC=0
grep -c -E '^[+-].*(occupancy|reqd_work_group_size|DPCT1007|DPCT1049|DPCT1110)' /tmp/f1_sycl.diff

START_OCC_LINES_RC=0
grep -n -E '^[+-].*(occupancy|reqd_work_group_size|DPCT1007|DPCT1049|DPCT1110)' /tmp/f1_sycl.diff

START_BARRIER_CALL_COUNT_RC=0
grep -c -E '^[+-][^/]*barrier[[:space:]]*\(' /tmp/f1_sycl.diff

START_COMMIT_STATS_RC=0
git log --format='%h|%s' --stat 18a27e02..HEAD

Recount first pass (before the plan's final state; superseded by the Check 1 run):

START_RECOUNT_RC=0
bun /tmp/f1_recount.js

(First-pass counts were 13 `[x]` / 8 `[~]` / 3 open, before rows 11 and F5 closed.)

VERDICT confirmed
