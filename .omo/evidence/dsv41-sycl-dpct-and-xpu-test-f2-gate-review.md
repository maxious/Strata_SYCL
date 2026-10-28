# Gate Review: F2 — code quality review of the dsv41 SYCL port

- **recommendation:** REJECT
- **checkbox mark:** [~]
- **reviewer:** omo-native-gate-reviewer (st_01a1788c), final gate for F2
- **date:** 2026-10-26
- **plan audited:** `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` line 228 (F2) + Verification-strategy gate commands + Must-NOT guardrails
- **evidence lane:** st_01a17872 (attempt 4, `f2evidence`) — receipts under `/tmp/dpct_run/`

> Report path: this F2 review follows the F1 precedent and is written to a distinct path
> `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md` so it does not clobber the
> todo-1 gate review at `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-gate-review.md`.

## originalIntent

The user wants a final code-quality pass over the merged dsv41 SYCL port before the plan is
declared complete: the port must still compile under the Intel `icpx` compiler in both JIT mode
and AOT mode (`-DSTRATA_SYCL_AOT=bmg-g21`), the parity harnesses must still print rows that are
byte-identical to the todo-1 baseline (no numerical regression from the comment/audit/fix work),
the todo-2 conversion notes must each state a reason rather than a bare TODO, and the branch must
have introduced no new abstraction and no new or changed default.

## desiredOutcome

A gate that demonstrates all five F2 sub-criteria with un-piped command receipts:
(1) JIT build green under `icpx`; (2) AOT build green under `icpx` with `-DSTRATA_SYCL_AOT=bmg-g21`;
(3) the parity rows captured from `ds4_parity` / `ds4_blas_parity` and shown byte-identical to
todo 1's recorded baseline rows; (4) every todo-2 note carries a substantive reason (no bare
TODO); (5) `git diff 18a27e02..HEAD` touches only `sycl/` and `docs/` with no CMakeLists default
change and no unplanned abstraction.

## userOutcomeReview

Three of the five sub-criteria are demonstrated; two are not.

- **JIT under icpx — SATISFIED.** `/tmp/dpct_run/f2_configure3.txt` names
  `/opt/intel/oneapi/compiler/2026.2/bin/icpx` as the CXX compiler (`IntelLLVM 2026.2.0`) and
  `/opt/intel/oneapi/compiler/2026.2/bin/icx` as the C compiler. `/tmp/dpct_run/f2_build3.txt`
  reaches `[244/244]`, linking `sycl/ds4_blas_parity` at `[187/244]` and `sycl/ds4_parity` at
  `[221/244]`. `/tmp/dpct_run/f2_ctest3.txt` shows `100% tests passed, 0 tests failed out of 2`
  (`ds4_parity` Passed 0.31 s, `ds4_blas_parity` Passed 0.33 s). The F2_CONFIGURE_RC / F2_BUILD_RC /
  F2_CTEST_RC = 0 claims reproduce from the artifacts.

- **AOT under icpx — NOT SATISFIED (stated gap).** No AOT configure/build exists anywhere in the
  F2 receipts. `grep STRATA_SYCL_AOT|bmg` over `/tmp/dpct_run/f2_*.txt` returns nothing, and the
  lane's own ledger records the AOT variant as "the un-compiled AOT variant
  (-DSTRATA_SYCL_AOT=bmg-g21) as a stated gap". The F2 criterion explicitly requires "both JIT and
  AOT"; one half is missing, so this criterion is unmet.

- **Parity rows byte-identical to todo 1 baseline — NOT DEMONSTRATED.** The only parity artifact
  is `/tmp/dpct_run/f2_ctest3.txt`, which contains the ctest *summary* only (`Passed` lines and the
  `100% tests passed` banner). It does not contain the actual parity row output, and there is no
  byte-identical comparison against todo 1's baseline rows anywhere in the F2 receipts (no
  `DIFF_PARITY` gate). The lane's DONECLAIM prose asserts the run "naming Intel(R) Arc(TM) Pro B60
  Graphics and ending ds4_blas_parity OK" — but that string is NOT in the referenced
  `f2_ctest3.txt` (`grep Arc|MATCH|OK` over all `f2_ctest*.txt` returns nothing). Passing `ctest`
  proves no parity *failure*; it does not prove the stronger, separately-named "byte-identical to
  baseline" claim, and the artifact that would carry that claim is absent.

- **Todo-2 notes state a reason — SATISFIED.** Todo 2's shipped artifact is
  `sycl/docs/sycl-dsv41-dpct-triage.md` (commit `ca0f6f94`). Every one of its 30 DPCT-class rows
  carries a substantive reason cell (RESOLVED cites the resolving line; ACCEPTED_AS_IS says why a
  change is not wanted; NOT_APPLICABLE says why the CUDA idea does not apply; ACTIONABLE hands the
  class to a named owner todo). The only "todo" strings are owner pointers (`todo 3`, `todo 4`) on
  the two ACTIONABLE rows, not placeholders. The prior todo-2 gate review independently reproduced
  `NOTES_REASON_RC=0` for the same check.

- **No new abstraction or default — SATISFIED.** `git diff 18a27e02..HEAD --name-only` lists only
  `sycl/` and `docs/` paths and contains no `CMakeLists.txt` (top-level or `sycl/`). The
  `STRATA_SYCL_AOT` option and the two opt-in switches (`STRATA_ENABLE_SYCL`, `STRATA_SYCL_DSV41`,
  `STRATA_SYCL_DSV41_EXE`) are unchanged from `18a27e02`. The `sycl/include/dpct/device.hpp`
  additions are the plan-specified todo-4 ordinal fix (`detail::ordered_gpu_devices()`), and the
  `sycl/include/strata/sycl_doorbell.hpp` additions are the plan-specified todo-5 copy-engine
  hand-back (`dev_load`, `AnsRec`/`AnsSlot`/`SpinRec`/`SpinBuf`, `answer_spin_max`,
  `spin_record`). `dev_load` is a multi-use semantic alias for "read a copy-engine-written
  device-memory flag" (8 call sites across 5 files), not a single-use pass-through wrapper, and the
  structs are the actual data layout of the mechanism the plan names, not speculative indirection.

## Plan-wording question (task item 1) — a correction, not a gap

The plan's literal gate command `ctest --test-dir build-sycl-dsv41 -R ds4` is imprecise: with the
top-level configure path the lane used, `enable_testing()` and both `add_test(NAME ds4_parity /
ds4_blas_parity ...)` calls live inside `sycl/CMakeLists.txt` (lines 271 / 277 / 291), which
registers the tests in the `sycl` sub-directory of the build tree, so the correct ctest target is
`build-sycl-dsv41/sycl` (the lane used `/tmp/f2_sycl_build/sycl`). This is a **correction**, not a
criterion failure: the plan's own references cite `sycl/CMakeLists.txt:277` and `:291` and the
`:269` comment ("names carry the `ds4` prefix so `ctest -R ds4` selects them"), so the plan's
intent is to run those two tests, and the lane's corrected command genuinely runs them (2/2, named
in `f2_ctest3.txt`). I record it as a NOTE (plan wording should say `build-sycl-dsv41/sycl`), not
a blocker.

## Facts reproduced this session (not taken from the lane's prose)

- `git rev-parse HEAD` = `28bd2aaac830...` (branch `sycl/dsv41-sycl-support`); the task's stated
  "HEAD 39b128ad" is one commit behind — `39b128ad` (todo-4 ordinal merge) is an ancestor, and
  `28bd2aaa` adds the todo-8 K-quant GEMV sign fix (`sycl/src/dsv41/ds4_iq.dp.cpp`, 4 lines).
- `git log --oneline 18a27e02..HEAD` = 7 commits, all `[Engine]`-prefixed, all `sycl/`-only or
  `docs/`-only in scope.
- `git diff 18a27e02..HEAD --stat` = 12 files, +1038/-177, all under `sycl/`.
- `git diff 18a27e02..HEAD --name-only | grep -i cmake` = empty (no default changed).
- `git show ca0f6f94 --stat` = `sycl/docs/sycl-dsv41-dpct-triage.md | 106 +` (todo-2 note triage,
  single docs file, no source edit).
- `git show 2e5c53b9 --stat` = `sycl/docs/sycl-dsv41-queue-binding.md | 175 +` (the todo-4 audit
  table; note this is the queue-binding doc, NOT the todo-2 notes commit — the task's parenthetical
  mislabels `2e5c53b9` as "the notes commit"; the notes commit is `ca0f6f94`).
- `grep -rn "STRATA_SYCL_AOT|bmg" /tmp/dpct_run/f2_*.txt` = 0 lines (AOT never attempted).
- `cat /tmp/dpct_run/f2_ctest3.txt` = ctest summary only; `grep "Arc|MATCH|OK"` over
  `f2_ctest*.txt` = 0 lines (no parity row output captured).
- `sycl/CMakeLists.txt:271` `enable_testing()`, `:277` `add_test(NAME ds4_parity ...)`, `:291`
  `add_test(NAME ds4_blas_parity COMMAND ds4_blas_parity --selftest)` — confirms the sycl
  sub-project owns the ds4 tests.

## Skill-perspective check (remove-ai-slops / programming)

Both skills were loaded from `/home/maxious/.omo/binary-runtime/5.1.48/plugin/skills/` and applied
directly over the diff, tests, and production code relevant to this gate:

- **Tests.** The F2 lane added, removed, and modified no tests; it only ran the pre-existing
  `ds4_parity` / `ds4_blas_parity` harnesses. There are therefore no excessive, tautological,
  implementation-mirroring, deletion-only, or removal-verifying tests introduced. No test pins
  prose (none of the changed files are prose).
- **Production extraction / parsing / normalization.** The only new production code is the
  plan-named todo-4 and todo-5 machinery (`ordered_gpu_devices`, the doorbell hand-back structs and
  `dev_load`). These are the actual mechanism the plan specifies, not needless abstraction;
  `dev_load` has 8 call sites (not a one-off). Todo 2's parsing/analysis scripts (`analyze.py`,
  `verify_cites.py`, `enum_check.py`, `stats.py`) stayed in `/tmp/dpct_w2` and were removed at
  cleanup — nothing was committed, and the `ENUM_SUM_NEG_RC=1` control is a legitimate can-fail
  check, not a removal-verifying test.
- **Maintenance burden / false confidence.** The two residual DPCT1098/DPCT1093 classes are
  documented as "unconfirmed static analysis" and handed to named owner todos; no new default
  surface. The one false-confidence risk is the lane's prose claiming the ctest output "names
  Intel(R) Arc(TM) Pro B60 Graphics ... ds4_blas_parity OK" when the referenced `f2_ctest3.txt`
  contains only the summary — that overstatement is the parity-rows gap named in the blockers.

## checked artifact paths

- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (line 228 F2; Verification-strategy gate commands; Must-NOT guardrails; full read)
- `/tmp/dpct_run/f2_configure3.txt` (icpx/icx named, `IntelLLVM 2026.2.0`)
- `/tmp/dpct_run/f2_build3.txt` (`[244/244]`, `ds4_blas_parity` at 187/244, `ds4_parity` at 221/244)
- `/tmp/dpct_run/f2_ctest3.txt` (2/2 passed — summary only, no parity row output)
- `/tmp/dpct_run/f2_gates3.json`, `/tmp/dpct_run/f2_gates2.json`, `/tmp/dpct_run/f2_gates.json`
- `/tmp/dpct_run/ledger_f2a4g.jsonl`, `/tmp/dpct_run/ledger_f2a4.jsonl` (attempt-4 receipts, AOT gap, ctest-dir correction)
- `/home/maxious/Strata/sycl/docs/sycl-dsv41-dpct-triage.md` (todo-2 notes: reason per row, no bare TODO)
- `/home/maxious/Strata/sycl/CMakeLists.txt` (lines 271 / 277 / 291 — sycl sub-project owns the ds4 tests)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-2-dsv41-sycl-dpct-and-xpu-test.log` (todo-2 receipts)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test-todo2-gate-review.md` (corroborating APPROVE + N-5 DIFF_PARITY note)
- Git state: `18a27e02` (integration base), `39b128ad` (todo-4 ordinal merge), HEAD `28bd2aaa`; `git diff 18a27e02..HEAD --stat`; `git show ca0f6f94`; `git show 2e5c53b9`

## blockers

1. **violatedCriterion:** F2 — "the port still compiles under `icpx` for both JIT and AOT
   (`-DSTRATA_SYCL_AOT=bmg-g21`)". **evidencePointer:** `/tmp/dpct_run/f2_configure3.txt`,
   `/tmp/dpct_run/f2_build3.txt` (JIT only), `/tmp/dpct_run/ledger_f2a4g.jsonl` (records the AOT
   variant as a stated gap; no AOT receipt exists anywhere under `/tmp/dpct_run/f2_*`).
   Observation: only the JIT half of the criterion was exercised; the AOT build was never run, so
   the criterion is unmet by an explicit missing artifact.

2. **violatedCriterion:** F2 — "parity rows are byte-identical to todo 1's baseline".
   **evidencePointer:** `/tmp/dpct_run/f2_ctest3.txt` (contains only the ctest `Passed` summary and
   the `100% tests passed` banner; no parity row output and no byte-identical comparison against
   todo 1's baseline). Observation: the artifact that would prove the byte-identical claim is
   missing, and the lane's DONECLAIM prose ("naming Intel(R) Arc(TM) Pro B60 Graphics and ending
   ds4_blas_parity OK") is not supported by the referenced file (`grep Arc|MATCH|OK` = 0). Passing
   ctest proves no parity failure, not the stronger byte-identical-to-baseline claim.

## evidence gaps (non-blocker NOTES)

- **N-1 (plan-wording correction).** The plan's `ctest --test-dir build-sycl-dsv41 -R ds4` is
  imprecise; the correct target is `build-sycl-dsv41/sycl`. The lane's correction is sound and the
  intent (run the two ds4 tests) is met.
- **N-2 (task hint mislabel).** The task's parenthetical calls `git show 2e5c53b9` "the notes
  commit"; `2e5c53b9` is actually the todo-4 queue-binding doc. The todo-2 notes commit is
  `ca0f6f94`. Both were inspected; neither introduces a bare TODO or a new abstraction.
- **N-3 (branch HEAD moved).** The task names "HEAD 39b128ad"; the live HEAD is `28bd2aaa`
  (todo-8 K-quant sign fix, 4 lines in `sycl/src/dsv41/ds4_iq.dp.cpp`). That commit changes a
  parity-covered IQ path, which is why the byte-identical-parity check (blocker 2) is not
  trivially satisfied and must be measured, not assumed.
