# F1 plan-compliance audit - dsv41-sycl-dpct-and-xpu-test

Auditor: independent F1 child lane `st_01a17926` (deep-low, tool-driven; no executor lane reviewed its own work here).
Date: 2026-10-26. Branch `sycl/dsv41-sycl-support`, HEAD `d337362b`, base `18a27e02`, 12 commits.
Plan: `.omo/plans/dsv41-sycl-dpct-and-xpu-test.md`.
Audited checkboxes: only the `[x]` rows 1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 18, F2. Rows 8, 10, F3, F4 (`[~]`) and rows 11, 15, 16, 17, 19, F1, F5 (open) were not audited as final.

Method note (assumption, recorded per task rules): a row's "independent verdict of confirmed/APPROVE" is an entry in `.omo/ulw-execute/ledger.jsonl` where a gate reviewer (a child other than the executor lane) returns confirmed/APPROVE for that row's work; the entry may name the gate-review file as its artifact while the row names its task log. An executor lane's own `DONECLAIM`/`VERDICT` is a self-claim and does not count. Rows 12 and 18 name no artifact of their own ("tracked there"), so they resolve to the artifacts of the rows they defer to (todo 9 and F2 respectively).

## Check 1 - checkbox vs evidence

| Row | Claim (from the plan) | Named artifact under `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/` | Independent ledger verdict | Result |
| --- | --- | --- | --- | --- |
| 1 | Re-record the baseline: SYCL dsv41 exe built, parity and run rows captured (measurement). | `task-1-dsv41-sycl-dpct-and-xpu-test.log` (5,530,478 B) - exists | ledger L256 `checkbox` "verdict":"confirmed" (gate reviewer `st_01a1774e`; L277 "VERDICT confirmed (APPROVE, no blockers)") | PASS |
| 2 | Replace residual dpct advisory blocks with accepted one-line notes (commit `[Engine] triage the DPCT conversion notes in the SYCL dsv41 port`, `ca0f6f94`). | `task-2-dsv41-sycl-dpct-and-xpu-test.log` (18,996 B) - exists | ledger L278 `checkbox`: "gate review st_01a1777c VERDICT confirmed, 8 gates PASS" | PASS |
| 3 | Audit DPCT1098: every host-mapped payload read uncached (commit `[Engine] fix the wrong DPCT1098 device read sites in the SYCL dsv41 port`, `0857e37e`). | `task-3-dsv41-sycl-dpct-and-xpu-test.log` (26,882 B) - exists | ledger L288 `checkbox`: "gate reviewer VERDICT confirmed, 5 gates PASS" (artifact `dsv41-sycl-dpct-and-xpu-test-todo3-gate-review.md`) | PASS |
| 4 | Audit DPCT1093: every queue created on the thread that owns its device (commit `[Engine] audit DPCT1093 queue binding and record the queue-creation table`, `b07f91f0`). | `task-4-dsv41-sycl-dpct-and-xpu-test.log` (22,904 B) - exists | supersession chain in the ledger: L281 FAIL (`st_01a17784`), L296 FAIL (audit gates pass, two-card smoke unmet), L352 REJECT (`st_01a17886`, DEVICE_LOST wedge) - then L373 "gate reviewer st_01a178b9 APPROVE both checkboxes" covering todo 4 and todo 9 after the ordinal fix and the two-card fix landed; L374 marks the row | PASS |
| 5 | Move the answer payload off host memory: the copy-engine hand-back, every wait bounded (commit `[Engine] dsv41 SYCL: settle the host->GPU signalling path and keep every wait bounded`, `433868e5`). | `task-5-dsv41-sycl-dpct-and-xpu-test.log` (10,542 B) + `todo5-gate-review.md` - exist | ledger L310 `todo-complete`: "gate reviewer: APPROVE - VERDICT PASS, todo 5 lands [x]" | PASS |
| 6 | Make the host CPU lane contribute experts (commit `[Engine] dsv41 SYCL: the CPU lane computes...` / `a698bb84` family; default run `CPU lane: 132 of 1680 experts`). | `task-6-dsv41-sycl-dpct-and-xpu-test.log` (45,211 B) + `todo6-gate-review.md` (pre-fix FAIL), `todo6b-gate-review.md`, `todo6c-gate-review.md` - exist | ledger L315 REJECT superseded by L355 "gate review st_01a178a APPROVE - todo 6 lands [x]" and L367 "gate reviewer st_01a178ae APPROVE ... todo 6 [x] correct on 28bd2aaa" | PASS |
| 7 | Chase the degenerate token; verified no code change needed on merged HEAD `ce6218ee` (fix landed in `28bd2aaa`); head argmax gate against independent torch CPU reference. | `task-7-dsv41-sycl-dpct-and-xpu-test.log` (57,724 B), `todo7-gate-review.md`, `todo7b-gate-review.md` - exist | ledger L398 `gate-verdict`: "VERDICT confirmed: APPROVE" (reviewer `st_01a178f8`, artifact `todo7b-gate-review.md`); earlier L323 APPROVE landed the row as `[~]` before the merged-build re-verification | PASS |
| 9 | Two-card run against the same-day single-card baseline (commit `[Engine] dsv41 SYCL: avoid cross-context events in two-card hand-offs`). | `task-9-dsv41-sycl-dpct-and-xpu-test.log` (17,592 B) + `todo9-gate-review.md` - exist | ledger L373 "gate reviewer st_01a178b9 APPROVE both checkboxes ... todo 9 all 12 RCs reproduced un-piped and adjacent" | PASS |
| 12 | Chase the two-card `UR_RESULT_ERROR_DEVICE_LOST` and land the two-card run against todo 8's same-day single-card baseline; "Same work as todo 9's open gap; tracked there." | resolves to todo 9's artifacts (both exist); fix commit is in `18a27e02..HEAD` | ledger L374 "row 12 -> [x] (tracked by todo 9)" under `st_01a178b9`'s APPROVE (L373) | PASS |
| 13 | Amend the measured-record criteria the evidence proved wrong (first id 671, EOS id 1 exemption, `--vram-experts 0` control, per-device load receipt, ctest target path). | `row13-gate-review.md` (8,727 B) - exists | ledger L378 "gate reviewer st_01a178c0 APPROVE row 13 against primary evidence" | PASS |
| 14 | Add a K-quant GEMV parity row (`iq::mv` -> `mv_kernel` -> `sub_dot`, Q2_K/Q4_K/Q5_K) to `ds4_parity` (commit `4e003d34`). | `row14-gate-review.md` (9,348 B) - exists | ledger L401 `gate-verdict`: "VERDICT confirmed: APPROVE" (reviewer `st_01a178f6`) | PASS |
| 18 | Close F2's two gaps: compile the AOT variant (`-DSTRATA_SYCL_AOT=bmg-g21`) and diff the ds4 parity rows byte-identically against todo 1's baseline rows; "Same work as F2's open gap; tracked there." | resolves to F2's artifacts. Closure receipts do exist in `task-2-dsv41-sycl-dpct-and-xpu-test.log` L128-182 (`DONECLAIM f2gaps gates=AOT_CONFIGURE_RC=0 AOT_BUILD_RC=0 AOT_RECONFIGURE_RC=0 AOT_REBUILD_RC=0 AOT_EXE_RC=0 PARITY_DIFF_RC=0 CLEANUP_RC=0`; both parity row files 12,152 bytes at identical SHA256 `3f026239...`), but they were written by the executor lanes (`st_01a17892`, `st_01a178bb`) | no independent confirmed/APPROVE covers the closure. The only independent review of these criteria is REJECT: ledger L356 "gate review st_01a1788c REJECT - F2 lands [~] ... 2 unmet: AOT variant ... never compiled, parity rows never captured/diffed". The checkbox was flipped on the lanes' own DONECLAIM (L375, L368) | FLAG |
| F2 | Code quality review: JIT under icpx 244/244, 2/2 ds4 tests, notes state reasons, no new abstraction/default; AOT variant compiled and parity rows byte-identical. | named evidence is `/tmp/dpct_run/f2_configure3.txt` (2,049 B), `f2_build3.txt` (4,818,486 B), `f2_ctest3.txt` (309 B) - outside the evidence dir - and `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md` (12,856 B), which is a sibling of the evidence dir, not under `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/` as this check requires | that review file records REJECT with two blockers (AOT never compiled at review time; parity rows never diffed), matching ledger L356. Post-closure state rests on the executor lanes' self-receipts (task-2 log `DONECLAIM f2gaps` + lane "VERDICT:" line at L182); ledger L375 flips F2 to `[x]` on that DONECLAIM and L382 notes "F2 already [x]" - no independent re-review exists | FLAG |

Check 1 result: 11 PASS, 2 FLAG (rows 18 and F2).

## Check 2 - scope

Non-conforming diff paths (expect 0):

```
0
bash -c 'git diff 18a27e02..HEAD --name-only | grep -vc -E "^(sycl/|docs/)"'
```

`git diff 18a27e02..HEAD --name-only` lists 25 paths, all under `sycl/` or `docs/`; `grep -c` of non-conforming names returns 0 (grep exit 1 = no matches).

Commit subjects (12/12 prefixed, 0 bad):

```
0
bash -c 'git log --format=%s 18a27e02..HEAD | grep -vc -E "^(\[Engine\]|\[Docs\])"'
```

Full list: 11 subjects starting `[Engine]`, 1 starting `[Docs]` (`[Docs] dsv41 SYCL: measured report for the ported engine`, HEAD `d337362b`). No subject lacks the area prefix.

Committed paths under `.omo/` or `bench/results` (expect 0):

```
0
bash -c 'git log --format=%H 18a27e02..HEAD -- .omo bench/results | wc -l'
```

Check 2 result: PASS.

## Check 3 - regression gates on the merged branch

Both gate build dirs were `rm -rf`'d before configure, so every build below is a full build of the merged tree at `d337362b`, not an incremental re-use of a previous lane's build.

DEF_BUILD_RC=0
bash -c 'cmake -S . -B /tmp/f1_def_build -G Ninja && cmake --build /tmp/f1_def_build'

Default build green: 69/69 targets, output ends `[69/69] Linking CXX executable nvfp4_avx512_parity`, zero FAILED lines (ninja reaching 69/69 with exit 0 implies none).

DEF_CTEST_RC=8
bash -c 'ctest --test-dir build --output-on-failure'

The literal gate ran and finished, but its composition is not the plan's reference state: it sees 39 tests, "0% tests passed, 39 tests failed out of 39", of which 35 are `***Not Run` (missing test executables) and 4 are `Failed` (`router_dot_parity_avx`, `iq_avx2_parity_iq3s_mt1`, `q8k_quant_parity_avx`, `isa_guard_check`). Diagnosis: the shared `build` directory is not a build of this checkout - its `CMakeCache.txt` records `CMAKE_HOME_DIRECTORY=/home/maxious/Strata_Niko1221`, configured 2026-10-11, and only 14 files remain in it. The plan's reference state (52 passed / 4 skipped / 1 failed) cannot come from this directory as it stands.

DEF_TESTS_BUILD_RC=0
bash -c 'cmake -S . -B /tmp/f1_def_tests -G Ninja -DSTRATA_BUILD_TESTS=ON && cmake --build /tmp/f1_def_tests'

Supplementary (clearly labelled, same default configuration plus tests, this checkout): 161/161 targets, zero FAILED lines.

DEF_CTEST_SUPP_RC=8
bash -c 'ctest --test-dir /tmp/f1_def_tests --output-on-failure'

Supplementary default ctest on this checkout: 57 tests, "98% tests passed, 1 tests failed out of 57" (52 passed, 4 skipped), sole failure `55 - isa_guard_check` - exactly the pre-existing upstream allow-list gap the plan records (row 10 `[~]`, open row 17). This is the meaningful default-test state of the merged branch.

SYCL_BUILD_RC=0
bash -c 'mkdir -p /home/maxious/tmpbuild; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; TMPDIR=/home/maxious/tmpbuild cmake -S sycl -B /tmp/f1_sycl_build -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON && cmake --build /tmp/f1_sycl_build'

Green: 260/260 targets, zero FAILED lines, `sycl/strata-dsv41` linked (248/260), output ends `[260/260] Linking CXX executable strata`.

SYCL_CTEST_RC=1
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ctest --test-dir /tmp/f1_sycl_build/sycl -R ds4 --output-on-failure'

The literal gate cannot deliver its named result as specified: output `Failed to change working directory to "/tmp/f1_sycl_build/sycl": No such file or directory`. The gate configures with `cmake -S sycl`, so `enable_testing()` and both `add_test` calls (sycl/CMakeLists.txt:272/277/291) register at the build root (`/tmp/f1_sycl_build/CTestTestfile.cmake` exists; `/tmp/f1_sycl_build/sycl` does not). The expected `ds4_parity`/`ds4_blas_parity` naming is therefore MISSING from the literal gate - reported as observed, not guessed (same class of plan-wording defect the F2 row already recorded for `build-sycl-dsv41/sycl`).

SYCL_CTEST_ROOT_RC=0
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ctest --test-dir /tmp/f1_sycl_build -R ds4 --output-on-failure'

Supplementary run at the corrected path (build root) names both tests and passes: `ds4_parity` Passed 0.27 sec, `ds4_blas_parity` Passed 0.31 sec, `100% tests passed, 0 tests failed out of 2`.

Check 3 result: all four required RCs produced un-piped (DEF_BUILD_RC=0, DEF_CTEST_RC=8, SYCL_BUILD_RC=0, SYCL_CTEST_RC=1); the two non-zero RCs are fully characterised above - DEF_CTEST's failure is a stale foreign-checkout build directory plus the known `isa_guard_check` gap (confirmed on a fresh tests-enabled build of this checkout), SYCL_CTEST's failure is the command's own wrong directory.

## Check 4 - tracked-file cleanliness

```
0
bash -c 'git status --porcelain | grep -vc "^??"'
```

`git status --porcelain` shows zero tracked-file changes. Every entry is untracked and confined to `?? .omo/` and `?? bench/results/2026-10-*/`, exactly the expected untracked paths. Check 4 result: PASS.

## Check 5 - cleanup

RM_RC=0
rm -rf /tmp/f1_def_build /tmp/f1_sycl_build

(node:fs proof, run after the removal)

```
existsSync('/tmp/f1_def_build') === false -> true
existsSync('/tmp/f1_sycl_build') === false -> true
```

The supplementary scratch dir was removed with the same receipt: `existsSync('/tmp/f1_def_tests') === false -> true`.

## Summary

Checks 2, 4 and 5 pass outright. Check 3 produces all four required un-piped RCs; the default build and the SYCL build are green with zero FAILED lines and the ds4 suite passes 2/2, while the two non-zero RCs are command/state artefacts documented with their receipts. Check 1 confirms 11 of the 13 audited `[x]` rows against an existing artifact under the evidence dir plus an independent confirmed/APPROVE in the ledger. Rows 18 and F2 are FLAG: their only independent review verdict is REJECT (`st_01a1788c`), the gap closure was accepted from the executor lanes' own DONECLAIM receipts without an independent re-review, and F2's named gate-review artifact sits outside `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/`. The `[x]` on those two rows is therefore not backed by the evidence chain this audit requires.

VERDICT FAIL
