# Gate Review: F1 — default build and ds4 tests green

- **recommendation:** REJECT
- **checkbox mark:** [~]
- **reviewer:** omo-native-gate-reviewer (st_01a1787e), final gate for F1
- **date:** 2026-10-26
- **plan audited:** `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (todo 10 acceptance + Verification-strategy gate commands + IS-8)
- **evidence lane:** st_01a1786d (`/tmp/dpct_run/f1_*` receipts, `/tmp/dpct_run/ledger_f1ev.jsonl`)

> Report path: the fallback `.omo/evidence/<goal>-gate-review.md` already holds the todo-1 gate
> review. Following the repo's todo-10 precedent, this F1 review is written to a distinct path so it
> does not clobber that artifact.

## originalIntent

The user wants the default (no-SYCL) path proven untouched and green by the dsv41 SYCL port: the
default build builds, the full default `ctest` stays green, and any failure is shown to be
pre-existing on `origin/main` rather than port-caused. This is a verification gate, not a change
(plan todo 10: `Commit: N | gate only`, closes IS-8).

## desiredOutcome

A gate that demonstrates (1) default configure + build green, (2) default `ctest` green with
`CTEST_RC=0`, (3) any ctest failure proven pre-existing on `origin/main` by reference reading (the
port did not touch the failing path), and (4) the in-place gate command reproduced honestly rather
than a vacuous "zero tests" run read as green.

## userOutcomeReview

The lane proves half the checkbox and disproves the other half, honestly:

- **Default build is green.** `/tmp/dpct_run/f1_build.txt` ends `[69/69]` (no-tests configure),
  `/tmp/dpct_run/f1_build_tests.txt` ends `[161/161]` (tests-enabled configure). Both configure
  receipts show "Build files have been written to" `/tmp/f1_build` and `/tmp/f1_build_tests`.
- **Default ctest is NOT green.** `/tmp/dpct_run/f1_ctest_tests.txt` (tests-enabled) shows 52 passed,
  4 skipped (`expert_multi_test`, `native_hadamard_test`, `tq_cpu_parity`, `tq_cpu_parity_avx2`),
  1 failed (`isa_guard_check`), i.e. `CTEST_TESTS_RC=8`. `/tmp/dpct_run/f1_fail.txt` shows the
  violation: `strata::kernels::cpu::iq512x_rows(...) (777 wide instructions, first vmovq): wide-ISA
  code in an unlisted public function [iq_avx512.cpp.o]`.
- **The one failure is pre-existing, not port-caused.** `git diff 4b9e05a2 HEAD --
  tools/check_isa_guard.py src/kernels/cpu/iq_avx512.cpp` is empty (0 lines, `REF_DIFF_RC=0`);
  `grep -c iq512x_rows tools/check_isa_guard.py` = 0 on HEAD and on `git show
  4b9e05a2:tools/check_isa_guard.py` (rc=1 each); `iq512x_rows` is defined at
  `src/kernels/cpu/iq_avx512.cpp:607` in both. The branch touches only `sycl/` and `docs/` (45
  files vs `origin/main`), so no default-path file moved.
- **The literal acceptance is unmet.** Todo 10's `CTEST_RC=0` and the gate command
  `ctest --test-dir build --output-on-failure` require the full default ctest to pass; it exits 8.
- **The in-place build dir is unusable.** `/tmp/dpct_run/f1_build_stale_cache.txt` shows the
  `/home/maxious/Strata/build/CMakeCache.txt` was created in `/home/maxious/Strata_Niko1221/build`
  and that source tree has no `CMakeLists.txt`; `DEFAULT_BUILD_RC=1` for that path. A clean
  out-of-tree build was used instead, which is a valid reproduction of the default tree.
- **The zero-test trap was caught.** `f1_ctest.txt` on the no-tests build reads
  "No tests were found!!!"; the lane flagged it as vacuous rather than green.

Net: the "default build" half is satisfied; the "tests green" half is not (one pre-existing
failure). The checkbox therefore lands `[~]`, not `[x]`, with the gap named below. Because a stated
acceptance criterion (`CTEST_RC=0`) fails with concrete evidence, the gate recommendation is REJECT
of the full-green claim, not a port regression finding.

## Facts reproduced this session (not taken from the lane's prose)

- `git rev-parse origin/main` = `4b9e05a2a806...` (matches the lane's reference base); `4b9e05a2` is
  an ancestor of HEAD; HEAD = `a698bb84`, branch `sycl/dsv41-sycl-support`.
- `git diff 4b9e05a2 HEAD -- tools/check_isa_guard.py src/kernels/cpu/iq_avx512.cpp | wc -l` = 0.
- `grep -c iq512x_rows tools/check_isa_guard.py` = 0 (rc=1); `git show 4b9e05a2:tools/check_isa_guard.py | grep -c iq512x_rows` = 0 (rc=1).
- `grep -n iq512x_rows src/kernels/cpu/iq_avx512.cpp` → line 607 (definition) in HEAD and in `4b9e05a2`.
- `tools/check_isa_guard.py` `WIDE_PUBLIC` (lines 32–36) lists `"iq512_gu_rows"` and `"iq512_rows"`
  but no `iq512x_rows`; the `isa_guard_check` test is registered at `CMakeLists.txt:2315/2319`.
- `CMakeLists.txt:119` `set(_strata_tests_default OFF)`, `:121` (ON only if
  `tests/CMakeLists.txt` AND `bench/micro` exist), `:123` `option(STRATA_BUILD_TESTS ...)`, `:268`
  `if(STRATA_BUILD_TESTS)` with inline `add_test` registration.
- `tests/CMakeLists.txt` is absent on disk (`ls` rc=2) and untracked (`git ls-files
  tests/CMakeLists.txt` empty), so `_strata_tests_default` is OFF and the literal no-flag configure
  registers zero tests.
- Branch scope vs `origin/main`: 45 changed files, all `sycl/` or `docs/`; zero under `src/`,
  `include/`, `CMakeLists.txt`, `tools/`, `tests/`.

## Skill-perspective check (remove-ai-slops / programming)

Both skills were loaded from `/home/maxious/.omo/binary-runtime/5.1.48/plugin/skills/` and applied
directly over the diff, tests, and production code relevant to this gate:

- There is no default-path diff to slop-review: the branch's changed files are `sycl/` + `docs/`
  only, and the two files that own the ctest failure (`tools/check_isa_guard.py`,
  `src/kernels/cpu/iq_avx512.cpp`) have an empty diff vs `origin/main`. No production extraction,
  parsing, or normalization was added on this path.
- No tests were added, removed, or modified by this lane; it only ran the pre-existing `ctest`
  suite. There are therefore no excessive, tautological, implementation-mirroring, deletion-only, or
  removal-verifying tests.
- The failing `isa_guard_check` is a legitimate pre-existing test (it guards wide-ISA instructions
  against an explicit allow-list), not a slop artifact. Its failure is a real allow-list gap
  (`iq512x_rows` missing from `WIDE_PUBLIC`), unchanged from `origin/main`.
- The lane's `grep -c` controls are genuine two-sided controls (OFF count 0 / exit 1 vs ON count /
  exit 0), not prose pins; they measure machine-consumed compile-list membership. No
  `programming`-criteria finding (maintenance burden, false confidence, scope drift) applies to a
  zero-diff verification gate.
- No code review report or manual QA matrix was among the F1 inputs, consistent with zero product
  files changed. The related todo-10 gate review (st_01a17766) DID carry its own skill-perspective
  check and reached the same pre-existing conclusion; that coverage does not replace the direct
  check above.

## checked artifact paths

- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (todo 10 acceptance, gate commands, IS-8, full read)
- `/tmp/dpct_run/f1_ctest_tests.txt` (52 passed, 4 skipped, 1 failed; isa_guard_check)
- `/tmp/dpct_run/f1_fail.txt` (iq512x_rows VIOLATION, 777 wide instructions, first vmovq)
- `/tmp/dpct_run/f1_build_stale_cache.txt` (stale Strata_Niko1221 cache; in-place build unusable)
- `/tmp/dpct_run/f1_build.txt` (`[69/69]` default no-tests build)
- `/tmp/dpct_run/f1_build_tests.txt` (`[161/161]` tests-enabled build)
- `/tmp/dpct_run/f1_configure.txt`, `/tmp/dpct_run/f1_configure_tests.txt` (configure receipts)
- `/tmp/dpct_run/f1_ctest.txt` ("No tests were found!!!" — vacuous no-tests run)
- `/tmp/dpct_run/f1_isa_ref_diff.txt` (empty; branch diff on the two files = 0 lines)
- `/tmp/dpct_run/ledger_f1.jsonl`, `/tmp/dpct_run/ledger_f1ev.jsonl` (lane receipts and adversarial classes)
- `/home/maxious/Strata/CMakeLists.txt` (lines 119–123, 268, 2315–2322)
- `/home/maxious/Strata/tools/check_isa_guard.py` (WIDE_PUBLIC allow-list, lines 32–36)
- `/home/maxious/Strata/src/kernels/cpu/iq_avx512.cpp` (iq512x_rows at line 607)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test-todo10-gate-review.md` (corroborating reproduction on origin/main)
- Git state: `origin/main` = `4b9e05a2`, HEAD = `a698bb84`, branch `sycl/dsv41-sycl-support`, 45 changed files all `sycl/`/`docs/`

## blockers

1. **violatedCriterion:** todo-10 acceptance `CTEST_RC=0 on the default build` (plan gate:
   `ctest --test-dir build --output-on-failure` must pass)
   **observation:** default ctest exits 8 — test #55 `isa_guard_check` fails (`iq512x_rows` wide-ISA
   code unlisted in `tools/check_isa_guard.py` allow-list); pre-existing on `origin/main`, not
   port-caused.
   **evidencePointer:** `/tmp/dpct_run/f1_ctest_tests.txt`, `/tmp/dpct_run/f1_fail.txt`

## gap line text (for [~])

Default build green (69/69 no-tests, 161/161 tests-enabled targets) but default ctest exits 8 on
test #55 `isa_guard_check` (`iq512x_rows`: 777 wide-ISA instructions, unlisted in
`tools/check_isa_guard.py` allow-list); pre-existing on `origin/main` `4b9e05a2`, so `CTEST_RC=0` /
"ctest --test-dir build must pass" is unmet — not a port-caused regression.

## exact evidence gaps

- (Advisory, non-blocking) The task title says "ds4 tests green", but this lane ran the **default**
  ctest (57 CPU tests), not the SYCL `ctest -R ds4` parity harnesses (`ds4_parity`,
  `ds4_blas_parity`). If the parent's F1 literally requires the ds4 parity tests green, that
  artifact is absent from this lane; the plan's own hint ties the criterion to "ctest --test-dir
  build must pass" (the default ctest), which is what this review evaluates.
- (Advisory, non-blocking) `origin/main` was not rebuilt/re-run by this lane; the pre-existing claim
  rests on static reference reading (empty diff + allow-list grep). It is corroborated by the
  earlier todo-10 gate review, which reproduced `CTL_CTEST_RC=8` on a clean `origin/main` control
  tree (`4b9e05a2`).

## Advisory notes (non-blocking)

1. The plan's todo-10 acceptance text writes `CTEST_RC=0`, but the measured result is 8. That is a
   plan-wording lag versus a proven pre-existing upstream gap; it does not implicate the SYCL port,
   but it is still a non-green default ctest and is why the checkbox is `[~]`, not `[x]`.
2. The in-place `/home/maxious/Strata/build` is stale and cannot reproduce the literal gate command
   as-written; the clean out-of-tree build (`/tmp/f1_build_tests`) is an acceptable stand-in and the
   lane disclosed it.
