# Regression gates re-run - dsv41-sycl-dpct-and-xpu-test

Task st_01a1793d, run 2026-10-26T17:23Z (2026-10-27 +11:00) on lain (linux x64, AMD Ryzen 7 5700X3D, 2x Intel Arc Pro B60).
Repo `/home/maxious/Strata`, branch `sycl/dsv41-sycl-support`, HEAD `d337362b8a9fa09402fd41e2636cbd7beae66161` (as expected).

Convention: every `*_RC` line below sits directly above the command whose exit code it is. Each value is the
un-piped exit code captured as `rc=$?` immediately after its command (never through a pipe). Raw captures live in
`.omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates-artifacts/` (same names as the logs cited).

## 1. What configures `build-sycl-dsv41`

STEP1_GREP_RC=1 (un-piped grep: 0 matches)
CMD: grep -rn "build-sycl-dsv41" tools/ setup.py sycl/ serve/ 2>/dev/null

STEP1_GREP_PIPELINE_RC=0 (pipeline rc = head's rc, as given in the task)
CMD: grep -rn "build-sycl-dsv41" tools/ setup.py sycl/ serve/ 2>/dev/null | head -20

**Finding (static reading):** no repo script names `build-sycl-dsv41`; the name exists only in `.omo/` plan
documents. The repo's own SYCL build script is `sycl/tools/build.sh`, which configures any SYCL build dir as
(`b=${BUILD_DIR:-$repo/build-sycl}`):

```
cmake -S $repo/sycl -B $b -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DSTRATA_SYCL_AOT="${AOT:-}" -DSTRATA_SYCL_SPIN_MAX="${SPIN_MAX:-}" -DSTRATA_SYCL_I8_ESIMD="${I8_ESIMD:-AUTO}"
```

It wraps `source /opt/intel/oneapi/setvars.sh` at the top and builds the whole dir. The exact configure command
the dir `build-sycl-dsv41` is meant to use is the one this plan names at line 67 (same shape as build.sh, plus the
two dsv41 switches), and it is the command run in section 3:

```
cmake -S sycl -B build-sycl-dsv41 -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON
```

## 2. Default gate at the literal path `build/`

### 2a. Stale-cache pre-step (why a cache reset was needed)

The prior audit's `build/` was configured from `/home/maxious/Strata_Niko1221` (2026-10-11). Snapshot of its cache
(`regression-gates-artifacts/stale-cache-keys.txt`, `STALE_CACHE_SNAPSHOT_RC=0`):
`CMAKE_HOME_DIRECTORY:INTERNAL=/home/maxious/Strata_Niko1221`,
`CMAKE_CACHEFILE_DIR:INTERNAL=/home/maxious/Strata_Niko1221/build`, `CMAKE_BUILD_TYPE:STRING=Release`,
`STRATA_BUILD_TESTS:BOOL=ON`, `CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++`.

STALE_CONFIGURE_RC=1
CMD: bash -c 'cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release'   (run first, against the stale cache)

Error (log `def-configure-stale.log`): `The source "/home/maxious/Strata/CMakeLists.txt" does not match the source
"/home/maxious/Strata_Niko1221/CMakeLists.txt" used to generate cache.  Re-run cmake with a different source
directory.` This is the mechanism behind the prior audit's garbage numbers: CMake refuses the stale cache outright.

CACHE_RM_RC=0
CMD: rm -f build/CMakeCache.txt   (cache-generation reset only; `build/` and its artifacts are kept)

CMAKEFILES_RM_RC=0
CMD: rm -rf build/CMakeFiles   (cache-generation reset only; `build/` and its artifacts are kept)

`build/` was NOT deleted; only the cache-generation files were cleared so the literal command can configure this
checkout. `TESTS_DEFAULT_INPUT_RC=2` for `ls tests/CMakeLists.txt bench/micro` (log `def-tests-present.txt`):
this checkout has no `tests/CMakeLists.txt`, so `STRATA_BUILD_TESTS` defaults to OFF here (CMakeLists.txt:119-123),
while the stale cache carried `STRATA_BUILD_TESTS:BOOL=ON` and the plan's named default gate passes it explicitly.

### 2b. The task's verbatim step-2 command, exactly as written

Run 1's transcript echoes these as `DEF_*`; they are shown here as `DEF_VERBATIM_*` to keep them apart from the
plan-named gate in 2c. Values unaltered.

DEF_VERBATIM_CONFIGURE_RC=0
CMD: bash -c 'cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release'

DEF_VERBATIM_BUILD_RC=0
CMD: bash -c 'cmake --build build'

DEF_VERBATIM_FAILED_GREP_RC=1 (count 0 - "grep -c FAILED must be 0" holds for the verbatim build too)
CMD: grep -c FAILED regression-gates-artifacts/def-build.log

DEF_VERBATIM_CTEST_RC=8
CMD: bash -c 'ctest --test-dir build --output-on-failure'

This ctest is STILL the prior audit's artifact and reproduces it exactly: the verbatim configure registers no
tests (tests default OFF above), so it leaves the stale `build/CTestTestfile.cmake` in place (dated Oct 11 22:14,
its own header says `# Source directory: /home/maxious/Strata_Niko1221`). Its 39 tests all point at
`/home/maxious/Strata_Niko1221/build/*`: **35 Not Run** (`Unable to find executable: /home/maxious/Strata_Niko1221/build/...`)
plus 4 fast failures against stale paths (`router_dot_parity_avx`, `iq_avx2_parity_iq3s_mt1`,
`q8k_quant_parity_avx`, `isa_guard_check` - the latter via `can't open file
'/home/maxious/Strata_Niko1221/tools/check_isa_guard.py'`), summary `0% tests passed, 39 tests failed out of 39`
(log `def-ctest.log`). This matches the prior audit's "ctest exit was 8 with 35 tests Not Run" - that report was
this stale test file, not this branch.

### 2c. The plan's named default gate (same literal path, `-DSTRATA_BUILD_TESTS=ON`)

The plan's "Default regression" gate is `cmake -S . -B build -G Ninja -DSTRATA_BUILD_TESTS=ON` then
`ctest --test-dir build --output-on-failure`; `-DSTRATA_BUILD_TESTS=ON` is also the value the stale cache carried.
It regenerates `build/CTestTestfile.cmake` for THIS checkout (`CTESTFILE_ID_RC=0`: `# Source directory:
/home/maxious/Strata`, `# Build directory: /home/maxious/Strata/build`). These are the `DEF_*` gate RCs:

DEF_CONFIGURE_RC=0
CMD: bash -c 'cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON'

DEF_BUILD_RC=0
CMD: bash -c 'cmake --build build'

DEF_FAILED_GREP_RC=1 (DEF_FAILED_COUNT=0 - "grep -c FAILED must be 0" holds)
CMD: grep -c FAILED regression-gates-artifacts/def2-build.log

CTESTFILE_ID_RC=0
CMD: head -3 build/CTestTestfile.cmake

DEF_CTEST_RC=8
CMD: bash -c 'ctest --test-dir build --output-on-failure'

DEF_FAILED_NAMES_RC=0
CMD: grep -E '% tests passed|\(Failed\)|\(Not Run\)|iq512x_rows' regression-gates-artifacts/def2-ctest.log

DEF_FAILED_NAMES_CAT_RC=0
CMD: cat regression-gates-artifacts/def2-ctest-failed.txt

**Failing tests (complete list): `55 - isa_guard_check (Failed)` and nothing else.** The run is
`98% tests passed, 1 tests failed out of 57` (0 Not Run). Said explicitly, as asked: **the only failure is
`isa_guard_check` on `iq512x_rows`** - `check_isa_guard: .../build/libstrata_kernels_cpu.a: 755 functions, 117
hold wide-ISA instructions, 1 not allowed`, `VIOLATION strata::kernels::cpu::iq512x_rows(...) (777 wide
instructions, first vmovq): wide-ISA code in an unlisted public function [iq_avx512.cpp.o]`. That is open row 17,
a `tools/` change outside this plan's scope.

## 3. SYCL gate at the literal path `build-sycl-dsv41`

`build-sycl-dsv41` did not exist; it was configured with the section-1 command plus `TMPDIR=/home/maxious/tmpbuild`
inside `source /opt/intel/oneapi/setvars.sh` (both configure and build are setvars-wrapped - a bare build fails on
the oneMKL headers, per prior lanes' evidence). Compiler: IntelLLVM 2026.2.0.

SYCL_CONFIGURE_RC=0
CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; TMPDIR=/home/maxious/tmpbuild cmake -S sycl -B build-sycl-dsv41 -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON'

SYCL_BUILD_RC=0
CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; TMPDIR=/home/maxious/tmpbuild cmake --build build-sycl-dsv41'

SYCL_FAILED_GREP_RC=1 (SYCL_FAILED_COUNT=0)
CMD: grep -c FAILED regression-gates-artifacts/sycl-build.log

SYCL_TESTDIR_LS_RC=2
CMD: ls build-sycl-dsv41/CTestTestfile.cmake build-sycl-dsv41/sycl   (where the ds4 tests register)

Output: `build-sycl-dsv41/CTestTestfile.cmake` exists, `build-sycl-dsv41/sycl` does not - with `cmake -S sycl` the
`enable_testing()`/`add_test` calls in `sycl/CMakeLists.txt` register at the build root.

SYCL_CTEST_LIST_RC=0
CMD: bash -c 'ctest --test-dir build-sycl-dsv41 -N -R ds4'

Inventory at the build root: `Test #1: ds4_parity`, `Test #2: ds4_blas_parity` (Total Tests: 2).

SYCL_CTEST_RC_LITERAL=1
CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ctest --test-dir build-sycl-dsv41/sycl -R ds4 --output-on-failure'

The plan-line-67 literal subdir path cannot work: `Failed to change working directory to
"/home/maxious/Strata/build-sycl-dsv41/sycl": No such file or directory`. Real value recorded as asked; the
working path follows.

SYCL_CTEST_RC=0
CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ctest --test-dir build-sycl-dsv41 -R ds4 --output-on-failure'

SYCL_CTEST_NAMES_RC=0
CMD: grep -E 'Test #|tests passed|tests failed' regression-gates-artifacts/sycl-ctest.log

Both named tests pass: `1/2 Test #1: ds4_parity ... Passed 0.27 sec`, `2/2 Test #2: ds4_blas_parity ... Passed
0.31 sec`, `100% tests passed, 0 tests failed out of 2`.

## 4. git status

GIT_STATUS_RC=0
CMD: git status --porcelain

GIT_TRACKED_CHANGES_COUNT=0
CMD: git status --porcelain | grep -v '^??' | wc -l

Output carries only untracked `??` entries (`.omo/`, `bench/results/...`; `build/` and `build-sycl-dsv41/` are
ignored). **No tracked-file changes.** HEAD `d337362b8a9fa09402fd41e2636cbd7beae66161` on
`sycl/dsv41-sycl-support` throughout; no product file was edited; `build/` and `build-sycl-dsv41/` were not deleted.

## Assumptions recorded

1. The stale `build/` cache made the literal configure impossible (STALE_CONFIGURE_RC=1); only `CMakeCache.txt`
   and `CMakeFiles/` were cleared (documented above). The directory and every other artifact were kept, and the
   stale cache identity was snapshotted first (`stale-cache-keys.txt`).
2. `DEF_*` names denote the plan's named default gate (its `-DSTRATA_BUILD_TESTS=ON` = the stale cache's own
   value); the task's verbatim transcription is recorded separately as `DEF_VERBATIM_*`. Both ran at the literal
   path `build/`.
3. Both SYCL configure and build are setvars-wrapped with `TMPDIR=/home/maxious/tmpbuild`; a bare build fails on
   the oneMKL headers (`#error SYCLomatic runtime requires oneMath/oneMKL support`), per prior lanes' evidence.

## Verdict

The two gates confirm this branch's state at their literal paths: the default build is green
(`grep -c FAILED` = 0) and its ctest's only failure is `isa_guard_check` on `iq512x_rows` (open row 17, `tools/`
change, outside this plan's scope); the SYCL gate builds green and `ds4_parity` + `ds4_blas_parity` pass 2/2
(`SYCL_CTEST_RC=0`); `git status --porcelain` shows no tracked-file changes.

VERDICT confirmed
