#!/bin/bash
# F3 two-card manual QA (task st_01a178c4) - todo 10 default-build regression re-run.
# Format: every CMD line is followed by its own un-piped *_RC= line (rc=$? taken immediately after the command).
# Plain configure (no SYCL flags), default build, ctest --test-dir build; cwd is the scratch dir so `build`
# is <scratch>/build and nothing is built under /home/maxious/Strata.
E=/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test
R=$E/f3tc-receipts-build-def.txt
D=/tmp/f3tc_bld_def

echo "=== F3 two-card manual QA: todo 10 default-build regression re-run (task st_01a178c4) ===" > "$R"
echo "scratch dir: $D (fresh), build dir: $D/build (never under /home/maxious/Strata)" >> "$R"
mkdir -p "$D"
cd "$D" || { echo "CD_TO_SCRATCH_FAILED" >> "$R"; exit 9; }

echo "CMD: (cwd $D) bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S /home/maxious/Strata -B build -G Ninja -DSTRATA_BUILD_TESTS=ON > $E/f3tc-def-configure.txt 2>&1'" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S /home/maxious/Strata -B build -G Ninja -DSTRATA_BUILD_TESTS=ON > $E/f3tc-def-configure.txt 2>&1"
rc=$?
echo "DEFAULT_CONFIGURE_RC=$rc" >> "$R"
grep -m1 'CXX compiler identification' "$E/f3tc-def-configure.txt" >> "$R"

echo "CMD: (cwd $D) bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake --build build > $E/f3tc-def-build.txt 2>&1'" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake --build build > $E/f3tc-def-build.txt 2>&1"
rc=$?
echo "DEFAULT_BUILD_RC=$rc" >> "$R"
tail -3 "$E/f3tc-def-build.txt" >> "$R"

echo "CMD: (cwd $D) bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ctest --test-dir build --output-on-failure > $E/f3tc-ctest.txt 2>&1'" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ctest --test-dir build --output-on-failure > $E/f3tc-ctest.txt 2>&1"
rc=$?
echo "DEFAULT_CTEST_RC=$rc" >> "$R"

echo "CMD: sed -n '/The following tests FAILED/,\$p' $E/f3tc-ctest.txt > $E/f3tc-ctest-failed-names.txt 2>&1" >> "$R"
sed -n '/The following tests FAILED/,$p' "$E/f3tc-ctest.txt" > "$E/f3tc-ctest-failed-names.txt" 2>&1
rc=$?
echo "FAILED_NAMES_RC=$rc" >> "$R"
cat "$E/f3tc-ctest-failed-names.txt" >> "$R"

echo "CMD: grep -n 'iq512x_rows' $E/f3tc-ctest.txt > $E/f3tc-isa-violation.txt 2>&1" >> "$R"
grep -n 'iq512x_rows' "$E/f3tc-ctest.txt" > "$E/f3tc-isa-violation.txt" 2>&1
rc=$?
echo "IQ512X_VIOLATION_RC=$rc (0 = the iq512x_rows violation line is present in the isa_guard_check output)" >> "$R"
cat "$E/f3tc-isa-violation.txt" >> "$R"

echo "CMD: grep -c 'tests passed' $E/f3tc-ctest.txt" >> "$R"
grep -c 'tests passed' "$E/f3tc-ctest.txt" >> "$R"
rc=$?
echo "CTEST_SUMMARY_GREP_RC=$rc" >> "$R"
grep -nE 'tests passed|tests failed' "$E/f3tc-ctest.txt" >> "$R"

echo "DEFAULT_PHASE_DONE $(date -Is)" >> "$R"
