#!/bin/bash
# F3 two-card manual QA (task st_01a178ea, plan row 9) - fresh build phase.
# Every CMD line is followed by its own un-piped *_RC= line (rc=$? taken immediately after the command).
set -u
E=/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test
R=$E/f3twc-receipts-build.txt
B=/tmp/f3tc_build

echo "=== F3 two-card manual QA: build phase (task st_01a178ea) ===" > "$R"
echo "build dir: $B | branch HEAD: $(git -C /home/maxious/Strata rev-parse HEAD)" >> "$R"
echo "PIPELINE_START $(date -Is)" >> "$R"

echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S sycl -B /tmp/f3tc_build -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON' (cwd /home/maxious/Strata, stdout -> $E/f3twc-configure.txt)" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S sycl -B /tmp/f3tc_build -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON" > "$E/f3twc-configure.txt" 2>&1
rc=$?
echo "SYCL_CONFIGURE_RC=$rc" >> "$R"

if [ "$rc" -eq 0 ]; then
  echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake --build /tmp/f3tc_build' (stdout -> $E/f3twc-build.txt)" >> "$R"
  bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake --build /tmp/f3tc_build" > "$E/f3twc-build.txt" 2>&1
  rc2=$?
  echo "SYCL_BUILD_RC=$rc2" >> "$R"

  echo "CMD: grep -c 'FAILED' $E/f3twc-build.txt" >> "$R"
  grep -c 'FAILED' "$E/f3twc-build.txt" > "$E/f3twc-build-failed-count.txt" 2>&1
  rc=$?
  echo "BUILD_FAILED_GREP_RC=$rc (1 means zero FAILED lines)" >> "$R"
  echo "BUILD_FAILED_COUNT=$(cat "$E/f3twc-build-failed-count.txt")" >> "$R"

  echo "CMD: grep -oE '\\[[0-9]+/[0-9]+\\]' $E/f3twc-build.txt | tail -1 (last ninja progress)" >> "$R"
  grep -oE '\[[0-9]+/[0-9]+\]' "$E/f3twc-build.txt" | tail -1 > "$E/f3twc-build-progress-last.txt" 2>&1
  rc=$?
  echo "BUILD_PROGRESS_GREP_RC=$rc" >> "$R"
  echo "BUILD_PROGRESS_LAST=$(cat "$E/f3twc-build-progress-last.txt")" >> "$R"

  echo "CMD: node -e 'import{existsSync,statSync}from\"node:fs\"; const p=\"/tmp/f3tc_build/sycl/strata-dsv41\"; console.log(JSON.stringify({exists:existsSync(p), size:existsSync(p)?statSync(p).size:null}))'" >> "$R"
  node -e 'import{existsSync,statSync}from"node:fs"; const p="/tmp/f3tc_build/sycl/strata-dsv41"; console.log(JSON.stringify({exists:existsSync(p), size:existsSync(p)?statSync(p).size:null}))' > "$E/f3twc-exe-receipt.txt" 2>&1
  rc=$?
  echo "EXE_RECEIPT_RC=$rc" >> "$R"
  cat "$E/f3twc-exe-receipt.txt" >> "$R"

  echo "CMD: sha256sum /tmp/f3tc_build/sycl/strata-dsv41 (build identity)" >> "$R"
  sha256sum /tmp/f3tc_build/sycl/strata-dsv41 > "$E/f3twc-build-identity.txt" 2>&1
  rc=$?
  echo "BUILD_IDENTITY_RC=$rc" >> "$R"
  cat "$E/f3twc-build-identity.txt" >> "$R"
fi

echo "BUILD_PHASE_DONE $(date -Is)" >> "$R"
echo "PIPELINE_DONE $(date -Is)"
