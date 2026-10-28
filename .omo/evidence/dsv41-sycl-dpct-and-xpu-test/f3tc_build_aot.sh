#!/bin/bash
# F3 two-card manual QA (task st_01a178c4) - AOT SYCL build phase.
# Format: every CMD line is followed by its own un-piped *_RC= line (rc=$? taken immediately after the command).
# Every build is wrapped in bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; <command>'.
E=/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test
R=$E/f3tc-receipts-build-aot.txt
B=/tmp/f3tc_bld_aot
export PATH=/home/maxious/.omo/agent/bundled-bun/daff063de3f967ea:$PATH

echo "=== F3 two-card manual QA: AOT SYCL build phase (task st_01a178c4) ===" > "$R"
echo "scratch build dir: $B (fresh, created by this run; never under /home/maxious/Strata)" >> "$R"
echo "configure flags: -DSTRATA_SYCL_AOT=bmg-g21 -DSTRATA_SYCL_DSV41_EXE=ON (plus the enabling base flags" >> "$R"
echo "  -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_DSV41=ON without which sycl/ is not configured and the EXE" >> "$R"
echo "  option is nested inside if(STRATA_SYCL_DSV41), sycl/CMakeLists.txt:257-258/:297-298)" >> "$R"

echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S /home/maxious/Strata -B $B -G Ninja -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON -DSTRATA_SYCL_AOT=bmg-g21 -DCMAKE_CXX_COMPILER=icpx > $E/f3tc-configure-aot.txt 2>&1'" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S /home/maxious/Strata -B $B -G Ninja -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON -DSTRATA_SYCL_AOT=bmg-g21 -DCMAKE_CXX_COMPILER=icpx > $E/f3tc-configure-aot.txt 2>&1"
rc=$?
echo "AOT_CONFIGURE_RC=$rc" >> "$R"
grep -m1 'CXX compiler identification' "$E/f3tc-configure-aot.txt" >> "$R"

echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake --build $B > $E/f3tc-build-aot.txt 2>&1'" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake --build $B > $E/f3tc-build-aot.txt 2>&1"
rc=$?
echo "AOT_BUILD_RC=$rc" >> "$R"
tail -3 "$E/f3tc-build-aot.txt" >> "$R"

echo "CMD: bun -e node:fs existsSync+statSync of $B/sycl/strata-dsv41 > $E/f3tc-exe-receipt.txt 2>&1" >> "$R"
P="$B/sycl/strata-dsv41" bun -e 'import { existsSync, statSync } from "node:fs"; const p = process.env.P; const e = existsSync(p); console.log("BINARY=" + p); console.log("EXISTS=" + e); if (e) { const s = statSync(p); console.log("SIZE=" + s.size); console.log("MTIME=" + s.mtime.toISOString()); } process.exit(e ? 0 : 1);' > "$E/f3tc-exe-receipt.txt" 2>&1
rc=$?
echo "AOT_EXE_RC=$rc (0 = the binary exists at $B/sycl/strata-dsv41)" >> "$R"
cat "$E/f3tc-exe-receipt.txt" >> "$R"

echo "CMD: git -C /home/maxious/Strata rev-parse HEAD > $E/f3tc-build-identity.txt 2>&1" >> "$R"
git -C /home/maxious/Strata rev-parse HEAD > "$E/f3tc-build-identity.txt" 2>&1
rc=$?
echo "GIT_HEAD_RC=$rc" >> "$R"
cat "$E/f3tc-build-identity.txt" >> "$R"
echo "BUILD_FLAGS=STRATA_ENABLE_SYCL=ON STRATA_SYCL_DSV41=ON STRATA_SYCL_DSV41_EXE=ON STRATA_SYCL_AOT=bmg-g21 CMAKE_CXX_COMPILER=icpx" >> "$R"
echo "AOT_PHASE_DONE $(date -Is)" >> "$R"
