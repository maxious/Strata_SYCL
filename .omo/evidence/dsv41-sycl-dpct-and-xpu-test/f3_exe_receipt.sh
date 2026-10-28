R=/tmp/dpct_run/f3_receipts.txt
echo "CMD: bun -e node:fs existsSync+statSize of /tmp/f3_sycl_build/sycl/strata-dsv41 > /tmp/dpct_run/f3_exe_receipt.txt 2>&1" >> "$R"
/home/maxious/.omo/agent/bundled-bun/daff063de3f967ea/bun -e 'const fs = await import("node:fs"); const p = "/tmp/f3_sycl_build/sycl/strata-dsv41"; const ok = fs.existsSync(p); console.log("EXISTS=" + ok); if (!ok) process.exit(1); const st = fs.statSync(p); console.log("SIZE=" + st.size); console.log("MTIME=" + st.mtime.toISOString())' > /tmp/dpct_run/f3_exe_receipt.txt 2>&1
rc=$?
echo "SYCL_EXE_RC=$rc (0 = binary present at the real path sycl/strata-dsv41)" >> "$R"
cat /tmp/dpct_run/f3_exe_receipt.txt >> "$R"
echo "CMD: git -C /home/maxious/Strata rev-parse HEAD > /tmp/dpct_run/f3_build_identity.txt 2>&1" >> "$R"
git -C /home/maxious/Strata rev-parse HEAD > /tmp/dpct_run/f3_build_identity.txt 2>&1
rc=$?
echo "GIT_HEAD_RC=$rc" >> "$R"
cat /tmp/dpct_run/f3_build_identity.txt >> "$R"
echo "BUILD_FLAGS=STRATA_ENABLE_SYCL=ON STRATA_SYCL_DSV41=ON STRATA_SYCL_DSV41_EXE=ON CMAKE_CXX_COMPILER=icpx (JIT, no STRATA_SYCL_AOT)" >> "$R"
echo "PACK=/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 CARD=--gpus 0 DATE=$(date -Is)" >> "$R"
