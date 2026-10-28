#!/bin/bash
# F3 QA - SYCL build phase. Every *_RC is the un-piped exit code of the CMD line printed above it.
R=/tmp/dpct_run/f3_receipts.txt
echo '=== F3 QA receipts (task st_01a1789d, manual QA executor) ===' >> "$R"
echo "SYCL_PHASE_START $(date -Is)" >> "$R"

echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S /home/maxious/Strata -B /tmp/f3_sycl_build -G Ninja -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_DSV41=ON -DCMAKE_CXX_COMPILER=icpx > /tmp/dpct_run/f3_configure.txt 2>&1'" >> "$R"
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S /home/maxious/Strata -B /tmp/f3_sycl_build -G Ninja -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_DSV41=ON -DCMAKE_CXX_COMPILER=icpx > /tmp/dpct_run/f3_configure.txt 2>&1'
rc=$?
echo "CFG_RC=$rc" >> "$R"

echo "CMD: cmake --build /tmp/f3_sycl_build > /tmp/dpct_run/f3_build.txt 2>&1" >> "$R"
cmake --build /tmp/f3_sycl_build > /tmp/dpct_run/f3_build.txt 2>&1
rc=$?
echo "BUILD_RC=$rc" >> "$R"

echo "CMD: grep -c strata-dsv41 /tmp/f3_sycl_build/build.ninja > /tmp/dpct_run/f3_exe_grep_pre.txt 2>&1" >> "$R"
grep -c strata-dsv41 /tmp/f3_sycl_build/build.ninja > /tmp/dpct_run/f3_exe_grep_pre.txt 2>&1
rc=$?
echo "EXE_TARGET_PRE_GREP_RC=$rc" >> "$R"
cat /tmp/dpct_run/f3_exe_grep_pre.txt >> "$R"

echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S /home/maxious/Strata -B /tmp/f3_sycl_build -G Ninja -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON -DCMAKE_CXX_COMPILER=icpx > /tmp/dpct_run/f3_configure_exe.txt 2>&1'" >> "$R"
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S /home/maxious/Strata -B /tmp/f3_sycl_build -G Ninja -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON -DCMAKE_CXX_COMPILER=icpx > /tmp/dpct_run/f3_configure_exe.txt 2>&1'
rc=$?
echo "CFG_EXE_RC=$rc" >> "$R"

echo "CMD: grep -c strata-dsv41 /tmp/f3_sycl_build/build.ninja > /tmp/dpct_run/f3_exe_grep_post.txt 2>&1" >> "$R"
grep -c strata-dsv41 /tmp/f3_sycl_build/build.ninja > /tmp/dpct_run/f3_exe_grep_post.txt 2>&1
rc=$?
echo "EXE_TARGET_POST_GREP_RC=$rc" >> "$R"
cat /tmp/dpct_run/f3_exe_grep_post.txt >> "$R"

echo "CMD: cmake --build /tmp/f3_sycl_build > /tmp/dpct_run/f3_build_exe.txt 2>&1" >> "$R"
cmake --build /tmp/f3_sycl_build > /tmp/dpct_run/f3_build_exe.txt 2>&1
rc=$?
echo "BUILD_EXE_RC=$rc" >> "$R"

echo "SYCL_PHASE_DONE $(date -Is)" >> "$R"
