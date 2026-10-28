#!/bin/bash
# F3 QA - corrected SYCL build (setvars-wrapped: oneMKL headers are only visible under setvars.sh,
# proven by BUILD_RC=1/BUILD_EXE_RC=1 failing at sycl/src/prefill/gemm.dp.cpp on the bare cmake --build).
R=/tmp/dpct_run/f3_receipts.txt
echo "CMD: timeout 1200 bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake --build /tmp/f3_sycl_build > /tmp/dpct_run/f3_build2.txt 2>&1'" >> "$R"
timeout 1200 bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake --build /tmp/f3_sycl_build > /tmp/dpct_run/f3_build2.txt 2>&1'
rc=$?
echo "BUILD2_RC=$rc" >> "$R"
echo "BUILD2_PHASE_DONE $(date -Is)" >> "$R"
