#!/bin/bash
# F3 QA todo 10 ctest phase on the default build (no SYCL flags).
R=/tmp/dpct_run/f3_receipts.txt
echo "CMD: ctest --test-dir /tmp/f3_default_build --output-on-failure > /tmp/dpct_run/f3_ctest.txt 2>&1" >> "$R"
ctest --test-dir /tmp/f3_default_build --output-on-failure > /tmp/dpct_run/f3_ctest.txt 2>&1
rc=$?
echo "CTEST_RC=$rc" >> "$R"
echo "CMD: grep -n '% tests passed' /tmp/dpct_run/f3_ctest.txt > /tmp/dpct_run/f3_ctest_summary.txt" >> "$R"
grep -n '% tests passed' /tmp/dpct_run/f3_ctest.txt > /tmp/dpct_run/f3_ctest_summary.txt
rc=$?
echo "CTEST_SUMMARY_RC=$rc" >> "$R"
cat /tmp/dpct_run/f3_ctest_summary.txt >> "$R"
echo "CMD: grep -n 'Failed' /tmp/dpct_run/f3_ctest.txt > /tmp/dpct_run/f3_ctest_failed.txt" >> "$R"
grep -n 'Failed' /tmp/dpct_run/f3_ctest.txt > /tmp/dpct_run/f3_ctest_failed.txt
rc=$?
echo "CTEST_FAILED_NAMES_RC=$rc" >> "$R"
cat /tmp/dpct_run/f3_ctest_failed.txt >> "$R"
echo "CTEST_PHASE_DONE $(date -Is)" >> "$R"
