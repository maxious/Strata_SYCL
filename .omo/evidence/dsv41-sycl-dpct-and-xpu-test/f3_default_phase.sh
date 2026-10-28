#!/bin/bash
# F3 QA - todo 10 default-build phase (no SYCL flags). Every *_RC is the un-piped exit code of the CMD line above it.
R=/tmp/dpct_run/f3_receipts.txt
echo "DEFAULT_PHASE_START $(date -Is)" >> "$R"

echo "CMD: cmake -S /home/maxious/Strata -B /tmp/f3_default_build -G Ninja -DSTRATA_BUILD_TESTS=ON > /tmp/dpct_run/f3_def_configure.txt 2>&1" >> "$R"
cmake -S /home/maxious/Strata -B /tmp/f3_default_build -G Ninja -DSTRATA_BUILD_TESTS=ON > /tmp/dpct_run/f3_def_configure.txt 2>&1
rc=$?
echo "DEF_CONFIGURE_RC=$rc" >> "$R"

echo "CMD: cmake --build /tmp/f3_default_build > /tmp/dpct_run/f3_def_build.txt 2>&1" >> "$R"
cmake --build /tmp/f3_default_build > /tmp/dpct_run/f3_def_build.txt 2>&1
rc=$?
echo "DEF_BUILD_RC=$rc" >> "$R"

echo "DEFAULT_PHASE_DONE $(date -Is)" >> "$R"
