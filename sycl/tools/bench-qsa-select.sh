#!/usr/bin/env bash
# Kernel-level control for the TK_PER_MAX change: runs qsa_select_bench at the shipped capacity (262,144 cells)
# through 135,168 (the old register-kernel cliff edge), 200,000 and 262,144 cells, and reports the top-k time and
# the register kernel's id agreement with the reference.
set -uo pipefail
b=$(cd "$(dirname "$0")/../.." && pwd)
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; set -u
bin="$b/build-sycl/qsa_select_bench"
if [ ! -x "$bin" ]; then echo "build it first: cmake --build $b/build-sycl --target qsa_select_bench" >&2; exit 2; fi
echo "bin: $bin  ($(stat -c %y "$bin"))"
for ctx in 131072 135168 200000 262144; do
    echo "== ctx=$ctx capacity=262144 reps=200 =="
    ONEAPI_DEVICE_SELECTOR=level_zero:0 "$bin" "$ctx" 1 200 262144 2>&1 \
        | grep -E 'top-k|ctx |refused|not available' | head -4
done
