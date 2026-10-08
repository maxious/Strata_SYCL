#!/usr/bin/env bash
# One arm of an A/B: build the engine and the qsa_select_bench control, prove the binary is the one just built and
# which lever it carries, run the kernel control, and (when given) the perf_matrix rows.
#   sycl/tools/arm-run.sh <arm-name> [comma-separated prompt sizes]
set -uo pipefail
b=$(cd "$(dirname "$0")/../.." && pwd)
arm=${1:?usage: arm-run.sh <arm-name> [sizes]}
sizes=${2:-}
cd "$b"
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; set -u
echo "=== [$arm] head $(git rev-parse --short HEAD) ==="
echo "=== [$arm] build ==="
cmake --build build-sycl -j 12 --target strata qsa_select_bench 2>&1 | tail -3
echo "engine md5: $(md5sum build-sycl/strata | cut -d' ' -f1)"
printf 'STRATA_TK_PER_MAX in the binary: '; strings build-sycl/strata | grep -c STRATA_TK_PER_MAX || true
printf 'sources newer than the binary (want empty): '; find sycl/src sycl/include -newer build-sycl/strata -type f | head -3
echo "=== [$arm] kernel control ==="
sycl/tools/bench-qsa-select.sh
if [ -n "$sizes" ]; then
    echo "=== [$arm] perf_matrix --sizes $sizes ==="
    sudo env PATH=$PATH STRATA_NATIVE=1 STRATA_SYCL_BIN="$b/build-sycl/strata" \
      python3 sycl/tools/perf_matrix.py --configs strata-q2_0.json --sizes "$sizes" \
      --out "$b/sycl/benchy-results/v1-$(date +%F)-$arm" 2>&1 | tail -30
    sudo chown -R "$(id -u):$(id -g)" "$b/sycl/benchy-results"
fi
