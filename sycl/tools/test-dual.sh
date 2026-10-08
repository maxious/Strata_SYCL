#!/usr/bin/env bash
# One row of the dual (--layer-split auto) config, which is the #1054 gate: does the latter card's expert cache open,
# does the engine reach "session is up", and does it survive the first request?
#   sycl/tools/test-dual.sh <arm-name>
set -uo pipefail
b=$(cd "$(dirname "$0")/../.." && pwd)
arm=${1:?usage: test-dual.sh <arm-name>}
cd "$b"
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; set -u
echo "=== [$arm] head $(git rev-parse --short HEAD) ==="
cmake --build build-sycl -j 12 --target strata 2>&1 | tail -3
echo "engine md5: $(md5sum build-sycl/strata | cut -d' ' -f1)"
sudo pkill -x strata; sleep 3
sudo rm -rf "$b/sycl/benchy-results/dual-$arm"
sudo env PATH=$PATH STRATA_NATIVE=1 STRATA_SYCL_BIN="$b/build-sycl/strata" \
  python3 sycl/tools/perf_matrix.py --configs strata-q2_0-dual.json --sizes 20 \
  --out "$b/sycl/benchy-results/dual-$arm" 2>&1 | tail -6
echo '--- the decisive startup lines ---'
sudo grep -E 'layer split, CUDA|expert cache|session is up|ENGINE EXIT|DEVICE_LOST|Did not complete|"exit"' \
  "$b/sycl/benchy-results/dual-$arm"/q2_0-dual-20.log "$b/sycl/benchy-results/dual-$arm"/matrix.jsonl 2>/dev/null | tail -14
sudo chown -R "$(id -u):$(id -g)" "$b/sycl/benchy-results"
