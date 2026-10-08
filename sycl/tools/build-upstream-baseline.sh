#!/usr/bin/env bash
# Configure + build the SYCL port of REAL UPSTREAM (Niko1221/Strata main) for the baseline benchmark.
# Mirrors the local build-b60 configuration: icx/icpx, Release, SPIR-V JIT (no AOT), ccache on.
set -uo pipefail
src=$(cd "$(dirname "$0")/../.." && pwd)
log="$src/sycl/benchy-results/upstream-baseline-build.log"
mkdir -p "$(dirname "$log")"
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; set -u
{
  echo "=== $(date -Is) configure ==="
  cmake -S "$src/sycl" -B "$src/build-sycl" -G Ninja \
    -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_SYCL_AOT= \
    -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=/home/maxious/Strata_SYCL/sycl/build-b60/_deps/strata_llamacpp-src
  echo "CONFIGURE_EXIT=$?"
  echo "=== $(date -Is) build ==="
  cmake --build "$src/build-sycl" -j "${JOBS:-12}" --target strata
  echo "BUILD_EXIT=$?"
  ls -la "$src/build-sycl/strata" && "$src/build-sycl/strata" --help >/dev/null 2>&1; echo "RUN_EXIT=$?"
  echo "=== $(date -Is) done ==="
} >"$log" 2>&1
tail -5 "$log"
