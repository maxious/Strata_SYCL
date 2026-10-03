#!/usr/bin/env bash
# Configure + build the SYCL port inside strata-sycl-dev.  sycl/build.sh [target...]
set -uo pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
repo=${REPO:-/work/Strata_B70}
b=${BUILD_DIR:-$repo/build-sycl}
[ -f $b/build.ninja ] || cmake -S $repo/sycl -B $b -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DSTRATA_SYCL_AOT="${AOT:-}" 2>&1 | tail -15
cmake --build $b -j ${JOBS:-12} ${1:+--target "$@"} -- -k 0 2>&1 | tee $b/build.log | grep -E '^FAILED|error:|^ninja: build stopped|Linking|^\[[0-9]+/[0-9]+\] Linking' | tail -40
echo "BUILD EXIT ${PIPESTATUS[0]}"
grep -c 'error:' $b/build.log | sed 's/^/errors: /'
grep -E '^FAILED' $b/build.log | sed 's#.*/##' | sort | uniq | head -60

# XMX CI gate (README P2): the fused-dequant XMX quantized GEMM must stay parity-clean
# against dequant + oneMKL. xmx_gemm_bench already returns non-zero when more than
# 0.1% of its elements fall outside 1% (or the XMX path is refused/can't run), so an
# XMX change that regresses the quantized GEMM fails the build like any other kernel.
# STRATA_XMX_GATE=0 turns this check off.
if [ "${STRATA_XMX_GATE:-1}" != "0" ] && [ -x $b/xmx_gemm_bench ]; then
  xmx_fail=0
  for ty in 18 20 21 22 42 7 8 12 13; do
    if ! $b/xmx_gemm_bench $ty > $b/gate-$ty.log 2>&1; then
      echo "XMX GATE FAIL (type $ty):"
      tail -3 $b/gate-$ty.log
      xmx_fail=1
    fi
  done
  if [ $xmx_fail != 0 ]; then
    echo "BUILD EXIT 1"; exit 1
  fi
  echo "XMX GATE OK"; rm -f $b/gate-*.log
fi
