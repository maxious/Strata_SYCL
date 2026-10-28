# Todo 5 brief: tile-path output parity rows + ctest registration (plan dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2). Never edit /home/maxious/Strata. Never push. Every icpx/cmake/ninja/ctest run inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'. Blocked by todos 3 and 4 — start only after both land with confirmed verdicts.

## What to do
Tile-path output parity vs an independent scalar reference built inside ds4_parity.cpp (pattern: ref_iq_set_gate_up at :354; do NOT edit src/dsv41/ds4_tq_cpu.cpp), from the exact F16 operands staged and the same kb ranges, reduced in fixed order. Bound: abs_err <= 32*2^-24*sum_abs + 4*ulp_f32(ref); gate/up F16 output within one BF16 ulp (ds4_parity.cpp:976-983). Copy the non-vacuity pattern of ds4_parity.cpp:1149-1267: scale-sensitive data, corrupt-one-scale-byte negative control (:1186-1191), max_rows=32 > kLightRows=16. Shapes M=2,15,16,17,31,32,33,160 with K and N tails. Register xmx_group_parity and xmx_gemm_bench as ctest names after sycl/CMakeLists.txt:477-480 (no add_test today).

## Closes / blocks
Closes GAP-4. Blocks todo 6.

## Acceptance criteria
One ctest run (after source /opt/intel/oneapi/setvars.sh) exits 0 with stdout containing MATCH: for ds4_parity, ds4_blas_parity, xmx_group_parity and xmx_gemm_bench. B60 device lock per ds4_blas_parity.cpp:29. Tolerances: xmx_gemm_bench.cpp:89-91 (1% rel, 0.1% element budget), xmx_group_parity.cpp:115.

## QA scenarios
- happy: all four MATCH lines, ctest RC 0.
- failure: corrupt-one-scale-byte negative control must be caught (non-vacuity proof logged).
Evidence: append every CMD+un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-5-parity.log ending VERDICT PASS or VERDICT FAIL. Never end a turn before the VERDICT line exists.

## Commit
'[Engine] add sg16 tile-path parity rows and ctest registration' on dsv41-trellis-w2 (never push).

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"} for item key todo5-parity-ctest. rm -rf your scratch dir with ls absence proof RC 2.
