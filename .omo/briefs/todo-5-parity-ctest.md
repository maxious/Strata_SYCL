# Todo 5 brief — Output parity rows + ctest registration (dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (HEAD def4288b, contains todo 3 kernel + todo 4 dispatch arm).

## Must do
- Add tile-path output parity vs an independent scalar reference built INSIDE ds4_parity.cpp (follow pattern ref_iq_set_gate_up at ds4_parity.cpp:354). Do NOT edit src/dsv41/ds4_tq_cpu.cpp.
- Reference computes from the exact F16 operands staged and the same kb ranges, reduced in fixed order.
- Bound: abs_err <= 32*2^-24*sum_abs + 4*ulp_f32(ref); gate/up F16 out within one BF16 ulp (ds4_parity.cpp:976-983).
- Copy the non-vacuity pattern of ds4_parity.cpp:1149-1267: scale-sensitive data, corrupt-one-scale-byte negative control (:1186-1191), max_rows=32 > kLightRows=16.
- Shapes: M=2,15,16,17,31,32,33,160 with K and N tails.
- Register xmx_group_parity and xmx_gemm_bench as ctest names after sycl/CMakeLists.txt:477-480 (no add_test today).

## Acceptance (assert BOTH directions)
- One ctest run (after source /opt/intel/oneapi/setvars.sh) exits 0 with stdout containing MATCH: for ds4_parity, ds4_blas_parity, xmx_group_parity, xmx_gemm_bench.
- The negative-control invocation exits 1 with stdout containing NO MATCH.
- QA: happy = all parity rc=0; failure = one payload byte flip must make output parity rc=1.

## Environment rules
- Every strata/icpx/cmake/ninja/ctest invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; <command>'.
- AOT build: cmake -S sycl -B <scratch>/buildAOT -G Ninja -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_SYCL_AOT=bmg-g21.
- GPU serialization: pgrep -x strata-dsv41 must return RC 1 before any device run; DEVICE_LOST = contention, retry once.
- Never modify ~/llama.cpp. Never push. Commit title area-prefixed: test(sycl): stageA output parity + ctest registration — if that fails the area rule, use [Engine] prefix instead.
- Scratch dir: /home/maxious/.scratch-t5-<lane-id>; remove at end with ls RC 2 proof.

## Evidence log
/home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-5-dsv41-trellis-sg16-perf.log — one-line CMD+RC entries; end with VERDICT PASS or VERDICT FAIL + failure_class. Never end a turn before the VERDICT line exists. Use real tools, never XML tool-call text.
