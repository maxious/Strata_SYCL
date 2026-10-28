# Todo 3 attempt-6 repair brief (AOT sg16 A-stage defect)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2, HEAD c0e7439c). Never edit /home/maxious/Strata. Never push. Scratch dir: your own /home/maxious/.scratch-a6-<id>. Every icpx/cmake/ninja/ze run inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'.

## Prior findings (attempt 5, VERDICT FAIL, evidence task-3-attempt5.log)
- Multi-expert mapping VERIFIED correct on both flavors (--diag + G1 pass incl. down k=512). No mapping fix needed; kernel git-clean at c0e7439c.
- binAOT (-DSTRATA_XMX_SG16=1, spir64_gen bmg-g21): outputs ALL ZERO on one-hot DOWN probe at k=128 (Shape{2,{16,16},128,128}); deterministic across two fresh runs. Same driver on JIT scalar stand-in: exact RC 0.
- Nondeterministic SIGSEGV inside libsycl DeviceKernelInfo::setCompileTimeInfoIfNeeded on queue::submit of launch_stageA<kIQ4_NL,false> in the AOT binary.
- UR_RESULT_ERROR_OUT_OF_RESOURCES when the light route runs after tile kernels are resident (G5 fallback, G7).

## Repro recipe (from attempt 5)
Driver: one-hot plane at (k=m,m), oracle strata::dequantize_iq4_nl(row r)[m%32]; down contract k=N n_out=K; plane LOCAL rows, outputs GLOBAL rows; link against flavor libstrata_dsv41.a + libze_loader.so; run ZES_ENABLE_SYSMAN=1 STRATA_DS4_NO_LIGHT=1.

## Your job
1. Reproduce the k=128 DOWN zero-output on a fresh AOT build (buildAOT: cmake -S sycl -B <scratch>/buildAOT -G Ninja -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_SYCL_AOT=bmg-g21; ninja strata_dsv41; then driver compile+link per recipe). Confirm RC 1 zero outputs.
2. Root-cause it. Investigate, in order: (a) the XMX sg16 A-stage block in sycl/src/dsv41/ds4_iq.dp.cpp (~lines 1285-1493) for how B-plane rows are read when k=128 spans 4 blocks vs k=512; (b) ext_intel_packed / joint_matrix packed-B layout for the sparse one-hot plane on the AOT path (JIT stand-in differs from XMX here); (c) the SIGSEGV: check launch_stageA<kIQ4_NL,false> kernel registration for AOT (per_kernel device-code-split, template instantiation, host-side get_pointer patterns — the known local_accessor::get_pointer-on-host defect family in ds4_tq.dp.cpp/ds4_iq.dp.cpp launch helpers). Pin the root cause with a minimal experiment (e.g. dense-plane k=128 DOWN probe, or a single-block k=32 DOWN probe) before editing.
3. Fix minimally in the kernel/launch helpers (only files under sycl/src/dsv41). Commit '[Engine] fix XMX sg16 A-stage AOT path' ONLY if root cause confirmed by experiment; otherwise leave git-clean and report.
4. Rebuild BOTH flavors fresh (JIT: same cmake without AOT flag; AOT with -DSTRATA_SYCL_AOT=bmg-g21), relink drivers.
5. Run gates FOREGROUND one at a time, both flavors: G1 tolerance shapes, G3 determinism, G4 nibbles, G5 guards, G6 negative control, G7 legacy (retry G7 up to 2x only on UR_RESULT_ERROR_DEVICE_LOST). Append every CMD+un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-3-attempt6.log; write VERDICT PASS or VERDICT FAIL as the last line. Never end a turn before the VERDICT line exists.
6. rm -rf your scratch dir; ls -d must return RC 2.
7. Yield keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"} for your item key.

## Success criteria
VERDICT PASS requires: G1+G3+G4 exact/tolerance MATCH on BOTH flavors, G5 guards pass, G6 negative control caught, G7 legacy runs (or documented device-stack limitation), commit landed if kernel changed.
