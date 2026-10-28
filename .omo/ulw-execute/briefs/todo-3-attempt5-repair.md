# Todo 3 attempt 5 — REPAIR the XMX A-stage sg16 tile (product bug)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2, HEAD c0e7439c).
Edit ONLY sycl/src/dsv41/ds4_iq.dp.cpp (the XMX branch under STRATA_XMX_SG16). Do not touch the scalar stand-in path (it is verified correct). Do not touch other files. Commit the fix on this branch with title `[Engine] fix XMX sg16 A-stage multi-expert mapping` (never push).

## Evidence from attempt-4 independent verification (log .omo/evidence/dsv41-trellis-sg16-perf/task-3-attempt4-verify.log)
- JIT flavor (no STRATA_XMX_SG16, scalar stand-in): ALL gates PASS — G1 bit-exact all shapes (gate/up 4096/4096, 4224/4224 fp16 bits; down fp32 exact), G3 byte-identical, G4 nibbles 16/16, G5 guards, G6 negative control, G7 --legacy PASS RC 0.
- AOT flavor (-DSTRATA_XMX_SG16=1, joint_matrix path): G1 RC 1 — half outputs wrong (caseA gate/up 2045/4096 exact, 2048 outside, max diff 65216; down max diff 88485). G3 deterministic (cmp byte-identical RC 0). G6 RC 1 with baseline already wrong (outside=4606, no corruption). G7 --legacy RC 1 on 3 fresh processes (gate/up 15848/15872 outside, max diff inf -> NaN). Never DEVICE_LOST.
- Pattern: outputs for experts past the first get garbage. Suspect the joint_matrix B load or the multi-expert grid mapping: the tile reads the B plane by the expert's LOCAL row (ds4_iq.dp.cpp:1425, bpack[at + 2*(m0+m)]) but writes h/y at the GLOBAL row lo+m0+m (:1443).

## Harness (rebuild it yourself; prior scratch dir was deleted)
Write your own driver in a fresh scratch dir (e.g. /home/maxious/.scratch-v5-<id>). Oracle per repo: sycl/test/ds4_parity.cpp dec_ref_block (:972) + strata::dequantize_iq4_nl (include/strata/artifact/dequant.hpp), F16 matmul ((p0+p1)+p2)+p3 with round-to-nearest-even; tolerance one BF16 ulp for gate/up, kDownAbsTol 1e-3 for down (ds4_parity.cpp:226). Contract traps found by the prior verifier: the DOWN call takes k=N, n_out=K (not k=n_out=n_ff); B plane and outputs are indexed by the expert's LOCAL row for plane reads and GLOBAL row lo+m0+m for outputs; rebuild binaries after every source edit (a stale binary once produced false failures).

## Build gates (both flavors, fresh dirs)
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S /home/maxious/Strata-wt/dsv41-trellis-w2/sycl -B <scratch>/buildJIT -G Ninja -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_NATIVE_EXPERTS=OFF' ; ninja strata_dsv41
Same for buildAOT adding -DSTRATA_SYCL_AOT=bmg-g21. Link drivers against libstrata_dsv41.a + /usr/lib/x86_64-linux-gnu/libze_loader.so (JIT: icpx -fsycl -Xsycl-target-backend=spir64 '-cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required'; AOT: icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen '-device bmg-g21 -options -cl-fp32-correctly-rounded-divide-sqrt').

## Gates to run on BOTH flavors after the fix
G1 tolerance shapes (caseA gate/up, down, and the K%32 tail shapes incl. 1984/1984), G3 determinism (two dumps cmp byte-identical), G4 nibbles 16/16, G5 --guards, G6 --negative control, G7 --legacy (retry up to 2x only on UR_RESULT_ERROR_DEVICE_LOST). Run every gate FOREGROUND inside your turn — never leave run_in_background sessions; the turn must not end before the VERDICT line is written.

## Deliverables
- Append every CMD and its un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-3-attempt5.log; end with VERDICT PASS or VERDICT FAIL.
- Fix commit on dsv41-trellis-w2; show git log -1 and the diff stat (only ds4_iq.dp.cpp).
- rm -rf scratch with ls absence proof (un-piped RC 2).
- Yield keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"} for your pushed item key.
