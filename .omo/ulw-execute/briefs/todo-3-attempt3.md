# Todo 3 attempt 3 - repair lane for `iq::stageA_set` (branch dsv41-trellis-w2, commit cf2ead1c)

Verifier verdict: needs-fix, failure_class product. Two measured defects to fix in
`sycl/src/dsv41/ds4_iq.dp.cpp` (and its hpp if the signature must change).

## Defect 1 - uneven-K split races on the scalar stand-in
Measured: caseC K=160 (nblk=5), N=32, rows 24/17/21 -> stand-in gate/up wrong in exactly
the columns n%4 in {0,1} (reduction threads tid = 16*(n%4)+m, i.e. tid<32), reruns
byte-identical. Packing is proven a bijection, the same plane is correct on the AOT
(XMX) flavor, and caseC2 K=256 is bit-exact on the stand-in.
Root cause (static): work-group barriers inside the per-sub-group kb loop have differing
trip counts when K/32%4 != 0 (kb0 = (s*nblk)/4), so the four-partial reduction races
sub-group 3's partial store.
Fix: make every sub-group execute the same number of barrier points (hoist the barriers
out of the uneven loop, or restructure the partial reduction so all 4 sub-groups reach the
same barriers), without changing the XMX path's semantics.

## Defect 2 - non-deterministic down on both flavors
Measured: same binary, same inputs, two runs differ byte-for-byte (RC_DIFF_AOT_RUNS=1,
RC_DIFF_JIT_RUNS=1); persists with fresh device buffers, serialized launches, plane slack.
Candidate (static): no barrier between `load_tab<T>(tab)` (written by tid<4) and the first
`deq8` read of `tab`, whereas the proven `set_tc_kernel` has `item_ct1.barrier()` exactly
there (ds4_iq.dp.cpp:1107).
Fix: add the missing barrier(s) at that point in `set_tc_stageA_kernel` (both flavors),
matching the proven kernel's placement.

## Scope rules
- Touch only `sycl/src/dsv41/ds4_iq.dp.cpp` and `ds4_iq.dp.hpp`. Do not edit
  `set_gate_up`, `set_down`, `is_tq`, or any dispatch wiring. Keep `kSetSmemStageA*`
  constants and the `Tiles`/`Out` static_asserts intact.
- Keep `[[sycl::reqd_sub_group_size(16)]]`, 64 threads / 4 sub-groups, and the guards
  (refuse M==1, K%32!=0, K/32<4, n_out%16!=0).
- Gate/up parity is judged against the repo's one-BF16-ulp standard (plan todo 5); the
  XMX 1-F16-ulp reduction-order flips are NOT a defect and must not be "fixed" by
  reordering the XMX accumulation.

## Gates to run (fresh dirs, un-piped RCs, setvars.sh + ZES_ENABLE_SYSMAN=1)
1. AOT build: cmake -S sycl -B <fresh> -G Ninja -DCMAKE_CXX_COMPILER=icpx
   -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_AOT=bmg-g21 -DSTRATA_NATIVE_EXPERTS=OFF;
   ninja strata_dsv41 -> RC 0. Same without STRATA_SYCL_AOT into a second fresh dir -> RC 0.
2. Determinism: run the same driver twice per flavor, byte-diff the reports -> RC 0.
3. Uneven-K: caseC K=160 N=32 rows{24,17,21} bit-exact within the one-BF16-ulp gate/up
   standard and down within tolerance on BOTH flavors.
4. Even cases: caseA K=128 N=32 rows{16,16}; caseC2 K=256 rows{24,17,21};
   caseD K=256 N=16 rows{8,1,5} -> pass on both flavors.
5. Negative control: flip one plane half, device readback matches, exactly the expected
   output row changes -> proves corruption reaches the kernel.
6. Guards: refuse M==1, K=96, K=100, N=20; accept M=17.
7. Static_assert control: drift kSetSmemStageAOutDn by +16 in a scratch copy, compile with
   the AOT build's own compile command -> RC 1 with message
   "Out drifted from the kSetSmemStageA* constants".
8. Legacy cross-check against proven iq::set_gate_up/set_down using plain row-major x rows.

## Deliverables
- One commit on branch dsv41-trellis-w2 (never pushed), message
  `[Engine] fix sg16 A-stage barrier races for dsv41`.
- Evidence log at `/home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-3-attempt3.log`
  ending with `VERDICT PASS` or `VERDICT FAIL`, every gate's un-piped RC line printed
  directly above its RC.
- Scratch dir deleted with an ls absence proof. Never modify /home/maxious/llama.cpp;
  never link its prebuilt libggml-base.so; reference must be dec_ref_block from
  sycl/src/dsv41/ds4_parity.cpp plus an own F16 matmul with reduction ((p0+p1)+p2)+p3.
- Do not end the turn on a promise of a future notification: read your own logs and finish.
