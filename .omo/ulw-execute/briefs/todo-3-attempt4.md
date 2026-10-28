# Todo 3 repair attempt 4 - JIT scalar stand-in uneven-K tail

Scope: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2, HEAD 3ff0bd81).
Only sycl/src/dsv41/ds4_iq.dp.cpp / ds4_iq.dp.hpp may change. Never push. Never edit /home/maxious/Strata.

Measured defect (verifier st_01a17e8b, durable log .omo/evidence/dsv41-trellis-sg16-perf/task-3-verify.log):
- JIT flavor (no -DSTRATA_SYCL_AOT, the #ifndef STRATA_XMX_SG16 scalar stand-in) caseC K=160/N=32 rows{24,17,21}:
  gate/up 992/1984 fp16 bits exact, 992 outside, max diff 0.207458; down 992 exact, 9 within kDownAbsTol,
  983 outside, max diff 0.761062. Per-row dump: 16 of 32 columns wrong on every row (uneven split nblk=5 vs kStageASplit=4).
- Same counts byte-identical at cf2ead1c baseline JIT -> the defect PRE-EXISTS the sg16 change.
- AOT/XMX flavor at 3ff0bd81 passes all six shapes, 0 outside; determinism cmp RC=0 everywhere; guards PASS both
  flavors; negative control reaches the kernel both flavors; legacy cross-check within one BF16 ulp.

Task: fix the scalar stand-in's uneven-K split/tail so caseC K=160 is bit-exact (gate/up within one BF16 ulp,
down within kDownAbsTol ds4_parity.cpp:226) on the JIT flavor, without touching the XMX kernel path.

Gates (each command printed directly above its un-piped RC, appended to
/home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-3-attempt4.log, ending VERDICT PASS or VERDICT FAIL):
1. Fresh buildAOT (-DSTRATA_SYCL_AOT=bmg-g21) and buildJIT (without it), -DSTRATA_SYCL_DSV41=ON -DSTRATA_NATIVE_EXPERTS=OFF, ninja strata_dsv41 -> RC 0 each.
2. Driver: x = BYTE values scaled ~±0.05; oracle = dec_ref_block-equivalent IQ4_NL decode + F16 matmul ((p0+p1)+p2)+p3; plane per expert at bstride/2 uint16_t elements; blob slab per expert at stride bytes.
3. Shapes on BOTH flavors: caseA K=128/N=32 rows{16,16}, caseB K=640/N=48 rows{40}, caseC K=160/N=32 rows{24,17,21},
   caseC2 K=256/N=32 rows{24,17,21}, caseD K=256/N=16 rows{8,1,5}, caseM17 K=128/N=32 rows{17}.
4. Determinism: two runs per shape per flavor, cmp -s -> RC 0.
5. Guards: refuse M==1, K=96, K=100, N=20; accept M=17.
6. Negative control: flip one plane half, device readback matches, expected output rows move, unsaturated data.
7. Legacy cross-check vs iq::set_gate_up/set_down on plain row-major x rows (K=256/N=256 IQ4_XS), within one BF16 ulp.

Deliverables: commit `[Engine] fix sg16 A-stage scalar stand-in uneven-K tail` on dsv41-trellis-w2 (COMMIT_RC printed, never pushed);
the durable log ending VERDICT PASS/FAIL; scratch dir deleted with an absence proof.
Device runs inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 STRATA_DS4_NO_LIGHT=1 ...'.
Never link /home/maxious/llama.cpp's prebuilt libggml-base.so. If the fix cannot be made without touching the XMX kernel,
report VERDICT FAIL with the reason and propose marking the checkbox [~] with a one-line gap instead.
