ROLE: you are an independent verifier; do it yourself; do NOT delegate to another agent.
UNIT: independently adjudicate plan todo 3 at commit c0e7439c on branch dsv41-trellis-w2 (worktree /home/maxious/Strata-wt/dsv41-trellis-w2). The executor's log /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-3-attempt4.log ends VERDICT FAIL: gates G1-G6 passed on BOTH flavors (G1 TOTAL outside tolerance: 0, G3 determinism byte-identical, G4 nibbles 16/16, G5 guards, G6 negative control), but G7 (--legacy cross-check vs iq::set_gate_up/set_down) hung twice with UR_RESULT_ERROR_DEVICE_LOST (JIT then AOT) and was killed RC=137. Your job: re-run EVERY gate fresh in your own scratch and decide whether G7's failure is product (the fix broke the legacy path) or harness/environment (transient device fault).

WORKTREE: read-only on /home/maxious/Strata-wt/dsv41-trellis-w2 (HEAD must be c0e7439c; verify with git log --oneline -1 before building). NEVER touch /home/maxious/Strata. Scratch: /home/maxious/.scratch-v4-<your id>.

BUILD (fresh dirs, one per flavor):
cmake -S sycl -B <scratch>/buildAOT -G Ninja -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_SYCL_AOT=bmg-g21
cmake -S sycl -B <scratch>/buildJIT -G Ninja -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_NATIVE_EXPERTS=OFF
ninja strata_dsv41 in each. Every invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; <cmd>'. Long builds: run_in_background:true + bash_output; never a bare sleep.

GATES (run the parity driver's modes on BOTH flavors; device env ZES_ENABLE_SYSMAN=1 STRATA_DS4_NO_LIGHT=1):
G1 shapes caseA/caseB/caseC/caseC2/caseD/caseM17: gate/up within one BF16 ulp, down within kDownAbsTol, TOTAL outside tolerance: 0.
G3 determinism: run each shape twice, byte-identical (cmp RC 0).
G4 fidelity: nibbles 16/16 per shape.
G5 guards: refuse M==1, K=96, K=100, N=20; accept M=17.
G6 negative control: corrupt one scale byte, expect outside-tolerance counts > 0 (non-vacuity).
G7 legacy cross-check (--legacy): compare stageA output vs iq::set_gate_up/set_down within one BF16 ulp. If it hangs or returns UR_RESULT_ERROR_DEVICE_LOST, retry up to 2 times (fresh process each); if a retry passes, that is harness/environment, not product. If it consistently fails with wrong values, that is product.

EVIDENCE: append every command and its UN-PIPED RC incrementally to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-3-attempt4-verify.log (new file; never delete the executor's task-3-attempt4.log). End the log with VERDICT PASS or VERDICT FAIL.
CLEANUP: rm -rf your scratch AFTER the log is complete; print ls absence proof with its un-piped RC.
DELIVER: exactly one JSON object {"verdict":"confirmed|needs-fix|false-positive","failure_class":"product|harness|none","evidence":[...],"repro":"...","confidence":0..1,"coverage_limits":[...]}.

Never report a piped exit code; print each command then its un-piped RC. No throughput claims unless a run completed.
