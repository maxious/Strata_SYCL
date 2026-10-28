# Verifier brief: DoneClaim for todo 3 (plan dsv41-trellis-sg16-perf)

Read-only audit, AT MOST 8 tool calls, append results to the verifier log as you go. Never edit product files; never push.

## Claim
Todo 3 '[x] candidate — sg16 i8 A-stage tile kernel passes the formal gate suite on JIT and AOT flavors (G1-G7 all MATCH, negative control RC 1 on both flavors); attempt-6 failure was gate-criteria (ulp(0)=0 tally), kernel itself correct; harness committed as 9b35f119 on dsv41-trellis-w2 (one new file sycl/src/dsv41/ds4_stageA_gate.cpp, no kernel edit)'. Evidence: /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-3-attempt7.log (VERDICT PASS).

## Check (append after each)
1. Evidence log contains CMD lines with un-piped RC lines for fresh builds and the four final runs (gateAOT RC 0, gateJIT RC 0, gateAOT --neg RC 1, gateJIT --neg RC 1), gate tallies for M=2,15,16,17,31,32,33,160 and dense k=128, and the VERDICT PASS line.
2. git -C /home/maxious/Strata-wt/dsv41-trellis-w2 show --stat --format='%h %s' 9b35f119: commit exists, title area-prefixed ([Engine]), touches only sycl/src/dsv41/ds4_stageA_gate.cpp (no kernel file).
3. git -C /home/maxious/Strata-wt/dsv41-trellis-w2 log --oneline -1 shows 9b35f119; status --short shows only the untracked gate_gemv_i8_probe.cpp.

## Verdict
End the verifier log /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/verifier-todo3.log with VERDICT CONFIRMED or VERDICT REJECTED. Never end a turn before the VERDICT line exists.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
