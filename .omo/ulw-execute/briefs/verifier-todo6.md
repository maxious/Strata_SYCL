# Verifier brief: DoneClaim for todo 6 (plan dsv41-trellis-sg16-perf)

Read-only audit, AT MOST 8 tool calls, append results to the verifier log as you go. Never edit product files; never push.

## Claim
Todo 6 '[~] candidate — B60 measurement complete: parity clean (gateAOT/JIT G1-G5,G7 MATCH RC 0, G6 --neg CAUGHT RC 1); stageA AOT faster at small M (M=2 145 vs 235 us, M=32 72 vs 171 us) but 15-31% slower at M=160 (gate/up 311-360 vs 238-280 us); decode baseline 0.84 one-card / 1.28 two-card tok/s completed runs; default stays OFF; measurement-only, no edits, no commits'. Evidence: /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-6-b60-measurement.log (VERDICT FAIL, perf-regression-at-target-batch).

## Check (append after each)
1. Evidence log has CMD lines with un-piped RC lines (12), us numbers for small M and M=160 on both paths, tok/s only for completed runs, and the VERDICT FAIL line with failure_class.
2. git -C /home/maxious/Strata-wt/dsv41-trellis-w2 log --oneline -1 shows 9b35f119 and status --short shows only the untracked gate_gemv_i8_probe.cpp (no todo-6 commit).
3. Scratch dir /home/maxious/.scratch-t6-st_01a180b3 absent (ls RC 2).

## Verdict
End the verifier log /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/verifier-todo6.log with VERDICT CONFIRMED or VERDICT REJECTED. Never end a turn before the VERDICT line exists.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
