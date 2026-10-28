# Verifier brief: DoneClaim for todo 13 (plan dsv41-trellis-sg16-perf)

Read-only audit, AT MOST 8 tool calls, write the log EARLY and append as you go. Never edit product files; never push.

## Claim
Todo 13 '[x] candidate — trellis A-stage feasibility probe delivered: measured us comparison of portable-FMA tq:: lane vs joint_matrix A-stage probe tile, plus written contract note; measurement-only, no commit, no adoption'. Evidence: /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-13-trellis-a-stage-probe.log (VERDICT PASS).

## Check (in this order, appending results to the verifier log after each)
1. Evidence log contains CMD+un-piped RC lines and the VERDICT PASS line.
2. It contains a measured us comparison (both variants) and a contract note about the 128-value {fp16 d; 4K stream bytes} block shape vs the 32-value contract.
3. git -C /home/maxious/Strata-wt/dsv41-trellis-w2 log --oneline -1 shows HEAD unchanged (c0e7439c) and status shows no todo-13 product edits.

## Verdict
End the verifier log /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/verifier-todo13.log with VERDICT CONFIRMED or VERDICT REJECTED. Never end a turn before the VERDICT line exists.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
