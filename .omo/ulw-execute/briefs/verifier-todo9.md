# Verifier brief: DoneClaim for todo 9 (plan dsv41-trellis-sg16-perf)

Read-only audit, AT MOST 8 tool calls, append results to the verifier log as you go. Never edit product files; never push.

## Claim
Todo 9 '[~] candidate — tile-vs-light crossover measured at T=8/T=16, not adopted: light path statically ineligible for dsv41 (guard requires n_ff % 256 == 0, n_ff=640 -> remainder 128); tile us recorded both flavors with RC 0; kLightRows and window-verification bits unchanged'. Evidence: /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-9-window-crossover.log (VERDICT FAIL).

## Check (append after each)
1. Evidence log contains CMD+un-piped RC lines, tile us at T=8/T=16 (JIT and AOT), and the VERDICT FAIL line.
2. The ineligibility claim: read the light-path guard cited (ds4_iq.dp.cpp around :2514/:2519) in /home/maxious/Strata-wt/dsv41-trellis-w2 and confirm it requires n_ff % 256 == 0.
3. git -C /home/maxious/Strata-wt/dsv41-trellis-w2 diff HEAD -- sycl/src/dsv41/ds4_iq.dp.cpp is empty and log --oneline -1 shows HEAD c0e7439c (no todo-9 commit).

## Verdict
End the verifier log /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/verifier-todo9.log with VERDICT CONFIRMED or VERDICT REJECTED. Never end a turn before the VERDICT line exists.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
