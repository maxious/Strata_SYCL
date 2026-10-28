# Verifier brief: DoneClaim for todo 8 (plan dsv41-trellis-sg16-perf)

Read-only audit, AT MOST 8 tool calls, append results to the verifier log as you go. Never edit product files; never push. Every strata invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'.

## Claim
Todo 8 '[x] candidate — two-card SYCL expert sharding verified: layer split CUDA0 [0,20] / CUDA1 [21,39]+head via describe(); completed prompts one-card 1.41 tok/s vs two-card 4.10 tok/s (--experts 5, rc=0); no silent single-card fallback; measurement-only, no commit'. Evidence: /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-8-two-card-sharding.log (VERDICT PASS).

## Check (append after each)
1. Evidence log contains CMD lines with un-piped RC_ENGINE/RC_DRIVER lines, tok/s only for completed runs, describe() sharding lines, and the VERDICT PASS line.
2. git -C /home/maxious/Strata-wt/dsv41-trellis-w2 log --oneline -1 shows c0e7439c and status --short shows only the untracked gate_gemv_i8_probe.cpp (no todo-8 commit).
3. No strata-dsv41 process is still running: pgrep -af strata-dsv41 must return RC 1.

## Verdict
End the verifier log /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/verifier-todo8.log with VERDICT CONFIRMED or VERDICT REJECTED. Never end a turn before the VERDICT line exists.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
