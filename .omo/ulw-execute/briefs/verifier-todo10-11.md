# Verifier brief: DoneClaims for todos 10 and 11 (plan dsv41-trellis-sg16-perf)

Read-only audit. Never edit product files; never push. Every strata/icpx/binary invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'.

## Claims to verify
- Todo 10: '[~] candidate — i8 gemv router-gate probe measured, not adopted (bit drift)'. Evidence: /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-10-gate-gemv-i8.log. Plan acceptance: measured us for both variants, rc=0, answers bit-identical. Claimed gap: bit-identical impossible for per-row int8 of f32 gate weights (0/256, max abs diff 0.942); timing noise-level.
- Todo 11: '[~] candidate — launch counts captured, fusion already shipped, no delta to adopt; no end-to-end tok/s'. Evidence: /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-11-launch-fusion.log. Plan acceptance: launch counts captured plus measured tok/s delta, rc=0. Claimed gap: gate+up already fused (gate_up_kernel ds4_iq.dp.cpp:710-803, mx_gate_up_kernel ds4_gguf.dp.cpp:707-); end-to-end decode tok/s not measured.

## Check
1. Both logs exist, contain CMD+un-piped RC lines and a VERDICT line.
2. The claimed gaps are consistent with the plan's acceptance criteria and QA failure scenarios (report-only outcomes are legitimate).
3. Spot-check the fusion claim by reading the cited source regions in /home/maxious/Strata-wt/dsv41-trellis-w2 (single submit per gate+up path).
4. Confirm no product commit was made for these todos (git log/status on dsv41-trellis-w2 must show no todo-10/11 commits; probe file untracked is acceptable).

## Verdict
Print VERDICT CONFIRMED (both claims correctly [~] with accurate one-line gaps) or VERDICT REJECTED with the specific failing item per todo. Write your audit to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/verifier-todo10-11.log ending with the VERDICT line. Never end a turn before the VERDICT line exists.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
