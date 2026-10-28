# Todo 10 brief: decode probe D-x4 gate gemv i8 (plan dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2). Never edit /home/maxious/Strata or ~/llama.cpp. Never push. Every strata/icpx/binary invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'.

## What to do
Probe an i8 gemv variant for the f32 router gate (ds4_model.dp.cpp:8936, baseline 22.3 us at M=1; gemv_f32 sites ds4_model.dp.cpp:8936, 11305, 11315, 11412). Measure us for BOTH variants at M=1 and check answers are bit-identical. Only worth adopting if bytes read drop AND it is faster; adoption requires measurement — measure, do not adopt without it.

## Must NOT
Never state a throughput number for a prompt that did not complete. Report only un-piped exit codes. No adoption without measurement.

## Acceptance
Measured us for both variants, rc=0 lines, answers bit-identical.

## QA scenarios
happy: i8 faster; failure: no gain or bit drift => report only, nothing adopted.

## Evidence
Append every CMD+un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-10-gate-gemv-i8.log ending VERDICT PASS (i8 faster + bit-identical; adoption decision recorded) or VERDICT FAIL (no gain or bit drift; numbers recorded, nothing adopted). Never end a turn before the VERDICT line exists. rm -rf your scratch dir with ls absence proof RC 2.

## Commit
Only if adopted: '[Engine] i8 gemv for router gate after measurement' on dsv41-trellis-w2 (never push). Probe-only runs: no commit.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
