# Todo 9 brief: decode probe D-x3 window crossover (plan dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2). Never edit /home/maxious/Strata or ~/llama.cpp. Never push. Every strata/icpx/binary invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'. Repro env: ZES_ENABLE_SYSMAN=1 STRATA_DS4_NO_LIGHT=1 only when the probe needs it.

## What to do
Measure tile-vs-light at T=8 and T=16 for win_experts (ds4_gguf.dp.cpp:2076). Adopt the tile path at those T values ONLY if faster and parity-clean; otherwise report the numbers and adopt nothing. kLightRows change is a separate decision — do NOT touch kLightRows (ds4_iq.dp.cpp:1316-1317) in this todo. Window-verification bits must stay unchanged.

## Must NOT
Never state a throughput number for a prompt that did not complete. Report only un-piped exit codes.

## Acceptance
Measured us per T with rc=0 lines; window-verification bits unchanged.

## QA scenarios
happy: crossover found; failure: no gain => report, no adoption.

## Evidence
Append every CMD+un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-9-window-crossover.log ending VERDICT PASS (crossover adopted, faster+parity-clean) or VERDICT FAIL (no gain or parity issue; numbers recorded, nothing adopted). Never end a turn before the VERDICT line exists. rm -rf your scratch dir with ls absence proof RC 2.

## Commit
Only if adopted: '[Engine] sg16 window crossover at T=8/T=16 after measurement' on dsv41-trellis-w2 (never push). Otherwise no commit.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
