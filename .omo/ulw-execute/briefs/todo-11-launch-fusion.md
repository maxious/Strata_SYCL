# Todo 11 brief: decode probe D-x5 launch fusion (plan dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2). Never edit /home/maxious/Strata or ~/llama.cpp. Never push. Every strata/icpx/binary invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'.

## What to do
Count per-token launches (q8_quant, gate, up, down, MoE routing; entry points ds4_gguf.dp.cpp:1991, 2031, 2076; q8_quant :280) and measure fusing gate+up into one grouped launch. Adopt the fusion ONLY if measured faster.

## Must NOT
Never state a throughput number for a prompt that did not complete. Report only un-piped exit codes. No adoption without measurement.

## Acceptance
Launch counts captured plus measured tok/s delta, rc=0 lines.

## QA scenarios
happy: fusion faster; failure: no gain => report, no adoption.

## Evidence
Append every CMD+un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-11-launch-fusion.log ending VERDICT PASS (fusion adopted, faster) or VERDICT FAIL (no gain or parity break; counts and numbers recorded, nothing adopted). Never end a turn before the VERDICT line exists. rm -rf your scratch dir with ls absence proof RC 2.

## Commit
If adopted: '[Engine] fuse gate+up into one grouped launch after measurement' on dsv41-trellis-w2 (never push). If not adopted: bench/probe evidence only, no commit.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
