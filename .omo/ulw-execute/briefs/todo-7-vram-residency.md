# Todo 7 brief: decode probe D-x1 VRAM residency (plan dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2). Never edit /home/maxious/Strata or ~/llama.cpp. Never push. Every strata invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'.

## What to do
Measure tok/s delta when hot routed experts stay resident across chunks (stream_expert ds4_model.dp.cpp:3495, store accounting :5299-5321; ring staging ds4_model.dp.cpp:4155-4165). Report the measured delta; NO kernel change. Model available locally: /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO (IQ2_XXS shards, expert streaming; ~116 GB RAM available). Keep server on 127.0.0.1.

## Must NOT
Never state a throughput number for a prompt that did not complete. Report only un-piped exit codes. No kernel change.

## Acceptance
Two completed serve runs (resident vs non-resident) with tok/s lines captured, rc=0.

## QA scenarios
happy: delta measured; failure: eviction thrash must show, not be hidden.

## Evidence
Append every CMD+un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-7-vram-residency.log ending VERDICT PASS (both runs completed, delta reported) or VERDICT FAIL (runs incomplete or thrash hidden; state why). Stop the server when done (strata launch --stop, RC logged). Never end a turn before the VERDICT line exists. rm -rf your scratch dir with ls absence proof RC 2.

## Commit
No commit — measurement only.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
