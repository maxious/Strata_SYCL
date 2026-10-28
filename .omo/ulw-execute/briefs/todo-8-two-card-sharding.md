# Todo 8 brief: decode probe D-x2 two-card sharding (plan dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2). Never edit /home/maxious/Strata or ~/llama.cpp. Never push. Every strata invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'.

## What to do
Verify --parallel/--gpu actually splits expert sets for the SYCL engine; capture tok/s for one vs two cards from completed prompts. Prior measured reference: 1.65 tok/s one-card, 12.08 two-card (draft F-f). Model available locally: /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO (IQ2_XXS shards). Keep server on 127.0.0.1.

## Must NOT
Never state a throughput number for a prompt that did not complete. Report only un-piped exit codes. A silent single-card fallback must be reported, not hidden.

## Acceptance
Completed runs on both cards, rc=0, sharding visible in the engine log.

## QA scenarios
happy: two-card faster; failure: silent single-card fallback reported.

## Evidence
Append every CMD+un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-8-two-card-sharding.log ending VERDICT PASS (sharding verified + completed runs on both cards) or VERDICT FAIL (fallback or incomplete runs; state why). Stop the server when done (strata launch --stop, RC logged). Never end a turn before the VERDICT line exists. rm -rf your scratch dir with ls absence proof RC 2.

## Commit
If a sharding defect is found and fixed minimally: '[Engine] fix SYCL two-card expert sharding' on dsv41-trellis-w2 (never push). Measurement-only outcome: no commit.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
