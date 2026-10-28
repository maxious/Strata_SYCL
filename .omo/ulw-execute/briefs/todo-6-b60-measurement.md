# Todo 6 brief: B60 measurement — tile vs legacy at M=160 (plan dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2). Never edit /home/maxious/Strata. Never push. Every strata/icpx/binary invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'. Blocked by todo 5 — start only after todos 3,4,5 land with confirmed verdicts.

## What to do
Run the parity/bench binaries with the tile-path flag off and on on both B60 cards. Record gate/up/down microseconds per shape and full decode tok/s. Adopt the new path default-on ONLY if measured faster AND parity-clean; otherwise keep default off and report the numbers. Baselines (measured on the B60 pair, this checkout): M=160 gate+up 69.5 us f16 / 34.0 us i8 / 52.0 us i8-block, down 34.2 / 17.7 / 26.8 us; M=1 floor gate+up 22.3 / 11.1 us, down 11.9 / 5.9 us; decode 1.65 tok/s one-card, 12.08 two-card.

## Must NOT
Never state a throughput number for a prompt that did not complete.

## Closes / blocks
Closes GAP-1 (measured outcome IS-1). Blocks todo 12.

## References
Bench invocation: xmx_gemm_bench [ty=18] [M=96] [mode=gu|down]; baselines draft F-f in the plan.

## QA scenarios
- happy: flag-off and flag-on runs on both cards, per-shape us + decode tok/s recorded with the prompt completing.
- failure: parity-clean check fails → default stays off, numbers still reported.
Evidence: append every CMD+un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-6-measurement.log ending VERDICT PASS (faster+parity-clean → default-on) or VERDICT FAIL (keep default off, numbers recorded). Never end a turn before the VERDICT line exists.

## Commit
Only if default-on adopted: '[Engine] enable sg16 tile path by default after B60 measurement' on dsv41-trellis-w2 (never push). If default stays off, no product commit — evidence log only.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"} for item key todo6-b60-measurement. rm -rf your scratch dir with ls absence proof RC 2.
