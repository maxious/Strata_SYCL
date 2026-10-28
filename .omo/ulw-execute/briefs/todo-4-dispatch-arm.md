# Todo 4 brief: dispatch arm behind kill-switch and eligibility mask (plan dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2). Never edit /home/maxious/Strata. Never push. Every icpx/cmake/ninja/ctest run inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'. Blocked by todo 3 — start only after todo 3 lands with a confirmed verdict.

## What to do
Route eligible types to set_tc_stageA in iq::set_gate_up (ds4_iq.dp.cpp:2265) and iq::set_down (:2298) behind an env kill-switch STRATA_DS4_NO_STAGEA mirroring tc_ok()/light_on() at :1533/:1864, plus a per-type eligibility mask: a type enters the tile path only when its parity row exists (precedent: the kIQ4_NL refusal at :2267).

## Must NOT
- trellis branch is_tq() (:2266/:2282/:2297-2307) stays on tq:: untouched.
- Keep light kernels for experts <= kLightRows=16 (:1316); threshold unchanged this plan.
- Do not touch dense gemv sites in ds4_model.dp.cpp (:8936/:11305/:11315/:11412).
- Do not change SetArgs or the ds4_gguf.dp.cpp set_gate_up (:2173)/set_down (:2236) contract.

## Closes / blocks
Closes GAP-3. Blocks todos 5 and 6.

## Acceptance criteria
With STRATA_DS4_NO_STAGEA unset, existing ctest suite output byte-identical to the baseline run captured before todo 3.

## QA scenarios
- happy: flag off => legacy kernel selected (log line shows legacy route).
- failure: flag on with an ineligible type must fall back to legacy, logged.
Evidence: append every CMD+un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-4-dispatch.log ending VERDICT PASS or VERDICT FAIL. Baseline capture: run the ctest suite on the pre-change HEAD first, save output hash, re-run after change, diff must be empty with flag unset.

## Commit
'[Engine] route eligible iq types to sg16 stage-A behind kill-switch' on dsv41-trellis-w2 (never push).

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"} for item key todo4-dispatch-arm. Never end a turn before the VERDICT line exists. rm -rf your scratch dir with ls absence proof RC 2.
