# Todo 13 brief: Trellis A-stage feasibility probe (measurement only) (plan dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2). Never edit /home/maxious/Strata or ~/llama.cpp. Never push. Every strata/icpx/binary invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'.

## What to do
Measure whether a joint_matrix A-stage tile can serve the trellis types (TQK6/TQK7/TQK5X/TQK4X/TQK5T/TQK4T, ds4_iq.dp.hpp:30), whose blocks are 128-value {fp16 d; 4K stream bytes} decoded through 15-bit state windows (ds4_tq.dp.cpp:46-56/:63-80) — a different block shape than the 32-value contract. Deliver a measured us comparison of the current portable-FMA tq:: lane against a probe tile, plus a written contract note. Adoption returns to the approval gate — this todo is measurement only.

## Must NOT
Must NOT edit src/dsv41/ds4_tq_cpu.cpp. Must NOT change trellis decode semantics or adopt anything. Never state a throughput number for a prompt that did not complete. Report only un-piped exit codes.

## Acceptance
Measured us comparison lines with rc=0 plus a written contract note (in the evidence log; no product commit required).

## QA scenarios
happy: tile feasible and faster — note contract, adoption deferred to approval gate; failure: block-shape mismatch makes the tile infeasible — documented with numbers.

## Evidence
Append every CMD+un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-13-trellis-a-stage-probe.log ending VERDICT PASS (measurement delivered with contract note) or VERDICT FAIL (measurement could not be completed; state why). Never end a turn before the VERDICT line exists. rm -rf your scratch dir with ls absence proof RC 2.

## Commit
No product commit. Probe code stays in your scratch dir; the contract note lives in the evidence log.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
