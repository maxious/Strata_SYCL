# Todo 12 brief — Docs measured-report update (dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (HEAD def4288b). Docs only — do NOT touch src/, sycl/, tests/.

## Must do
- Update docs/DETAILS.md with every measured number from todos 6-11 and 13, each tagged with what it was measured on (B60 pair / single B60, build commit) per docs discipline in AGENTS.md.
- Mark unconfirmed static-analysis findings separately from measured claims.
- Must NOT publish a number not captured in an evidence log. Every number in the diff must map to a captured evidence-log line under /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/ (task-6-b60-measurement.log, task-7-vram-residency.log, task-8-two-card-sharding.log, task-9-window-crossover.log, task-10-gate-gemv-i8.log, task-11-launch-fusion.log, task-13-trellis-a-stage-probe.log).
- Include the todo-13 hazard note (kernel ordering constraint) and the todo-6 gap (stageA slower at M=160, default stays OFF).

## Acceptance
- Docs diff obeys the measured-numbers rule: every number maps to an evidence line; no unmeasured claim appears.

## Rules
- Commit title: docs(sycl): measured stageA and decode numbers — if the area rule rejects it, use [Docs] prefix. Never push.
- Evidence log: /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-12-dsv41-trellis-sg16-perf.log — one-line CMD+RC entries, mapping table number->evidence line, end with VERDICT PASS or VERDICT FAIL + failure_class. Never end a turn before the VERDICT line exists. Use real tools, never XML tool-call text.
