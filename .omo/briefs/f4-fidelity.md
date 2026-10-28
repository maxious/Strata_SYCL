# F4 — Ideal-state fidelity (dsv41-trellis-sg16-perf)

Compare the plan's promised end state (/home/maxious/Strata/.omo/plans/dsv41-trellis-sg16-perf.md) against current state. Checks with un-piped RC:
1. Every top-level checkbox in the plan is [x] or [~] with a one-line gap (read the plan file; list each).
2. Evidence logs exist for todos 1-13 (ls .omo/evidence/dsv41-trellis-sg16-perf/task-N-*.log; each ends with a VERDICT line — grep -c VERDICT per file).
3. Ledger entries exist per todo with verifier CONFIRMED (grep ledger.jsonl).
4. Integration branch dsv41-trellis-int in /home/maxious/Strata-wt/dsv41-trellis-base contains all verified commits (git log --oneline | head -20); nothing pushed (git log origin/dsv41-trellis-int..HEAD non-empty is expected; confirm origin ref unchanged, RC printed).
5. Claims that need measurement (throughput numbers in docs/DETAILS.md) trace to completed prompts in task-6/7/8 logs — spot-check 3 numbers via the mapping table.
6. Scratch dirs gone: ls /home/maxious/.scratch-t* → RC=2.
Do not modify anything. End with exactly one line: VERDICT APPROVED or VERDICT REJECTED + reason. Use real tools inside eval; never XML-as-text; never end before the VERDICT line.
