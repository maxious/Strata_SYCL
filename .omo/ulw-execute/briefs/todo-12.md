ROLE: you are the executor of this unit; do it yourself; do NOT delegate.
UNIT: plan todo 12 - docs measured-report update.
WORKTREE: supplied in the dispatch message (final integration base). NEVER touch /home/maxious/Strata.
FILES IN SCOPE: docs/DETAILS.md only (plus docs/TERMINAL.md or docs/AMD_HIP.md only if a number you must publish belongs there - say why).
READ FIRST: every evidence log under .omo/evidence/dsv41-trellis-sg16-perf/task-{6,7,8,9,10,11,13}-*.log; AGENTS.md docs rule (measured numbers with what they were measured on; no claim without a measurement; plain words, short sentences, non-technical first).
DO: update docs/DETAILS.md with every measured number from todos 6-11 and 13, each tagged with the hardware (B60 pair) and the build commit it was measured on. Mark unconfirmed static analysis separately from measured numbers. Publish nothing that has no evidence-log line.
VERIFY: a reproducibility script over the docs diff - extract every number in the added lines and show the evidence-log line that contains it; the script must exit 0 and print one line per number with its source. Any number without a source must be removed before the unit is done.
MANUAL QA: render the changed section (sed -n of the changed hunk plus a plain-text render) and capture it in the log; confirm every claim's tag is present.
ADVERSARIAL CLASSES: misleading_success_output (the mapping script is the gate, not a self-review); stale_state (re-read the evidence logs this run, never memory); dirty_worktree (git status --short limited to docs). malformed_input / prompt_injection / cancel_resume / flaky_tests / hung_or_long_commands: not applicable to a prose-only edit.
ARTIFACT: /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-12-dsv41-trellis-sg16-perf.log ending with VERDICT PASS or VERDICT FAIL.
COMMIT: allowlist only the doc files, title '[Docs] measured stageA and decode numbers'.
CLEANUP: remove the mapping script temp files; CLEANUP_RM_RC=0.
RETURN a DoneClaim JSON: task, changed_files, tests, manual_qa, cleanup, risks.