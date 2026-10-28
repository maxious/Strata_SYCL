# Gate review — todo 6c "re-measured default-path evidence on the merged build"

- recommendation: **APPROVE**
- todo6 landing mark: **[x]** is correct (default-path CPU-lane contribution now proven N > 0 on one card)
- lane audited: `st_01a1788e` (DONECLAIM todo6c), measurement-only lane, no commit produced
- audited build: merged `28bd2aaa` (HEAD), `git log 18a27e02..HEAD` byte-exact against the 7 expected commits

## originalIntent

Plan todo 6 (`/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md`, line ~155) wants proof that the host CPU lane actually computes the experts the card does not hold: a one-card engine run prints `CPU lane: <N> of 1680 experts` with N > 0 and a non-zero `CPU lane` per-token time, the count is real work (not a display artefact), the decode is no longer degenerate, and the feature holds on the merged build (`28bd2aaa`), not only under forced residency.

## desiredOutcome

The todo 6 `[x]` mark and its re-measured evidence text are correct: the default one-card greedy run now clears N > 0 on the quality path, the denominators reconcile, the emitted ids decode to `The capital of France is **Paris**.` + EOS, the gates are genuine un-piped exit codes, cleanup is proven, and no guardrail was broken.

## userOutcomeReview

Every audited claim checks out against the files. The `[x]` mark and its new evidence text are correct.

### Claim 1 — CPU lane line (default + --n 8) — VERIFIED

- `/tmp/dpct_run/w6c_n8.log` (the `--n 8` run) contains verbatim:
  - `prompt: 10 tokens in 272 ms (36.77 tok/s, windows of 8192)` (present, before any decode tok/s)
  - `decode: 8 tokens in 2565 ms (2.73 tok/s); CPU lane: 132 of 1680 experts; the cache moved 3 (3 in all)`
  - `decode per token: layers 36.33 ms (of it: routing waits 0.00, CPU lane 11.79, the rest host launches), head wait 329.76, Engram reads 0.04`
  - => `CPU lane: 132 of 1680 experts` with CPU lane 11.79 ms/token, verbatim.
- `/tmp/dpct_run/w6c_default.log` (the step's exact `--n 32` command) contains verbatim:
  - `prompt: 10 tokens in 707 ms (14.14 tok/s, windows of 8192)` (before decode tok/s)
  - `decode: 9 tokens in 4944 ms (1.62 tok/s); CPU lane: 195 of 1920 experts; the cache moved 5 (5 in all)`
  - `decode per token: ... CPU lane 43.28 ...`
  - and `experts 14.86 GB (1684 of 15360 in VRAM)` — the card holds its usual 1684 resident slots.
- The plan text's numbers (132/1680 @ 11.79, 195/1920 @ 43.28, 1684 resident) match the logs verbatim.

### Claim 2 — gates in the evidence log — VERIFIED (wording NOTE only)

All 12 gates appear in `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-6-dsv41-sycl-dpct-and-xpu-test.log` (DONECLAIM at line 523):
`SYCL_CONFIGURE_RC=0 SYCL_BUILD_RC=0 SYCL_RECONFIGURE_RC=0 SYCL_REBUILD_RC=0 EXE_RC=0 RUN_RC=0 RUN_N8_RC=0 RUN_EXP0_RC=0 RUN_MIX_RC=0 RUN_V0_RC=0 SYCL_CTEST_RC=0 CLEANUP_RC=0`.
Each is the un-piped `$?` of the `CMD:` line it is adjacent to (commands redirect stdout/stderr to logs but do not pipe the exit code). NOTE: the audit brief said each gate sits "directly above" its command; in the file each `NAME_RC=` sits directly BELOW its `CMD:` line. Association is unambiguous and correct — cosmetic wording inversion only, not a defect.

### Claim 3 — ctest named and passed — VERIFIED (evidence-preservation NOTE only)

- `SYCL_CTEST_RC=0` is recorded un-piped under the `ctest --test-dir /tmp/dpct_w6c_build/sycl -R ds4` command.
- The log records `2/2 named and passed: ds4_parity 0.27 s, ds4_blas_parity 0.31 s; 100% tests passed, 0 failed.`
- `No tests were found!!!` does NOT appear anywhere in the todo6c region (grep count = 0).
- The two tests are genuinely defined (`sycl/CMakeLists.txt` `add_test(NAME ds4_parity ...)` at 277, `add_test(NAME ds4_blas_parity ... --selftest)` at 291), and the configure used the correct `-DCMAKE_CXX_COMPILER=icpx` recipe (the F2-documented fix for the "zero tests matched" failure).
- NOTE: the final build's raw ctest output is not preserved in a separate file (the final `ctest` ran inline; `/tmp/dpct_w6c_build` was then removed). The prior same-source build's raw output is preserved at `/tmp/dpct_run/w6c.txt` (`Test project /tmp/dpct_w6_build` → `ds4_parity ... Passed`, `ds4_blas_parity ... Passed`, `100% tests passed, 0 tests failed out of 2`). This is a documentation-preservation gap, not a criterion failure.

### Claim 4 — cleanup receipt — VERIFIED

`/tmp/dpct_w6c_build` is absent now, checked with `node:fs` via the bundled Bun runtime (system `node` is not installed):
- `existsSync("/tmp/dpct_w6c_build") === false`
- `statSync` throws `ENOENT`
`ls`/`test`/`stat` exit codes were not used, per instruction.

### Claim 5 — denominator reconciliation + tokenizer decode — VERIFIED

- Arithmetic (checked directly):
  - `1680 = (8-1) x 40 x 6 = 7 x 240` ✓
  - `1920 = (9-1) x 40 x 6 = 8 x 240` ✓
  - `1240 = (32-1) x 40 x 1 = 31 x 40` ✓
- Independent decode with `/tmp/tokvenv/bin/python` using the repo's `tools/strata_tokenizer.py` (same path `serve/server.py` uses) on ids `671 6102 294 8760 344 2619 51119 42499 1`:
  - pieces: `['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis', 'Ġ**', 'Paris', '**.', '<｜end▁of▁sentence｜>']`
  - decode: `'The capital of France is **Paris**.<｜end▁of▁sentence｜>'`
  - => `The capital of France is **Paris**.` + EOS, exactly as claimed (id 1 = EOS, type 3, the pack's `eos_token_id`).
- The ids emitted by the engine run (`671,6102,294,8760,344,2619,51119,42499,1` in `w6c_default.log`) match this sequence byte-identically.

### Claim 6 — guardrails — VERIFIED

- `git log 18a27e02..HEAD` is exactly, in order:
  `28bd2aaa`, `39b128ad`, `a698bb84`, `433868e5`, `2e5c53b9`, `b18f6c40`, `ca0f6f94`; `HEAD == 28bd2aaa`. No new commit was added by this lane (measurement-only).
- No `.omo/` or `bench/results` path appears in any of those 7 commits (grep = NONE).
- `git diff --name-only 18a27e02..HEAD` lists only `sycl/` paths (12 files: 3 docs, 2 headers, 7 sources) — no `src/`, no `include/`, no `CMakeLists.txt`.
- Must-NOT rules held:
  - **no barrier-scope conversion**: the DPCT1065 advisories were replaced with notes but the `item_ct1.barrier()` full-barrier calls are kept (current file has 103 `barrier()` sites); the one new `ext_oneapi_submit_barrier` is todo 5's sanctioned doorbell/payload publish, not a scope narrowing.
  - **no occupancy rewrite**: the only "occupancy" hit in the diff is a triage-doc row recording `DPCT1007 ... ACCEPTED_AS_IS` (no code rewrite).
  - **no ggml_sycl build**: zero `ggml_sycl`/`GGML_SYCL` references in the diff.
- Acceptance criterion `GG_RC=0` (CPU reference still runs) was met in the todo6b evidence (log lines 225-226, 377-379: `GG_RC=0`, "0 of 1567344 differ in bits"); todo6c skips re-running it with the stated reason "no CPU expert kernel was touched", which is accurate for this measurement-only lane.

## blockers

None.

## notes (NOT blockers)

1. The audit brief's phrase "each [gate] directly above the command" is inverted relative to the file, where each `NAME_RC=` sits directly below its `CMD:` line. Cosmetic only.
2. The final (`/tmp/dpct_w6c_build`) ctest's raw output is not persisted as a standalone file; the log carries a named 2/2 summary and the earlier same-source build's raw output (`/tmp/dpct_run/w6c.txt`) shows both tests named and passed. The substance (tests found and passed, no "No tests were found!!!") is established; this is a documentation-preservation NOTE.
3. `GG_RC=0` is not re-run in todo6c; it was met in todo6b (log line 377) and the skip rationale is sound for a no-code-change lane.

## skill-perspective check (remove-ai-slops / programming) — re-run directly

- This lane is measurement-only: no production code, no test code, no diff was produced by it. There is therefore no diff/tests to slop-scan for the lane's own output — no deletion-only tests, tautological tests, implementation-mirroring tests, or unnecessary extraction/parsing/normalization can exist in this lane's work.
- The evidence log and plan text are measurement records, not prose to be slop-trimmed; their claims are machine-consumed numbers (exit codes, CPU-lane counts, tok/s, ids) that I reproduced from the underlying logs above.
- The upstream code review (`todo6b-gate-review.md`) contains an explicit "skill-perspective check (remove-ai-slops / programming)" section that names the same overfit/slop criteria (no new tests, no tautological/deletion-only tests, no gratuitous extraction) and confirms it ran the pass independently over the publish commit. Coverage there matches this review's own findings; this review re-ran the checks directly over the lane's evidence and the branch diff, not by delegation.

## checked artifact paths

- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (todo 6 section + new evidence text)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-6-dsv41-sycl-dpct-and-xpu-test.log` (full log; todo6c DONECLAIM at line 523)
- `/tmp/dpct_run/w6c_default.log`, `/tmp/dpct_run/w6c_n8.log` (claim 1)
- `/tmp/dpct_run/w6c_exp0.log`, `/tmp/dpct_run/w6c_mix.log`, `/tmp/dpct_run/w6c_v0.log` (controls referenced by the plan text)
- `/tmp/dpct_run/w6c.txt` (prior same-source build's raw ctest output)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo6b-gate-review.md` (upstream review + skill check)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo6-gate-review.md` (existence confirmed)
- `/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/tokenizer/` (`vocab.json`, `merges.txt`, `token_type.json`, `tokenizer.json`)
- `/home/maxious/Strata/tools/strata_tokenizer.py` (decode path), `/tmp/tokvenv/bin/python`
- `git` range `18a27e02..HEAD` (diff stat, name-only, commit list)

## evidence gaps (exact)

- None that block approval. The two NOTES above (final-build raw ctest output not persisted; `GG_RC` not re-run in todo6c) are both explained by the lane's measurement-only nature and do not contradict any stated success criterion.
