# Gate review — todo 7 "chase the degenerate token: hidden-state correctness, then the head and sampler"

- recommendation: **APPROVE** (checkbox lands **[~]** with an honest gap line, not `[x]`)
- verdict: **VERDICT CONFIRMED ([~])** — the divergence-naming branch of the acceptance is satisfied; the happy path is not, and that gap is named.
- lane audited: `st_01a177f6` (deep-low), worktree `/home/maxious/Strata-wt/dsv41-dpct-w7` @ `a698bb84`
- reviewer: omo-native-gate-reviewer (this report is the only write; no code changed)
- date: 2026-10-26

## originalIntent

The plan (`/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md`, todo 7, lines ~167-185) tasked the lane with chasing the degenerate-token symptom: remove the `129276` x8 repeat by separating the sampler from routing — run greedy (`--temperature 0`, fixed seed), dump logits, compare each step's top candidate against the CPU-only reference, and fix the root cause found, not the symptom. The todo explicitly owns the head path *only if* a type-3 id survives todos 5 and 6; the plan's own diagnosis attributes the repeat to the missing CPU-lane mass (todo 6's `[~]` gap).

## desiredOutcome

Either (a) a greedy decode whose emitted ids are not a single repeated token and whose first-answer sequence matches the CPU reference, or (b) the log naming the exact divergence with the logits that prove it — plus an un-piped `RUN_RC` and a bit-identical negative control (`--temperature 0` twice). The user-visible outcome the plan ultimately wants (todo 8) is the decoded answer "Paris", which is downstream of this todo.

## userOutcomeReview

Not achieved as the happy path: the emitted ids are still `129276` x8 (verified in `run_w7_a.log`, `run_w7_b.out`, `run_w7_c.out`, `run_w7_d.out`), so the repeated-token symptom is **not removed** and the answer "Paris" is not produced. This is the honest gap the checkbox must carry.

However, the acceptance's alternative branch — "the log names the exact divergence with the logits that prove it" — is satisfied in substance, and I reproduced every acceptance-critical number from the artifacts on disk:

- **Divergence named exactly:** SYCL argmax = `129276` (token type 3, the `<｜box｜>` vision control token) at logit `989.406982` vs the plan's expected first answer `11111` (" Paris", type 1) at logit `437.116119`, rank **86,498** of 129,280 in the same row. All 11 logit rows have argmax `129276` (989.39..989.54), every logit positive (min 74.99, mean 454.00), rows near-identical (max |row8−row9| = 0.011658). All verified byte-level against `w7_logits.f32`.
- **Why the head and sampler are not the cause:** `w7_hidden.f32` (17×40×4×5120, no NaN/Inf) shows the residual exploding (layer 0 RMS ~88 → layer 39 RMS ~296,115 by my parse) and collapsing to one DC direction — at layer 39, position 9, all four streams are collinear (pairwise cosine 1.0000) with means 323,104 / 250,107 / 294,352 / 284,057 and within-vector std ≈ 12,445 (3.9% of mean); stream 0's first 8 dims `324270.9, 323558.1, 321722.5, 324468.0, 327333.0, 316057.0, 324609.0, 329907.7` match the lane's claim verbatim. The first visible anomaly is layer 0's output stream 3 (RMS ~176 vs streams 0/1/2 at ~4.6/~5.5/~0.13), so the defect enters upstream of the head path — in the layer blocks' Hyper-Connections path. The lane's attribution to `hc_split_kernel`/`hc_post_kernel` + per-layer pre mixes is a reasonable localization, consistent with the dump.
- **Sampler sound and untouched:** `sampler_greedy_kernel` (`sycl/src/dsv41/vendor/sampler.dp.cpp:96-197`, `out[t] = (wi < n_vocab) ? wi : 0;` at :197) preserves the lowest-index tie rule; exe defaults to temperature 0 (`src/dsv41/ds4_main.cpp:208`). The negative control proves determinism: `run_w7_b.out` and `run_w7_c.out` are byte-identical (`cmp` rc 0, the log's `DIFF_RUNS_RC7=1` under `! cmp -s` is correct).
- **Receipts green:** SYCL build `w7_build2.txt` ends `[53/53] Linking CXX executable strata`, 0 error lines; `w7_ctest.log` = 2/2 ds4 tests passed; `RUN_RC7=0` un-piped with a `prompt:` line; worktree clean at `a698bb84` with no commit (NOCHANGE confirmed — `git status --porcelain` empty, `git diff --stat a698bb84` empty); `/tmp/dpct_w7_build` removed (`LS_RC=2`).

The negative control and the un-piped `RUN_RC` are satisfied; the "non-repeated text" happy path is not. On balance the todo's diagnostic purpose — separate sampler from routing and name the root cause — is fulfilled with evidence I reproduced, so this is an honest `[~]`, not a failure.

## blockers

None. The acceptance's OR branch (name the exact divergence with the logits that prove it) is met by the SYCL-side logits + hidden-state dump; the missing reference logits are a disclosed limitation (see notes), not a stated-criterion failure, because the OR branch is satisfied by the SYCL's own logits naming and localizing the divergence.

## gate results (mandated gates)

- **G1 plan reread — PASS with gap.** The happy path ("emitted ids decode to non-repeated text" AND "first-answer matches CPU reference") is unmet: still `129276` x8. The OR branch ("names the exact divergence with the logits that prove it") is met: divergence `129276` (type 3) vs `11111` (type 1) named, SYCL logits + hidden-state dump proving the degenerate/content-free state and localizing the defect upstream. `RUN_RC` un-piped; negative control bit-identical. The corrupted-logits failure harness (`--dump-logits` with the head forced to zeros) was **not run**; the lane substituted the tie-rule reading + negative control + logits evidence, and the plan's steering accepted that substitution.
- **G2 receipts match disk — PASS.** Build (`w7_build2.txt` tail `[53/53]`), ctest (`100% tests passed, 0 failed out of 2`), negative-control `cmp` rc 0, `run_w7_a.log` ids line, worktree clean `a698bb84`, `/tmp/dpct_w7_build` absent — all reproduced.
- **G3 honesty — PASS.** `FIRST_ID_RC7=1` (first id is 129276, not 11111) is stated, not hidden; the `REF_GGUF_RC=1`/`REF_GGUF_RC7=1` oracle failures are disclosed with verbatim tracebacks; NOCHANGE/COMMIT_RC=0 is stated. The lane's evidence log corrects its own earlier empty `FIRST_ID=`.
- **G4 mechanism fidelity — PASS (no code change).** NOCHANGE is the correct outcome for a root cause proven upstream of todo 7's head-path scope; the plan's own "owns the head path only if a type-3 id survives" clause is honored by diagnosing rather than touching the sampler. No repetition penalty, no clamp, no tolerance relaxation.
- **G5 cleanup — PASS.** `CLEANUP_RC=0`, `LS_RC=2`, build dir gone.

## skill-perspective check (remove-ai-slops + programming), run directly

No diff exists (NOCHANGE), so there is no production extraction/parsing/normalization to audit and no new tests to overfit. Applied over the evidence and the (unchanged) test/parity surface:

- **Overfit/slop pass — none found.** No tests were added, so no deletion-only, tautological, implementation-mirroring, or prose-pinning test exists. The pre-existing `ds4_parity`/`ds4_blas_parity` are behavior (bit-parity) tests and passed unchanged (2/2). The dump files are machine-consumed evidence, not pinned prose.
- **Programming — root-cause vs symptom: PASS.** The lane correctly declined to patch the sampler or clamp logits; it localized the defect upstream via the dumps. This is the plan's own out-of-scope example, not scope drift. The `exl3_torch`-blocked oracle is a disclosed environment gap, not a fabricated reference.

## notes (NOT blockers)

1. **The "CPU-reference logit 437.116119" is misattributed.** `437.116119` is the SYCL's own `logit[11111]` at row 9 of `w7_logits.f32` (reproduced exactly), not a CPU-reference value. The true CPU-reference logits were never produced: `tools/dsv41/ref_gguf.py` fails at `import ref` → `ref.py:33 import exl3_torch as ET`, and `exl3_torch.py` exists nowhere on disk (dropped in commit `a0dbad80`, "drop ... the EXL3 packer"); confirmed by `git log --all -S exl3_torch`. The oracle is structurally un-runnable in this environment regardless of numpy/torch. The lane discloses "the CPU-reference logits comparison was NOT produced by an oracle run" and the expected ids come from the plan's recorded sequence — this is the honest framing; only the "CPU-reference … at logit" wording overstates the source of that one number.
2. **Dependency installs were irrelevant, not "failed".** `pip_numpy.log` ends `Successfully installed numpy-2.5.4` and `pip_torch.log` ends `Successfully installed torch-2.14.1+cpu` into `/tmp/tokvenv` — the lane's "pip fix apparently failed" is inaccurate as stated. It does not change the verdict, because the `exl3_torch` import blocks the oracle either way.
3. **Layer-0 RMS secondary figures do not reproduce.** My parse of `w7_hidden.f32` gives layer 0 RMS 88.00 (all 17 positions) / 70.60 (position 9) and layer-0 stream 3 RMS 175.85 (vs streams 0/1/2 = 4.58/5.46/0.13), not the claimed 57.59 / 141.11 / 3.81/3.39/0.12. The layer-39 numbers (stream means, std, cosines, first-8 dims) reproduce exactly. The qualitative finding — stream 3 anomalously large at layer 0, residual explosion, collinear content-free layer-39 streams — is confirmed under every reasonable aggregation, so this is a measurement-methodology discrepancy in a secondary diagnostic, not a blocker.
4. **Corrupted-logits failure harness not run.** The plan's QA failure scenario (head forced to zeros) was substituted with the tie-rule reading + negative control + logits evidence. Acceptable for the divergence-naming branch; it remains the one happy-path QA that would independently prove the sampler is not the cause.

## checked artifact paths

- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (todo 7 criteria + IS-5 row, read)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-7-dsv41-sycl-dpct-and-xpu-test.log` (read, 248 lines)
- `/tmp/dpct_run/w7_logits.f32` (parsed: 11×129,280 f32, all claims reproduced)
- `/tmp/dpct_run/w7_hidden.f32` (parsed: 17×40×4×5120, layer-39 claims reproduced exactly; layer-0 RMS noted)
- `/tmp/dpct_run/run_w7_a.log`, `run_w7_b.out`, `run_w7_c.out`, `run_w7_d.out` (read; ids `129276` x8, `cmp` b/c rc 0)
- `/tmp/dpct_run/w7_build2.txt`, `w7_ctest.log`, `w7_diffscope.txt` (read)
- `/tmp/dpct_run/ref_gguf.log`, `ref_gguf2.log`, `pip_numpy.log`, `pip_torch.log` (read)
- `/tmp/q_france.ids` (read: `128803,3085,344,270,6102,294,8760,33,128804,128822` — the input prompt, 10 ids)
- `/home/maxious/Strata-wt/dsv41-dpct-w7/tools/dsv41/ref_gguf.py`, `ref.py`, `compare.py` (read; `exl3_torch` import confirmed)
- `/home/maxious/Strata-wt/dsv41-dpct-w7/sycl/src/dsv41/vendor/sampler.dp.cpp` (read; tie rule + `out[t]` confirmed)
- `/home/maxious/Strata-wt/dsv41-dpct-w7/sycl/src/dsv41/ds4_model.dp.cpp` (read; `DS4_DUMP_H` layout `T × nl × H × D` confirmed)
- worktree git state (`git log --oneline -5`, `git status --porcelain`, `git diff --stat a698bb84`) — clean, NOCHANGE
- gguf-py dequantizer support (`/tmp/ds4_cpu/_deps/strata_llamacpp-src/gguf-py/gguf`) — IQ1_M/IQ2_XXS/IQ2_XS/Q5_K/IQ4_NL all map to a `GGMLQuantizationType`, so the earlier "loader rejects IQ types" claim was correctly withdrawn

## exact evidence gaps

1. **No CPU-reference logits exist anywhere on disk.** No `.npz`, no reference logit file. `437.116119` is `w7_logits.f32` row 9 `logit[11111]` (the SYCL's own value), not an oracle value. The oracle is blocked by `import exl3_torch` (module absent, dropped in `a0dbad80`), so the reference side of the divergence rests on the plan's recorded expected ids (`11111, 51119, 6102, 8760`), not on a measured reference.
2. **No corrupted-logits failure harness artifact.** The plan's QA failure scenario was not run; the substituted evidence (tie rule + negative control + logits) is on disk and reproduced.
3. **Layer-0 RMS figures (57.59 / 141.11 / 3.81 / 3.39 / 0.12) are not reproducible** from `w7_hidden.f32` under any aggregation I tried (all-positions, position 9, decode positions); the layer-39 figures are exact. Qualitative conclusion unaffected.

## code review report / manual QA matrix coverage

As with todos 5 and 6, no separate code-review report or manual-QA-matrix artifact exists for todo 7 (they are plan F2/F3 deliverables, still unchecked). The skill-perspective and overfit/slop pass above is my direct review, run over the diff-less evidence and the unchanged parity tests; it is not a substitute for F2/F3 when those run. This does not change the verdict, because the todo's own acceptance criteria are independently satisfied by the dump receipts and the negative control.
