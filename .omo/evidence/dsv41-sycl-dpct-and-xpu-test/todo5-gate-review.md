# Gate review — todo 5: the copy-engine hand-back (host->GPU signalling)

- **Recommendation:** APPROVE
- **Verdict:** VERDICT PASS
- **Todo 5 checkbox:** land `[x]` — every acceptance criterion is met; the BOUNDED-GAP outcome is the plan's own sanctioned fallback ("keep the bounded spin and record the gap in the docs") and is recorded in the docs file.
- **Reviewer:** omo-native-gate-reviewer (st_01a177e4) — independent, no code changes made; only this report written.
- **Date:** 2026-10-26

## originalIntent

The user's plan (`.omo/plans/dsv41-sycl-dpct-and-xpu-test.md`, todo 5, lines 144-153) asked for the host->GPU answer-payload direction to be settled on the Intel Arc Pro B60 cards: either prove the card observes the host's store, or bound every wait and record the measured gap. The named mechanism was the copy-engine hand-back (second in-order queue, payload-then-flag, device-memory flag read with a device-scope atomic), and the plan explicitly required: (a) per-spin-site counts before/after, (b) a `STRATA_SPIN_MAX=1` run that fails loudly instead of silently producing zeros, (c) a default-bound run completing with an un-piped `RUN_RC` and a `prompt:` line, (d) the direction stated as OBSERVED or BOUNDED-GAP with the measurement backing it.

## desiredOutcome

- GAP-3 (answer payload crossing host memory in both directions) and GAP-7 (flaky `sycl::atomic_ref` on host USM) closed or their gap documented.
- Every wait bounded (no unbounded spin; the 5 s xe tile reset avoided).
- The host->GPU direction classified OBSERVED or BOUNDED-GAP, honestly, from un-piped exit codes.
- No default change; the Intel path stays opt-in; `ctest -R ds4` green.

## userOutcomeReview

The shipped commit (worktree `/home/maxious/Strata-wt/dsv41-dpct-w5` @ `eed419f0`) does what the plan asked: the default path's host store is **OBSERVED** by the card's uncached read (64 waits, most reads 1, last value 255, 0 hit the bound, decode 381 ms vs 384 ms base), and the copy-engine hand-back (`DS4_ANSWER_COPY=1`) is **BOUNDED-GAP** (most reads 1,000,000, last value 191, 63 hit the bound, decode 11,530 ms) and is off by default. Every wait is now bounded, including the previously-unbounded window-path wait. The `STRATA_SPIN_MAX=1` run fails loudly (RUN_SPIN_RC=1, no decode line). This is the plan's authorized fallback outcome, not silent success prose.

The copy-engine hand-back is implemented but does not work on this card once a slot is cached; the default path therefore remains the polled mailbox (answer still crosses host memory). The plan's `Mechanism to use` paragraph named device-scope atomic as the only sanctioned read for a copy-engine-written flag; the shipped code uses the uncached read (`strata::dev_load` -> `sys_load_mapped`) for that flag, a deliberate, documented deviation justified by a measured stale-line result. This is a fidelity note, not a success-criterion failure: the plan's own "Original experiment still required" paragraph directs the executor to settle the read path by measurement and record the gap, which was done.

## Acceptance criteria check (G1)

| Criterion | Result | Evidence |
| --- | --- | --- |
| Per-spin-site counts + last value, before and after | MET | log: BEFORE run_t5.log (1 wait, most 1000000, last 192, 1 at bound); AFTER default run_t5e.log (64 waits, most 1, last 255, 0 at bound); AFTER copy run_t5d.log (64 waits, most 1000000, last 191, 63 at bound). Base build had no answer-spin instrumentation, disclosed in the log, with base decode 384 ms as the reference. |
| STRATA_SPIN_MAX=1 fails loudly | MET | `RUN_SPIN_RC=1`, `SPIN_DEC_RC=0` (grep `^decode:` count 0), bound-hit lines verbatim: "the card never saw an answer (its wait ran to its bound)". |
| Default-bound run completes, RUN_RC un-piped, prompt line | MET | `RUN_RC=0`, `PROMPT_RC=0`, `prompt: 10 tokens in 160 ms (62.69 tok/s, windows of 8192)`. |
| OBSERVED or BOUNDED-GAP stated with measurement | MET | OBSERVED for host store; BOUNDED-GAP for copy engine, each with the spin record and decode time. |

## Receipts vs disk (G2)

- Evidence log: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-5-dsv41-sycl-dpct-and-xpu-test.log` — present, 108 lines, RC lines sit directly beside their commands. PASS.
- Commit scope: `eed419f0` diff is 8 files, all under `sycl/` (verified with `git show --stat`). PASS. No `docs/` top-level file and no `.omo/`/`bench/results` in the commit. PASS.
- Docs gap file: `sycl/docs/sycl-dsv41-copy-engine-handback.md` — present, read in full, records OBSERVED and BOUNDED-GAP with the runs and the open item. PASS.
- Build receipt: evidence log names `w5_build9.log` (`SYCL_BUILD_RC=0`, `ERR_RC=1`); `w5_build9.log` tail is `[64/64] Linking CXX executable strata` and `grep -c "error:"` is 0. PASS. (Note: the task's stated artifact `w5_build4.log` is a FAILED intermediate build — "ninja: build stopped: subcommand failed" — not the final receipt. The correct final receipt is `w5_build9.log`.)
- ctest receipt: `w5_ctest.log` shows 2/2 passed (`ds4_parity`, `ds4_blas_parity`), matching `SYCL_CTEST_RC=0`. PASS.
- Integration base `/home/maxious/Strata` HEAD is `2e5c53b9`, unchanged (dirty with untracked `.omo/` and `bench/results/`, pre-existing). PASS.

## Honesty (G3)

- Every `*_RC` in the evidence log is an un-piped `$?` beside its own command; no piped exit code reported as a result. PASS.
- Throughput is taken from a completed prompt (`prompt:` line present before the tok/s quote); no throughput claimed from the spin-failure run. PASS.
- BOUNDED-GAP is not presented as OBSERVED; the summary and docs state the copy engine "does not work on this card" and is off by default. PASS.
- CPU_LANE_NONZERO_RC=1 is honestly 0 of 1680 and attributed to todo 6's routing defect, not to this todo. PASS.

## Mechanism fidelity (G4)

| Plan requirement | Disk state | Result |
| --- | --- | --- |
| Second in-order queue, created next to `Dev::st` on the owning thread | `ds4_model.dp.cpp:2097` (`d.st = ...create_queue()`) then `:2100` (`d.st2 = ...create_queue()`) in the same device-selection block | PASS |
| Payload first, then flag, on `st2` | `serve_lane()` answer block: `ans_cpu` + `ans_part` copies, then the whole `AnsSlot` (flag last), then `q2->wait()` | PASS |
| Flag in device memory | `d.ans` is a `dalloc` device allocation; `ans_flag = &d.ans[slot].seq` when `answer_copy` | PASS |
| Device-scope atomic read of the copy-engine flag | **DEVIATION** — `strata::dev_load` is `sys_load_mapped` (uncached annotated read), not a device-scope atomic | NOTE (see below) |
| No `sycl::atomic_ref` on host USM in the answer path | grep over all changed `.dp.cpp` files: no `atomic_ref`; host-mapped flags keep `strata::sys_load`; `atomic_ref` appears only in the header for `spin_record`/`sys_atomic_u32` | PASS |
| No bound removed or widened | bound stays 1,000,000 (`answer_spin_max`); window-path wait was unbounded and is now bounded to the same bound (a tightening, not a widening) | PASS |
| Upstream SYCL library not rebuilt | diff touches only `sycl/include`, `sycl/src/dsv41`, `sycl/docs`; no `ggml_sycl` build | PASS |

**Device-scope-atomic deviation (NOTE, not a blocker).** The plan's `Mechanism to use` named a device-scope atomic as the only sanctioned read of a copy-engine-written flag. The code uses the uncached read for both the host store and the copy-engine flag, and the docs ("The reads that decide the answer") state the measured reason: on a slot the card has cached, the device-scope atomic also reads the stale line (contradicting the plan's `/tmp/dbprobe*.cpp` 10/10 result, which was on a never-cached slot). This is a documented, measured deviation within the plan's "settle the read path by experiment" latitude, and the affected path (`DS4_ANSWER_COPY=1`) is off by default. It is not a success-criterion failure, but the maintainer should be aware that the shipped mechanism differs from the plan's sanctioned read.

## Cleanup (G5)

- `CLEANUP_RC=0`, `LS_RC=2`; `/tmp/dpct_w5_build` is gone (re-verified: `ls` exits 2). PASS.
- Worktree `/home/maxious/Strata-wt/dsv41-dpct-w5` kept at `eed419f0`, clean status. PASS.

## Skill-perspective check (remove-ai-slops + programming), run directly over the diff

- **No new tests in the commit.** The todo's acceptance is gated by an engine smoke (the run logs), not by ctest; that matches the plan ("todos 3, 4, 5, 6 are gated by an engine smoke, not by ctest"). No excessive, tautological, deletion-only, or implementation-mirroring tests were added. The `grep ^decode:` / `^prompt:` receipts assert machine-consumed values (a completed decode/prompt line), not prose. PASS.
- **No prose pinning.** The docs file is prose but ships no test pinning its wording. PASS.
- **Instrumentation is the acceptance evidence, not slop.** `AnsRec`/`SpinRec`/`report_spins`/`spin_record` exist to produce the spin-count evidence the acceptance criteria literally require. They are not dead code; they are printed at run end and quoted in the evidence. PASS.
- **`dev_load` pass-through wrapper.** It is a one-line semantic alias for `sys_load_mapped`, giving a name ("device-memory flag read") distinct from `sys_load` (host-memory flag read). Marginal, but it carries the naming intent and is not a speculative abstraction. NOTE.
- **Boundary error handling.** The `try/catch` around `serve_lane()` and the copy-failure `fprintf`/`faulted=true` paths are genuine boundary handling for a device path that can fail; not over-defensive. PASS.
- **Module size.** `ds4_model.dp.cpp` is ~10,573 pure LOC and `sycl_doorbell.hpp` 135. The former is far over the 250-LOC ceiling, but it is a pre-existing dpct-port file, not created by this commit; the commit adds ~220 lines of mechanism/instrumentation directly tied to the acceptance criteria. Recorded as a pre-existing maintenance-burden NOTE, not introduced by this change and out of this todo's scope.
- **Scope drift / maintenance burden / false confidence.** None found. The commit is confined to the mechanism plus its instrumentation and the one docs file. The report does not overstate: BOUNDED-GAP is labeled as such everywhere, the CPU-lane zero is correctly disclaimed, and the throughput is qualified to a completed prompt.

## Code review report / QA matrix coverage

The task input lists a "code review report" and a "manual QA matrix" as expected inputs. Neither exists on disk for todo 5 (searched `.omo/evidence/` and the task evidence directory; the plan's F2/F3 final-verification todos are still unchecked). This is an **evidence gap**: the skill-perspective check above is my direct review, and it is not a substitute for the plan's F2/F3 artifacts when those run. It does not change this todo's verdict because the todo's own acceptance criteria are independently satisfied by the engine-run receipts.

## Checked artifact paths

- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (todo 5 + success criteria + F-wave, read)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-5-dsv41-sycl-dpct-and-xpu-test.log` (read)
- `/tmp/dpct_run/run_t5.log`, `run_t5b/c/d.log`, `run_t5_copy.log`, `run_t5e.log`, `run_t5_spin4.log` (read; spin counts and decode times compared)
- `/tmp/dpct_run/w5_build4.log`, `w5_build9.log`, `w5_ctest.log` (read)
- `/home/maxious/Strata-wt/dsv41-dpct-w5/sycl/docs/sycl-dsv41-copy-engine-handback.md` (read)
- `git diff b18f6c40..eed419f0` over all 8 files (read: `sycl_doorbell.hpp`, `ds4_kernels.dp.cpp/.hpp`, `ds4_model.dp.cpp`, `ds4_gguf.dp.cpp`, `vendor/exl3.dp.cpp`, `vendor/exl3_legacy.dp.cpp`, `sycl/docs/sycl-dsv41-copy-engine-handback.md`)
- Worktree git state (`git log`, `git show --stat`, `git status`) and integration base `2e5c53b9` (read)
- `/tmp/dpct_w5_build` cleanup receipt (re-verified absent, `ls` exit 2)

## Exact evidence gaps (all NOTES; none is a stated-criterion failure)

1. **Stale gate-6 receipts in the evidence log.** The log's `git log --oneline -1` quotes `2e5c53b9` and its `git diff --name-only` quotes only `sycl/docs/sycl-dsv41-queue-binding.md` — both are the integration-base/todo-4 state, not the todo-5 worktree `eed419f0`. The actual commit and its 8-file `sycl/` scope are correct on disk, so this is a stale receipt, not a wrong commit.
2. **Task artifact list vs evidence log.** The task names `w5_build4.log` as the build receipt, but `w5_build4.log` is a FAILED intermediate build; the final green build the log cites is `w5_build9.log`. The final receipt is valid; the named artifact is just the wrong file.
3. **Decode-time mismatch between the evidence log/docs and the named copy-run artifact.** The log and docs cite the copy-engine decode as 11,530 ms (0.61 tok/s), which is `run_t5d.log`; the task-named artifact `run_t5_copy.log` shows 8,344 ms (0.84 tok/s) with the identical spin record (64 waits, most 1,000,000, last 191, 63 at bound). Both are BOUNDED-GAP; the qualitative finding is identical, but the two files record different magnitudes of the same failure.
4. **No separate code-review report or manual QA matrix for todo 5.** These are plan F2/F3 deliverables still unchecked; this review's skill pass covers the same ground directly but is not the F2/F3 artifact.
5. **Device-scope-atomic deviation** (documented above under G4): the shipped read for the copy-engine flag is the uncached read, not the plan's sanctioned device-scope atomic. Measured and documented, but a literal departure from the plan's `Mechanism to use` wording.
