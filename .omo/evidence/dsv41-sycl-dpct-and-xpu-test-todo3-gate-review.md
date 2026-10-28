# Gate review — todo 3: DPCT1098 host-mapped read-path audit (dsv41-sycl-dpct-and-xpu-test)

Date: 2026-10-26
Reviewer: omo-native-gate-reviewer (independent; no product edits made; report artifact only)
Skills consulted: remove-ai-slops, programming (both read; overfit/slop pass run directly over the diff)

## recommendation

APPROVE

## blockers

(none — no stated success criterion is failed)

## originalIntent

(From the plan, todo 3) Audit DPCT1098 in the SYCL dsv41 port: enumerate every read of a host-mapped or host-allocated payload in `sycl/src/dsv41`, confirm each goes through the uncached path (`ld_fresh` / `strata::sys_load` / `strata::sys_load_mapped`), fix any site that still reads with a plain `*` load, and record a note for the site that legitimately reads device-resident memory. Deliver a table of every host-mapped read site (file:line, read path, UNCACHED/DEVICE_RESIDENT mark, reason), keep the build and `ctest -R ds4` green, and prove the fix matters with a throwaway control that observes the stale value the fix removes.

## desiredOutcome

A reviewer can see, site by site, that no host-mapped payload read in the port is served from a stale cache line; the two conversion classes that can silently produce wrong values are audited and fixed; and the fix is demonstrated to change the observed behavior on this card (plain read misses a store already in memory, the uncached read sees it).

## userOutcomeReview

The shipped commit (0857e37e) is exactly the read-path audit the todo asked for. Every host-mapped read that still used a plain load was converted to `strata::sys_load` / `strata::sys_load_mapped` / `ld_fresh`; the device-resident sites were left untouched with a one-line audit note naming why; and the two plan attributions that were wrong in code were corrected with evidence rather than silently followed. The failure control is real and can fail: on this card a plain spin missed a store that was already clflushed into memory in 8 of 10 trials, while the fenced uncached read (the fixed path) saw it in 10 of 10. Build, ctest, and the engine smoke all pass. The todo's acceptance criteria are satisfied.

## Checked artifact paths

- /home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md (todo 3 acceptance text, ~line 124; GAP-2/GAP-3 rows)
- /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-3-dsv41-sycl-dpct-and-xpu-test.log (300 lines, full)
- /home/maxious/Strata-wt/dsv41-dpct-w3 (worktree, branch dsv41-dpct-w3 @ 0857e37e)
- git: log --oneline -1; diff --name-only 18a27e02 HEAD; status --short; rev-parse HEAD~1; rev-parse 18a27e02
- /tmp/gate_diff_full.txt (full commit diff, 207 lines, re-read)
- sycl/src/dsv41/ds4_kernels.dp.cpp, ds4_gguf.dp.cpp, vendor/exl3.dp.cpp, vendor/exl3_legacy.dp.cpp (committed files, greps + reads)
- sycl/src/dsv41/ds4_kernels.dp.hpp (Combine struct, lines 225-236)
- sycl/src/dsv41/ds4_model.dp.cpp (lines 1408-1427 hmap, 2515 dalloc, 2650-2668 E host-mapped, 11035-11140 combine call site)
- sycl/include/strata/sycl_doorbell.hpp (sys_load / sys_load_mapped / load_mapped_float4 signatures)
- /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test-todo4-gate-review.md (sibling format reference)

## Evidence gaps (exact)

All gaps below are NOTES, not blockers — each ties to no stated success criterion.

- No separate code-review-report artifact exists for todo 3 (`grep -rln "code review"` over the evidence dir returns nothing). The audit table + MANUAL-QA-CHANNEL inside the evidence log serve as the review. That material shows the site-by-site read-path audit (the skill-perspective check) but does not explicitly claim an overfit/slop pass; because the commit adds no tests and no extraction/parsing/normalization, the slop surface is nil — I ran that check directly and found nothing to flag (see Slop/overfit pass below).
- `WC_SITES_RC=0` / `WC_POST_RC=0` record the `wc -l` exit codes but the log does not quote the `wc -l` counts verbatim; the count is stated in prose ("grep count of DPCT1098 ... is 5") and I independently reproduced 5.
- `SHASUM_RC=0` / `MTIME_RC=0` / `MTIME2_RC=0` record exit codes but not the hash/mtime values themselves.
- `CTEST_RECHECK_RC` could not be run: `/tmp/dpct_w3_build` was removed at lane end (independent `ls` → BUILD_LS_RC=2, "No such file or directory"). Judged: the recorded receipts are internally consistent and sufficient (configure → build → sha → mtime → ctest with verbatim 2/2 summary → engine smoke, all RC=0).
- The commit message ("[Engine] fix the wrong DPCT1098 device read sites in the SYCL dsv41 port") differs in wording from the plan's suggested line ("[Engine] dsv41 SYCL: route every host-mapped payload read through the uncached path"). Both are `[Engine]`-prefixed; the exact message is not an acceptance criterion.

## Gate results

### Gate 1 — Plan reread and commit scope: PASS
- LOG_RECHECK_RC=0 — `git -C /home/maxious/Strata-wt/dsv41-dpct-w3 log --oneline -1` = `0857e37e [Engine] fix the wrong DPCT1098 device read sites in the SYCL dsv41 port`.
- SCOPE_RECHECK_RC=0 — `git diff --name-only 18a27e02 HEAD` = exactly 4 paths, all under sycl/: `sycl/src/dsv41/ds4_gguf.dp.cpp`, `sycl/src/dsv41/ds4_kernels.dp.cpp`, `sycl/src/dsv41/vendor/exl3.dp.cpp`, `sycl/src/dsv41/vendor/exl3_legacy.dp.cpp`. Diff stat: 34 insertions(+), 15 deletions(-).
- ST_RECHECK_RC=0 — `git status --short` = 0 lines (clean).
- Parent check: `git rev-parse HEAD~1` = `18a27e02fd5d3b63dddf9a5a1863abaf5581f769` = the base `18a27e02`; `18a27e02..HEAD` contains only 0857e37e. Worktree branch = `dsv41-dpct-w3`.

### Gate 2 — Read-site completeness: PASS
- Log RCs recorded: ATTR_CORRECTION_RC=0, SITE_SUM_RC=0, WC_SITES_RC=0, SITE_SUM_POST_RC=0, WC_POST_RC=0.
- RECOUNT_RC=0 — independent `grep -rn DPCT1098 sycl/src/dsv41` = **5** sites, matching the lane's stated count: `ds4_kernels.dp.cpp:125` (DPCT1098:79), `ds4_kernels.dp.cpp:1859` (DPCT1098:109), `vendor/exl3.dp.cpp:356` (DPCT1098:62), `ds4_gguf.dp.cpp:127` (DPCT1098:35), `ds4_gguf.dp.cpp:613` (DPCT1098:49).
- ATTRIBUTION CORRECTION independently confirmed correct. The plan's GAP-3 cited `ds4_kernels.dp.cpp:1983` as "a plain load of host USM". The code says otherwise: `combine_kernel` reads `cpu[...]`/`ed[...]`; the sole call site `ds4_model.dp.cpp:11138` passes `d.cpu_part`/`d.ed`, where `d.cpu_part` is `dalloc` device memory (`:2515`) filled by `s->memcpy(d.cpu_part, h_part, ...)` at `:11136`. So `:1983` is DEVICE_RESIDENT, not host-mapped — the plan attribution is wrong and the executor's correction is right. Separately, the plan prose called `:1857` "device-resident E"; the code shows `E = h_mk_embed` is `hmap()` = `sycl::malloc_host` (`ds4_model.dp.cpp:2654/2663`, `hmap` at `:1408-1427`) — host-mapped, not device-resident — so the executor fixed it (`erow[i] = *(E + ...)` → `erow[i] = ld_fresh(E + ...)`).
- LEFTOVER check — no unconverted plain read of a host-mapped flag remains. Targeted grep for `*cpu_part` / `*a.cpu_flag` / `cpu_flag[t]` / `while (*(volatile` / `erow[i] = *(E` / `*(volatile const unsigned*)(slots` returns only 3 matches, all function-signature pointer declarations (`const float *cpu_part, float sh) {`), not reads. Every host-mapped flag/part read in the four files now goes through `strata::sys_load` / `strata::sys_load_mapped` / `ld_fresh`. The remaining plain `volatile` reads are device-resident (mixes scratch `ds4_kernels.dp.cpp:520`, h/u activations, expert part `a.part`, J job structs, d_mat_frag) or are `sys_store` writes or the `sys_load` helper body — all correctly left alone per the audit table.
- Constraint check: `sycl/include/strata/sycl_doorbell.hpp` untouched; no new read helper added (`ld_fresh`/`sys_load`/`sys_load_mapped` pre-existing); spin bounds untouched (the `while (sys_load(...) != v)` conversions keep the empty loop body; the audit notes say "spin bound untouched" and defer the unbounded-spin shapes to todo 5).

### Gate 3 — Failure control that can fail: PASS
- CTRL4_RC=0. Control 4 (clflushed host store; plain spin vs fenced-uncached read like `strata::sys_load`, 10 trials) — verbatim summary: `SUMMARY trials=10 plain_seen=2 uncached_fenced_seen=10` and `CONTROL RESULT: plain_spin_stale=OBSERVED uncached_fenced_spin_stale=NOT_REPRODUCED`. Per-trial: 8 of 10 plain spins `seen=0 iters=20000000` (never saw the store already in memory); the 2 that saw it (`trial 08 ... iters=18325061`, `trial 09 ... iters=477666`) needed millions of iterations; every fenced-uncached arm saw it in 14-17K iterations. This is exactly the required contrast: the plain read misses the store (8/10 stale) while the fixed path sees it (10/10).
- Controls 1-3 recorded honestly, not as passes: control 1 (between-kernel, 20 trials) "did NOT reproduce a stale read; recorded as measured" (0/20); control 2 "confounded and inconclusive" (plain store not device-visible mid-run, uncached load compiler-hoistable); control 3 "plain ... stale OBSERVED ... the uncached arm was invalid (missing acquire fence, load hoisted)". Each labels its own limitation.

### Gate 4 — Build/test receipts: PASS
- SYCL_BUILD_RC=0 and SYCL_CTEST_RC=0 recorded in the log (also SYCL_CONFIGURE_RC=0, SHASUM_RC=0, MTIME_RC=0).
- ctest summary quoted verbatim in the log: `100% tests passed, 0 tests failed out of 2` with `ds4_parity ... Passed 0.21 sec` and `ds4_blas_parity ... Passed 0.31 sec`, `Total Test time (real) = 0.53 sec`.
- Engine smoke receipt: RUN_RC=0, PROMPT_RC=0; the run (`--gpus 0 --ids 271 --n 8 --max-context 512`) prints `prompt: 1 tokens in 105 ms (9.54 tok/s, windows of 8192)` and `decode: 8 tokens in 378 ms (18.51 tok/s); CPU lane: 0 of 1680 experts`. Both throughput figures come from a completed prompt/decode — no throughput is quoted without a completed prompt line.
- CTEST_RECHECK_RC: not re-runnable (build dir removed); recorded receipts judged sufficient (see Evidence gaps).
- No parity/trace run beyond ctest and the smoke is claimed in this todo; the plan gates todos 3-6 by engine smoke, which is present.

### Gate 5 — Cleanup receipts: PASS
- CLEANUP_RC=0 (`rm -rf /tmp/dpct_w3_build`) and LS_RC=2 (`ls /tmp/dpct_w3_build` → gone) recorded.
- BUILD_LS_RC=2 — independent `ls /tmp/dpct_w3_build` → "No such file or directory".
- WT_LS_RC=0 — worktree `/home/maxious/Strata-wt/dsv41-dpct-w3` retained for merge (branch dsv41-dpct-w3, HEAD 0857e37e).

### RC / pipe / throughput hygiene: PASS
- Every `*_RC` in the log sits on the line immediately after its own command; redirections are `> file` / `2>&1` (files), never pipes. No piped exit code is reported as a result. No throughput number is quoted without a completed `prompt:`/`decode:` line.

## Slop / overfit pass (remove-ai-slops + programming, run directly)

- Tests: the commit adds zero test files and zero test lines (diff = 4 `.dp.cpp` files, 34 insertions/15 deletions). Therefore no excessive, deletion-only, tautological, implementation-mirroring, or "verify the requested removal" tests exist to flag. The parity gates (`ds4_parity`, `ds4_blas_parity`) are pre-existing and still green; the plan explicitly gates todos 3-6 by engine smoke, not ctest, and the smoke ran.
- Production extraction / parsing / normalization: none. The diff is a pure read-path conversion plus audit-note comments and three `#include "strata/sycl_doorbell.hpp"` lines; no new helper, abstraction, or wrapper was introduced (the plan forbade a new read helper, and none was added).
- Comment slop: the added `// audit note: CORRECT/WRONG - ...` lines are WHY comments (they name the allocation and the reason a read is device-resident vs host-mapped), which is the todo's own deliverable ("record a note ... naming the class and the reason"), not an obvious-comment restatement. Not slop.
- The throwaway controls (`/tmp/dpct_w3_ctrl{,2,3,4}.cpp`) are evidence, not committed tests; control 4 is a can-fail control (it observed the stale value 8/10). The "failure-control must actually run and be measured, not asserted impossible" clause is satisfied.
- Programming-criteria note: the changed files are C++/SYCL, outside the skill's per-language reference set (Python/Rust/TS/JS/Go). Applied the shared axioms: behavior-preserving conversion (plain load → uncached load of the same location with the same element indexing — `[t]` sites gained `+ t`, `*flag` sites read element 0), no new validation, no scope drift, and the run is the proof (build + ctest + smoke all green).

## Verdict

APPROVE — all five gates PASS. The DPCT1098 audit enumerated every host-mapped read, fixed the plain reads, corrected both plan attributions against the code, and demonstrated the fix with a control that can (and did) fail on the plain path. No blocker tied to a stated success criterion exists.
