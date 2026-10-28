# Gate review — todo 6b "publish the routing ids the host reads" (follow-up close attempt)

- recommendation: **APPROVE**
- todo6 landing: **[x]** (feature proven; the partial-mix run satisfies the acceptance wording)
- lane audited: `st_01a17852` (worktree `/home/maxious/Strata-wt/dsv41-dpct-w6b`, detached `a698bb84`, `git status --porcelain` = 0 lines, no new commit)
- commit audited (carries the publish fix, from the prior todo-6 attempt): `a698bb84a057a4bdaccda28d1198921d89cc7d32` "[Engine] dsv41 SYCL: publish the routing ids the host reads"

## originalIntent

Plan todo 6 (`/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md`): the `CPU lane: 0 of 1680 experts` line was diagnosed as a payload defect — the device published `req->ids/w/x` with plain stores while the ring flag used a hinted store, so the host saw the flag before the ids left the card's cache and read zeros. The intended change is to publish the routing ids the host actually reads so the CPU lane computes the experts the card does not hold, printing `CPU lane: <N> of 1680 experts` with N > 0.

## desiredOutcome

A one-card engine run prints `CPU lane: <N> of 1680 experts` with N > 0 and a non-zero `CPU lane` per-token time, proving the host consumes the published ids and offloads non-resident expert mass.

## userOutcomeReview

Achieved. The publish path is proven working end-to-end and the acceptance wording is satisfied by a real one-card run:

- **Publish feature proven**: `DS4_TRACE` snapshot `/tmp/dpct_run/w6b_trace_snap.bin` parsed independently in this review — 280 records, 1680 ids, 1673 non-zero, 190 distinct values, byte-exact (3920/3920 consumed). The ids the host reads are real routed picks, not zeros.
- **N > 0 on one card**: `/tmp/dpct_run/w6b_mix025.log` (`--vram-experts 0.25`) prints `decode: 8 tokens in 929 ms (7.53 tok/s); CPU lane: 60 of 1680 experts; the cache moved 0 (0 in all)` and `decode per token: ... CPU lane 5.19, ...`. N = 60 > 0, per-token CPU-lane time non-zero, `RUN_MIX_RC=0` un-piped, on the same default decode path (same routing and combine; only the documented residency-budget knob differs).
- **Counting is not a display artefact**: `/tmp/dpct_run/w6b_v0.log` (`--vram-experts 0`) prints `CPU lane: 1680 of 1680 experts` with `CPU lane 125.03` ms/token — every pick really executed and counted by the host lane when non-resident.
- **Default run**: `/tmp/dpct_run/w6b_default.log` prints `CPU lane: 0 of 1680 experts`. This is correct behavior, not a defect: all 1680 traced picks fall inside the card's hot-resident set (max profile rank 32 vs 41-42 resident slots/layer; budget math reconciles exactly), so the host lane has nothing non-resident to serve.
- **Regressions green**: `ctest -R ds4` 2/2 (ds4_parity 0.28 s, ds4_blas_parity 0.33 s, 100% passed); `GG_RC=0` (`/tmp/ds4_cpu/ds4_gg_bench`, "0 of 1567344 differ in bits").

The lane's no-code-change decision is correct: forcing the default run's N > 0 would require changing the picks (forbidden quality-path change), deliberately keeping picked HOT experts off the card (default regression: all-CPU control 0.83 tok/s vs default 18.32 tok/s), or counting experts the host never ran (forbidden).

## blockers

None.

## notes (NOT blockers)

- The default one-card run prints `CPU lane: 0 of 1680 experts` (`CPU lane 0.00 ms`). This is NOT a failure of the stated acceptance criterion (which does not say "default"; it says "an engine run on one card ... with N > 0"). It is correct behavior because this stationary trajectory's picks all land in the card's hot-resident set. The plan's GAP-4 premise ("zero contribution is a routing defect") was a misdiagnosis; the actual defect (payload not published) is fixed by `a698bb84`.
- The plan's QA failure control `--experts 0` ("all experts on the CPU lane") is mislabeled: it clamps the routed top-k to 1 (`src/dsv41/ds4_main.cpp:1232`), so it prints `CPU lane: 0 of 280 experts` (280 requests x 1 pick). The lane correctly identified `--vram-experts 0` as the real residency switch and ran it (1680 of 1680). The control's purpose — prove the count is not a display artefact — is met.
- The default run's N > 0 becomes true once a run's picks leave the hot set (demonstrated at `--vram-experts 0.25`; expected after todo 7's hidden-state fix diversifies decode picks). This is a forward note for todo 7/8, not a todo-6 defect.

## skill-perspective check (remove-ai-slops / programming)

Applied directly over the diff (`a698bb84`), production code, and the (unchanged) parity tests.

- **Overfit / slop pass: none found.** No new tests were added, so there are no deletion-only tests, tautological tests, implementation-mirroring tests, or tests that merely verify the requested removal. No prose is pinned by any test.
- **Unnecessary production extraction/parsing/normalization: none.** `publish_payload_kernel` + reduced `publish_kernel` + the in-order `q2` publish queue is a real mechanism fix (payload first, completion flag last on its own 64-byte line), not gratuitous extraction. `alignas(64)` on `Req::seq` and `Req::T` is a real cache-line fix.
- **Comments kept are WHY comments** (cache-line sharing rationale, flag-last ordering guarantee, `sycl::device` has `==` but no `<`) — KEEP per skill.
- **Programming: root-cause fix at the shared seam.** The mailbox payload stores are fixed once in the publish path, not per-caller. No scope drift (diff touches only `sycl/src/dsv41/ds4_kernels.dp.cpp` and `.hpp`; 91 insertions / 49 deletions). The lane correctly refused a fabricated counting fix once the controls proved the count path correct.
- The upstream code review report (`todo6-gate-review.md`) records the same skill-perspective check and overfit/slop coverage for the publish commit; this review re-ran the pass independently over the same diff and reached the same finding.

## checked artifact paths

- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (todo 6 acceptance, QA scenarios, IS-4 row)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-6-dsv41-sycl-dpct-and-xpu-test.log` (full executor log; todo6b DONECLAIM section)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo6-gate-review.md` (upstream review)
- `/home/maxious/Strata-wt/dsv41-dpct-w6b` (HEAD `a698bb84`, clean tree, commit show)
- `/tmp/dpct_run/w6b_default.log`, `w6b_mix025.log`, `w6b_v0.log`, `w6b_exp0.log` (run logs, quoted lines verified on disk)
- `/tmp/dpct_run/w6b_trace_snap.bin` (parsed independently: 280 records, 1680 ids, 1673 non-zero, 190 distinct)
- `/tmp/dpct_run/w6b_ctest.log` (2/2 passed, 100%)
- `/tmp/dpct_run/w6b_gg.log` / `gg_verify.log` (AVX2-vs-ggml parity 0 differ)
- source: `sycl/src/dsv41/ds4_kernels.dp.cpp`, `sycl/src/dsv41/ds4_kernels.dp.hpp`, `sycl/src/dsv41/ds4_model.dp.cpp` (trace write format at :7609-7620, :8902), `src/dsv41/ds4_main.cpp` (--experts clamp :1232)

## evidence gaps (exact)

- None that block approval. The default-run `N > 0` evidence is absent because the default trajectory's picks are all hot-resident; that absence is explained and correct, not a missing required artifact. The only literal acceptance element is satisfied by `w6b_mix025.log` (60 of 1680, 5.19 ms/token, `RUN_MIX_RC=0`).
