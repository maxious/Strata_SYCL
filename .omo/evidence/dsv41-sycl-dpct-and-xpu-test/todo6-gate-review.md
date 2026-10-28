# Gate review — todo 6 "publish the routing ids the host reads"

- recommendation: **REJECT**
- verdict: **VERDICT FAIL**
- todo6 landing: **[~]** (partial — the publish-path fix is correct and necessary, but the headline acceptance criterion is unmet)
- commit audited: `71e069e986c2bb81c94cd92834198e7e44c5ad8a` on `18a27e02fd5d3b63dddf9a5a1863abaf5581f769` (worktree `/home/maxious/Strata-wt/dsv41-dpct-w6`)
- lane audited: `st_01a177ba` (deep-low), attempts 6-9

## originalIntent

From the plan (`/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md`, todo 6): the zero CPU-lane count was diagnosed as a payload defect — the device published `req->ids/w/x` with plain stores while the ring flag used a hinted store, so the host saw the flag before the ids left the card's cache and read zeros; every pick then resolved to expert 0 (a VRAM-resident expert) and no pick ever hit the CPU lane. The intended change is to publish the routing ids the host actually reads, so the CPU lane contributes the experts the card does not hold.

## desiredOutcome

A one-card engine run where the CPU lane performs real expert compute: the run prints `CPU lane: <N> of 1680 experts` with **N > 0**, and a **non-zero `CPU lane` per-token time**, proving the host is consuming the routed ids and offloading the non-resident expert mass.

## userOutcomeReview

Not achieved. The default one-card run still prints `CPU lane: 0 of 1680 experts` with `CPU lane 0.00 ms` per token, and the emitted ids are still the degenerate repeat (`129276,129276,...` — the exact symptom todo 6 was meant to cure via the restored CPU-lane mass). The only N > 0 evidence is a `--vram-experts 0` run that forces *zero* experts into VRAM (all 15360 experts on the CPU lane), which is a degenerate all-CPU control, not the "CPU lane contributes the experts the card does not hold" scenario the todo names.

The lane's *technical diagnosis is correct and honest*: the trace proves the publish fix delivers 1680 real routed ids (1673 non-zero, 190 distinct, per-layer coherent), and the `--vram-experts 0` control proves the count path works (1680 of 1680, CPU lane 102.13 ms/token). But the fix is **necessary, not sufficient**: with real picks delivered, the picks still all land on the card's hot-resident set (1687 of 15360 experts in VRAM), so the CPU lane contributes 0 experts during a normal decode. The todo's stated goal is therefore unmet even though the commit is a correct root-cause fix for the payload defect.

## blockers

1. violatedCriterion: `todo6-accept-nonzero` — "an engine run on one card prints `CPU lane: <N> of 1680 experts` with N > 0 and a non-zero `CPU lane` per-token time" (plan lines ~155-165 acceptance criteria, quoted verbatim in the plan).
   observation: The default one-card run prints `CPU lane: 0 of 1680 experts` and `CPU lane 0.00 ms`; N > 0 appears only under `--vram-experts 0` (0 experts in VRAM, all-CPU), a control the plan never names. The desired outcome — the CPU lane contributing experts during normal decode — is not demonstrated. The lane records this itself as `CPU_LANE_NONZERO_RC=1`.
   evidencePointer: `/tmp/dpct_run/run_t6b.log` (default: "decode: 8 tokens in 381 ms (18.38 tok/s); CPU lane: 0 of 1680 experts; … CPU lane 0.00") vs `/tmp/dpct_run/run_t6b_v0.log` (`--vram-experts 0`: "CPU lane: 1680 of 1680 experts; … CPU lane 102.13").

## gate results (the five mandated gates)

- **G1 plan reread — FAIL.** The acceptance criterion "N > 0 and non-zero CPU-lane time on a one-card run" is not met by the default run (N = 0, 0.00 ms). It is met only by `--vram-experts 0`. The plan's QA "failure" control names `--experts 0` as "all experts on the CPU lane"; the lane correctly proved `--experts 0` is actually the routed top-k clamped to 1 (ds4_main.cpp:1230-1233) and that the real residency switch is `--vram-experts 0` (ds4_main.cpp:1214).
- **G2 receipts match disk — PASS.** `w6_build8.log` tail = `[53/53] Linking CXX executable strata`, `grep -c "error:" = 0`; `w6_ctest8.log` verbatim `100% tests passed, 0 tests failed out of 2`; commit scope `git diff 18a27e02..HEAD` = exactly the 2 files; `/tmp/dpct_w6_build` absent (LS_RC=2). Worktree clean, HEAD = the audit commit.
- **G3 honesty — PASS.** `CPU_LANE_NONZERO_RC=1` is stated (default N=0 is reported, not hidden). All `*_RC` captured un-piped beside their commands. Throughput only from completed prompts.
- **G4 mechanism fidelity — PASS.** The diff adds `publish_payload_kernel` (ids/w/x/na/T via `strata::sys_store_mapped(plain, ...)`), `publish_kernel` reduced to the flag store last (`strata::sys_store` on `alignas(64) seq`), and a per-device in-order `publish_queue` (q2) with payload `depends_on(routed)` + `s->ext_oneapi_submit_barrier({payload})` + flag submitted after payload on q2. `sys_store_mapped(bool plain, ...)` / `doorbell_plain_payload()` confirmed in `sycl/include/strata/sycl_doorbell.hpp:99-112`. **No counting logic altered**: the diff touches only `sycl/src/dsv41/ds4_kernels.dp.cpp` and `.hpp`; the count path (`ds4_model.dp.cpp` `serve_lane` `cpu_experts += nslots`; `ds4_main.cpp` print) is unchanged.
- **G5 cleanup — PASS.** `LS_RC=2`, `/tmp/dpct_w6_build` is gone.

## evidence gaps (exact)

- A normal one-card run with `N > 0` and a non-zero `CPU lane` time is **missing**. The two runs on disk are the default (`N=0`) and the degenerate `--vram-experts 0` control (`N=1680`). No run shows the intended partial mix (0 < N < 1680).
- No prompt that exercises the long-tail expert mass was tried, so the lane's "hot-set residency" explanation for N=0 is inferred from the residency split code (`ds4_model.dp.cpp:4514-4522`) and the run's own 0-count, not from a run whose picks fall outside the hot set.
- The plan's "latent drift" (`plan_chunk` consuming mailbox seqs without publishing) is reported as `UNVERIFIED-STATIC` (does not reproduce on static reading); no dedicated empirical test isolates it, though the trace + all-CPU runs are offered as indirect coverage.

## skill-perspective check (remove-ai-slops / programming)

Applied directly over the diff, production code, and the (unchanged) parity tests.

- **Overfit / slop pass: none found.** No new tests were added, so there is no deletion-only test, tautological test, implementation-mirroring test, or prose-pinning test. The pre-existing `ds4_parity` / `ds4_blas_parity` tests passed unchanged (100%, 2/2) and are behavior tests (bit-parity), not diff-certifiers.
- **Unnecessary production extraction/parsing/normalization: none.** The change is a focused root-cause fix: it moves the mailbox payload stores out of `route_publish_kernel` into a dedicated `publish_payload_kernel` and puts the flag last on its own cache line. Comments are WHY comments (the doorbell-publish design, the flag-last guarantee, `sycl::device` has `==` but no `<`), which the skill says to KEEP.
- **Programming: root-cause fix confirmed.** The lane fixed the shared seam (payload stores) once, not a per-caller symptom; it did not touch the CPU expert kernel semantics (AVX2 bit-parity preserved, `GG_VERIFY_RC=0`: "0 of 1567344 differ in bits", reproduced independently). No new test is warranted under the repo's conventions (the parity tests are the machine-consumed seam; the doorbell fix is verified by the DS4_TRACE artifact and the residency control).

## notes (NOT blockers)

- The plan's QA control `--experts 0` is mislabeled: it is the routed top-k clamped to 1, not "all experts on the CPU lane". The lane correctly substituted `--vram-experts 0` and proved the count path is not a display artefact (1680 of 1680). Count-path correctness is established; only the happy-path N > 0 is missing.
- The plan's GG criterion "runs the same prompt" is imprecise: `/tmp/ds4_cpu/ds4_gg_bench` takes no prompt argument (it is the AVX2-vs-ggml parity benchmark). `GG_RC=0` is reproduced.
- `publish_queue` statics are intentionally never freed (comment), consistent with the parent commit's "never free the codebook statics at exit".

## checked artifact paths

- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (todo 6 criteria, lines ~155-165)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-6-dsv41-sycl-dpct-and-xpu-test.log`
- `/home/maxious/Strata-wt/dsv41-dpct-w6` (HEAD `71e069e9`, diff vs `18a27e02`)
- `/tmp/dpct_run/run_t6b.log` (default), `/tmp/dpct_run/run_t6b_v0.log` (`--vram-experts 0`), `/tmp/dpct_run/run_t6b_e0.log` (`--experts 0`), `/tmp/dpct_run/run_t6.log`, `/tmp/dpct_run/run_t6_trace.log`
- `/tmp/dpct_run/w6_trace.bin` (decoded: 280 records, 1680 ids, 1673 non-zero, 190 distinct, max 383)
- `/tmp/dpct_run/w6_build8.log`, `/tmp/dpct_run/w6_ctest8.log`, `/tmp/dpct_run/w6_ctest.log`
- `/tmp/dpct_run/gg_verify.log` (re-run of `/tmp/ds4_cpu/ds4_gg_bench`, rc 0)
- source: `sycl/src/dsv41/ds4_kernels.dp.cpp`, `sycl/src/dsv41/ds4_kernels.dp.hpp`, `sycl/include/strata/sycl_doorbell.hpp`, `sycl/src/dsv41/ds4_model.dp.cpp`, `src/dsv41/ds4_main.cpp`
