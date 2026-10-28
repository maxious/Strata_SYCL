# Todo 11 gate review — measured report and push (dsv41-sycl-dpct-and-xpu-test)

Date: 2026-10-26. Branch: `sycl/dsv41-sycl-support`. HEAD `d337362b8a9fa09402fd41e2636cbd7beae66161`
("[Docs] dsv41 SYCL: measured report for the ported engine"), one file
`sycl/docs/sycl-dsv41-measured-report.md` (72 lines). Reviewer: independent gate review; no product
file was edited, nothing was pushed, no monitor was used.

Every RC below is the un-piped exit code of the command printed directly below it. The two `grep -c`
count checks are the caller's sanctioned form; per this evidence tree's convention (see
`dsv41-sycl-dpct-and-xpu-test-manual-qa.md`), a grep exit of 1 means "no match found", which is the
expected result for those checks.

## RC1 — the report and its numbers against the evidence

### Cross-check of every quoted number (report line numbers from the 72-line file)

| Report quote | Evidence | Verdict |
| --- | --- | --- |
| commit `ce6218ee` | `dsv41-sycl-dpct-and-xpu-test-manual-qa-two-card.md` header ("Commit: ce6218ee2642eaaba29be0735c4a8e45c27d11f2"), `row16d-gate-review.md` | backed |
| 2x Intel Arc Pro B60, 24 GB each | `dsv41-sycl-dpct-and-xpu-test-manual-qa.md` ("2x Intel Arc Pro B60 `8086:e211`"); engine logs "24 GiB of graphics memory" | backed (see hardware flag below) |
| Intel oneAPI 2026.2, Level Zero | `row16d-gate-review.md` ("icx/icpx from oneapi 2026.2"), `SYCL_BE=PI_LEVEL_ZERO` in every invocation | backed |
| "Q4_K quantised, about 115 GB" | no evidence log quotes any quant label or model size; pack manifest contradicts both | **FAIL 1 (below)** |
| 10 prompt tokens, 8 answer tokens | `artifacts/one-card-baseline.log` "prompt: 10 tokens ..."; `--n 8` in every invocation | backed |
| one card 2026-10-27, exit code 0 | `artifacts/EXECUTION-COMPLETE.txt` ("Date: 2026-10-27 ... Exit code: 0") | backed |
| 7.92 tok/s (10 tokens in 1262 ms, windows of 8192) | `artifacts/one-card-baseline.log` `prompt: 10 tokens in 1262 ms (7.92 tok/s, windows of 8192)` | backed, completed prompt |
| 1.65 tok/s (8 tokens in 4231 ms) | `artifacts/one-card-baseline.log` `decode: 8 tokens in 4231 ms (1.65 tok/s); CPU lane: 109 of 1680 experts` | backed, completed decode |
| first id 671; ids `671,6102,294,8760,344,2619,51119,42499` | `artifacts/one-card-baseline.log`, `artifacts/decoded-answers.txt` | backed |
| decoded "The capital of France is \*\*Paris\*\*." | `artifacts/decoded-answers.txt` (pack tokenizer) | backed |
| 109 of 1680 experts on CPU (one-card run) | `artifacts/one-card-baseline.log` `CPU lane: 109 of 1680 experts` | backed (see CPU-lane flag below) |
| ddc3c9e7 recorded 157 of 1920 | `todo7b-gate-review.md` (merged build ddc3c9e7: "CPU lane: 157 of 1920 experts") | backed (see CPU-lane flag below) |
| two-card runs 2026-10-27, exit code 0 x2 | `artifacts/EXECUTION-COMPLETE.txt` (run 1 and run 2 "Exit code: 0") | backed |
| card 0 layers 0-20, card 1 layers 21-39 + final | `artifacts/two-card-run1.log` / `two-card-run2.log` (`CUDA0 layers [0, 20]`, `CUDA1 layers [21, 39] + head`) | backed |
| run 1: 19.88 tok/s (10 in 503 ms); 2.44 tok/s (8 in 2873 ms) | `artifacts/two-card-run1.log` `prompt: 10 tokens in 503 ms (19.88 tok/s ...)` and `decode: 8 tokens in 2873 ms (2.44 tok/s)` | backed, completed prompt and decode |
| run 2: 38.09 tok/s (10 in 263 ms); 12.08 tok/s (8 in 579 ms) | `artifacts/two-card-run2.log` `prompt: 10 tokens in 263 ms (38.09 tok/s ...)` and `decode: 8 tokens in 579 ms (12.08 tok/s)` | backed, completed prompt and decode |
| same eight ids in both two-card runs | `artifacts/two-card-run1.log` / `two-card-run2.log`, `artifacts/decoded-answers.txt` ("Token IDs from both two-card runs") | backed |
| run 2 copied all 11011 experts, 94.7 GB in 36.0 s | `artifacts/two-card-run2.log` `11011 of the CPU lane's 11011 experts copied into RAM (94.7 GB, 1 NUMA part) in 36.0 s` | backed |
| journal reset check found no matches (exit code 1) | `dsv41-sycl-dpct-and-xpu-test-manual-qa-two-card.md` scenario 5 (grep exit code 1 after each run) | backed; see note 2 (single-BDF scope) |
| parity rows 0 of 51 exact / 51 of 51 within one BF16 step / 0 outside | `row14-gate-review.md` lines 117-119 (Q2_K/Q4_K/Q5_K gemv rows) and `task-7-dsv41-sycl-dpct-and-xpu-test.log:353` | backed; see note 3 (measurement commit) |
| "pass on Linux, CUDA, HIP and SYCL" | SYCL/Linux backed (`row15-gate-review.md` Gate 3, `row16d-gate-review.md` Gate 2); CUDA/HIP contradicted | **FAIL 2 (below)** |
| loud-spin prints "the card never saw an answer", exits 0 instead of 1 | `artifacts/loud-spin-control.log` (both message lines), `artifacts/SUMMARY.txt` ("exit code 0 vs expected 1") | backed |
| "on the earlier build the same control ended with exit code 1" | `f4-fidelity.md:63` (`RUN_SPIN_RC=1`), `task-5-dsv41-sycl-dpct-and-xpu-test.log:55` (`RUN_SPIN_RC=1`), `task-9-dsv41-sycl-dpct-and-xpu-test.log:172` (`SPIN_RUN_RC=1`), `f3tc-receipts-runs.txt:37` (`SPIN_RUN_RC=1` at 4e003d34) | backed |
| `--n 8` stops before EOS id 1 | `dsv41-sycl-dpct-and-xpu-test-manual-qa.md` S-8-4 note; `todo7b-gate-review.md` (ddc3c9e7 run emitted `...42499,1`) | backed |
| Version 0.2.6, engine 0.1.51 | repo doc header convention (AGENTS.md); not a measurement | accepted |

### Flag class (a) — quoted throughput with no completed prompt behind it

NONE. All six quoted throughput figures (7.92, 1.65, 19.88, 2.44, 38.09, 12.08 tok/s) sit directly
beside a completed `prompt:` line and a completed `decode:` line in `artifacts/one-card-baseline.log`,
`artifacts/two-card-run1.log` and `artifacts/two-card-run2.log`. The loud-spin run's own
`prompt: 10 tokens in 257 ms (38.87 tok/s)` is correctly NOT quoted anywhere in the report (its
answer never completed), matching the discipline `todo5-gate-review.md` line 47 requires.

### Flag class (b) — hardware mislabel

The REPORT is correct: it says "two Intel Arc Pro B60 cards, 24 GB of graphics memory each",
matching `dsv41-sycl-dpct-and-xpu-test-manual-qa.md` ("2x Intel Arc Pro B60 `8086:e211`") and
`row16d-gate-review.md` ("Intel Arc Pro B60"). The mislabel the caller warns about lives in the
EVIDENCE, not the report, and is flagged here: `dsv41-sycl-dpct-and-xpu-test-manual-qa-two-card.md`
says "Hardware: Intel Arc A770 x2 (devices 0,1)" and "A770 24GB sufficient", and
`artifacts/SUMMARY.txt` says "Hardware: 2x Intel Arc A770". The true hardware is 2x Intel Arc Pro
B60, 24 GB, device ids `8086:e211`. The report did not propagate the A770 mislabel.

### Flag class (c) — CPU-lane figures vs the run each names

All consistent; no figure is swapped:
- 109 of 1680 belongs to the ce6218ee one-card run (`artifacts/one-card-baseline.log`, the same run
  whose 7.92/1.65 tok/s the report quotes) — the report names it there. Matches the caller's truth.
- 157 of 1920 belongs to the ddc3c9e7 `--n 32` run (`todo7b-gate-review.md`) — the report names it
  "an earlier build, ddc3c9e7" and says "That run asked for a longer answer", which matches `--n 32`
  vs the measured runs' `--n 8`. Matches the caller's truth.
- The ce6218ee gate run's 105 of 1680 (`row16d-gate-review.md`) is NOT quoted in the report; no
  contamination. (The 4e003d34 AOT receipts lane's 112 of 1680, `f3tc-keylines-base.txt`, is also
  correctly not quoted.)

### FAIL 1 — model identity "Q4_K quantised, about 115 GB" (report line 12) is not backed and is contradicted by the pack the runs used

The measured runs all passed `--pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4`,
whose own manifest (`maya-ds4/tensors.txt`, `maya-ds4/ds4.txt` `model_dir ...GSQ-RCO`) records the
pack's real identity. Tensor census over `tensors.txt` (per-tensor quant labels and bytes):

- IQ1_M: 23040 tensors, 59.45 GB; IQ2_XXS: 16128 tensors, 49.05 GB; IQ2_XS: 6912 tensors, 23.57 GB;
  Q5_K: 524 tensors, 5.33 GB; IQ4_NL: 2 tensors (the Engram file), 110.60 GB; F32 0.16 GB;
  Q8_0 0.39 GB; BF16 0.28 GB. **Q4_K: zero tensors.**
- On-disk GGUF files: four `...GSQ-RCO-IQ2_XXS-0000x-of-00004.gguf` shards (39.08 + 39.39 + 39.25 +
  20.53 = 138.24 GB) plus `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (110.60 GB) = **248.83 GB total
  (231.74 GiB)**. No accounting in any evidence log or in the pack produces "about 115 GB"
  (closest figures: the one-card CPU-lane experts 117.9 GB and the RAM store 93.4 GB, neither of
  which is the model's size). `docs/MODELS.md:49` itself says DeepSeek V4.1-Flash needs "257 GB of
  disk".

No evidence log under `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/` quotes a quant label or a model
size at all (grep over the directory for `115 GB` and `Q4_K quant`: zero hits outside the report).
Plan todo 11's own text requires "pack identity" and states "Do not paste any number not produced in
this session"; `115` was produced by nothing in this session. The correct pack identity is
DeepSeek-V4.1-Flash-1017-GSQ-RCO (IQ1_M/IQ2_XXS/IQ2_XS experts, Q5_K, IQ4_NL engram), 248.83 GB on
disk.

### FAIL 2 — "`ds4_parity` and `ds4_blas_parity` pass on Linux, CUDA, HIP and SYCL" (report line 58) overclaims what was measured

Only the Linux SYCL build was measured: `row15-gate-review.md` Gate 3 and `row16d-gate-review.md`
Gate 2 run `ctest -R ds4` on the icpx SYCL build (2/2 pass). `task-15-dsv41-sycl-dpct-and-xpu-test.log:78`
states plainly: "not run: the CUDA and HIP builds - no CUDA or HIP hardware here, and no CUDA or HIP
file is touched." The two tests exist only as SYCL targets
(`sycl/CMakeLists.txt:273-292`: `add_executable(ds4_parity ...)` / `ds4_blas_parity` under
`STRATA_SYCL_DSV41`, linked to `strata_dsv41` and oneMKL); there is no CUDA or HIP build of either
test in the tree, and the plan's IS-8 only claims "`ctest -R ds4` passes on the SYCL build". The
phrase "pass on Linux, CUDA, HIP and SYCL" occurs nowhere in the evidence logs, the plan, or the
docs except in this report. Unbacked as written; the supported claim is "both suites pass on the
Linux SYCL build".

## RC2 — scope of the commit range

SHOW_PATHS_RC=0
git show --name-only --format= d337362b

(sycl/docs/sycl-dsv41-measured-report.md — the only path, under sycl/.)

SHOW_NONSYCL_COUNT=0 (grep -c; RC 1 = no match = expected)
git show --name-only --format= d337362b | grep -cvE '^sycl/'

LOG_SUBJECTS_RC=0
git log --format=%s 18a27e02..HEAD

(12 subjects, every one starting `[Engine]` or `[Docs]`: d337362b `[Docs]`, 83417a6f `[Engine]`,
ce6218ee `[Engine]`, 4e003d34 `[Engine]`, ddc3c9e7 `[Engine]`, 28bd2aaa `[Engine]`, 39b128ad
`[Engine]`, a698bb84 `[Engine]`, 433868e5 `[Engine]`, 2e5c53b9 `[Engine]`, b18f6c40 `[Engine]`,
ca0f6f94 `[Engine]`.)

SUBJECT_PREFIX_VIOLATIONS=0 (grep -c; RC 1 = no match = expected)
git log --format=%s 18a27e02..HEAD | grep -cvE '^\[(Engine|Docs)\] '

OUTSIDE_SYCL_DOCS_COUNT=0 (grep -c; RC 1 = no match = expected)
git diff 18a27e02..HEAD --name-only | grep -cvE '^(sycl|docs)/'

COMMITTED_OMO_BENCH_COUNT=0 (grep -c; RC 1 = no match = expected)
git diff 18a27e02..HEAD --name-only | grep -cE '^(\.omo/|bench/results)'

COMMITTED_OMO_BENCH_COUNT_D337=0 (grep -c; RC 1 = no match = expected)
git show --name-only --format= d337362b | grep -cE '^(\.omo/|bench/results)'

RC2 result: PASS. d337362b lists only sycl/ paths; every subject in 18a27e02..HEAD carries the
`[Engine]`/`[Docs]` prefix; 0 paths outside sycl/ and docs/ in the range; no `.omo/` or
`bench/results` path is committed.

## RC3 — push receipt and working tree

PUSH_REV_PARSE_RC=0
git rev-parse fork/sycl/dsv41-sycl-support

(d337362b8a9fa09402fd41e2636cbd7beae66161 — equal to `git rev-parse HEAD`
d337362b8a9fa09402fd41e2636cbd7beae66161. The fork ref is at the gated commit.)

STATUS_PORCELAIN_RC=0
git status --porcelain

(Only `??` untracked entries: `?? .omo/` and `?? bench/results/2026-10-*` directories.)

TRACKED_CHANGES_COUNT=0 (grep -c; RC 1 = no match = expected)
git status --porcelain | grep -cvE '^\?\?'

STATUS_PORCELAIN_TRACKED_RC=0
git status --porcelain --untracked-files=no

(no output: zero tracked-file changes.)

RC3 result: PASS. `fork/sycl/dsv41-sycl-support` == HEAD == d337362b8a9fa09402fd41e2636cbd7beae66161;
no tracked-file changes are present (only untracked `.omo/` and `bench/results/`, which are the
plan's forbidden commit paths and are correctly uncommitted).

## RC4 — diffstat of the report commit

DIFFSTAT_RC=0
git diff --stat 83417a6f..HEAD

(sycl/docs/sycl-dsv41-measured-report.md | 72 +++++... ; 1 file changed, 72 insertions(+) — exactly
the caller's expected stat.)

RC4 result: PASS.

## Notes (not the verdict drivers)

1. Evidence-side provenance split: `f3tc-receipts-runs.txt` / `f3tc-build-identity.txt` are the
   earlier card-run phase (task st_01a178c4) on the AOT build at 4e003d34 (base/run1/run2 RC=0,
   `SPIN_RUN_RC=1`); the numbers the report quotes come from the later manual QA (st_01a178f9) at
   ce6218ee (`artifacts/*.log`, `dsv41-sycl-dpct-and-xpu-test-manual-qa-two-card.md`). The two sets
   differ (e.g. 4e003d34 base: 36.01 tok/s prompt, 2.61 decode, CPU lane 112 of 1680) and the report
   consistently quotes only the ce6218ee set. No cross-set contamination found.
2. The two-card QA's journal reset check greps only `0000:11:00.0` (device 1's BDF); "Neither run
   reset a card" is slightly broader than that single-BDF check, though the run logs show no reset
   line. Suggest narrowing the report's wording to what the check covered.
3. The 0/51-51/51-0 parity counts were printed by the row-14 harness at 4e003d34 (and re-verified at
   28bd2aaa/ddc3c9e7 per todo 7); at ce6218ee the same rows pass in `row16d-gate-review.md`'s ctest.
   "Measured ... at commit ce6218ee" is therefore substantively right (comment-only commits between)
   but the printed counts themselves come from the parent commit's run.
4. Plan-literal deviation, recorded for the orchestrator: plan todo 11's body asks to tag claims in
   `docs/SYCL_DSV41_PLAN.md` and `docs/SYCL_DSV41_PLAN2.md` and to quote a hardware/software block
   including `8086:e211`, PCIe, CPU/RAM, kernel driver, NEO runtime; the delivered commit instead
   adds `sycl/docs/sycl-dsv41-measured-report.md` with a shorter hardware block (no PCI ids, CPU/RAM,
   NEO runtime). This gate's RC set is scoped to the delivered report plus the push, so this is a
   note, not a verdict driver; but the report also omits the plan-required `8086:e211` identity line.

## Checked artifact paths

- sycl/docs/sycl-dsv41-measured-report.md (report under review, 72 lines)
- .omo/evidence/dsv41-sycl-dpct-and-xpu-test/dsv41-sycl-dpct-and-xpu-test-manual-qa.md
- .omo/evidence/dsv41-sycl-dpct-and-xpu-test/dsv41-sycl-dpct-and-xpu-test-manual-qa-two-card.md
- .omo/evidence/dsv41-sycl-dpct-and-xpu-test/row15-gate-review.md
- .omo/evidence/dsv41-sycl-dpct-and-xpu-test/row16d-gate-review.md
- .omo/evidence/dsv41-sycl-dpct-and-xpu-test-todo4-gate-review.md
- .omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo7b-gate-review.md
- .omo/evidence/dsv41-sycl-dpct-and-xpu-test/row14-gate-review.md (parity row counts)
- .omo/evidence/dsv41-sycl-dpct-and-xpu-test/artifacts/{one-card-baseline.log,two-card-run1.log,two-card-run2.log,loud-spin-control.log,decoded-answers.txt,SUMMARY.txt,EXECUTION-COMPLETE.txt}
- .omo/evidence/dsv41-sycl-dpct-and-xpu-test/{f3tc-receipts-runs.txt,f3tc-build-identity.txt,f3tc-keylines-base.txt,f4-fidelity.md,task-5/7/9/15-dsv41-sycl-dpct-and-xpu-test.log}
- /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/{ds4.txt,tensors.txt} and the five GGUF files beside it (sizes, quant census)
- sycl/CMakeLists.txt:273-292 (ds4_parity/ds4_blas_parity target definitions)
- .omo/plans/dsv41-sycl-dpct-and-xpu-test.md (todo 11 row, IS-8, commit strategy)

## Verdict

RC2, RC3 and RC4 pass exactly as specified, and the three named flag classes come back clean: every
quoted throughput has a completed prompt behind it, the report's hardware label is the correct 2x
Intel Arc Pro B60 24 GB (the Arc A770 mislabel is in the two-card QA log and SUMMARY only), and the
CPU-lane figures 109/1680 and 157/1920 name the right runs with the gate run's 105/1680 unquoted.
RC1 fails on the umbrella requirement that the report's content be backed by evidence: the model is
described as "Q4_K quantised, about 115 GB" when the measured pack's own manifest and files show a
IQ1_M/IQ2_XXS/IQ2_XS + Q5_K + IQ4_NL mix with zero Q4_K tensors and 248.83 GB on disk, and the
parity suites are claimed to pass "on Linux, CUDA, HIP and SYCL" when only the Linux SYCL build was
ever run and no CUDA/HIP build of either test exists. Both figures/claims are unbacked by any
evidence log and contradicted by primary evidence, violating the plan's own "pack identity" and "do
not paste any number not produced in this session" requirements.

VERDICT FAIL
