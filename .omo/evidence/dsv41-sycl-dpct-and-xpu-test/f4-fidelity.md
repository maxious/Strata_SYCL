# F4 — ideal-state fidelity, IS-1..IS-8 (dsv41-sycl-dpct-and-xpu-test)

Date: 2026-10-27 (evidence dated 2026-10-26/27). Branch HEAD `28bd2aaa` on `sycl/dsv41-sycl-support`.
Audit scope: the IS rows in `.omo/plans/dsv41-sycl-dpct-and-xpu-test.md`, every log under
`.omo/evidence/dsv41-sycl-dpct-and-xpu-test/`, the gate reviews (`todo2/todo4/todo5/todo6/todo6b/todo7/todo8/todo10`,
`dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md`, `dsv41-sycl-dpct-and-xpu-test-f1-gate-review.md`), and
`.omo/ulw-execute/ledger.jsonl`. No card work, no code changes, no plan edits.

Method rules applied: a lane's prose is not evidence when the log file it names lacks the value (named
`/tmp/dpct_run` artifacts were opened and checked); no throughput is stated without a `prompt:` line in the
log; knob-chosen runs (`--vram-experts 0.25`, `--vram-experts 0`) are kept separate from default-knob runs;
findings from reading code are marked **unconfirmed static analysis**.

## Row verdicts

### IS-1 — partially met
Statement: every dpct advisory block in `sycl/src/dsv41` is either rectified or carries a one-line accepted
note naming the class and the reason it was left.

- Evidence pointer: `task-2-dsv41-sycl-dpct-and-xpu-test.log` (triage commit `ca0f6f94`, one file,
  `sycl/docs/sycl-dsv41-dpct-triage.md`, "1 file changed, 106 insertions(+)"); `todo2-gate-review.md` gate 8
  (`NOTES_REASON_RC=0`); `dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md` ("Every one of its 30 DPCT-class rows
  carries a substantive reason cell"); tree state at `28bd2aaa` (this audit).
- Verbatim figure: `total id count: 30 ids, 776 notes.` with `| RESOLVED | 4 | 25 |`, `| ACTIONABLE | 2 | 112 |`,
  `| ACCEPTED_AS_IS | 15 | 563 |`, `| NOT_APPLICABLE | 9 | 76 |` (task-2 log, MANUAL-QA-CHANNEL); the triage doc
  itself says `This file changes no code. It is the review record.`
- Gap: the accepted reasons are per class in a doc, not one-line notes on the blocks. The raw blocks remain in
  the source (this audit: `grep -rn 'DPCT' sycl/src/dsv41` = 1051 lines at `28bd2aaa`); inline `// audit note:`
  lines exist only at the DPCT1098 / read-path sites from todo 3 (17 lines in `ds4_kernels.dp.cpp`,
  `ds4_gguf.dp.cpp`, `vendor/exl3*.dp.cpp`; zero in `ds4_model.dp.cpp`, where DPCT1093's 107 notes sit). The
  563 ACCEPTED_AS_IS + 76 NOT_APPLICABLE + 25 RESOLVED and the 107 DPCT1093 blocks carry no inline note. Also
  IS-1's reviewer-facing half is assigned to todo 11, which is open.

### IS-2 — met
Statement: every host-mapped payload read goes through the uncached read path, and every queue is created on
the thread that owns its device.

- Evidence pointer: `task-3-dsv41-sycl-dpct-and-xpu-test.log` (22-row read-site AUDIT TABLE; commit
  `b18f6c40`); `sycl/docs/sycl-dsv41-queue-binding.md:95-99` (five `MATCH` rows, commit `2e5c53b9`);
  `task-4-dsv41-sycl-dpct-and-xpu-test.log` ("todo 4 gap close": ordinal fix `39b128ad`; `RUN_G1_RC=0`,
  `RUN_G1B_RC=0`); `todo4-gate-review.md` ("The queue-creation table (all 5 sites MATCH) is unchanged ... and
  still valid").
- Verbatim figure (read side, the failure control that proves the fix matters):
  `SUMMARY trials=10 plain_seen=2 uncached_fenced_seen=10` /
  `CONTROL RESULT: plain_spin_stale=OBSERVED uncached_fenced_spin_stale=NOT_REPRODUCED` (task-3, control 4);
  `Read sites changed by this todo: 14 device read sites`.
- Verbatim figure (binding side): `ds4: CUDA1 layers [0, 39] + head: weights 5.86 GB, experts 14.87 GB (1685 of
  15360 in VRAM), caches 0.00 GB, 601 MiB free` with `prompt: 10 tokens in 166 ms (60.40 tok/s, windows of 8192)`
  (task-4, `--gpus 1` control, 2/2).
- Notes: the plan's GAP-3 attribution of `ds4_kernels.dp.cpp:1983` as a plain load of host USM is corrected in
  task-3 — that site is `DEVICE_RESIDENT` (Combine captured by value, `d.cpu_part` device buffer); the
  correction is code reading backed by cites (**unconfirmed static analysis** for the attribution; the read-site
  fixes and the stale-read control are measured). The two-card half of todo 4's own acceptance is unmet and is
  reported under IS-7 and the todo-4 `[~]` gap; the row statement above is a code property and is evidenced.

### IS-3 — met
Statement: the host->GPU signalling direction is either proven on this card or bounded with a documented gap,
and no run ever spins unbounded.

- Evidence pointer: `task-5-dsv41-sycl-dpct-and-xpu-test.log` (gates 2-4); `todo5-gate-review.md`.
- Verbatim figure (default path, OBSERVED): `ds4: the card's waits for the answer (device 0): 64 waits (the
  token path 64, the window path 0), the most reads 1, the last value read 255, 0 hit the bound`.
- Verbatim figure (loud failure at a tight bound): `RUN_SPIN_RC=1` with
  `ds4: the card never saw an answer (its wait ran to its bound): the run ends here` and
  `SPIN_DEC_RC=0` (no completed decode) at `STRATA_SPIN_MAX=1`.
- The copy-engine hand-back is recorded as the documented gap the row allows: `BOUNDED-GAP` —
  `the most reads 1000000, the last value read 191, 63 hit the bound`, off by default. The previously unbounded
  window-path wait now takes the answer bound; every named spin site is bounded.

### IS-4 — met
Statement: the host CPU lane computes experts the card does not hold and its share is printed as a non-zero
count.

- Evidence pointer: `task-6-dsv41-sycl-dpct-and-xpu-test.log` (todo6c re-test on the merged build `28bd2aaa`,
  default knobs, `RUN_N8_RC=0`); `task-8-dsv41-sycl-dpct-and-xpu-test.log` (G2, default run, `RUN_RC=0`);
  `todo6b-gate-review.md` (APPROVE, `[x]`).
- Verbatim figure (default run, merged build): `decode: 8 tokens in 2565 ms (2.73 tok/s); CPU lane: 132 of 1680
  experts; the cache moved 3 (3 in all)` with `decode per token: layers 36.33 ms (of it: routing waits 0.00, CPU
  lane 11.79, the rest host launches), head wait 329.76, Engram reads 0.04`.
- Verbatim figure (default quality run, task-8 G2): `decode: 9 tokens in 4017 ms (1.99 tok/s); CPU lane: 211 of
  1920 experts; the cache moved 14 (14 in all)`.
- Knob-chosen runs, distinguished: `--vram-experts 0.25` prints `CPU lane: 60 of 1680 experts` at `CPU lane 5.19`
  ms/token (`/tmp/dpct_run/w6b_mix025.log`, checked on disk; quoted in the task-6 log and todo6b review);
  `--vram-experts 0` prints `CPU lane: 1680 of 1680 experts` at `CPU lane 125.03` — the count-path control.
- Prose-vs-log reconciliation: the plan's todo-8 note figure `1481 of 1680` is not one of the four quality runs
  (they print `211/156/158/152 of 1920`); it is the `--n 8` fix run at `/tmp/dpct_run/w8/fix_run.log`
  (`decode: 8 tokens in 8446 ms (0.83 tok/s); CPU lane: 1481 of 1680 experts; the cache moved 27 (27 in all)`),
  verified in that file (todo8-gate-review G4 says the same). Both shapes are explained by one accounting
  ((tokens-1) x 40 layers x 6 picks: 8x40x6=1680, 9x40x6=1920).

### IS-5 — partially met
Statement: the one-card decode of "what is the capital of france" has first greedy id 11111 (` Paris`, type 1),
decoded text containing Paris, no type-3 control token, repeated at least 3 times.

- Evidence pointer: `task-8-dsv41-sycl-dpct-and-xpu-test.log` G3-G6; `todo8-gate-review.md` (APPROVE `[~]`,
  gaps G1/G2).
- Verbatim figure (met substance): emitted ids `671 6102 294 8760 344 2619 51119 42499 1`; decoded answer
  `The capital of France is **Paris**.<｜end▁of▁sentence｜>` (`PARIS_RC=0`); repeats `REPEAT2_RC=0`,
  `REPEAT3_RC=0`, `REPEAT4_RC=0` with byte-identical ids in all four runs; `129276` (`<｜box｜>`) is gone.
- Gap (literal criteria unmet, both mis-derived):
  1. First id is `671`, not 11111: `FIRST_ID_RC=1`. The plan's 11111 expectation traces to the defective run's
     own logits (todo-7 gap text: `437.116119 is the SYCL's own logit for 11111, not a reference value`).
  2. The no-type-3 clause trips on the terminator: `emitted 9 ids, 1 of them type 3 [1]` (`CONTROL_RC=0`) —
     id 1 is the pack's `eos_token_id`, which `token_type.json` classifies as type 3, so a terminating answer
     can never satisfy "no type-3". Excluding EOS, no emitted id is type 3.
- Verdict rationale: the row's substance (a real Paris answer, no vision control token, >=3 repeats) is met;
  the two literal numbers are wrong as specified. The plan must amend them (recommendation row 13 below).

### IS-6 — met
Statement: a tokens-per-second figure is published only for a run whose prompt completed, with its build,
flags, pack, card count, and date.

- Evidence pointer: `task-8-dsv41-sycl-dpct-and-xpu-test.log` G2/G6 and header.
- Verbatim figure: `prompt: 10 tokens in 414 ms (24.13 tok/s, windows of 8192)` and
  `decode: 9 tokens in 4017 ms (1.99 tok/s)` (`RUN_RC=0`, `PROMPT` line present); run 3:
  `prompt: 10 tokens in 281 ms (35.61 tok/s, windows of 8192)`.
- Provenance in the same log: date `date=2026-10-26`; build `commit=a698bb84a057a4bdaccda28d1198921d89cc7d32`
  plus the fix `COMMIT_RC=0 (fed551b7 [Engine] dsv41 SYCL: subtract the K-quant min term in the GEMV)`
  (cherry-picked to the branch as `28bd2aaa`); flags and pack in the literal G2 command (`--gpus 0
  --ids @/tmp/q_france.ids --n 32 --temperature 0 --seed 0 --max-context 512 --verbose`, pack
  `/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4`); card count one card
  (`one device only (grep -c CUDA1 = 0)`, card 1 idle in the xpu snapshot).
- Guardrails recorded: the malformed-ids control printed `prompt: 0 tokens in 0 ms (0.00 tok/s, windows of
  8192)` and is explicitly `never quoted for speed or first id` — that line must never be published. The docs
  half of publication (todo 11) and any two-card figure (todo 9) are still open; every figure published so far
  complies with the row.

### IS-7 — partially met
Statement: the two-card run is reported against a same-day single-card run of the same build, or not reported
at all.

- Evidence pointer: `task-4-dsv41-sycl-dpct-and-xpu-test.log` ("todo 4 gap close" section);
  `todo4-gate-review.md` (REJECT `[~]`, blockers).
- Verbatim figure: `RUN_TWO_RC=130`, `RUN_TWOB_RC=130`, `RUN_TWO2_RC=124`, `prompt:` count 0 in every attempt;
  the wedge is `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` and
  `maya-ds4: DPCT_CHECK_ERROR(d.ev->wait_and_throw()): error` at `ds4_model.dp.cpp:11213`.
- Gap: the guardrail branch holds today — no two-card throughput or scaling number is claimed anywhere in the
  evidence (the wedged runs print no `prompt:` line and nothing is quoted from them). But the delivery branch
  is missing and blocked: todo 9 is open, 3/3 recorded attempts wedge deterministically before a prompt, and
  no "not reported at all" decision is recorded. The same-day baseline exists (todo 8, 2026-10-26), so the row
  can be discharged either by fixing the wedge (recommendation row 12) or by recording the no-report decision.

### IS-8 — partially met
Statement: the default build stays green and lists no dsv41 source, and `ctest -R ds4` passes on the SYCL
build.

- Evidence pointer: `task-10-dsv41-sycl-dpct-and-xpu-test.log`; `dsv41-sycl-dpct-and-xpu-test-todo10-gate-review.md`
  (`VERDICT confirmed`, all five gates PASS); SYCL ctest rows in `task-2/3/5/6/8` logs and `f2_ctest3.txt`.
- Verbatim figure (met halves): `DSV41_OFF_GREP_RC=1 (count=0)`, `OFF_OBJGREP_RC=1 (count=0)`,
  `OFF_DSV41_RC=1 (count=0)`, positive control `DSV41_ON_GREP_RC=0 (count=75)`;
  `100% tests passed, 0 tests failed out of 2` (`ctest -R ds4`, SYCL build, 2/2). The default build compiles:
  `DEFAULT_BUILD_RC=0 (default no-SYCL build, 161/161 targets)`.
- Verbatim figure (gap): `DEFAULT_CTEST_RC=8` — `98% tests passed, 1 tests failed out of 57`, solely test #55
  `isa_guard_check`; reproduced identically on `origin/main` `4b9e05a2` (`CTL_CTEST_RC=8`, `ISA_CTEST_RC=8`,
  `FAILDIFF_RC=0`), verbatim violation
  `VIOLATION strata::kernels::cpu::iq512x_rows(int, unsigned char const*, unsigned long, int, void const* const*, int, float* const*, int, int) (777 wide instructions, first vmovq): wide-ISA code in an unlisted public function [iq_avx512.cpp.o]`.
  The failure and its pre-existing status are measured; the cause attribution (allow-list gap:
  `ALLOW_X_RC=1 (count=0)` for `iq512x_rows` vs positive control `ALLOW_Y_RC=0 (count=1)` for `iq512_rows`) is
  **unconfirmed static analysis** of `tools/check_isa_guard.py`.
- Gap: "default stays green" is unmet at the ctest level (1/57, pre-existing, not port-caused); the plan's own
  `CTEST_RC=0` criterion cannot pass until the allow-list is fixed upstream (recommendation row 17).

## Known gaps on `[~]` rows (F4 obligation)

- **todo 4 `[~]`** — ordinal mapping fixed and merged (`39b128ad`; `--gpus 1` completes a prompt 2/2), but the
  two-card acceptance is unmet: 3/3 `UR_RESULT_ERROR_DEVICE_LOST` wedges at `ds4_model.dp.cpp:11213`,
  `RUN_TWO_RC=130`, `RUN_TWOB_RC=130`, `RUN_TWO2_RC=124`, no `prompt:` line. Last evidence: task-4 log
  ("todo 4 gap close"), `todo4-gate-review.md`.
- **todo 7 `[~]`** — at its own gate the repeated-token symptom remained (`129276` x8) and the acceptance's OR
  branch was met: the log names the divergence (`row 9 argmax 129276 @ 989.406982, 11111 @ 437.116119 rank
  86,498`) and localizes the fault upstream (hidden RMS `57.59 -> 3,360.04 (layer 1) -> 299,127.56 (layer 39)`).
  The named CPU oracle is structurally un-runnable (`ref.py:33 import exl3_torch`; `exl3_torch.py` dropped in
  `a0dbad80`) — a limitation, not a lie; a later torch oracle covered embedding + layer 0 (ledger line 347).
  The symptom itself is since removed by `28bd2aaa` (task-8).
- **todo 8 `[~]`** — FIRST_ID/CONTROL criteria mis-derived (IS-5 gap above). Also recorded in its log:
  `NEG_IDS_RC=0` (the `hello world` ids file does not fail — the shared parser stops silently at the first
  non-numeric token; the substitute `NEG2_RC=1` control names `token 999999999 out of range`), the parity
  harness does not cover the K-quant GEMV (it passed before and after the fix), and `xpu-smi` `memory.used`
  does not reflect Level Zero allocations (the one-card load receipt is the engine's per-device line).
- **todo 10 `[~]`** — `CTEST_RC=8` on the pre-existing `isa_guard_check` failure (IS-8 gap above).
- **F2 `[~]`** — REJECT: the AOT variant (`-DSTRATA_SYCL_AOT=bmg-g21`) was never compiled, and the parity rows
  were never captured or diffed byte-identically against todo 1's baseline (`f2_ctest3.txt` holds only the
  Passed summary). JIT/notes/scope sub-criteria pass. (An F2-gap lane was running at audit time.)

## Recommendations — shortfalls that must become new `- [ ] N.` rows (orchestrator adds them; F4 does not edit the plan)

1. Two-card wedge / IS-7 delivery (also the todo-4 `[~]` gap):

   `- [ ] 12. Chase the two-card UR_RESULT_ERROR_DEVICE_LOST at ds4_model.dp.cpp:11213 and land the two-card run against todo 8's same-day single-card baseline`

   Acceptance: `--gpus 0,1` prints a `prompt:` line with un-piped `RUN_RC=0`, both cards busy, numbers beside
   todo 8's on the same build (`SAME_BUILD_RC=0`); if the wedge cannot be closed on this driver, record the
   no-report decision with the receipts and close IS-7's second branch.

2. Mis-derived criteria (IS-5 gap, todo-8 G1/G2, todo-8 G3, todo-6 control mislabel, F2 ctest path):

   `- [ ] 13. Amend the measured-record criteria the evidence proved wrong: IS-5 first id is 671 ("The") and its no-type-3 clause exempts the pack's EOS id 1; todo 6's failure control is --vram-experts 0 (not --experts 0); the one-card load receipt is the engine's per-device line, not xpu-smi memory.used; the SYCL ctest gate is ctest --test-dir <build>/sycl -R ds4`

3. Parity coverage for the bug class that shipped (todo-8 G5):

   `- [ ] 14. Add a K-quant GEMV parity row (iq::mv -> mv_kernel -> sub_dot, Q2_K/Q4_K/Q5_K) to ds4_parity`

   Acceptance: the row fails on the pre-`28bd2aaa` sign and passes at `28bd2aaa`; `ctest -R ds4` selects it.

4. Duplicate launcher symbols / exe-vs-harness coverage (task-8 finding; nm and byte-identical machine code
   are measured, the coverage consequence is **unconfirmed static analysis**):

   `- [ ] 15. Give the exe-linked kernel copies one home: remove the duplicate hc_pre_norm / hc_mix_scale / sparse_attn definitions, or cover the ds4_kernels.dp.cpp copies in the parity harness`

5. IS-1's literal per-block note requirement:

   `- [ ] 16. Add the one-line accepted note (class + reason) to every remaining dpct advisory block in sycl/src/dsv41, or amend IS-1 to accept sycl/docs/sycl-dsv41-dpct-triage.md as the per-class accepted record`

6. IS-8's default-green criterion (pre-existing upstream, outside this plan's `sycl/`-only scope):

   `- [ ] 17. Make the default ctest gate green: add iq512x_rows to WIDE_PUBLIC in tools/check_isa_guard.py (separate upstream change)`

   Acceptance: `ctest` on a default build exits 0 with 57/57, or the plan records the pre-existing failure as
   an accepted deviation with the `origin/main` `4b9e05a2` receipts attached.

7. F2's two unmet sub-criteria (already being chased by lane `st_01a17892` at audit time; collapse to receipts
   if that lane lands):

   `- [ ] 18. Close F2's two gaps: compile the AOT variant (-DSTRATA_SYCL_AOT=bmg-g21) and diff the ds4 parity rows byte-identically against todo 1's baseline rows`

8. Todo-8's failed failure-control (the harness cannot fail on non-numeric ids):

   `- [ ] 19. Make --ids @file fail loudly on a non-numeric token (parse error naming the token; src/dsv41/ds4_main.cpp:71, a scope exception to the sycl/-only rule)`

   Acceptance: the plan's literal control (ids file `hello world`) exits non-zero with `NEG_IDS_RC` non-zero and
   prints the offending token.

## Scope receipts

No plan edit, no checkbox added, no code change, no card work: this audit only read files and wrote this
report. All measured numbers above are quoted from the named logs (or from a named `/tmp/dpct_run` artifact
that was opened and checked); every code-reading conclusion is marked unconfirmed static analysis.
