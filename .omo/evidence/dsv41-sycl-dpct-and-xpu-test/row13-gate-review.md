# Row 13 Gate Review — criteria amendment

- **recommendation**: APPROVE
- **reviewed by**: st_01a178c0 (omo-native-gate-reviewer), 2026-10-26
- **artifact reviewed**: `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md`, row 13 (line 228, `[x]`)
- **verdict basis**: each amended criterion is reproduced from the evidence logs and pack artifacts it cites, not from the lane prose. Five-of-five mandated checks pass.

## originalIntent

The plan's own fidelity audit (F4, `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f4-fidelity.md`) proved that three measured-record criteria were mis-derived: IS-5's literal first-id "11111" and its bare no-type-3 clause, todo 6's `--experts 0` failure control, and todo 9's `xpu-smi memory.used` load receipt. Row 13 was opened (and F4 recommended it verbatim) to amend the plan text so the criteria match what the evidence actually measured.

## desiredOutcome

Plan text whose acceptance criteria name the correct first greedy id (671), exempt the pack's EOS id 1 from the no-type-3 clause, name `--vram-experts 0` as todo 6's failure control, and use the engine's own per-device lines (not `xpu-smi memory.used`) as the todo 9 load receipt — each change traceable to an evidence log. No product code change; `.omo/` stays untracked; only row 13's checkbox moves to `[x]`.

## userOutcomeReview

The four measured-record amendments are present at the stated lines and each is confirmed against primary evidence (details below). The amendment is text-only: `git status --porcelain` shows zero tracked modifications and the plan lives under the untracked `.omo/` tree. The ledger shows the only checkbox transition attributable to this amendment is row 13 opening `[ ]` → `[x]`; every other `[x]` was already set by its own lane before row 13. A reader following the amended criteria would reproduce the real answer (671-first, EOS-terminated, Paris), the real CPU-lane count control, and the real two-card load receipt.

## Check 1 — IS-5 (line 33) and todo 8 acceptance (line 186): 671 + EOS id 1 exemption

**PASS.**

- Line 33 (IS-5) now reads: first greedy id is "671 `The` on this pack; the earlier 11111 figure was the defective run's own logit, not a reference value" and "no type-3 control token other than the pack's EOS id 1 (`eos_token_id`, which `token_type.json` classifies as type 3, so a terminating answer can never satisfy a bare no-type-3 clause)".
- Line 186 (todo 8 acceptance) now reads: "first emitted id is the answer's first token (671 `The`; `FIRST_ID_RC=1` is the correct outcome ... since 11111 was the defective run's own logit), no emitted id has token type 3 other than the pack's EOS id 1 (`CONTROL_RC=0` ...)".
- Evidence for 671: `task-8-dsv41-sycl-dpct-and-xpu-test.log` lines 116/142/147/152 — all four runs emit `671 6102 294 8760 344 2619 51119 42499 1`; G3 reports `671`, `FIRST_ID_RC=1`.
- Evidence id 1 is the pack's EOS and classified type 3: `token_type.json` is a length-129280 array with `token_type.json[1] == 3` (verified directly: `id0..3: [3, 3, 3, 1]`, `id1 type: 3`); `ds4.txt:27` reads `eos_token_id 1`. Task-8 G4: `emitted 9 ids, 1 of them type 3 [1]`, `CONTROL_RC=0`.
- Evidence 11111 traces to the defective run's own logit: `task-7-dsv41-sycl-dpct-and-xpu-test.log` lines 186-187, 289-290 — `129276 @ 989.406982`, `11111 @ 437.116119 rank 86,497`, and "437.116119 is the SYCL's own logit for 11111, not a reference value". `task-8` line 237/288 repeats the same provenance.

## Check 2 — todo 6 QA failure control (line 163): `--vram-experts 0`, task-6 log shows the full count

**PASS.**

- Line 163 now reads: "failure - a run with `--vram-experts 0` (all experts on the CPU lane) ... that must still complete and report a full CPU-lane count".
- Evidence in `task-6-dsv41-sycl-dpct-and-xpu-test.log`: line 234-240 (pre-fix run) — `--vram-experts 0` prints `decode: 8 tokens in 921 ms (7.60 tok/s); CPU lane: 1680 of 1680 experts` and `CPU lane 102.13` ms/token; lines 359-362 — `--vram-experts 0` prints `CPU lane: 1680 of 1680 experts` with `CPU lane 125.03` ms/token. Post-fix shape also present: line 489 `CPU lane: 1920 of 1920 experts` (9-token shape, denominator 1920), and the log explains `1680 of 1680` is the 8-token shape.
- The log explicitly records why `--experts 0` is NOT the correct control (lines 229-232, 252-253, 277): it clamps routed top-k to 1, yielding `CPU lane: 0 of 280 experts`, not a full CPU-lane count.

## Check 3 — todo 9 acceptance (line 198): engine per-device lines as load receipt

**PASS.**

- Line 198 now reads: "the engine's own per-device lines show both cards loaded (`XPU_TWO_RC=0` with two GPU device lines; `xpu-smi` `memory.used` is not the load receipt)".
- Evidence in `task-9-dsv41-sycl-dpct-and-xpu-test.log` lines 129-130: the acceptance run prints two engine device lines — `ds4: CUDA0 layers [0, 20] ...` and `ds4: CUDA1 layers [21, 39] + head ...` — with `RUN_RC=0`; `XPU_TWO_RC=0` and `XPU_TWO_COUNT_RC=0` (count 2) follow.

## Check 4 — IS-to-todo map row for IS-5 (line 255): 671 + EOS exemption

**PASS.**

- Line 255 reads: "todo 7 first-id 671 plus no type-3 id beyond the pack's EOS id 1; todo 8 decoded answer containing Paris, 3 repeats". Contains both 671 and the EOS-id-1 exemption.

## Check 5 — Guardrail: text-only amendment, no stray `[x]`

**PASS.**

- `git status --porcelain` shows zero tracked modifications (no `M`/`A`/`D`/`R` lines); only `?? .omo/` and `?? bench/results/...` untracked entries. The plan file is untracked (`git ls-files --error-unmatch .omo/plans/dsv41-sycl-dpct-and-xpu-test.md` fails).
- Ledger transition (`.omo/ulw-execute/ledger.jsonl`): state before row 13 = `[x] 1,2,3,4,5,6,9,12,18,F2; [~] 7,8,10,F4; open 11,13,14,15,16,17,19,F1,F3,F5`; state after row 13 = same except 13 moves from open to `[x]`. No other checkbox gained `[x]` from this amendment; the current plan file matches the post-amendment state exactly.

## Adversarial / skill-perspective check

- **remove-ai-slops**: the amendment's scope is prose (plan criteria), not production code or tests. There is no new test to be tautological or deletion-only, no production extraction/parsing/normalization, and no mirrored implementation — the amendments are corrections of mis-derived literals backed by the measured logs (671, `--vram-experts 0`, per-device lines), not restatements of a lane claim. No slop introduced. The PROSE no-test-pinning rule applies: nothing machine-consumed was added that requires a pin.
- **programming**: N/A for code; the amendment does not add code, tests, or abstractions. Maintenance-burden and false-confidence findings: none.
- No dedicated code-review report exists for row 13 (correct: it is a plan-text amendment with no code diff), so "code review report coverage" is N/A; the F4 report and the task-8/todo8 gate reviews are the provenance for the amendments and agree with them.

## Checked artifact paths

- `.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (lines 33, 163, 186, 198, 228, 255)
- `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-8-dsv41-sycl-dpct-and-xpu-test.log`
- `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-7-dsv41-sycl-dpct-and-xpu-test.log`
- `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-6-dsv41-sycl-dpct-and-xpu-test.log`
- `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-9-dsv41-sycl-dpct-and-xpu-test.log`
- `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f4-fidelity.md`
- `/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/tokenizer/token_type.json` (id 1 == type 3, verified)
- `/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/ds4.txt` (`eos_token_id 1`)
- `.omo/ulw-execute/ledger.jsonl` (checkbox state transition around row 13)
- `git status --porcelain` (no tracked modification)

## exact evidence gaps

None blocking. Two non-blocking consistency notes (not among the five mandated checks, no stated criterion they violate):

- **NOTE** (IS-4 map row): the success-criteria map row for IS-4 (line 254) still reads "plus the `--experts 0` control", while todo 6's QA control (line 163) was amended to `--vram-experts 0`. The IS-4 map row was not part of the row 13 amendment as written and is not one of the checks this gate is asked to enforce.
- **NOTE** (SYCL ctest gate path): row 13's text asserts "the SYCL ctest gate is `ctest --test-dir <build>/sycl -R ds4`", but the Verification strategy line 67 still reads `ctest --test-dir build-sycl-dsv41 -R ds4` (without `/sycl`). The correction is supported by evidence (F2 log uses `ctest --test-dir /tmp/f2_sycl_build/sycl -R ds4`) but was not propagated to line 67. Not one of the five checks, so noted rather than blocked.

## blockers

None.
