# Gate review — todo 4: DPCT1093 queue-binding audit (dsv41-sycl-dpct-and-xpu-test) — third delivery (ordinal-mapping fix)

Date: 2026-10-26
Reviewer: omo-native-gate-reviewer (independent; no edits made; report artifact only)

> This review supersedes the second-delivery review written to this same path (which REJECTed
> because `--gpus` ordinals bound the host CPU: `RUN_TWO_RC=134`, `RUN_ONE_RC=134`, commit
> `2e5c53b9`). This third delivery fixes that ordinal mapping (commit `ba1230a2`). This review
> re-checks the lane's claims against the code, the evidence log, and the on-disk run artifacts.

## recommendation

REJECT

## checkbox mark and gap

todo 4 must be marked `[~]`, not `[x]`.

Gap line: the ordinal-mapping gap is closed (`--gpus 1` now binds physical card 1 and completes
a prompt 2/2; `--gpus 0,1` now shows two GPU device lines and zero
`INVALID_EVENT_WAIT_LIST`), but the todo's own two-card acceptance is still unmet: 3/3 attempts
wedge deterministically with `UR_RESULT_ERROR_DEVICE_LOST` (level_zero error 20) at
`ds4_model.dp.cpp:11213` (`d.ev->wait_and_throw()`) and never print a `prompt:` line —
`RUN_TWO_RC=130`, `RUN_TWOB_RC=130`, `RUN_TWO2_RC=124`, prompt count 0. Last evidence:
`.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-4-dsv41-sycl-dpct-and-xpu-test.log`
("todo 4 gap close" section, lines 316+).

## blockers

1. violatedCriterion: todo 4 acceptance criterion — "a two-card smoke `SYCL_BE=PI_LEVEL_ZERO <build>/strata-dsv41 --pack <pack> --gpus 0,1 --ids 271 --n 8 --max-context 512` exits with `RUN_RC` un-piped, prints a `prompt:` line, and `xpu-smi` shows both cards busy during the run (`XPU_BUSY_RC=0` with two device lines)."
   observation: The two-card smoke does not pass. `RUN_TWO_RC=130`, `RUN_TWOB_RC=130`, `RUN_TWO2_RC=124`; `prompt:` count is 0 in every attempt. After both cards bind correctly (two GPU device lines, both holding layers and experts, zero `INVALID_EVENT_WAIT_LIST`, no CPU/OpenCL marker), the run dies in the cross-device compute path with `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` and `maya-ds4: DPCT_CHECK_ERROR(d.ev->wait_and_throw()): error`, then error-loops without self-terminating (killed with SIGINT for the gate runs, timeout 240 for the control). This is a different defect from the ordinal-mapping bug this todo was re-opened to fix — the old mapping aborted at load (`:10362`, `INVALID_EVENT_WAIT_LIST`) before reaching this path — but the todo's stated acceptance criterion (a two-card run that exits 0 and prints a prompt) is still not satisfied.
   evidencePointer: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-4-dsv41-sycl-dpct-and-xpu-test.log` (`RUN_TWO_RC=130`, `RUN_TWOB_RC=130`, `RUN_TWO2_RC=124` each beside its un-piped command; grep counts `prompt:`=0, `DEVICE_LOST`=574/77); `/tmp/dpct_run/w4ord_two.log` (verbatim `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` and `DPCT_CHECK_ERROR(d.ev->wait_and_throw()): error`); `/tmp/dpct_run/tb_prompt.txt` (=0); `sycl/src/dsv41/ds4_model.dp.cpp:11213` (verified: `CK(DPCT_CHECK_ERROR(d.ev->wait_and_throw()));`).

## originalIntent

(From the plan, todo 4, and its recorded gap) Audit DPCT1093: check each `select_device(d.id)` / `create_queue()` pair in `ds4_model.dp.cpp` against the thread that owns that device (threads mode runs one thread per GPU), fix only a site that is wrong, deliver a table of every queue-creation site, and prove the binding dynamically with a two-card smoke (`--gpus 0,1`) plus the `--gpus 1` single-card control. The gap discovered in the second delivery: `dpct::select_device` numbered index 0 as the SYCL default device and the rest of `get_devices(all)` in runtime order, so slot 1 bound `[opencl:cpu]` — `--gpus 0,1` aborted with `INVALID_EVENT_WAIT_LIST` at `ds4_model.dp.cpp:10362`. The re-opened scope was that ordinal-to-device mapping only.

## desiredOutcome

A reviewer sees the queue-creation table (all sites MATCH), the build and `ctest -R ds4` stay green, and — the point of the acceptance criteria — the binding is proven dynamically: `--gpus 0,1` exits 0 with a `prompt:` line and both cards busy, and `--gpus 1` alone still completes.

## userOutcomeReview

The ordinal-mapping fix itself is real and correct. I read commit `ba1230a2` (2 files: `sycl/include/dpct/device.hpp`, `sycl/docs/sycl-dsv41-queue-binding.md`; `git status --porcelain` clean; scope `sycl/` only). `dev_mgr::dev_mgr()` now fills `_devs` from `detail::ordered_gpu_devices()`, which numbers Level Zero GPU devices by `zeDeviceGet` discovery index and gives no ordinal to a CPU device or to the OpenCL copy of a card that already has a Level Zero interface; a machine with no Level Zero GPU falls back to SYCL GPU runtime order. On this PC that maps ordinal 0 to `zeDeviceGet` index 0 (uuid `868011e2-0000-0000-0d00-000000000000`, `xpu-smi` device 0, PCI `0000:0d:00.0`) and ordinal 1 to index 1 (PCI `0000:11:00.0`). The probe (`/tmp/dpct_run/w4ord_devprobe.cpp`, binary present) shows `sycl::default_selector_v` picks the same uuid the old code gave ordinal 0, so `--gpus 0` keeps its exact device (the "byte-identical" claim holds at the binding level, not the binary level).

The dynamic binding proof now passes where it failed before: `--gpus 1` completes a prompt 2/2 (`RUN_G1_RC=0`, `RUN_G1B_RC=0`; `g1_keylines.txt`/`g1b_keylines.txt` show `ds4: CUDA1 layers [0, 39] + head` + `prompt: 10 tokens`, no `Ryzen`/`opencl`/`not supported` markers), and `--gpus 0,1` shows `ds4: CUDA0 layers [0, 20]` and `ds4: CUDA1 layers [21, 39] + head` with zero `INVALID_EVENT_WAIT_LIST`. The queue-creation table (all 5 sites MATCH) is unchanged from the second delivery and still valid.

What still fails is the todo's own happy-path acceptance: the two-card smoke never reaches a prompt. After both cards bind and load, the cross-device wait at `ds4_model.dp.cpp:11213` throws `UR_RESULT_ERROR_DEVICE_LOST` and the process error-loops. The lane records this honestly and flags it as a separate compute-path defect (consistent with the plan's recorded "xe driver resets the GPU tile after a 5 s job" signature), and does not claim a throughput from an incomplete prompt. But the criterion as written — `RUN_RC=0` un-piped, a `prompt:` line, both cards busy — is not met, and that is a stated acceptance criterion, not a stylistic or out-of-scope note. The todo therefore stays `[~]`, with the gap text updated to the DEVICE_LOST wedge.

## Checked artifact paths

- /home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md (todo 4 block and gap annotation)
- /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-4-dsv41-sycl-dpct-and-xpu-test.log (full; "todo 4 gap close" section)
- /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test-todo4-gate-review.md (second-delivery review being superseded)
- /home/maxious/Strata-wt/dsv41-dpct-w4ord (git: HEAD=ba1230a2, parent a698bb84, status --porcelain clean)
- git show ba1230a2: sycl/include/dpct/device.hpp (+93/-16), sycl/docs/sycl-dsv41-queue-binding.md (+27)
- sycl/include/dpct/device.hpp (read full `detail::ordered_gpu_devices()` and `dev_mgr::dev_mgr()`)
- sycl/src/dsv41/ds4_model.dp.cpp:11213 (verified `d.ev->wait_and_throw()` context)
- sycl/include/dpct/device.hpp `_cpu_device` / `cpu_device()` / `get_cpu_device_id()` (no callers in sycl/src — no regression from dropping the CPU device)
- /tmp/dpct_run/w4ord_two.log (verbatim DEVICE_LOST lines, two GPU device lines, no prompt)
- /tmp/dpct_run/g1_keylines.txt, /tmp/dpct_run/g1b_keylines.txt (--gpus 1 → CUDA1 + completed prompt, 2/2)
- /tmp/dpct_run/w4ord_devprobe.cpp and binary (topology probe source; default_selector vs zeDeviceGet order)
- /tmp/dpct_run/w4ord_smi_discovery.txt (xpu-smi device 0/1 PCI BDFs)
- /tmp/dpct_run/w4ord_diff.txt (=2 sycl files), /tmp/dpct_run/w4ord_scope.txt (=0 non-sycl)
- /tmp/dpct_run/tb_prompt.txt (=0), /tmp/dpct_run/tb_lost.txt, t_lost.txt (DEVICE_LOST counts)

## Evidence gaps (exact)

- The probe's stdout (the run of `/tmp/dpct_run/w4ord_devprobe`) is not persisted as a separate file — the topology results are documented inline in the evidence log only. This is not a blocker: the "default_selector picks discovery index 0" and "--gpus 0 byte-identical" claims are independently corroborated by the engine-run artifacts (`g1`/`g1b` bind `--gpus 1` → CUDA1, and the two-card run shows CUDA0 + CUDA1 with no CPU marker).
- The new lane records no `xpu-smi` busy sample for its two-card run (the `XPU_BUSY_RC=0 with two device lines` half of the acceptance). This is moot given the harder failure: `RUN_TWO_RC=130` and `prompt:`=0 already fail the criterion.
- No other gap: every gate RC in the lane's DONECLAIM was reproduced from the log or the on-disk artifacts.

## Skill-perspective check (remove-ai-slops + programming)

Both skills were unavailable in this session; their criteria were applied directly.

- No test code was added or changed (the change is one header + one docs file), so none of the overfit/slop test classes apply: no deletion-only tests, no tests that merely verify a requested removal, no tautological or implementation-mirroring tests, no excessive tests. The gates are engine-run evidence, consistent with the plan's "todos 3/4/5/6 are gated by an engine smoke, not ctest" decision.
- `detail::ordered_gpu_devices()` is a necessary extraction, not slop: the dev_mgr constructor previously did "default device + everything else", and the fix needs a stable GPU-only ordering; a free function is the smallest change. No parsing or normalization was added for its own sake.
- Maintenance burden (note, not blocker): the Level Zero discovery walk (`zeInit`/`zeDriverGet`/`zeDeviceGet`) duplicates the sysman-style enumeration the plan already mirrors in `ds4_mem.cpp` with attribution. It is ~40 lines, scoped, and self-contained; acceptable, but a future consolidation point.
- Edge case (note, not blocker): on a machine with a mix of Level Zero and OpenCL-only GPUs, the OpenCL-only cards are dropped from `_devs` (the doc states the intent is to drop the OpenCL *copy* of a card that already has a Level Zero interface, which is correct here; a genuinely separate OpenCL-only card would become unreachable). This box has two cards each exposing both interfaces, so it does not affect this goal.
- No scope drift: the commit touches `sycl/` only, changes no queue-creation/`--tp`/device-grouping code, and the docs file records the fix with measured facts rather than prose padding.

## Verdict

The re-opened gap (wrong `--gpus` ordinal → device mapping) is FIXED and the fix is correct and
verifiable. The todo's stated two-card acceptance is still not met — now by a deterministic
`UR_RESULT_ERROR_DEVICE_LOST` wedge at `ds4_model.dp.cpp:11213`, not by the ordinal mapping.
todo 4 must land `[~]` with the DEVICE_LOST wedge named as the gap; the artifact is REJECT.
