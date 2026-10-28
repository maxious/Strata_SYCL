# Manual QA matrix — F3 (dsv41-sycl-dpct-and-xpu-test)

Run: st_01a1789d (omo-native-qa-executor), 2026-10-26/27 (+11:00), host linux x64 (AMD Ryzen 7 5700X3D, 2x Intel
Arc Pro B60 `8086:e211`). Integration tree `/home/maxious/Strata`, branch `sycl/dsv41-sycl-support`, HEAD
`28bd2aaac8300a2d3952284c1559c4d3bf35529d`. Pack `/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4`.
Attempt directory: no ulw-loop plan exists (`agentToolkit.status()` -> `ULW_LOOP_PLAN_MISSING`), so artifacts live
in the caller's evidence directory (this directory). Every `*_RC` in `f3qa-receipts.txt` is the un-piped exit code
of the `CMD` line printed directly above it.

**Scope of this run:** todo 8's one-card invocation and todo 10's default-build regression commands.
**Deferred (plainly): the todo 9 two-card (`--gpus 0,1`) portion of F3 was NOT run** — it is blocked by a separate
defect being fixed elsewhere; no `--gpus 0,1` launch exists in this run's receipts.

**Build identity:** HEAD `28bd2aaa`, flags `STRATA_ENABLE_SYCL=ON STRATA_SYCL_DSV41=ON STRATA_SYCL_DSV41_EXE=ON`,
`CMAKE_CXX_COMPILER=icpx`, JIT (no `STRATA_SYCL_AOT`), binary `/tmp/f3_sycl_build/sycl/strata-dsv41` (real path;
`SYCL_EXE_RC=0`, `SIZE=37896656`), date 2026-10-27 (+11:00), card `--gpus 0` (DRM card0, 0000:0d:00.0).

**Deviations from the literal step commands (each recorded, none silent):**
1. Step 2's configure leaves `STRATA_SYCL_DSV41_EXE` OFF (`sycl/CMakeLists.txt:297`): the literal build produces
   no `strata-dsv41` (`EXE_TARGET_PRE_GREP_RC=1`, count 0). Reconfigured with `-DSTRATA_SYCL_DSV41_EXE=ON`
   (`CFG_EXE_RC=0`, post-grep count 84) per the caller's correction; the binary's real path
   `<build>/sycl/strata-dsv41` is used in every invocation (w6c's build-root entry was a 17-byte symlink).
2. Step 2's bare `cmake --build` fails (`BUILD_RC=1`, `BUILD_EXE_RC=1`) at `sycl/src/prefill/gemm.dp.cpp`
   (`#error "SYCLomatic runtime requires oneMath/oneMKL support"`): oneMKL headers are only on the include path
   under `setvars.sh`. The setvars-wrapped build (the pattern every prior lane used) succeeds: `BUILD2_RC=0`.
3. The step-3 invocation spells the pack `DeepSeek-V4-1-...`; that path does not exist. The instruction's exact
   pack path `DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4` was used (existence-checked).
4. The plan's todo-8 literal uses `--n 32 --temperature 0 --seed 0`; this run executes the caller's end-state
   literal `--n 8 --max-context 512 --verbose` (engine defaults to temperature 0 / greedy). `--n 8` truncates the
   decode before the EOS token (see S-8-4).

**Answer (decoded from the emitted ids with the pack tokenizer):** `The capital of France is **Paris**.`

**Verbatim lines (run 1; all four runs in `f3qa-keylines.txt`):**
```
prompt: 10 tokens in 470 ms (21.27 tok/s, windows of 8192)
decode: 8 tokens in 3139 ms (2.23 tok/s); CPU lane: 112 of 1680 experts; the cache moved 5 (5 in all)
```
Runs 2–4: prompt `444 ms (22.50 tok/s)` / `388 ms (25.76 tok/s)` / `439 ms (22.76 tok/s)`; decode
`2876 ms (2.43 tok/s); CPU lane: 108 of 1680` / `2817 ms (2.48 tok/s); CPU lane: 110 of 1680` /
`2990 ms (2.34 tok/s); CPU lane: 114 of 1680` (full verbatim lines in `f3qa-keylines.txt`).

**Emitted ids (all four runs, byte-identical):** `671 6102 294 8760 344 2619 51119 42499` — first id `671` ("The"),
zero type-3 ids per `<pack>/tokenizer/token_type.json` (the decode tool was golden-checked byte-identical against
the prior lane's known `w8/ids1.txt` + `decoded1.txt` before use: `f3qa-decode-goldencheck.txt`).

## surfaceEvidence

| id | criterion reference | surface | exact invocation | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| S-ENC | todo 8 `ENC_RC=0` (ids via pack tokenizer + chat template) | CLI (pack tokenizer in `/tmp/tokvenv`) | `<INV-ENC>` then un-piped `diff` of the encoded ids vs `/tmp/q_france.ids` | **PASS** (`ENC_RC=0`, `ENC_DIFF_RC=0`; `IDS=128803,3085,344,270,6102,294,8760,33,128804,128822`) | A-ENC, A-RECEIPTS |
| S-BUILD | step 2 SYCL build gates (`CFG_RC`, `BUILD_RC`, `SYCL_EXE_RC`) | cmake/Ninja build in `/tmp/f3_sycl_build` | `<INV-CFG>`, `<INV-BUILD>`, `<INV-CFG-EXE>`, `<INV-BUILD2>` (verbatim below) | **PASS** with recorded deviations 1–2 (`CFG_RC=0`; literal `BUILD_RC=1` bare-build gap; `BUILD2_RC=0`; `SYCL_EXE_RC=0`, 37896656 bytes at `sycl/strata-dsv41`) | A-CFG, A-BUILD2, A-EXE, A-RECEIPTS |
| S-8-1 | todo 8 happy: one-card run completes with a `prompt:` line (IS-5/IS-6 carrier) | card run: `strata-dsv41 --gpus 0` on DRM card0 | `<INV-RUN1>` | **PASS** (`RUN_RC=0`, `PROMPT_LINE_RUN1_RC=0`) | A-RUN1, A-KEY, A-RECEIPTS |
| S-8-2 | todo 8/IS-5: decoded text contains `Paris` (`PARIS_RC=0`) | decoded run-1 ids via pack tokenizer | `f3_decode.py --log /tmp/dpct_run/f3_n8.log`, `grep -ci paris` | **PASS** (count 1 per run ×4; answer `The capital of France is **Paris**.`) | A-DEC1, A-ANSWERS, A-RECEIPTS |
| S-8-3 | todo 8 literal `FIRST_ID=11111` (" Paris") | first emitted id, run 1–4 | `<INV-RUN1>` + decode extraction | **FAIL (literal criterion)** — observed `671` ("The") in all 4 runs (`FIRST_ID_RC=1`); the plan's own todo-8 `[~]` gap line records 11111 as mis-derived from the defective run's logits | A-DEC1, A-RECEIPTS |
| S-8-4 | todo 8 literal `CONTROL`: no emitted id of token type 3 | emitted ids vs `token_type.json` | `f3_decode.py --types` per run | **PASS for these runs** (`type3_ids: none`, `CONTROL_RC=0`); note: `--n 8` truncates before the EOS id 1, which `token_type.json` classifies type 3 — at `--n ≥ 9` the EOS is emitted and the literal criterion trips on it (plan `[~]` gap) | A-DEC1, A-RECEIPTS |
| S-8-5 | todo 8/IS-5: ≥3 repeats all answering, byte-identical | 4 card runs of the same invocation | `<INV-RUN1>` + `<INV-RUN2/3/4>` (log path only) | **PASS** (`RUN2/3/4_RC=0`; `REPEAT2/3/4_RC=0`; `IDS_REPEAT2/3/4_RC=0`) | A-RUN1..A-RUN4, A-RECEIPTS |
| S-8-6 | todo 8/IS-6: throughput quoted from a completed prompt with build/pack/card/date identity | run logs + identity block | `<INV-RUN1>`; identity `<INV-GIT>` | **PASS** (prompt/decode tok/s lines verbatim above; identity in A-IDENTITY and A-RECEIPTS) | A-KEY, A-IDENTITY, A-RECEIPTS |
| S-8-7 | todo 8 (IS-4 substance): CPU-lane expert count non-zero | run logs | `grep -n 'CPU lane:'` per run | **PASS** (112/108/110/114 of 1680, `CPULANE_RUNn_RC=0`) | A-KEY, A-RECEIPTS |
| S-8-8 | todo 8 “xpu-smi showing one card loaded” + step-5 `XPU_ONE_RC=0` | xpu-smi during/after + engine device lines | `<INV-XPU-ONE>` (one device line + figure), `<INV-XPU-LOOP>` (3 s trajectory across run 4) | **PASS** (`XPU_ONE_RC=0`, one device line, figures 254.578125 MiB idle / 22,649–22,854 MiB loaded card0; card1 ~505 MiB; `DEV1_RUNn_RC=1` ×4 = zero `ds4: CUDA1` lines) | A-SMILOOP, A-SMIREC, A-KEY |
| S-8-9 | todo 8/9: absence of a GPU reset (`RESET_RC=1` on zero matches) | run logs | reset-pattern `grep -icE` per run (pattern in A-RECEIPTS) | **PASS** (`RESET_RUN1..4_RC=1`, counts 0) | A-RECEIPTS, A-RUN1..A-RUN4 |
| S-10-1 | todo 10: default build (no SYCL flags) | cmake/Ninja in `/tmp/f3_default_build` | `<INV-DEF-CFG>`, `<INV-DEF-BUILD>` | **PASS** (`DEF_CONFIGURE_RC=0`, `DEF_BUILD_RC=0`) | A-DEF, A-RECEIPTS |
| S-10-2 | todo 10 literal acceptance `CTEST_RC=0` | `ctest` on the default build | `<INV-CTEST>` | **FAIL (literal criterion, known `[~]` gap)** — `CTEST_RC=8`, 56 passed / 1 failed of 57 | A-CTEST, A-CTS |
| S-10-3 | todo 10 regression substance: the default path's failures are pre-existing (origin/main control) | failing-test-name comparison | `grep -F '***Failed'` + name extract + `diff` vs `/tmp/dpct_w10_ctl_ctestout.log` | **PASS** — only `Test #55: isa_guard_check`, name-set identical to the origin/main `4b9e05a2` control (`FAILDIFF2_RC=0`) | A-FAILDIFF, A-FN, A-CTLN |
| S-10-4 | todo 10: OFF configure lists no dsv41 source (`DSV41_OFF_GREP_RC=1`) | default build compile list | `grep -c 'ds4_' /tmp/f3_default_build/build.ninja` | **PASS** (count 0) | A-RECEIPTS |
| S-10-5 | todo 10: ON configure lists dsv41 sources (`DSV41_ON_GREP_RC=0`, count > 0) | SYCL build compile list | `grep -c 'ds4_' /tmp/f3_sycl_build/build.ninja` | **PASS** (count 71) | A-RECEIPTS |
| S-9-1 | todo 9 / IS-7: two-card run vs same-day single-card baseline | card run `--gpus 0,1` | not launched | **DEFERRED** — blocked by a separate defect being fixed elsewhere (caller instruction: do not run `--gpus 0,1`); this portion of F3 is deferred, missing prerequisite = that defect's fix | A-RECEIPTS (no two-card launch recorded) |

## adversarialCases

| id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| ADV-8-1 | todo 8 failure scenario (`NEG_IDS_RC` non-zero + parse error naming the token) | input validation: genuinely malformed ids file (`hello world`) | non-zero exit, parse error naming the offending token | **FAIL (criterion)** — `NEG_IDS_RC=0`, `NEG_ERR_GREP_RC=1` (no error printed; the shared parser `src/dsv41/ds4_main.cpp:71` silently stops at the first non-numeric token and decodes an empty context). Reproduces the plan's `[~]` gap `NEG_IDS`; the defect is outside this lane's `sycl/` scope | A-NEG, A-RECEIPTS |
| ADV-8-2 | companion control to ADV-8-1 (harness can fail) | negative control: out-of-range id (`128803 999999999`) | non-zero exit naming the offending token | **PASS** — `NEG2_RC=1`, `maya-ds4: token 999999999 out of range` (`NEG2_ERR_GREP_RC=0`) | A-NEG2, A-RECEIPTS |
| ADV-10-1 | todo 10 failure scenario (`NEG_GREP_RC=0` on a tree listing dsv41 sources) | detection control: the OFF/ON grep must discriminate | the grep on the ON build returns a non-zero count, exit 0 | **PASS** — `DSV41_ON_GREP_RC=0`, count 71 (vs OFF count 0) | A-RECEIPTS |
| ADV-9-1 | todo 9 failure scenario (`--gpus 0,1` + `STRATA_SPIN_MAX=1` must fail loudly) | two-card hand-off failure | loud failure, no silent hand-off | **DEFERRED** with S-9-1 (two-card portion blocked by a separate defect under fix; not run) | A-RECEIPTS |

## Verbatim invocations

```
INV-ENC:   /tmp/tokvenv/bin/python /tmp/enc_q_france.py enc > /tmp/dpct_run/f3_enc_check.txt 2>&1; echo ENC_RC=$?
INV-CFG:   bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S /home/maxious/Strata -B /tmp/f3_sycl_build -G Ninja -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_DSV41=ON -DCMAKE_CXX_COMPILER=icpx > /tmp/dpct_run/f3_configure.txt 2>&1'; echo CFG_RC=$?
INV-BUILD: cmake --build /tmp/f3_sycl_build > /tmp/dpct_run/f3_build.txt 2>&1; echo BUILD_RC=$?
INV-CFG-EXE: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake -S /home/maxious/Strata -B /tmp/f3_sycl_build -G Ninja -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON -DCMAKE_CXX_COMPILER=icpx > /tmp/dpct_run/f3_configure_exe.txt 2>&1'; echo CFG_EXE_RC=$?
INV-BUILD2: timeout 1200 bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; cmake --build /tmp/f3_sycl_build > /tmp/dpct_run/f3_build2.txt 2>&1'; echo BUILD2_RC=$?
INV-GIT:   git -C /home/maxious/Strata rev-parse HEAD
INV-RUN1:  bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/f3_sycl_build/sycl/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 --gpus 0 --ids @/tmp/q_france.ids --n 8 --max-context 512 --verbose > /tmp/dpct_run/f3_n8.log 2>&1'; echo RUN_RC=$?
INV-RUN2:  (INV-RUN1 with output > /tmp/dpct_run/f3_n8_r2.log; RUN2_RC=)
INV-RUN3:  (INV-RUN1 with output > /tmp/dpct_run/f3_n8_r3.log; RUN3_RC=)
INV-RUN4:  (INV-RUN1 with output > /tmp/dpct_run/f3_n8_r4.log; RUN4_RC=)
INV-NEG:   (INV-RUN1 with --ids @/tmp/q_garbage.ids and output > /tmp/dpct_run/f3_neg.log; NEG_IDS_RC=)
INV-NEG2:  (INV-RUN1 with --ids @/tmp/q_badid.ids and output > /tmp/dpct_run/f3_neg2.log; NEG2_RC=)
INV-XPU-ONE: xpu-smi --query-gpu=memory.used --id=0 > /tmp/dpct_run/f3_smi_one_during.txt 2>&1; echo XPU_ONE_RC=$?
INV-XPU-LOOP: xpu-smi --query-gpu=memory.used --loop=3 --count=40 -f /tmp/dpct_run/f3_smi_during_loop.txt
INV-DEF-CFG: cmake -S /home/maxious/Strata -B /tmp/f3_default_build -G Ninja -DSTRATA_BUILD_TESTS=ON > /tmp/dpct_run/f3_def_configure.txt 2>&1; echo DEF_CONFIGURE_RC=$?
INV-DEF-BUILD: cmake --build /tmp/f3_default_build > /tmp/dpct_run/f3_def_build.txt 2>&1; echo DEF_BUILD_RC=$?
INV-CTEST: ctest --test-dir /tmp/f3_default_build --output-on-failure > /tmp/dpct_run/f3_ctest.txt 2>&1; echo CTEST_RC=$?
```
Every card launch was preceded by `pgrep -c strata-dsv41 > <file>` with the count read back from the file and a
launch only on `0` (readbacks 0 at all five launches; one mid-run snapshot read back 1, proving the during-state).

## artifactRefs

All paths are in this evidence directory (`/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/`);
every artifact is non-empty per `A-SIZES` (node:fs `statSync`, `EMPTY_COUNT=0`).

| id | kind | description | path |
| --- | --- | --- | --- |
| A-RECEIPTS | receipt log | consolidated CMD/RC receipts, all gates (raw interleaved capture: A-RAW) | `f3qa-receipts.txt` |
| A-RAW | receipt log | raw session capture (`/tmp/dpct_run/f3_receipts.txt`) | `f3qa-receipts-raw.txt` |
| A-RUN1 | engine log | end-state run 1 (`/tmp/dpct_run/f3_n8.log`) | `f3qa-run1.log` |
| A-RUN2 | engine log | repeat run 2 (`/tmp/dpct_run/f3_n8_r2.log`) | `f3qa-run2.log` |
| A-RUN3 | engine log | repeat run 3 (`/tmp/dpct_run/f3_n8_r3.log`) | `f3qa-run3.log` |
| A-RUN4 | engine log | repeat run 4, during-window xpu capture (`/tmp/dpct_run/f3_n8_r4.log`) | `f3qa-run4.log` |
| A-DEC1 | decode output | run-1 ids/first-id/type-3/decoded text | `f3qa-decode-RUN1.txt` |
| A-ANSWERS | decode output | run-2..4 decode outputs | `f3qa-decode-RUN2.txt`, `f3qa-decode-RUN3.txt`, `f3qa-decode-RUN4.txt` |
| A-KEY | key lines | verbatim prompt/decode/CPU-lane lines, all four runs | `f3qa-keylines.txt` |
| A-SMILOOP | xpu-smi log | 3 s memory trajectory across run 4 (load/decode/release) | `f3qa-smi-loop-during.txt` |
| A-SMIREC | receipt log | xpu-smi during/after receipts (incl. during-engine single-shots) | `f3qa-smi-receipts.txt` |
| A-NEG | engine log | adversarial ADV-8-1 (`--ids @/tmp/q_garbage.ids`) | `f3qa-neg-garbage-ids.log` |
| A-NEG2 | engine log | adversarial ADV-8-2 (`--ids @/tmp/q_badid.ids`) | `f3qa-neg2-badid.log` |
| A-ENC | tool output | pack-tokenizer re-encode of the prompt | `f3qa-enc-check.txt` |
| A-EXE | JS fs receipt | node:fs existsSync+size of `sycl/strata-dsv41` | `f3qa-exe-receipt.txt` |
| A-IDENTITY | receipt | build identity (HEAD, flags, JIT) | `f3qa-build-identity.txt` |
| A-CFG | build log | literal configure output | `f3qa-configure-literal.txt` |
| A-BUILD2 | build log | successful setvars-wrapped build output | `f3qa-build2-setvars.txt` |
| A-DEF | build log | default-build configure/build outputs | `f3qa-configure-exe.txt`, `f3qa-build-literal-failed.txt` (deviation evidence), `f3qa-receipts.txt` |
| A-CTEST | ctest log | full ctest output (57 tests) | `f3qa-ctest.txt` |
| A-CTS | ctest extract | summary line + failed-test lines | `f3qa-ctest-summary.txt`, `f3qa-ctest-failed.txt` |
| A-FN | extract | my failing-test names | `f3qa-failed-only.txt` |
| A-CTLN | extract | origin/main control failing-test names | `f3qa-ctl-failed-only.txt` |
| A-FAILDIFF | receipt | failed-only name diff vs control (`FAILDIFF2_EXIT=0`) | `f3qa-faildiff2.txt` |
| A-SIZES | JS fs receipt | node:fs non-empty proof for every artifact | `f3qa-artifact-sizes.txt` |
| A-GOLDEN | tool receipt | decode-tooling golden check vs prior lane output + prior repeat verification | `f3qa-decode-goldencheck.txt` |
| A-SCRIPTS | scripts | exact phase/run/analysis scripts (the `rc=$?` capture) | `f3_*.sh`, `f3_decode.py` in this directory |

## Gate ledger (see A-RECEIPTS for each gate's CMD line)

`CFG_RC=0` · `BUILD_RC=1` (literal bare build, oneMKL/setvars gap) · `EXE_TARGET_PRE_GREP_RC=1` · `CFG_EXE_RC=0` ·
`EXE_TARGET_POST_GREP_RC=0` · `BUILD_EXE_RC=1` (same gap) · `BUILD2_RC=0` · `SYCL_EXE_RC=0` (37896656 bytes) ·
`GIT_HEAD_RC=0` · `ENC_RC=0` · `ENC_DIFF_RC=0` · `PGREP_*_READBACK=0` ×5 launches · `RUN_RC=0` `RUN2_RC=0`
`RUN3_RC=0` `RUN4_RC=0` · `PARIS_RUN1..4_RC=0` · `REPEAT2/3/4_RC=0` · `IDS_REPEAT2/3/4_RC=0` · `FIRST_ID_RC=1`
(observed 671) · `CONTROL_RC=0` (zero type-3 ids) · `RESET_RUN1..4_RC=1` · `DEV1_RUN1..4_RC=1` · `XPU_ONE_RC=0` ·
`XPU_LOOP2_RC=0` · `NEG_IDS_RC=0` · `NEG2_RC=1` · `DEF_CONFIGURE_RC=0` · `DEF_BUILD_RC=0` · `CTEST_RC=8` (56/57) ·
`DSV41_OFF_GREP_RC=1` · `DSV41_ON_GREP_RC=0` · `FAILDIFF2_RC=0` · `EMPTY_COUNT=0`.

## Notes for the plan (evidence findings, no product change made)

1. `FIRST_ID=11111` and "no type-3 id" remain mis-specified literals (plan `[~]` gap): observed first id 671
   ("The") and, at `--n 8`, no type-3 id only because the type-3 EOS (id 1) is never reached.
2. The prior lane's G3 note ("xpu-smi `memory.used` never reflects the run") does not reproduce: a 3 s loop shows
   card 0 at 22,649–22,854 MiB while the engine runs and 254.58 MiB after release. Their single-shot evidently
   missed the ~18 s loaded window (the same trap that swallowed the first attempt in this run).
3. The step-2/step-3 literal commands need `-DSTRATA_SYCL_DSV41_EXE=ON`, a setvars-wrapped build, and the real
   binary path `<build>/sycl/strata-dsv41` (recorded as deviations 1–3 above).
4. `NEG_IDS` (malformed ids file) still fails silently via the shared parser (`src/dsv41/ds4_main.cpp:71`), out of
   `sycl/` scope — a standing plan gap, re-confirmed here with a live run.
