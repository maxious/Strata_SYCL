# F5 cleanup audit - dsv41-sycl-dpct-and-xpu-test

Auditor: independent F5 child lane `st_01a17948` (deep-low). Filesystem and process state only:
nothing was deleted, no product file was edited, no monitors were used. The only files this audit
writes are this report and its probe helper `f5_exists.cjs` beside it.

Receipt convention: every `*_RC=` line is the UN-PIPED exit code of the command printed directly
below it, captured as `$?` immediately after that command ran (redirection only, never a pipe).
Each probe's expected value is stated with its check. The node:fs existence probes run
`require("node:fs").existsSync` (the `node` binary is not installed on this host, so the node:fs API
runs under bun 1.4.3 via `f5_exists.cjs`, whose source is beside this file); that probe exits 0 when
`existsSync` is false and 1 when the path still exists, so `EXISTS_*_RC=0` is the expected "absent"
receipt, which is the same state the plan's F5 idiom records as an `ls` exit 2.

## Audit header

AUDIT_DATE_RC=0
date -u +%Y-%m-%dT%H:%M:%SZ
2026-10-26T17:49:59Z

GIT_BRANCH_RC=0
git -C /home/maxious/Strata rev-parse --abbrev-ref HEAD
sycl/dsv41-sycl-support

GIT_HEAD_RC=0
git -C /home/maxious/Strata rev-parse HEAD
d337362b8a9fa09402fd41e2636cbd7beae66161

## Check 1 - lane worktrees (expect: worktree list shows only /home/maxious/Strata; all seven lane paths absent, EXISTS_*_RC=0)

GIT_WORKTREE_LIST_RC=0
git -C /home/maxious/Strata worktree list
/home/maxious/Strata d337362b [sycl/dsv41-sycl-support]

EXISTS_WT_DPCT_NOTES_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /home/maxious/Strata-wt/dpct-notes
existsSync(/home/maxious/Strata-wt/dpct-notes) === false

EXISTS_WT_DSV41_W6C_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /home/maxious/Strata-wt/dsv41-dpct-w6c
existsSync(/home/maxious/Strata-wt/dsv41-dpct-w6c) === false

EXISTS_WT_DSV41_W9_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /home/maxious/Strata-wt/dsv41-dpct-w9
existsSync(/home/maxious/Strata-wt/dsv41-dpct-w9) === false

EXISTS_WT_ROW14_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /home/maxious/Strata-wt/row14
existsSync(/home/maxious/Strata-wt/row14) === false

EXISTS_WT_ROW15_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /home/maxious/Strata-wt/row15
existsSync(/home/maxious/Strata-wt/row15) === false

EXISTS_WT_ROW11_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /home/maxious/Strata-wt/row11
existsSync(/home/maxious/Strata-wt/row11) === false

EXISTS_WT_F1_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /tmp/f1_wt
existsSync(/tmp/f1_wt) === false

## Check 2 - scratch build dirs (expect: all eight absent, EXISTS_*_RC=0; pgrep process probes run only for a dir that still exists)

EXISTS_R15V_BUILD_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /tmp/r15v_build
existsSync(/tmp/r15v_build) === false

EXISTS_R16V5_BUILD_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /tmp/r16v5_build
existsSync(/tmp/r16v5_build) === false

EXISTS_R16V6_BUILD_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /tmp/r16v6_build
existsSync(/tmp/r16v6_build) === false

EXISTS_F1_DEF_BUILD_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /tmp/f1_def_build
existsSync(/tmp/f1_def_build) === false

EXISTS_F1_SYCL_BUILD_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /tmp/f1_sycl_build
existsSync(/tmp/f1_sycl_build) === false

EXISTS_F1_DEF_TESTS_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /tmp/f1_def_tests
existsSync(/tmp/f1_def_tests) === false

EXISTS_R14GATE_BUILD_SHM_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /dev/shm/r14gate_build
existsSync(/dev/shm/r14gate_build) === false

EXISTS_F3TC_BUILD_RC=0
bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs /tmp/f3tc_build
existsSync(/tmp/f3tc_build) === false

## Check 3 - live card processes (expect: pgrep -c prints 0 for both, PGREP_COUNT_*_RC=1)

PGREP_COUNT_STRATA_DSV41_RC=1
pgrep -c strata-dsv41
0

PGREP_COUNT_STRATA_RC=1
pgrep -c strata
0

## Check 4 - live watchers (expect: only pgrep own wrapper shell in the output; no real tail -n 0 -F watcher)

PGREP_TAIL_WATCHERS_RC=0
pgrep -af "tail -n 0 -F"
3429168 /bin/bash -c  F=/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/cleanup-audit.md echo '## Check 4 - live watchers (expect: only pgrep own wrapper shell in the output; no real tail -n 0 -F watcher)' >> "$F" echo >> "$F" out="$(pgrep -af "tail -n 0 -F" 2>&1)"; rc=$? printf 'PGREP_TAIL_WATCHERS_RC=%s\n%s\n%s\n\n' "$rc" 'pgrep -af "tail -n 0 -F"' "$out" >> "$F" printf 'PGREP_TAIL_WATCHERS_RC=%s\n%s\n%s\n\n' "$rc" 'pgrep -af "tail -n 0 -F"' "$out" 
3429169 /bin/bash -c  F=/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/cleanup-audit.md echo '## Check 4 - live watchers (expect: only pgrep own wrapper shell in the output; no real tail -n 0 -F watcher)' >> "$F" echo >> "$F" out="$(pgrep -af "tail -n 0 -F" 2>&1)"; rc=$? printf 'PGREP_TAIL_WATCHERS_RC=%s\n%s\n%s\n\n' "$rc" 'pgrep -af "tail -n 0 -F"' "$out" >> "$F" printf 'PGREP_TAIL_WATCHERS_RC=%s\n%s\n%s\n\n' "$rc" 'pgrep -af "tail -n 0 -F"' "$out" 

## Check 5 - repository state (expect: zero tracked-file changes, untracked .omo/ and bench/results only; HEAD equals fork/sycl/dsv41-sycl-support)

GIT_STATUS_PORCELAIN_RC=0
git -C /home/maxious/Strata status --porcelain
?? .omo/
?? bench/results/2026-10-22-gyro-coder-carveout-p2p/
?? bench/results/2026-10-22-gyro-coder-carveout/
?? bench/results/2026-10-22-gyro-coder-context/
?? bench/results/2026-10-22-gyro-coder-single-3072-carveout-gated/
?? bench/results/2026-10-22-gyro-coder-single-3072-carveout/
?? bench/results/2026-10-22-gyro-m-dual-3072-carveout-gated/
?? bench/results/2026-10-22-gyro-m-dual-3072-carveout/
?? bench/results/2026-10-22-gyro-m-dual-3072-host/
?? bench/results/2026-10-22-gyro-sycl-2xArcProB60/
?? bench/results/2026-10-23-023-regression/

GIT_DIFF_QUIET_RC=0
git -C /home/maxious/Strata diff --quiet


GIT_DIFF_CACHED_QUIET_RC=0
git -C /home/maxious/Strata diff --cached --quiet


GIT_REV_HEAD_RC=0
git -C /home/maxious/Strata rev-parse HEAD
74110114582500beb6bbd8f4fb30d550898e2c72

GIT_REV_FORK_TIP_RC=0
git -C /home/maxious/Strata rev-parse fork/sycl/dsv41-sycl-support
d337362b8a9fa09402fd41e2636cbd7beae66161

HEAD_EQ_FORK_TIP_RC=1
test 74110114582500beb6bbd8f4fb30d550898e2c72 = d337362b8a9fa09402fd41e2636cbd7beae66161


## Check 6 - evidence completeness (expect: every [x] checkbox resolves to a named artifact inside this evidence dir; ledger line count reported)

CHECKBOX_X_INVENTORY_RC=0
grep -nE '^- \[x\] ' /home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md
104:- [x] 1. Re-record the baseline: build the SYCL dsv41 exe and capture today's parity and run rows
114:- [x] 2. Replace the residual dpct advisory blocks with accepted one-line notes
124:- [x] 3. Audit DPCT1098: every host-mapped payload read must be uncached
134:- [x] 4. Audit DPCT1093: every queue must be created on the thread that owns its device
144:- [x] 5. Move the answer payload off host memory: the copy-engine hand-back
155:- [x] 6. Make the host CPU lane contribute experts: publish the routing ids the host reads — ids arrive real (trace re-parsed: 280 records, 1680 ids, 1673 non-zero, 190 distinct, byte-exact) and the host lane really computes and counts non-resident experts: an engine run on one card prints `CPU lane: 60 of 1680 experts` with CPU lane 5.19 ms/token on the default decode path (`--vram-experts 0.25`, `RUN_MIX_RC=0`, `/tmp/dpct_run/w6b_mix025.log`), and `--vram-experts 0` prints 1680 of 1680 at 125.03 ms/token, so the count is not a display artefact; the default run's `CPU lane: 0 of 1680` is correct hot-set residency (all traced picks rank at most 32 inside 41-42 resident slots/layer), not a criterion failure. Note: `--experts 0` clamps routed top-k to 1 so its line reads CPU lane 0 of 280. Re-measured on the merged build `28bd2aaa` (lane `st_01a1788e`, DONECLAIM todo6c: SYCL_CONFIGURE_RC=0 SYCL_BUILD_RC=0 SYCL_RECONFIGURE_RC=0 SYCL_REBUILD_RC=0 EXE_RC=0 RUN_RC=0 RUN_N8_RC=0 RUN_EXP0_RC=0 RUN_MIX_RC=0 RUN_V0_RC=0 SYCL_CTEST_RC=0 CLEANUP_RC=0): the **default** one-card greedy run now clears N > 0 on the quality path — `--n 8` prints `CPU lane: 132 of 1680 experts` (11.79 ms/token) and `--n 32` prints `CPU lane: 195 of 1920 experts` (43.28 ms/token; denominator 1920 = 9 decoded tokens × 40 layers × 6 picks), with the card holding its usual 1684 resident slots; `--vram-experts 0.25` gives 1448 of 1920 and `--vram-experts 0` gives 1920 of 1920, decode `The capital of France is **Paris**.` + EOS byte-identical to todo 8's sequence. Evidence: todo6b gates (SYCL_CONFIGURE_RC=0 SYCL_BUILD_RC=0 EXE_RC=0 RUN_RC=0 RUN_EXP0_RC=0 RUN_V0_RC=0 RUN_MIX_RC=0 SYCL_CTEST_RC=0 GG_RC=0 CLEANUP_RC=0), commit `a698bb84`, reviews `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo6-gate-review.md` (VERDICT FAIL, pre-fix) and `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo6b-gate-review.md` (APPROVE, `[x]`)
166:- [x] 7. Chase the degenerate token: hidden-state correctness, then the head and sampler - verified: no code change was needed or made (`COMMIT_RC=0`, `DIFF_SCOPE_RC=0`, worktree diff empty); the degenerate-token cause was fixed in `28bd2aaa` and holds on the merged build. Acceptance measured on merged HEAD `ce6218ee` (`RUN_RC=0` twice, `PROMPT_LINE_RC=0`, `FIRST_ID_RC=0` first id 671, `PARIS_RC=0`, `CPU_LANE_RC=0` CPU lane 157 of 1920, `SYCL_CTEST_RC=0` naming `ds4_parity` and `ds4_blas_parity`), and the head gate is real: SYCL argmax 671 @ 29.638208 against an independent whole-model torch CPU reference argmax 671 @ 30.843643, with the reconstruction method rejecting the old defective build (442.67 vs 457.72 RMS). Layer-0 residual matches (cos >= 0.9993, was 38.9-1010x off); two `--temperature 0` runs are id-identical by sha256 (`DIFF_RUNS_RC=1` is the negated-diff convention). Gate reviewer `st_01a178f8` returned VERDICT confirmed (APPROVE) on merged HEAD `ce6218ee`. Last evidence: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo7b-gate-review.md` and `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-7-dsv41-sycl-dpct-and-xpu-test.log` (`DONECLAIM todo7`, cleanup `existsSync('/tmp/dpct_w6c_build') === false`).
190:- [x] 9. Two-card run against the same-day single-card baseline
227:- [x] 12. Chase the two-card `UR_RESULT_ERROR_DEVICE_LOST` (level_zero error 20) at `ds4_model.dp.cpp:11213` and land the two-card run against todo 8's same-day single-card baseline. Same work as todo 9's open gap; tracked there.
228:- [x] 13. Amend the measured-record criteria the evidence proved wrong: IS-5's first id is 671 ("The") and its no-type-3 clause must exempt the pack's EOS id 1; todo 6's failure control is `--vram-experts 0` (not `--experts 0`); the one-card load receipt is the engine's per-device line, not `xpu-smi` `memory.used`; the SYCL ctest gate is `ctest --test-dir <build>/sycl -R ds4`.
229:- [x] 14. Add a K-quant GEMV parity row (`iq::mv` -> `mv_kernel` -> `sub_dot`, Q2_K/Q4_K/Q5_K) to `ds4_parity`. - verified on merged HEAD `ce6218ee` (branch commit `4e003d34`, one file `sycl/src/dsv41/ds4_parity.cpp`, 142 insertions / 1 deletion): `SYCL_CONFIGURE_RC=0`, `SYCL_BUILD_RC=0`, `SYCL_CTEST_RC=0` naming `ds4_parity` and `ds4_blas_parity`, with `Q2_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside`, `Q4_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside`, `Q5_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside`. Expected values derive from the repo's own sources (`src/dsv41/ds4_iq.cu:353-373`, `sycl/src/dsv41/ds4_iq.dp.cpp:391-425`, `ggml-common.h` block structures), min term `d * (s0 * lo + s1 * hi) - (m0 * sl + m1 * sh)`. Negative control corrupting `ref_k_sub_dot` puts all three rows at 51 outside while softmax stays 600/600 exact, `RESTORE_RC=0` with git status clean. Diff touches only `sycl/` and `docs/`, all commits `[Engine]`. Gate reviewer `st_01a178f6` returned `VERDICT confirmed` (APPROVE). Last evidence: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row14-gate-review.md`, cleanup `existsSync('/tmp/r14v_build') === false`.
230:- [x] 15. Give the exe-linked kernel copies one home: remove the duplicate `hc_pre_norm` / `hc_mix_scale` / `sparse_attn` definitions, or cover the `ds4_kernels.dp.cpp` copies in the parity harness. Merged as `83417a6f` (6 files under `sycl/`, 1197 insertions / 1529 deletions, subject `[Engine]`). Gate `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row15-gate-review.md` verdict confirmed: `SYCL_CONFIGURE_RC=0`, `SYCL_BUILD_RC=0` (260/260 edges, verification re-run `[53/53]`, `grep -c FAILED` = 0), `nm` shows exactly one `T` each of `hc_pre_norm`, `hc_mix_scale`, `sparse_attn` with 0 occurrences in `ds4_kernels.dp.cpp.o` and `ds4_launch.dp.cpp.o` (one home: `ds4_attn_hc.dp.cpp.o`), `SYCL_CTEST_RC=0` naming `ds4_parity` Passed 0.27 s and `ds4_blas_parity` Passed 0.33 s, `RUN_RC=0` with `prompt: 10 tokens in 262 ms (38.22 tok/s, windows of 8192)` and decoded ids `671,6102,294,8760,344,2619,51119,42499`; cleanup `existsSync('/tmp/r15v_build') === false`.
231:- [x] 16. Add the one-line accepted note (class + reason) to every remaining dpct advisory block in `sycl/src/dsv41`, or amend IS-1 to accept `sycl/docs/sycl-dsv41-dpct-triage.md` as the per-class accepted record. Merged as `ce6218ee` (756 insertions, 0 deletions, 12 files under `sycl/src/dsv41`). Gate `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row16d-gate-review.md` verdict confirmed: `SYCL_BUILD_RC=0` zero FAILED lines `[53/53]`, `SYCL_CTEST_RC=0` naming `ds4_parity` Passed 0.28 s and `ds4_blas_parity` Passed 0.34 s, `RUN_RC=0` with `prompt: 10 tokens in 268 ms (37.32 tok/s, windows of 8192)` and ids `671,6102,294,8760,344,2619,51119,42499` decoding "The capital of France is **Paris**.", CPU lane 105 of 1680; block parsing over `sycl/src/dsv41` finds 751 advisory blocks with zero unnoted; cleanup `CLEANUP_RC=0` with `existsSync === false`.
233:- [x] 18. Close F2's two gaps: compile the AOT variant (`-DSTRATA_SYCL_AOT=bmg-g21`) and diff the ds4 parity rows byte-identically against todo 1's baseline rows. Same work as F2's open gap; tracked there.
237:- [x] F2. Code quality review: JIT compiles under icpx and both ds4 tests pass (`F2_CONFIGURE_RC=0` naming `/opt/intel/oneapi/compiler/2026.2/bin/icpx`, `F2_BUILD_RC=0` 244/244, `F2_CTEST_RC=0` with `ctest --test-dir /tmp/f2_sycl_build/sycl -R ds4` = 100% tests passed, 0 failed out of 2), the todo-2 notes state reasons (`sycl/docs/sycl-dsv41-dpct-triage.md`, no bare TODO), and no new abstraction or default appeared (`git diff 18a27e02..HEAD` = sycl/ + docs only, zero CMakeLists change); but the AOT variant (`-DSTRATA_SYCL_AOT=bmg-g21`) and both gaps are now closed: the AOT variant compiled (`AOT_CONFIGURE_RC=0`, `AOT_RECONFIGURE_RC=0`, `AOT_REBUILD_RC=0` with `-DSTRATA_SYCL_AOT=bmg-g21` plus `-DSTRATA_SYCL_DSV41_EXE=ON`, `AOT_EXE_RC=0` proving `<build>/sycl/strata-dsv41` exists) and the parity rows diff byte-identically against todo 1's baseline (`PARITY_DIFF_RC=0`, both row files 12152 bytes at identical SHA256 3f026239…). Last evidence: `/tmp/dpct_run/f2_configure3.txt`, `f2_build3.txt`, `f2_ctest3.txt`, `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md` (REJECT, `[~]`). Plan-wording correction: the acceptance command must target `build-sycl-dsv41/sycl`, since `enable_testing()` and both `add_test` calls live in `sycl/CMakeLists.txt` (271/277/291).

CHECKBOX_ALL_INVENTORY_RC=0
grep -nE '^- \[[x~ ]\] ' /home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md
104:- [x] 1. Re-record the baseline: build the SYCL dsv41 exe and capture today's parity and run rows
114:- [x] 2. Replace the residual dpct advisory blocks with accepted one-line notes
124:- [x] 3. Audit DPCT1098: every host-mapped payload read must be uncached
134:- [x] 4. Audit DPCT1093: every queue must be created on the thread that owns its device
144:- [x] 5. Move the answer payload off host memory: the copy-engine hand-back
155:- [x] 6. Make the host CPU lane contribute experts: publish the routing ids the host reads — ids arrive real (trace re-parsed: 280 records, 1680 ids, 1673 non-zero, 190 distinct, byte-exact) and the host lane really computes and counts non-resident experts: an engine run on one card prints `CPU lane: 60 of 1680 experts` with CPU lane 5.19 ms/token on the default decode path (`--vram-experts 0.25`, `RUN_MIX_RC=0`, `/tmp/dpct_run/w6b_mix025.log`), and `--vram-experts 0` prints 1680 of 1680 at 125.03 ms/token, so the count is not a display artefact; the default run's `CPU lane: 0 of 1680` is correct hot-set residency (all traced picks rank at most 32 inside 41-42 resident slots/layer), not a criterion failure. Note: `--experts 0` clamps routed top-k to 1 so its line reads CPU lane 0 of 280. Re-measured on the merged build `28bd2aaa` (lane `st_01a1788e`, DONECLAIM todo6c: SYCL_CONFIGURE_RC=0 SYCL_BUILD_RC=0 SYCL_RECONFIGURE_RC=0 SYCL_REBUILD_RC=0 EXE_RC=0 RUN_RC=0 RUN_N8_RC=0 RUN_EXP0_RC=0 RUN_MIX_RC=0 RUN_V0_RC=0 SYCL_CTEST_RC=0 CLEANUP_RC=0): the **default** one-card greedy run now clears N > 0 on the quality path — `--n 8` prints `CPU lane: 132 of 1680 experts` (11.79 ms/token) and `--n 32` prints `CPU lane: 195 of 1920 experts` (43.28 ms/token; denominator 1920 = 9 decoded tokens × 40 layers × 6 picks), with the card holding its usual 1684 resident slots; `--vram-experts 0.25` gives 1448 of 1920 and `--vram-experts 0` gives 1920 of 1920, decode `The capital of France is **Paris**.` + EOS byte-identical to todo 8's sequence. Evidence: todo6b gates (SYCL_CONFIGURE_RC=0 SYCL_BUILD_RC=0 EXE_RC=0 RUN_RC=0 RUN_EXP0_RC=0 RUN_V0_RC=0 RUN_MIX_RC=0 SYCL_CTEST_RC=0 GG_RC=0 CLEANUP_RC=0), commit `a698bb84`, reviews `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo6-gate-review.md` (VERDICT FAIL, pre-fix) and `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo6b-gate-review.md` (APPROVE, `[x]`)
166:- [x] 7. Chase the degenerate token: hidden-state correctness, then the head and sampler - verified: no code change was needed or made (`COMMIT_RC=0`, `DIFF_SCOPE_RC=0`, worktree diff empty); the degenerate-token cause was fixed in `28bd2aaa` and holds on the merged build. Acceptance measured on merged HEAD `ce6218ee` (`RUN_RC=0` twice, `PROMPT_LINE_RC=0`, `FIRST_ID_RC=0` first id 671, `PARIS_RC=0`, `CPU_LANE_RC=0` CPU lane 157 of 1920, `SYCL_CTEST_RC=0` naming `ds4_parity` and `ds4_blas_parity`), and the head gate is real: SYCL argmax 671 @ 29.638208 against an independent whole-model torch CPU reference argmax 671 @ 30.843643, with the reconstruction method rejecting the old defective build (442.67 vs 457.72 RMS). Layer-0 residual matches (cos >= 0.9993, was 38.9-1010x off); two `--temperature 0` runs are id-identical by sha256 (`DIFF_RUNS_RC=1` is the negated-diff convention). Gate reviewer `st_01a178f8` returned VERDICT confirmed (APPROVE) on merged HEAD `ce6218ee`. Last evidence: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo7b-gate-review.md` and `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-7-dsv41-sycl-dpct-and-xpu-test.log` (`DONECLAIM todo7`, cleanup `existsSync('/tmp/dpct_w6c_build') === false`).
178:- [~] 8. One-card quality and throughput run for "what is the capital of france" — gap: the defect is fixed and merged (`28bd2aaa`, from lane commit `fed551b7`; `sub_dot()` added the K-quant min term's second half where CUDA `ds4_iq.cu:372` subtracts both halves), the answer is `The capital of France is **Paris**.` + EOS byte-identical over 4 runs, `129276` gone, CPU lane 1481 of 1680 experts, throughput `prompt: 10 tokens in 414 ms (24.13 tok/s)` then `281 ms (35.61 tok/s)`, `decode: 9 tokens in 4017 ms (1.99 tok/s)`; but the literal FIRST_ID criterion (id 11111 " Paris") is not met — the correct greedy answer starts with 671 ("The"), and the literal no-type-3 criterion trips on id 1, the pack's `eos_token_id`, which `token_type.json` classifies as type 3. Both criteria are mis-derived (11111 came from the defective run's logits; a terminating answer can never satisfy "no type-3"); IS-5 substance is met. Last evidence: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-8-dsv41-sycl-dpct-and-xpu-test.log` (G3 `FIRST_ID_RC=1` → 671; G4 `CONTROL_RC=0` → id 1 type 3), `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo8-gate-review.md` APPROVE `[~]`
190:- [x] 9. Two-card run against the same-day single-card baseline
202:- [~] 10. Default-build regression gate: the Intel path stays opt-in and the default stays green
215:- [ ] 11. Write the measured report and push the branch to the fork
227:- [x] 12. Chase the two-card `UR_RESULT_ERROR_DEVICE_LOST` (level_zero error 20) at `ds4_model.dp.cpp:11213` and land the two-card run against todo 8's same-day single-card baseline. Same work as todo 9's open gap; tracked there.
228:- [x] 13. Amend the measured-record criteria the evidence proved wrong: IS-5's first id is 671 ("The") and its no-type-3 clause must exempt the pack's EOS id 1; todo 6's failure control is `--vram-experts 0` (not `--experts 0`); the one-card load receipt is the engine's per-device line, not `xpu-smi` `memory.used`; the SYCL ctest gate is `ctest --test-dir <build>/sycl -R ds4`.
229:- [x] 14. Add a K-quant GEMV parity row (`iq::mv` -> `mv_kernel` -> `sub_dot`, Q2_K/Q4_K/Q5_K) to `ds4_parity`. - verified on merged HEAD `ce6218ee` (branch commit `4e003d34`, one file `sycl/src/dsv41/ds4_parity.cpp`, 142 insertions / 1 deletion): `SYCL_CONFIGURE_RC=0`, `SYCL_BUILD_RC=0`, `SYCL_CTEST_RC=0` naming `ds4_parity` and `ds4_blas_parity`, with `Q2_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside`, `Q4_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside`, `Q5_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside`. Expected values derive from the repo's own sources (`src/dsv41/ds4_iq.cu:353-373`, `sycl/src/dsv41/ds4_iq.dp.cpp:391-425`, `ggml-common.h` block structures), min term `d * (s0 * lo + s1 * hi) - (m0 * sl + m1 * sh)`. Negative control corrupting `ref_k_sub_dot` puts all three rows at 51 outside while softmax stays 600/600 exact, `RESTORE_RC=0` with git status clean. Diff touches only `sycl/` and `docs/`, all commits `[Engine]`. Gate reviewer `st_01a178f6` returned `VERDICT confirmed` (APPROVE). Last evidence: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row14-gate-review.md`, cleanup `existsSync('/tmp/r14v_build') === false`.
230:- [x] 15. Give the exe-linked kernel copies one home: remove the duplicate `hc_pre_norm` / `hc_mix_scale` / `sparse_attn` definitions, or cover the `ds4_kernels.dp.cpp` copies in the parity harness. Merged as `83417a6f` (6 files under `sycl/`, 1197 insertions / 1529 deletions, subject `[Engine]`). Gate `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row15-gate-review.md` verdict confirmed: `SYCL_CONFIGURE_RC=0`, `SYCL_BUILD_RC=0` (260/260 edges, verification re-run `[53/53]`, `grep -c FAILED` = 0), `nm` shows exactly one `T` each of `hc_pre_norm`, `hc_mix_scale`, `sparse_attn` with 0 occurrences in `ds4_kernels.dp.cpp.o` and `ds4_launch.dp.cpp.o` (one home: `ds4_attn_hc.dp.cpp.o`), `SYCL_CTEST_RC=0` naming `ds4_parity` Passed 0.27 s and `ds4_blas_parity` Passed 0.33 s, `RUN_RC=0` with `prompt: 10 tokens in 262 ms (38.22 tok/s, windows of 8192)` and decoded ids `671,6102,294,8760,344,2619,51119,42499`; cleanup `existsSync('/tmp/r15v_build') === false`.
231:- [x] 16. Add the one-line accepted note (class + reason) to every remaining dpct advisory block in `sycl/src/dsv41`, or amend IS-1 to accept `sycl/docs/sycl-dsv41-dpct-triage.md` as the per-class accepted record. Merged as `ce6218ee` (756 insertions, 0 deletions, 12 files under `sycl/src/dsv41`). Gate `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row16d-gate-review.md` verdict confirmed: `SYCL_BUILD_RC=0` zero FAILED lines `[53/53]`, `SYCL_CTEST_RC=0` naming `ds4_parity` Passed 0.28 s and `ds4_blas_parity` Passed 0.34 s, `RUN_RC=0` with `prompt: 10 tokens in 268 ms (37.32 tok/s, windows of 8192)` and ids `671,6102,294,8760,344,2619,51119,42499` decoding "The capital of France is **Paris**.", CPU lane 105 of 1680; block parsing over `sycl/src/dsv41` finds 751 advisory blocks with zero unnoted; cleanup `CLEANUP_RC=0` with `existsSync === false`.
232:- [~] 17. Make the default ctest gate green: add `iq512x_rows` to `WIDE_PUBLIC` in `tools/check_isa_guard.py` (a separate upstream change, outside this plan's `sycl/` scope). - gap: the fix lives in `tools/check_isa_guard.py`, outside the plan's `sycl/`+`docs/` scope, so it was not changed here; the default-build gate's own tests pass and only `isa_guard_check` fails on `iq512x_rows`. Last evidence: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates.md` verdict confirmed — `DEF_CONFIGURE_RC=0`, `DEF_BUILD_RC=0`, `DEF_FAILED_COUNT=0`, `DEF_CTEST_RC=8` over 57 tests whose sole failure is `isa_guard_check` on `iq512x_rows` (violation names `iq512x_rows`, `iq_avx512.cpp.o`); plus `row16d-gate-review.md` note 4 and `todo4-gate-review.md` `F2_CTEST_RC=0` naming `ds4_parity` and `ds4_blas_parity`.
233:- [x] 18. Close F2's two gaps: compile the AOT variant (`-DSTRATA_SYCL_AOT=bmg-g21`) and diff the ds4 parity rows byte-identically against todo 1's baseline rows. Same work as F2's open gap; tracked there.
234:- [~] 19. Make `--ids @file` fail loudly on a non-numeric token (parse error naming the token; `src/dsv41/ds4_main.cpp:71`) — a scope exception outside `sycl/`, needs the user's sign-off. - gap: the user's scope question (whether `src/dsv41/ds4_main.cpp` may be touched) was asked and no answer arrived within the wait, so the change was not made; the file is untouched. Last evidence: the unanswered scope question and `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/manual-qa.md`, whose `--ids @/tmp/q_france.ids` run exits 0 with the ids read from the file.
236:- [ ] F1. Plan compliance audit - every top-level checkbox is `[x]` or `[~]`, each `[~]` names its gap and its last evidence, every Must NOT have guardrail held (no barrier-scope conversion, no occupancy rewrite, no `ggml_sycl` build, no `.omo/`/`bench/results` commit), and each implementation todo's commit exists with its `[Engine]`/`[Docs]` prefix. Verifier is independent of every executor.
237:- [x] F2. Code quality review: JIT compiles under icpx and both ds4 tests pass (`F2_CONFIGURE_RC=0` naming `/opt/intel/oneapi/compiler/2026.2/bin/icpx`, `F2_BUILD_RC=0` 244/244, `F2_CTEST_RC=0` with `ctest --test-dir /tmp/f2_sycl_build/sycl -R ds4` = 100% tests passed, 0 failed out of 2), the todo-2 notes state reasons (`sycl/docs/sycl-dsv41-dpct-triage.md`, no bare TODO), and no new abstraction or default appeared (`git diff 18a27e02..HEAD` = sycl/ + docs only, zero CMakeLists change); but the AOT variant (`-DSTRATA_SYCL_AOT=bmg-g21`) and both gaps are now closed: the AOT variant compiled (`AOT_CONFIGURE_RC=0`, `AOT_RECONFIGURE_RC=0`, `AOT_REBUILD_RC=0` with `-DSTRATA_SYCL_AOT=bmg-g21` plus `-DSTRATA_SYCL_DSV41_EXE=ON`, `AOT_EXE_RC=0` proving `<build>/sycl/strata-dsv41` exists) and the parity rows diff byte-identically against todo 1's baseline (`PARITY_DIFF_RC=0`, both row files 12152 bytes at identical SHA256 3f026239…). Last evidence: `/tmp/dpct_run/f2_configure3.txt`, `f2_build3.txt`, `f2_ctest3.txt`, `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md` (REJECT, `[~]`). Plan-wording correction: the acceptance command must target `build-sycl-dsv41/sycl`, since `enable_testing()` and both `add_test` calls live in `sycl/CMakeLists.txt` (271/277/291).
238:- [~] F3. Real manual QA - drive the literal invocations in todos 8 and 9 yourself on the real card, capture the decoded answer text, the prompt/decode lines, `xpu-smi` state, and the absence of a GPU reset; confirm the default build and `ctest` still pass (todo 10's commands re-run). - gap: the todo 8 one-card portion and todo 10's regression gate are QA'd and PASS on substance (`st_01a1789d`: 4 byte-identical runs decoding `The capital of France is **Paris**.` from ids `671 6102 294 8760 344 2619 51119 42499`, prompt 21.27/22.50/25.76/22.76 tok/s and decode 2.23/2.43/2.48/2.34 tok/s each beside a completed `prompt:` line, CPU lane 112/108/110/114 of 1680, `RESET_RC=1` x4, `CTEST_RC=8` with the only failure `isa_guard_check` name-identical to the origin/main control), but the todo 9 two-card invocation was deferred because the DEVICE_LOST defect was open when the lane ran; it is now fixed (`dc3c9e7`) and needs a two-card QA pass. Runbook gaps recorded: the literal build needs `-DSTRATA_SYCL_DSV41_EXE=ON` plus a `setvars.sh`-wrapped build and the binary at `<build>/sycl/strata-dsv41`; `FIRST_ID=11111` is mis-derived (671 is correct); the no-type-3 clause passes only because `--n 8` truncates before EOS id 1; ADV-8-1 fails as predicted (`hello world` ids exit 0, parser at `src/dsv41/ds4_main.cpp:71`, row 19). Last evidence: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/dsv41-sycl-dpct-and-xpu-test-manual-qa.md`, `f3qa-receipts.txt`, and the two-card matrix `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/dsv41-sycl-dpct-and-xpu-test-manual-qa-two-card.md` (`DONECLAIM f3twocard`, `VERDICT: PASS with discrepancy`): two `--gpus 0,1` runs on commit `ce6218ee` both decoded "The capital of France is **Paris**." with per-device layer load across devices 0 and 1, no GPU resets (`journalctl` grep exit 1 after each run), and cleanup `existsSync('/tmp/f3tc_build') === false`; the remaining gap is the loud-spin control (`STRATA_SPIN_MAX=1`) printing the expected error but exiting 0 instead of 1 (`SPIN_RUN_RC`), and the matrix labels the hardware as Arc A770 while the runs were on the two Arc Pro B60s.
239:- [~] F4. Ideal-state fidelity - check IS-1..IS-8 one row at a time against the evidence logs, and report each `[~]` row as a known gap. A shortfall becomes new `- [ ] N.` rows, never a note. - gap: the audit is done (`st_01a178a1`); IS-2, IS-3, IS-4 and IS-6 are met, IS-1, IS-5, IS-7 and IS-8 are partially met. IS-1: the one-line accepted note does not exist at every dpct block (1051 DPCT lines remain at `28bd2aaa`, zero inline notes at DPCT1093's 107 blocks in `ds4_model.dp.cpp`) - row 16 is open. IS-5: substance holds (ids `671 6102 294 8760 344 2619 51119 42499 1` decode to "The capital of France is **Paris**." + EOS, 4 byte-identical runs, `129276` gone) but the literal criteria are mis-derived: first id is 671 not 11111, and the pack's EOS id 1 is classified type 3, so "no type-3 id" can never pass for a terminating answer - row 13 is open. IS-7: no two-card number is claimed anywhere and the two-card run now passes under todo 9's fix, but the measured report publishing it (todo 11) is open. IS-8: `ctest -R ds4` 2/2 and the default build are green, but `DEFAULT_CTEST_RC=8` - `isa_guard_check` fails on `iq512x_rows` (777 wide instructions, first vmovq), identical on `origin/main 4b9e05a2` - row 17 needs your scope sign-off. Last evidence: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f4-fidelity.md`.
240:- [ ] F5. Cleanup audit - every lane worktree removed or handed off, every `/tmp/<wave>_build` and scratch dir deleted with an un-piped receipt (`ls` exit 2), no live process holding a card, no live watcher, and the run-root temp dir removed.

EVIDENCE_DIR_LS_RC=0
ls -1 /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test
artifacts
cleanup-audit.md
dsv41-sycl-dpct-and-xpu-test-manual-qa-two-card.md
dsv41-sycl-dpct-and-xpu-test-manual-qa.md
f1-plan-compliance.md
f3_analyze.sh
f3_analyze4.sh
f3_ctest_phase.sh
f3_decode.py
f3_default_phase.sh
f3_exe_receipt.sh
f3_neg.sh
f3_neg2.sh
f3_run1.sh
f3_run2.sh
f3_run3.sh
f3_run4.sh
f3_sycl_build2.sh
f3_sycl_phase.sh
f3qa-artifact-sizes.txt
f3qa-build-identity.txt
f3qa-build-literal-failed.txt
f3qa-build2-setvars.txt
f3qa-configure-exe.txt
f3qa-configure-literal.txt
f3qa-ctest-failed.txt
f3qa-ctest-summary.txt
f3qa-ctest.txt
f3qa-ctl-failed-only.txt
f3qa-decode-RUN1.txt
f3qa-decode-RUN2.txt
f3qa-decode-RUN3.txt
f3qa-decode-RUN4.txt
f3qa-decode-goldencheck.txt
f3qa-enc-check.txt
f3qa-exe-receipt.txt
f3qa-faildiff2.txt
f3qa-failed-only.txt
f3qa-keylines.txt
f3qa-neg-garbage-ids.log
f3qa-neg2-badid.log
f3qa-pgrep-gates.txt
f3qa-receipts-raw.txt
f3qa-receipts.txt
f3qa-run1.log
f3qa-run2.log
f3qa-run3.log
f3qa-run4.log
f3qa-smi-loop-during.txt
f3qa-smi-receipts.txt
f3qa-smi-table-after.txt
f3qa-smi-table-during-engine.txt
f3tc-baseline-diff.txt
f3tc-build-aot.txt
f3tc-build-identity.txt
f3tc-configure-aot.txt
f3tc-ctest-failed-names.txt
f3tc-ctest.txt
f3tc-decode-base.txt
f3tc-decode-goldencheck.txt
f3tc-decode-run1.txt
f3tc-decode-run2.txt
f3tc-def-build.txt
f3tc-def-configure.txt
f3tc-exe-receipt.txt
f3tc-golden-out.txt
f3tc-ids-badid.ids
f3tc-ids-garbage.ids
f3tc-isa-violation.txt
f3tc-keylines-base.txt
f3tc-keylines-run1.txt
f3tc-keylines-run2.txt
f3tc-neg-badid.log
f3tc-neg-garbage.log
f3tc-pgrep-BASE.txt
f3tc-pgrep-NEG.txt
f3tc-pgrep-NEG2.txt
f3tc-pgrep-R1.txt
f3tc-pgrep-R2.txt
f3tc-pgrep-SPIN.txt
f3tc-pgrep-post.txt
f3tc-pipeline.log
f3tc-receipts-analyze.txt
f3tc-receipts-build-aot.txt
f3tc-receipts-build-def.txt
f3tc-receipts-runs.txt
f3tc-repeat-diff.txt
f3tc-run-base.log
f3tc-run1.log
f3tc-run2.log
f3tc-smi-after.txt
f3tc-smi-loop-stderr.txt
f3tc-smi-loop.txt
f3tc-spin-max1.log
f3tc_analyze.sh
f3tc_build_aot.sh
f3tc_build_def.sh
f3tc_cleanup.sh
f3tc_pipeline.sh
f3tc_runs.sh
f3twc-build-failed-count.txt
f3twc-build-identity.txt
f3twc-build-progress-last.txt
f3twc-build.txt
f3twc-configure.txt
f3twc-exe-receipt.txt
f3twc-receipts-build.txt
f3twc_build.sh
f3twc_decode.py
f3twc_runs.sh
f4-fidelity.md
f5_exists.cjs
ids-prep-q_france.log
planning-c1.txt
planning-c2.txt
planning-c3.txt
planning-c4.txt
planning-notepad.md
regression-gates-artifacts
regression-gates.md
row13-gate-review.md
row14-gate-review.md
row15-gate-review.md
row16b-gate-review.md
row16d-gate-review.md
task-1-dsv41-sycl-dpct-and-xpu-test.log
task-10-dsv41-sycl-dpct-and-xpu-test.log
task-11-dsv41-sycl-dpct-and-xpu-test.log
task-15-dsv41-sycl-dpct-and-xpu-test.log
task-2-dsv41-sycl-dpct-and-xpu-test.log
task-3-dsv41-sycl-dpct-and-xpu-test.log
task-4-dsv41-sycl-dpct-and-xpu-test.log
task-5-dsv41-sycl-dpct-and-xpu-test.log
task-6-dsv41-sycl-dpct-and-xpu-test.log
task-7-dsv41-sycl-dpct-and-xpu-test.log
task-8-dsv41-sycl-dpct-and-xpu-test.log
task-9-dsv41-sycl-dpct-and-xpu-test.log
todo11-gate-review.md
todo5-gate-review.md
todo6-gate-review.md
todo6b-gate-review.md
todo6c-gate-review.md
todo7-gate-review.md
todo7b-gate-review.md
todo8-gate-review.md
todo9-gate-review.md

ART_TASK1_LOG_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-1-dsv41-sycl-dpct-and-xpu-test.log


ART_TASK2_LOG_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-2-dsv41-sycl-dpct-and-xpu-test.log


ART_TASK3_LOG_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-3-dsv41-sycl-dpct-and-xpu-test.log


ART_TASK4_LOG_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-4-dsv41-sycl-dpct-and-xpu-test.log


ART_TASK5_LOG_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-5-dsv41-sycl-dpct-and-xpu-test.log


ART_TASK6_LOG_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-6-dsv41-sycl-dpct-and-xpu-test.log


ART_TASK7_LOG_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-7-dsv41-sycl-dpct-and-xpu-test.log


ART_TASK9_LOG_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-9-dsv41-sycl-dpct-and-xpu-test.log


ART_TASK10_LOG_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-10-dsv41-sycl-dpct-and-xpu-test.log


ART_TASK11_LOG_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-11-dsv41-sycl-dpct-and-xpu-test.log


ART_ROW13_GATE_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row13-gate-review.md


ART_ROW14_GATE_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row14-gate-review.md


ART_ROW15_GATE_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row15-gate-review.md


ART_ROW16D_GATE_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row16d-gate-review.md


ART_TASK15_LOG_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-15-dsv41-sycl-dpct-and-xpu-test.log


ART_ROW18_GATE_RC=1
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row18-gate-review.md


ART_F3TC_AOT_BUILD_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f3tc-build-aot.txt


ART_F3TC_BASELINE_DIFF_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f3tc-baseline-diff.txt


ART_F2_GATE_SIBLING_RC=0
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md


TASK2_F2GAPS_GREP_RC=0
grep -n 'DONECLAIM f2gaps' /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-2-dsv41-sycl-dpct-and-xpu-test.log
181:DONECLAIM f2gaps gates=AOT_CONFIGURE_RC=0 AOT_BUILD_RC=0 AOT_RECONFIGURE_RC=0 AOT_REBUILD_RC=0 AOT_EXE_RC=0 PARITY_DIFF_RC=0 CLEANUP_RC=0

LEDGER_WC_RC=0
wc -l /home/maxious/Strata/.omo/ulw-execute/ledger.jsonl
438 /home/maxious/Strata/.omo/ulw-execute/ledger.jsonl

## Check 5 - corrected re-run after the push (orchestrator steering: fork tip advanced to a53335fd; supersedes the earlier GIT_* receipts above)

GIT_STATUS_PORCELAIN_V2_RC=0
git -C /home/maxious/Strata status --porcelain
?? .omo/
?? bench/results/2026-10-22-gyro-coder-carveout-p2p/
?? bench/results/2026-10-22-gyro-coder-carveout/
?? bench/results/2026-10-22-gyro-coder-context/
?? bench/results/2026-10-22-gyro-coder-single-3072-carveout-gated/
?? bench/results/2026-10-22-gyro-coder-single-3072-carveout/
?? bench/results/2026-10-22-gyro-m-dual-3072-carveout-gated/
?? bench/results/2026-10-22-gyro-m-dual-3072-carveout/
?? bench/results/2026-10-22-gyro-m-dual-3072-host/
?? bench/results/2026-10-22-gyro-sycl-2xArcProB60/
?? bench/results/2026-10-23-023-regression/

GIT_DIFF_QUIET_V2_RC=0
git -C /home/maxious/Strata diff --quiet


GIT_DIFF_CACHED_QUIET_V2_RC=0
git -C /home/maxious/Strata diff --cached --quiet


GIT_REV_HEAD_V2_RC=0
git -C /home/maxious/Strata rev-parse HEAD
a53335fd598e5b9e976d830faa23badc1192a924

GIT_REV_FORK_TIP_V2_RC=0
git -C /home/maxious/Strata rev-parse fork/sycl/dsv41-sycl-support
a53335fd598e5b9e976d830faa23badc1192a924

HEAD_EQ_FORK_TIP_V2_RC=0
test a53335fd598e5b9e976d830faa23badc1192a924 = a53335fd598e5b9e976d830faa23badc1192a924


## Check 6 - close-time re-probe (row18/F2 gate artifact and ledger line count at file close)

ART_ROW18_GATE_V2_RC=1
test -f /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row18-gate-review.md


LEDGER_WC_V2_RC=0
wc -l /home/maxious/Strata/.omo/ulw-execute/ledger.jsonl
440 /home/maxious/Strata/.omo/ulw-execute/ledger.jsonl

## Supplementary probe - run-root temp dir clause of the F5 checkbox text (outside the six enumerated checks)

RUN_ROOT_DPCT_RUN_RC=0
ls -d /tmp/dpct_run
/tmp/dpct_run

LEFTOVER_TMPBUILD_RC=0
ls -d /home/maxious/tmpbuild
/home/maxious/tmpbuild

LEFTOVER_SHM_R14EV_RC=0
ls -d /dev/shm/r14ev
/dev/shm/r14ev

OTHER_TMP_ROOTS_RC=2
ls -d /tmp/senpi* /tmp/omo* /tmp/ulw* /tmp/*run-root*
ls: cannot access '/tmp/omo*': No such file or directory
ls: cannot access '/tmp/ulw*': No such file or directory
ls: cannot access '/tmp/*run-root*': No such file or directory
/tmp/senpi-codemode-132a69108433e59c
/tmp/senpi-codemode-5cfc8d70fe77f13d
/tmp/senpi-codemode-603b2a1b1daf742f
/tmp/senpi-codemode-b289a89eaa21e91a
/tmp/senpi-codemode-b81439115cfd4919
/tmp/senpi-codemode-c53778cad7ca8e09
/tmp/senpi-codemode-cabfd96ee76f1f35
/tmp/senpi-codemode-daaff17288f9e1b5
/tmp/senpi-codemode-e5d4e3b5351187c3
/tmp/senpi-computer-skill-a45cc4aadaa068db
/tmp/senpi-rpc-host-locks

## Findings

### Check 1 - lane worktrees: PASS
`git worktree list` (GIT_WORKTREE_LIST_RC=0) shows only the main checkout: `/home/maxious/Strata d337362b [sycl/dsv41-sycl-support]` (the pre-push tip at that moment; check 5's corrected re-run carries the pushed tip). All seven lane paths report `existsSync === false`, each with EXISTS_*_RC=0: `/home/maxious/Strata-wt/dpct-notes`, `/home/maxious/Strata-wt/dsv41-dpct-w6c`, `/home/maxious/Strata-wt/dsv41-dpct-w9`, `/home/maxious/Strata-wt/row14`, `/home/maxious/Strata-wt/row15`, `/home/maxious/Strata-wt/row11`, `/tmp/f1_wt`. No lane worktree remains to be removed or handed off.

### Check 2 - scratch build dirs: PASS
All eight named dirs report `existsSync === false`, each with EXISTS_*_RC=0: `/tmp/r15v_build`, `/tmp/r16v5_build`, `/tmp/r16v6_build`, `/tmp/f1_def_build`, `/tmp/f1_sycl_build`, `/tmp/f1_def_tests`, `/dev/shm/r14gate_build`, `/tmp/f3tc_build`. None exists, so the per-dir `pgrep -af <name>` owner probe (scoped by the brief to dirs that still exist) had nothing to run against. This independently reproduces the ledger's cleanup claims at L424-L430 from filesystem state.

### Check 3 - live card processes: PASS
`pgrep -c strata-dsv41` printed 0 (PGREP_COUNT_STRATA_DSV41_RC=1) and `pgrep -c strata` printed 0 (PGREP_COUNT_STRATA_RC=1). No process holds a card.

### Check 4 - live watchers: PASS
`pgrep -af "tail -n 0 -F"` printed exactly two full lines (pids 3429168 and 3429169). Read in full, both are pgrep's own wrapper shells - each is `/bin/bash -c` carrying this probe's own script text (which contains the pattern) - not tail processes. No live watcher survives.

### Check 5 - repository state: PASS on the corrected re-run
The first run predates the orchestrator's push and recorded HEAD `74110114` against fork tip `d337362b` (HEAD_EQ_FORK_TIP_RC=1); per the orchestrator's steering that receipt is superseded and kept as history, not deleted. The corrected re-run records GIT_REV_HEAD_V2 = GIT_REV_FORK_TIP_V2 = `a53335fd598e5b9e976d830faa23badc1192a924` with HEAD_EQ_FORK_TIP_V2_RC=0. `git status --porcelain` shows only `?? .omo/` and `?? bench/results/*` (the expected untracked roots) and zero tracked-file changes: GIT_DIFF_QUIET_V2_RC=0, GIT_DIFF_CACHED_QUIET_V2_RC=0.

### Check 6 - evidence completeness: NOT CONFIRMED (rows 18 and F2)
The plan's [x] inventory at receipt time is 15 rows - 1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16, 18, F2 (CHECKBOX_X_INVENTORY_RC=0). Named artifacts inside `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/` exist for 13 of them:

- rows 1-7 and 9 -> `task-N-dsv41-sycl-dpct-and-xpu-test.log` (ART_TASK1_LOG_RC=0 through ART_TASK9_LOG_RC=0; each named in its row's Evidence line).
- row 12 ("Same work as todo 9's open gap; tracked there") -> `task-9-dsv41-sycl-dpct-and-xpu-test.log`.
- row 13 -> `row13-gate-review.md` (ART_ROW13_GATE_RC=0; the row-named gate review).
- rows 14, 15, 16 -> `row14-gate-review.md`, `row15-gate-review.md`, `row16d-gate-review.md` (ART_ROW14_GATE_RC=0, ART_ROW15_GATE_RC=0, ART_ROW16D_GATE_RC=0; each named in its row text), with `task-15-dsv41-sycl-dpct-and-xpu-test.log` also present (ART_TASK15_LOG_RC=0).

Rows 18 and F2 have no named artifact inside the dir at close:

- Row 18 names no artifact ("Same work as F2's open gap; tracked there"), and the row-named gate review `row18-gate-review.md` is absent (ART_ROW18_GATE_RC=1; re-probed at file close, ART_ROW18_GATE_V2_RC=1).
- F2's row text names `/tmp/dpct_run/f2_configure3.txt`, `f2_build3.txt`, `f2_ctest3.txt` - outside this dir - and `.omo/evidence/dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md`, which is a sibling of the dir, not under it (ART_F2_GATE_SIBLING_RC=0).

Their closure is evidenced inside the dir under other names, so nothing is lost: `task-2-dsv41-sycl-dpct-and-xpu-test.log:181` carries `DONECLAIM f2gaps gates=AOT_CONFIGURE_RC=0 AOT_BUILD_RC=0 AOT_RECONFIGURE_RC=0 AOT_REBUILD_RC=0 AOT_EXE_RC=0 PARITY_DIFF_RC=0 CLEANUP_RC=0` (TASK2_F2GAPS_GREP_RC=0), `f3tc-build-aot.txt` and `f3tc-baseline-diff.txt` hold the AOT-variant build and the baseline-diff receipts (ART_F3TC_AOT_BUILD_RC=0, ART_F3TC_BASELINE_DIFF_RC=0), and the ledger entry that marked both rows [x] (L375) names `task-2-dsv41-sycl-dpct-and-xpu-test.log` as their artifact. The shortfall is the brief's literal condition - a named artifact for every [x] checkbox inside this evidence dir - and it is the same gap F1 flagged. The dispatched "rows 18/F2 independent gate" artifact `row18-gate-review.md` had not landed at close (ledger L438 notes it "not yet written"), and this audit re-probed it absent. Landing that file - or the orchestrator's explicit decision that the task-2/f3tc receipts are rows 18/F2's named artifacts - flips this check.

Ledger line count: 438 at the check-6 receipt (LEDGER_WC_RC=0), 440 at file close (LEDGER_WC_V2_RC=0). The ledger is appended live by the orchestrator, so the count moves.

### Outside the six enumerated checks: F5 text clause "run-root temp dir"
The only temp dir the receipts label the "run dir" is `/tmp/dpct_run` (task-1 log: `mkdir -p /tmp/dpct_run`, MKDIR_RUNDIR_RC=0). It still exists at close (RUN_ROOT_DPCT_RUN_RC=0) and is in live use as the ledger's artifact dir (L424-L438 name `/tmp/dpct_run/*.txt` artifacts), so the F5 checkbox text's "run-root temp dir removed" clause is unmet at close if that dir is what the clause names. Two further unlisted scratch leftovers exist: `/home/maxious/tmpbuild` (LEFTOVER_TMPBUILD_RC=0, the TMPDIR used by gate builds) and `/dev/shm/r14ev` (LEFTOVER_SHM_R14EV_RC=0). No `*run-root*` candidate exists under /tmp (OTHER_TMP_ROOTS_RC=2; it lists only harness `senpi-*` runtime dirs). None of the six enumerated checks covers these; they are filesystem facts reported for the orchestrator's F5 decision, and the verdict line below rests on the six enumerated checks.

## Summary before the verdict
Checks 1-5 are confirmed by the receipts above (check 5 on the corrected post-push re-run). Check 6 cannot be confirmed: rows 18 and F2 have no named artifact inside the evidence dir at close. The run-root clause of the F5 checkbox text is also unmet at close if it names `/tmp/dpct_run`. The cleanup state itself - worktrees, scratch dirs, card processes, watchers - is clean.

VERDICT FAIL

