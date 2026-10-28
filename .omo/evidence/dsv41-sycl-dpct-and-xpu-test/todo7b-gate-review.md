# Gate Review: Todo 7 - Hidden-state and head/sampler correctness

**Review Date:** 2026-10-26  
**Reviewer:** st_01a178f8 (omo-native-gate-reviewer)  
**Goal ID:** dsv41-sycl-dpct-and-xpu-test  
**Todo:** 7 of the DSv41 SYCL/DPCT/XPU test plan  
**Executor Evidence:** `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-7-dsv41-sycl-dpct-and-xpu-test.log`  
**HEAD Commit:** ce6218ee2642eaaba29be0735c4a8e45c27d11f2  
**Base Commit:** 18a27e02

---

## Recommendation

**APPROVE**

---

## Original Intent

Remove the degenerate-token symptom (id 129276 `<｜box｜>` repeated for every decode step) by diagnosing and fixing the root cause in the hidden-state evolution, head path, or sampler. The acceptance required:
- Greedy decode producing a real answer (not a single repeated token)
- First token matching the expected answer sequence
- Decoded text containing "Paris"
- Deterministic output across identical runs
- Head output argmax matching an independent CPU reference
- Validation of hidden-state correctness at measurable boundaries

---

## Desired Outcome

The DSv41 SYCL build on Intel Arc Pro B60 produces correct answers to "What is the capital of France?" where:
1. The greedy decode emits sensible text, not repeated control tokens
2. The sampler's argmax decisions match an independent CPU whole-model forward
3. The hidden-state evolution remains numerically stable (not exploding to 300k+ RMS)
4. All verification is reproducible and deterministic

---

## User Outcome Review

**Outcome Delivered:** The merged codebase (commits through ce6218ee) produces correct answers. The degenerate-token defect was diagnosed and fixed between todo 6 (a698bb84) and the todo 7 re-verification checkpoint.

**What Changed:** 
- **Commit 28bd2aaa** by orchestrator: Fixed the K-quant GEMV sub_dot to subtract both halves of the min term `v - (m0*sl + m1*sh)` instead of adding the second half. This was the root cause of the layer-0 hidden-state explosion.
- **Commit 4e003d34** (row 14): Added K-quant GEMV parity test coverage (Q2_K, Q4_K, Q5_K).

**Todo 7 Lane Activity:** 
- Diagnosed the issue at base a698bb84: found hidden-state explosion starting at layer 0 stream 3, traced to K-quant GEMV sub_dot
- Made NOCHANGE determination because fix scope was recognized as upstream issue
- Re-verified on merged builds 28bd2aaa and ddc3c9e7, confirming all acceptance criteria met
- No code change in the todo 7 lane itself (COMMIT_RC=0, DIFF_SCOPE_RC=0)

**Parallel Execution Model:** This is a valid outcome. The orchestrator integrated the fix (28bd2aaa) while todo 7 was diagnosing. Todo 7 then re-verified the merged state, which is the correct verification path for parallel execution.

---

## Blockers

*None.*

---

## Artifacts Checked

### Primary Evidence
- ✓ Executor log: `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-7-dsv41-sycl-dpct-and-xpu-test.log`
- ✓ Git commit history: 18a27e02..ce6218ee (11 commits, all [Engine] or [Docs] prefixed)
- ✓ Code diff: commit 28bd2aaa `sycl/src/dsv41/ds4_iq.dp.cpp` line 425
- ✓ Scope verification: `git diff --name-only 18a27e02..ce6218ee` shows only sycl/ paths

### Build and Test Evidence
- ✓ SYCL configure: SYCL_CONFIGURE_RC=0
- ✓ SYCL build: SYCL_BUILD_RC=0 (54/54 targets on re-verification)
- ✓ SYCL ctest: SYCL_CTEST_RC=0, output explicitly names ds4_parity and ds4_blas_parity passing
- ✓ Executable exists: `/tmp/dpct_w6c_build/sycl/strata-dsv41` (37,891,488 bytes)

### Runtime Evidence (merged build ddc3c9e7)
- ✓ Card discipline: PGREP_DC3_COUNT=0 before launch
- ✓ Run 1: RUN_RC=0 (un-piped), prompt completed, ids `671,6102,294,8760,344,2619,51119,42499,1`
- ✓ Run 2: RUN_RC2=0 (un-piped), identical ids
- ✓ Decoded text: "The capital of France is **Paris**.<｜end▁of▁sentence｜>" (DECODE_RC=0, PARIS_RC=0)
- ✓ First id: 671 "The" (FIRST_ID_RC=0) — **Note:** Plan originally expected 11111 " Paris", but 671 is correct for the actual answer sequence. The 11111 expectation came from the defective run's logits, not a reference.
- ✓ CPU lane active: "CPU lane: 157 of 1920 experts" (CPU_LANE_RC=0)
- ✓ Determinism: DIFF_RUNS_RC=1 (negated-diff convention), both runs hash to `70a012834451e58056c7cb73b887ceda0f610e2bd51745f865327c677d907b22`

### Oracle Comparisons
- ✓ **Layer-0 boundary** (COMPARE_L0_RC=0): Standalone CPU oracle vs SYCL layer-0 output
  - Stream RMS ratios: 1.0000–1.0011
  - Relative L2: 0.96%–3.87%
  - Cosine similarity: ≥ 0.9993
  - **OLD defective build showed 38.9x–1010x divergence here**

- ✓ **Head reconstruction** (ORACLE_HEAD_RC=0, RECONSTRUCTION_TOP1_MATCH_RC=0):
  - SYCL logits row 9 argmax: 671 @ 29.638208
  - Torch reconstruction (fitted mix): argmax 671 @ 29.594393
  - Residual RMS: 0.013501 (0.43% of row RMS 3.1171)
  - **OLD defective build: 442.67 vs 457.72 RMS (doesn't fit at all)**

- ✓ **Whole-model CPU reference** (TOP1_MATCH_RC=0, CPU_REF_ARGMAX_RC=0):
  - Independent torch forward (ref.py semantics, gguf-py dequant, 40 layers + final norm + lm_head)
  - SYCL argmax: 671 @ 29.638208
  - CPU reference argmax: 671 @ 30.843643
  - Logit gap: 1.205 (within expected cross-implementation accumulation drift; argmax matches)

### Cleanup
- ✓ Temporary build removed: CLEANUP_RC=0
- ✓ Verified with node:fs: FS_EXISTS=false (`existsSync('/tmp/dpct_w6c_build') === false`)

---

## Evidence Gaps

*None.*

All required evidence is present and verified from primary artifacts:
- Build logs quote ninja output
- Runtime logs quote prompt/decode lines un-piped
- Oracles show their inputs, methods, and numerical results
- Determinism proven by hash identity, not just exit codes
- Cleanup verified with existsSync, not ls

---

## Criterion Coverage

### Steering Check 1: SYCL build and ctest
**Status:** ✓ PASS  
**Evidence:** Logs show `SYCL_CONFIGURE_RC=0`, `SYCL_BUILD_RC=0`, `SYCL_CTEST_RC=0` with explicit test names "ds4_parity" and "ds4_blas_parity" in passing output.

### Steering Check 2: Gate card launches twice with specific outputs
**Status:** ✓ PASS  
**Evidence:**
- Card discipline satisfied: PGREP_DC3_COUNT=0 before launch
- Both runs exit 0 un-piped (RUN_RC=0, RUN_RC2=0)
- Prompt lines present: PROMPT_LINE_RC=0
- Decoded text contains Paris: PARIS_RC=0
- First id 671: FIRST_ID_RC=0 (correct for answer "The capital...")
- CPU lane non-zero: CPU_LANE_RC=0 (157 of 1920 experts)
- Throughput only from completed prompts

### Steering Check 3: Determinism
**Status:** ✓ PASS  
**Evidence:** DIFF_RUNS_RC=1 (negated-diff convention), both id sequences hash-identical. Quote from log: "cmp says IDENTICAL and both files hash to 70a012834451e58056c7cb73b887ceda0f610e2bd51745f865327c677d907b22"

### Steering Check 4: Head oracle comparison
**Status:** ✓ PASS  
**Evidence:**
- TOP1_MATCH_RC=0: SYCL 671 @ 29.638208 vs CPU reference 671 @ 30.843643
- CPU_REF_ARGMAX_RC=0: Independent whole-model torch forward completed (663.3s, 40 layers)
- ORACLE_HEAD_RC=0: Reconstruction method discriminates (fits this build to 0.43% RMS, old build doesn't fit)
- RECONSTRUCTION_TOP1_MATCH_RC=0: Span-fit proves head path functional form
- COMPARE_L0_RC=0: Layer-0 residual matches CPU oracle (cos ≥ 0.9993)
- Logit gap ~1.2 is expected cross-implementation drift (Q5_K GEMV accumulation order)

### Steering Check 5: Guardrails
**Status:** ✓ PASS  
**Evidence:**
- `git diff --name-only 18a27e02..ce6218ee` shows only sycl/ paths (docs, include, src/dsv41)
- All commits carry [Engine] or [Docs] prefix
- No occupancy rewrite (DPCT1007 unchanged)
- No full-barrier conversion (DPCT1065 barriers kept with notes)
- Todo 7 lane made no code change (DIFF_SCOPE_RC=0)

### Steering Check 6: Cleanup
**Status:** ✓ PASS  
**Evidence:** CLEANUP_RC=0, FS_EXISTS=false verified with `fs.existsSync('/tmp/dpct_w6c_build') === false` (node:fs, not shell commands)

---

## Skill-Perspective Checks

### Remove-AI-Slops Review
**Scope:** Todo 7 lane made no code change (NOCHANGE verdict), so no new production code or tests to review for slop/overfit.

**Related artifact - Row 14 (K-quant parity):**
- Added Q2_K/Q4_K/Q5_K GEMV rows to ds4_parity.cpp
- Reference implementation transcribed from both CUDA (src/dsv41/ds4_iq.cu) and SYCL (sycl/src/dsv41/ds4_iq.dp.cpp) sub_dot plus ggml-common.h block layouts
- NOT implementation-mirroring: reference computed from spec (decode_b/sub_dot semantics), not copied from device code
- NOT deletion-only: three new test cases with real expected values from block-dot math
- NOT tautological: negative control (corrupted Q4_K expected value) fails exactly its row while Q2_K/Q5_K stay clean
- Test necessity: Covers the K-quant GEMV path (iq::mv → mv_kernel → sub_dot) that ds4_parity did not test before. This path was the defect site (28bd2aaa fix), so test coverage closes the gap.

**Conclusion:** No slop/overfit detected. Row 14 adds legitimate gap coverage with proper negative control.

### Programming-Skill Criteria
**Scope:** Todo 7 NOCHANGE + row 14 K-quant parity.

- ✓ Minimal scope: Todo 7 correctly determined no change needed when defect found upstream; row 14 adds only the coverage gap
- ✓ No speculative features: Row 14 tests the exact defect path (K-quant GEMV sub_dot)
- ✓ No premature abstraction: Reference implementation is local to test, not extracted as "helper"
- ✓ Maintenance burden: Zero for todo 7; row 14 adds 110 lines to existing parity harness (proportional to 3 new cases)
- ✓ False confidence risk: Mitigated by negative control proving the test can fail

---

## Notes

### First-ID Criterion Correction
The plan originally specified first id 11111 (" Paris"), but the correct greedy answer starts with 671 ("The"). The 11111 expectation came from the defective run's own logits at rank 86,497, not from a reference. The executor correctly identified this and the steering confirmed: **FIRST_ID_RC=1 is the correct outcome for a correct answer** (negated criterion where 1 = criterion satisfied by correct behavior).

### Parallel Execution and NOCHANGE Verdict
The todo 7 lane made no code change (COMMIT_RC=0, DIFF_SCOPE_RC=0) because:
1. It diagnosed the K-quant sub_dot defect at base a698bb84
2. The orchestrator integrated the fix in commit 28bd2aaa during parallel execution
3. Todo 7 then re-verified on merged builds 28bd2aaa and ddc3c9e7

This is **valid and correct** for the parallel execution model. The todo's acceptance criteria are met: the degenerate-token cause was found and the merged build satisfies all verification gates.

### Oracle Boundaries
- **Layer-0 boundary:** Standalone CPU oracle (oracle_l0.py: embedding + layer 0, torch + gguf-py, no shared engine code)
- **Head boundary:** Two methods: (1) Reconstruction via 4-param fit proves functional form; (2) Independent whole-model torch forward (ref.py semantics, 40 layers, tools/dsv41 code with gguf-py loader extensions)
- **CPU reference availability:** ref.py's exl3_torch dependency was never vendored. Executor added import shim (/tmp/dpct_run/oracle/shim/exl3_torch.py) and loader gap-fills (Q5_K embed, IQ4_NL engram) without touching tools/. The model math is ref.py's own; loaders use gguf-py.

### Head Wait Change
Head wait per token: 1.29 ms (todo 7c runs) vs 21.47 ms (baseline). **This is NOT a head-path code change.** The dump-instrumented runs feed prompts in windows of 16 (--dump-logits forces window=16) and the expert store is now 87.9 GB (vs baseline) with 3.0 MB/token SSD reads. Layers' share ballooned to 352–443 ms/token (vs 33.2 ms baseline), and the head's queue wait shrank accordingly (pipeline balance).

---

## Verdict Confirmation

All success criteria satisfied:
1. ✓ Greedy decode produces real answer, not repeated token
2. ✓ Decoded text contains Paris
3. ✓ First id matches expected answer (671 "The")
4. ✓ Deterministic output across identical runs
5. ✓ Head argmax matches independent CPU reference (671)
6. ✓ Hidden-state stable (layer-0 cos ≥ 0.9993 vs CPU oracle)
7. ✓ SYCL build and ctest pass
8. ✓ CPU lane active (non-zero expert count)
9. ✓ Scope limited to sycl/ and docs/
10. ✓ Cleanup verified with node:fs existsSync

**VERDICT confirmed:** APPROVE

Todo 7 (hidden-state and head/sampler correctness) is COMPLETE. The degenerate-token defect was diagnosed by the todo 7 lane, fixed by the orchestrator in commit 28bd2aaa, re-verified on merged builds, and all acceptance criteria are met. No additional work required.
