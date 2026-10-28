# Row 14 gate review — K-quant GEMV parity row (Q2_K/Q4_K/Q5_K)

Work: `dsv41-sycl-dpct-and-xpu-test`, plan row 14
Lane commit: `675e3375` — merged by the orchestrator as branch commit `4e003d34` on
`sycl/dsv41-sycl-support`
Reviewer session: `senpi:st_01a178e7` (child of `01a1722c-55f2-713f-87ee-4fa64aab0e1f`)
Reviewed: 2026-10-27, host `lain`, Intel(R) Arc(TM) Pro B60 Graphics, oneAPI 2026.2 (`icx`/`icpx`).

## recommendation

**APPROVE**

No stated success criterion fails. Every claim below is reproduced from artifacts inspected in
this session; no lane prose was taken as evidence.

`blockers`: none. Each check below ties to a criterion with the artifact that proves it.

## originalIntent

Row 14 of `.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (line 229): "Add a K-quant GEMV parity row
(`iq::mv` -> `mv_kernel` -> `sub_dot`, Q2_K/Q4_K/Q5_K) to `ds4_parity`." The plan's acceptance
criteria (lines 242-251) require: the gate built and run with `STRATA_SYCL_DSV41=ON` and
`STRATA_SYCL_DSV41_EXE=ON`; `ds4_parity` and `ds4_blas_parity` named and passing, not "No tests
were found"; a negative control showing a corrupted expected value makes the row fail; guardrails
that the diff stays inside `sycl/` and no benchmark or `.omo/` commit lands; cleanup of the scratch
build dir. Row 14 exists because row 12's fix (`28bd2aaa` "subtract the K-quant min term in the
GEMV") shipped with no coverage — that commit's own message names the gap ("that coverage gap is a
finding for the plan").

## desiredOutcome

A SYCL developer merging this branch gets a `ds4_parity` row that checks the real K-quant GEMV dot
path on the card for all three K-quant types, with expected values derived from the repo's own
block-layout code rather than transcribed constants, so a regression in `sub_dot`'s min term fails
the gate loudly and names the type.

## userOutcomeReview

Satisfied. The merged file adds exactly one GEMV row per K-quant type to the existing harness, and
the harness on the card reports the three rows clean while an injected Q4_K reference error fails
the same harness naming `Q4_K gemv`. The outcome the user asked for — coverage of the dot path that
row 12 fixed — is present and load-bearing.

## checks against primary evidence

### 1. One K-quant GEMV row per type, driving the real dot path — PASS

Artifact: `sycl/src/dsv41/ds4_parity.cpp`, blob `ec63357cc4fd`, identical at `675e3375`,
`4e003d34`, and current branch HEAD (`d337362b`). Row added at lines 1286-1299:

```
for (int type : {strata::ds4::gguf::iq::kQ2_K, strata::ds4::gguf::iq::kQ4_K,
                 strata::ds4::gguf::iq::kQ5_K}) {
  ...
  bool launched = strata::ds4::gguf::iq::mv(queue, w.data(), n, row_bytes(type, k),
                                            xq.data(), y.data(), n, ncol, job);
  tally_f32(name(type) + " gemv", y.data(), yref.data(), n * ncol, exact, near, bad);
}
```

Real path confirmed by reading the callee chain, not by the row's own comment:
- `iq::mv` dispatch: `sycl/src/dsv41/ds4_iq.dp.cpp:2334-2355` — `kQ5_K` explicit, `kQ4_K`/`kQ2_K`
  through `DS4_IQ_SWITCH` (macro at line 2216 includes both) into `mv_nc` -> `mv_kernel`.
- `mv_kernel` dot: `ds4_iq.dp.cpp:1294-1297` calls `sub_dot(s, xp[0], xp[1])`; output write at
  line 1323 `y[c*ycs + row] = alpha*v + bias[row]`, matching the row's `job` fields
  (`alpha=1.0f`, `yscale=nullptr`, non-null `bias`, `ycs=n`).
- `Ty<kQ2_K>::MIN` and `Ty<kQ4_K>::MIN` are `1` (`ds4_iq.dp.cpp:123-125`), so the min term runs.
- `is_tq` is false for these types (`tq_kbits`, `ds4_iq.dp.hpp:23-32`), so the row does not fall
  into the trellis path.

Expected values derived from repo layout, not hand-transcribed constants. `ref_k_sub_dot`
(`ds4_parity.cpp:42-66`) decodes the block itself and applies the documented form
`d*(s0*lo + s1*hi) - (m0*sl + m1*sh)`, which matches the CUDA reference
`src/dsv41/ds4_iq.cu:353-373` and the SYCL kernel `ds4_iq.dp.cpp:391-425` line for line. Byte
offsets match the device decoder `decode_b` (`ds4_iq.dp.cpp:299-380`) and the vendored block
structs `sycl/third_party/ggml/ggml-common.h:272-284` (`block_q2_K`: `scales[16]`+`qs[64]`+`dm` =
84 bytes, static_asserted at `ds4_parity.cpp:145`), `:330-340` (`block_q4_K`: `dm`+`scales[12]`+
`qs[128]` = 144), `:342-353` (`block_q5_K`: `dm`+`scales[12]`+`qh[32]`+`qs[128]` = 170).
`get_scale_min_k4` in the harness mirrors the device helper `q4k_scale` (`ds4_iq.dp.cpp:382-389`).
Sizes come from production helpers `gguf::iq::block_bytes`/`row_bytes`, and the row body contains
no literal block-size or expected-value constants (grep for `[0-9]{3,}` over lines 822-935 returns
only source-path comments, RNG seed, and `n/k/ncol`).

### 2. Build and gate on the merged branch — PASS

Configure from a pinned detached checkout of `4e003d34` (`cmake -S <src>/sycl -B <build> -G Ninja
-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON
-DSTRATA_SYCL_DSV41_EXE=ON`):

```
SYCL_CONFIGURE_RC=0
```

Full build, 259 targets, zero `FAILED:` lines, no `ninja: build stopped`:

```
SYCL_BUILD_RC=0
```

Gate:

```
SYCL_CTEST_RC=0
```
```
Test project /dev/shm/r14gate_build
    Start 1: ds4_parity
1/2 Test #1: ds4_parity .......................   Passed    0.28 sec
    Start 2: ds4_blas_parity ...................   Passed    0.33 sec
100% tests passed, 0 tests failed out of 2
```
(verbatim log: `/dev/shm/r14ev/ctest_root.log`; both test names appear, not "No tests were found".)

The binary run directly on the card exits 0 and reports the three new rows clean
(`/dev/shm/r14ev/parity_direct.log`, `PARITY_DIRECT_RC=0`):
```
Q2_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
Q4_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
Q5_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
```

Deviations from the brief's literal command, both recorded rather than hidden:
- Scratch dir `/tmp/r14v_build` was deleted by a concurrent lane mid-run (observed: ninja cwd
  `(deleted)`; a `ctest` log at that path written by a process I did not start). A second unique
  path `/tmp/r14gate_build` was also deleted by a concurrent cleaner. The authoritative run used
  `/dev/shm/r14gate_build` from a detached checkout pinned at `4e003d34`; the reviewed blob is
  byte-identical to the merged commit.
- `ctest --test-dir <build>/sycl` cannot work with `-S sycl`: that makes `sycl` the top-level
  source dir, so `CTestTestfile.cmake` lands at the build root. Ran
  `ctest --test-dir /dev/shm/r14gate_build -R ds4 --output-on-failure`. Test selection and the
  pass/fail names are unchanged.
- `-j 4` and `TMPDIR=/dev/shm/r14tmp` were needed because four concurrent lane builds filled
  `/tmp` (`sycl-post-link`/`clang-offload-bundler` "No space left on device" on unrelated targets
  in earlier attempts). The clean run had zero failures.

### 3. Negative control — PASS (reproduced independently, stronger than the lane's)

Corruption applied in the pinned checkout: one line after the Q4_K/Q5_K scale decode in
`ref_k_sub_dot`, `if (type == ...kQ4_K) ++sc;` — Q4_K-only, host-side reference only, device path
untouched. Rebuilt, ran the gate (`/dev/shm/r14ev/ctest_neg.log`):

```
1/2 Test #1: ds4_parity .......................***Failed    0.90 sec
Q2_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
Q4_K gemv: 0/51 exact, 0 within one BF16 ulp, 51 outside
Q5_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
NO MATCH: 51 values outside one BF16 ulp of the reference
2/2 Test #2: ds4_blas_parity ..................   Passed    1.57 sec
50% tests passed, 1 tests failed out of 2
```

The row fails naming `Q4_K gemv`, Q2_K and Q5_K stay clean, and `ds4_blas_parity` still passes —
so the failure is attributable to the Q4_K row alone.

Restore byte-identity proven: `git -C <checkout> restore` returned 0, the file hash returned to
`ec63357cc4fd` (the merged blob), and `git status --porcelain` in the checkout was empty (0
entries). After restore, `ds4_parity` run directly exits 0 with all three rows clean
(`/dev/shm/r14ev/parity_restore_direct.log`), and the gate passes twice:
`RESTORE_CTEST_RC=0` (`/dev/shm/r14ev/ctest_restore_manual.log`) and the session's own
`/dev/shm/r14ev/ctest_restore.log` (both tests Passed).

The lane's reported control (`task-7-dsv41-sycl-dpct-and-xpu-test.log:335-342`) perturbed one
expected value by 5% and got "50 within / 1 outside"; that is a valid sensitivity check but a
narrow one. My control moves all 51 Q4_K outputs outside, which is the stronger statement of the
same criterion. The lane's `NEG_RC=8` numeric code is not independently confirmed here; the
criterion — the row fails and names Q4_K — is confirmed from `ctest_neg.log`.

### 4. Guardrails — PASS

- `git diff --name-only 18a27e02..HEAD`: 0 paths outside `sycl/`; 0 paths matching `.omo` or
  `bench/results`. (Branch HEAD advanced during review to `d337362b`, 12 commits since base; row 14
  is unchanged at every HEAD — `git diff --stat 4e003d34..HEAD -- sycl/src/dsv41/ds4_parity.cpp`
  is empty.)
- Row-14 commit `4e003d34` touches one file, +23 lines: `sycl/src/dsv41/ds4_parity.cpp`. Message
  prefix `[Engine]`; all 12 commits since base are `[Engine]` or `[Docs]`.
- No barrier-scope conversion (`fence_space::local_space`: 0 occurrences in the commit), no
  occupancy rewrite (`reqd_work_group_size`/`reqd_sub_group_size`/`occupancy`: 0), no CMakeLists
  change, no `ggml_sycl` target in the build tree (0 occurrences in build log and build dir).
- Main repo tracked dirt: 0 (`git status --porcelain` shows only pre-existing untracked `.omo/`,
  `bench/results/`, `docs/paper/`).

### 5. Cleanup — PASS

`bun -e "const fs=require('node:fs'); ..."`:
```
/tmp/r14v_build existsSync=false
/tmp/r14gate_build existsSync=false
/tmp/r14v_src existsSync=false
/dev/shm/r14gate_build existsSync=false
/dev/shm/r14tmp existsSync=false
```
The pinned worktree was removed with `git worktree remove --force` (RC=0) and pruned; the repo's
worktree list keeps only lanes owned elsewhere.

## remove-ai-slops / programming pass (applied directly; skills unavailable)

`remove-ai-slops` and `programming` are not installed in this session (skill roots contain only
`bun-1-4` and `gpt-image-gen`), so their criteria were applied from this prompt's documented list
directly over the diff, the harness code, and the production code it touches:

- **Excessive or useless tests**: none. 3 rows x 51 outputs = 153 comparisons against an existing
  harness that already runs tens of thousands; no new test binary, no new `add_test`.
- **Deletion-only tests / tests that merely verify a requested removal**: none. Nothing was
  removed.
- **Tautological tests**: none. The reference is computed host-side from block bytes and activations
  (`ref_k_sub_dot`, `fill_block`); it never calls `mv`, `mv_kernel`, or `sub_dot`, so agreement is
  not self-referential.
- **Implementation-mirroring tests**: `ref_k_sub_dot` reproduces `decode_b`'s byte arithmetic, which
  is the established pattern in this file (`ref_q4_K`/`ref_q5_K` at lines 66-92 do the same for the
  dequant path). It mirrors the *layout*, not the kernel's control flow; the negative control shows
  it can disagree with the device and fail.
- **Unnecessary production extraction / parsing / normalization**: none. The commit changes only the
  parity harness; no production file, no new helper in a shipped path, no new parsing.
- **Maintenance burden / false confidence / scope drift**: none at blocker level. Notes below.

Coverage note required by the brief: there is **no row-14 code review report artifact** in
`.omo/evidence/dsv41-sycl-dpct-and-xpu-test/` (no `task-14` log, no code-review file, no row-14
section in the manual QA matrix). So no report shows a skill-perspective or overfit/slop check for
this row; the direct check above is the only one. Recorded as an evidence gap, not a blocker — row
14's stated criteria do not require that artifact.

## notes (not blockers)

1. **No row-14 executor/QA/code-review artifact.** The lane's report is appended to
   `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-7-dsv41-sycl-dpct-and-xpu-test.log:300-389`;
   its numbers (`NEG_RC=8`, `RESTORE_RC=0`) are prose with no committed artifact behind them. All
   criteria were reproduced independently instead.
2. **All three new rows report 0 exact matches** (51 within one BF16 ulp each). That is the
   harness's documented rule for F32 GEMV whose activations are BF16-rounded
   (`ds4_parity.cpp:408-421`); a dropped min term moves outputs far beyond one ulp, which is what
   the negative control shows. Not a false-confidence risk at this tolerance.
3. **Device-side negative control not run.** Reverting row 12's fix (`28bd2aaa`,
   `ds4_iq.dp.cpp:418-421`) to show the three rows fail on a device regression would be a stronger
   guard, but it needs another full rebuild under heavy host contention. The host-side control
   already proves the comparison is load-bearing.
4. **Restore rebuild interrupted by my cleanup** at `[29/54]` (`ninja: build stopped: interrupted
   by user`) after the `ds4` binaries were relinked; the post-restore gate passed on both runs. The
   clean full-build `SYCL_BUILD_RC=0` was measured before the corruption.
5. **Host contention**: concurrent lanes filled `/tmp` and deleted two of my scratch build dirs
   mid-run, causing transient failures on targets unrelated to row 14. Evidence above comes from
   the pinned `4e003d34` run in `/dev/shm`.
6. **Cosmetic**: the header comment's line rewrap (`ds4_parity.cpp:19-22`) reads awkwardly
   ("one line per type, per set-tile row and per K-quant GEMV type, then MATCH when every decoded").
   Text-only, no behavioural effect.

## checked artifact paths

- `/home/maxious/Strata/sycl/src/dsv41/ds4_parity.cpp` (blob `ec63357cc4fd` at `675e3375`,
  `4e003d34`, `d337362b`)
- `/home/maxious/Strata/src/dsv41/ds4_iq.cu:353-373` (CUDA `sub_dot`)
- `/home/maxious/Strata/sycl/src/dsv41/ds4_iq.dp.cpp:123-125`, `:179-380` (`decode_b`),
  `:382-389` (`q4k_scale`), `:391-425` (`sub_dot`), `:1262-1325` (`mv_kernel`),
  `:2216-2222` (`DS4_IQ_SWITCH`), `:2334-2355` (`mv` dispatch)
- `/home/maxious/Strata/sycl/src/dsv41/ds4_iq.dp.hpp:23-32` (`tq_kbits`), `:57-58` (`row_bytes`)
- `/home/maxious/Strata/sycl/third_party/ggml/ggml-common.h:272-284`, `:330-340`, `:342-353`
- `/home/maxious/Strata/sycl/CMakeLists.txt:275-286` (`add_test` ds4_parity / ds4_blas_parity)
- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md:229`, `:242-251`
- `/home/maxious/Strata/.omo/ulw-execute/ledger.jsonl:388` (row-14 entry, lane `st_01a178c2`,
  lane commit `675e3375`, merge `4e003d34`)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-7-dsv41-sycl-dpct-and-xpu-test.log:300-389` (lane report, treated as untrusted)
- `/dev/shm/r14ev/cfg.log`, `build.log`, `ctest_root.log`, `parity_direct.log`, `build_neg.log`,
  `ctest_neg.log`, `parity_restore_direct.log`, `ctest_restore_manual.log`, `ctest_restore.log`,
  `build_restore.log`, `neg.py` (kept as this review's evidence; the build dirs are deleted)

## exact evidence gaps

- No committed artifact backs the lane's `NEG_RC=8` numeric exit code; `ctest_neg.log` proves the
  failure and its `Q4_K gemv` naming, not the code.
- No row-14 code-review report or manual-QA matrix exists, so the brief's "code review report
  shows the same skill-perspective check" cannot be confirmed for this row; the direct overfit/slop
  pass above is the only coverage.
- No device-side (revert row 12's fix) negative control was run.
