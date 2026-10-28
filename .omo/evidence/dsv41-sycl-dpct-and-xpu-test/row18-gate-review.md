# Row 18 / F2 gate review - AOT variant compile + ds4 parity rows vs todo 1 baseline

Date: 2026-10-26. Branch: sycl/dsv41-sycl-support. Reviewer: independent gate review (st_01a1793c).
Source state measured: d337362b8a9fa09402fd41e2636cbd7beae66161 (the branch tip when this review started).
Build tree: /tmp/r18v_build (clean), configured and built with TMPDIR=/home/maxious/tmpbuild. No product file
was edited; this review only reads the tree and writes this file.

Note on the tip: the branch gained two commits while this review ran - 74110114 `[Docs] dsv41 SYCL: correct the
pack identity and the test claim in the measured report` (sycl/docs/sycl-dsv41-measured-report.md) and a53335fd
`[Docs] dsv41 SYCL: tag every claim as measured, static, prior art, or not measured yet`
(docs/SYCL_DSV41_PLAN.md, docs/SYCL_DSV41_PLAN2.md). `git diff --name-only d337362b..HEAD` lists exactly those
three files, so every build and run below still describes the branch's code; the step-5 checks were re-run at
the new tip and both results are recorded.

## What is being verified

Plan `.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` row 18 (line 233) and F2 (line 237) are marked `[x]` on this
claim: "the AOT variant compiles (`-DSTRATA_SYCL_AOT=bmg-g21`) and the parity rows diff byte-identically against
todo 1's baseline (`PARITY_DIFF_RC=0`, both row files 12152 bytes at identical SHA256 3f026239...)". The
plan-compliance audit (`f1-plan-compliance.md`) flagged both rows because their only independent verdict was
REJECT (`.omo/evidence/dsv41-sycl-dpct-and-xpu-test-f2-gate-review.md`, reviewer st_01a1788c: "AOT never
compiled at review time; parity rows never diffed") and the gap closure was then accepted from the executor
lanes' own `DONECLAIM f2gaps` receipts (`task-2-dsv41-sycl-dpct-and-xpu-test.log` L128-182) without an
independent re-review. This review re-verifies the closure itself, un-piped, and ends with one verdict.

## Baseline provenance (todo 1's baseline rows)

Todo 1's baseline artifact is `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-1-dsv41-sycl-dpct-and-xpu-test.log`
- the artifact the plan names for todo 1 and the only todo-1 artifact in the directory (there is no standalone
baseline-rows file there; this resolution is recorded as an assumption). Todo 1 built in a worktree off 18a27e02
and captured six harness invocations at log lines 57006-57203. The baseline rows are the lines strictly between
each block's `----- OUTPUT -----` and `----- END OUTPUT -----` markers, concatenated in log order - the same
extraction rule the f2gaps receipt records. Re-extracted independently here:

```
6 blocks: ds4_parity --bad-scale (26 rows), --bad-scale-gu (26), ds4_blas_parity --corrupt (31),
--corrupt-scale (31), --corrupt-ldx (31), --selftest (30) = 175 rows
3f026239332fad4f2e6904d0ebbd50647e890d6b899ddcad61334de75018e8a9  /tmp/r18v_base_rows.txt   (12152 bytes)
```

That is exactly the SHA256 and size the DONECLAIM claims, so the receipts' baseline side is genuine. The
receipts' surviving run-side files were re-hashed too: `/tmp/dpct_run/f2_baseline_rows.txt` and
`/tmp/dpct_run/f2_parity_rows.txt` are both 12152 bytes at the same 3f026239... SHA256, and
`/tmp/dpct_run/f2_parity_diff.txt` is 0 bytes - internally consistent with `PARITY_DIFF_RC=0` as claimed.

## Gate 1 - the AOT variant compiles

AOT_CONFIGURE_RC=0
bash -c 'mkdir -p /home/maxious/tmpbuild; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; TMPDIR=/home/maxious/tmpbuild cmake -S sycl -B /tmp/r18v_build -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON -DSTRATA_SYCL_AOT=bmg-g21 > /tmp/r18v_configure.log 2>&1'

AOT_BUILD_RC=0
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; TMPDIR=/home/maxious/tmpbuild cmake --build /tmp/r18v_build > /tmp/r18v_build.log 2>&1'

(The one compound command in the step was split into configure and build so each has its own un-piped RC; flags
are identical. Output redirections to log files are captures, not pipes.)

The cache names the variant and the compiler:

```
CMAKE_CXX_COMPILER:STRING=/opt/intel/oneapi/compiler/2026.2/bin/icpx
STRATA_SYCL_AOT:STRING=bmg-g21
STRATA_SYCL_DSV41_EXE:BOOL=ON
```

The build reached `[260/260]` and its log carries no FAILED line:

GREP_FAILED_COUNT=0
grep -c FAILED /tmp/r18v_build.log

(grep prints 0 and exits 1 = "no matches"; the gate value is the count, 0. AOT device compilation is real in
this log - the ocloc `Compilation from IR` passes - so this is the `-DSTRATA_SYCL_AOT=bmg-g21` variant, not a
JIT build.)

## Gate 2 - the parity binary and its rows

PARITY_RC=0
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; /tmp/r18v_build/ds4_parity'

```
device: Intel(R) Arc(TM) Pro B60 Graphics
Q2_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
Q4_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
Q5_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
MATCH: 21744/59789 BF16 bits equal to the CPU reference, 38045 within one ulp, max float diff 4.66347
```

(28 rows total. The three K-quant GEMV rows are the coverage commit 4e003d34 added; they pass within one BF16
ulp. Todo 1's baseline predates that commit and contains no `Q*_K gemv` row - a fact that matters in Gate 3.)

## Gate 3 - the six closure-protocol rows vs todo 1's baseline rows

The closure's comparison is not the positive run (todo 1's artifact has none) but the six invocations whose
outputs form the baseline. All six were re-run un-piped from the AOT build and concatenated in baseline order:

NEG_BAD_SCALE_RC=1
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/r18v_build/ds4_parity --bad-scale > /tmp/r18v_rows_neg_bad_scale.txt 2>&1'

NEG_BAD_SCALE_GU_RC=1
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/r18v_build/ds4_parity --bad-scale-gu > /tmp/r18v_rows_neg_bad_scale_gu.txt 2>&1'

NEG_CORRUPT_RC=1
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/r18v_build/ds4_blas_parity --corrupt > /tmp/r18v_rows_neg_corrupt.txt 2>&1'

NEG_CORRUPT_SCALE_RC=1
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/r18v_build/ds4_blas_parity --corrupt-scale > /tmp/r18v_rows_neg_corrupt_scale.txt 2>&1'

NEG_CORRUPT_LDX_RC=1
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/r18v_build/ds4_blas_parity --corrupt-ldx > /tmp/r18v_rows_neg_corrupt_ldx.txt 2>&1'

BLAS_SELFTEST_RC=0
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/r18v_build/ds4_blas_parity --selftest > /tmp/r18v_rows_blas_selftest.txt 2>&1'

These six exit codes and the controls' verdict lines are identical to todo 1's baseline:
`NO MATCH: 27 / 29 / 63 / 192 / 256 values outside one BF16 ulp` for the five negative controls and
`ds4_blas_parity OK` for --selftest. The controls still fail loudly and by the same margins.

ROWS_CAT_RC=0
cat /tmp/r18v_rows_neg_bad_scale.txt /tmp/r18v_rows_neg_bad_scale_gu.txt /tmp/r18v_rows_neg_corrupt.txt /tmp/r18v_rows_neg_corrupt_scale.txt /tmp/r18v_rows_neg_corrupt_ldx.txt /tmp/r18v_rows_blas_selftest.txt > /tmp/r18v_new_rows.txt

DIFF_RC=1
diff /tmp/r18v_base_rows.txt /tmp/r18v_new_rows.txt

```
3f026239332fad4f2e6904d0ebbd50647e890d6b899ddcad61334de75018e8a9  /tmp/r18v_base_rows.txt   (12152 bytes)
2b8289d12487b17e59da84fc4920d04d67ed1abd4063010dad1419cffa7c07f5  /tmp/r18v_new_rows.txt    (12494 bytes)
```

The diff, verbatim (1460 bytes):

```
8c8
< MXFP4 set down tile: 729/4096 fp32 values exact, 3367 within one BF16 ulp, 0 outside
---
> MXFP4 set down tile: 742/4096 fp32 values exact, 3354 within one BF16 ulp, 0 outside
10,11c10,14
< IQ4_XS set gate/up tile: 8165/8192 fp16 bits exact, 27 within one BF16 ulp, 0 outside
< IQ4_XS set down tile: 932/8192 fp32 values exact, 7233 within one BF16 ulp, 27 outside
---
> IQ4_XS set gate/up tile: 8171/8192 fp16 bits exact, 21 within one BF16 ulp, 0 outside
> IQ4_XS set down tile: 884/8192 fp32 values exact, 7281 within one BF16 ulp, 27 outside
> Q2_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
> Q4_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
> Q5_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
34c37
< MXFP4 set down tile: 729/4096 fp32 values exact, 3367 within one BF16 ulp, 0 outside
---
> MXFP4 set down tile: 742/4096 fp32 values exact, 3354 within one BF16 ulp, 0 outside
36,37c39,43
< IQ4_XS set gate/up tile: 8135/8192 fp16 bits exact, 28 within one BF16 ulp, 29 outside
< IQ4_XS set down tile: 940/8192 fp32 values exact, 7252 within one BF16 ulp, 0 outside
---
> IQ4_XS set gate/up tile: 8141/8192 fp16 bits exact, 22 within one BF16 ulp, 29 outside
> IQ4_XS set down tile: 888/8192 fp32 values exact, 7304 within one BF16 ulp, 0 outside
> Q2_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
> Q4_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
> Q5_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
```

Decomposition: 169 of todo 1's 175 baseline rows are byte-identical in the AOT run. Six rows are not -
`MXFP4 set down tile` and `IQ4_XS set gate/up tile` / `IQ4_XS set down tile` in both ds4_parity blocks - and
six rows are added (the three `Q*_K gemv` rows per block). Every changed row keeps its "outside one BF16 ulp"
count identical to baseline (0, 0, 27, 0, 29, 0 respectively): the shifts are exact <-> within-one-ulp
reclassifications, not values leaving the tolerance band. The four ds4_blas_parity blocks are byte-identical.

### Provenance of the six changed rows

A surviving JIT build from the row-16 lanes (/tmp/r16v4_build, `STRATA_SYCL_AOT:STRING=`, icpx 2026.2, carries
the row-14 `Q*_K gemv` rows, i.e. built at or after 4e003d34) was run read-only for comparison. Its clean-input
tallies are todo 1's exactly (`MXFP4 set down tile: 729/4096 ... 3367`, `IQ4_XS set gate/up tile: 8165/8192
... 27`, `IQ4_XS set down tile: 940/8192 ... 7252`), and its six closure-protocol rows diff against the
baseline as pure additions:

JIT_DIFF_RC=1
diff /tmp/r18v_base_rows.txt /tmp/r18v_jit_rows.txt   (JIT rows: 12494 bytes; diff: 372 bytes)

```
11a12,14
> Q2_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
> Q4_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
> Q5_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
37a41,43
> Q2_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
> Q4_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
> Q5_K gemv: 0/51 exact, 51 within one BF16 ulp, 0 outside
```

So on the JIT build every one of todo 1's 175 baseline rows is byte-identical; only the row-14 rows are added.
The six tally shifts appear only in the AOT build and track the AOT device codegen (ocloc AOT for bmg-g21 vs
JIT runtime compilation) in the set-tile kernels - no range commit after the closure changes those kernels
(ce6218ee is notes-only per its own review, 83417a6f moves the hc/sparse_attn copies, ddc3c9e7 is the two-card
hand-off path, and the rows those touch are unchanged). This attribution is measured for the two build trees
named here; a same-commit JIT rebuild at d337362b was not run (a second full build, outside the prescribed
steps). The positive runs agree in shape: JIT `MATCH: 21777/59789 ... 38012 within one ulp, max float diff
4.66347` vs AOT `MATCH: 21744/59789 ... 38045 within one ulp, max float diff 4.66347` - same value count, same
max deviation, 33 values reclassified exact -> within-one-ulp.

## Gate 4 - ctest

SYCL_CTEST_RC=0
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ctest --test-dir /tmp/r18v_build -R ds4 --output-on-failure'

```
1/2 Test #1: ds4_parity .......................   Passed    0.19 sec
2/2 Test #2: ds4_blas_parity ..................   Passed    0.31 sec
100% tests passed, 0 tests failed out of 2
```

Both named suites pass. The literal plan gate path `build-sycl-dsv41/sycl` does not exist under
`cmake -S sycl`: enable_testing() and both add_test calls live in sycl/CMakeLists.txt, so with this configure
they register at the build root. The literal path fails:

LITERAL_CTEST_RC=1
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ctest --test-dir /tmp/r18v_build/sycl -R ds4 --output-on-failure > /tmp/r18v_literal_ctest.log 2>&1'

```
Failed to change working directory to "/tmp/r18v_build/sycl": No such file or directory
```

LS_SYCL_SUBDIR_RC=2 (`ls -d /tmp/r18v_build/sycl` - absent), LS_ROOT_CTEST_RC=0
(`/tmp/r18v_build/CTestTestfile.cmake` exists). The root path used in Gate 4 is the correct form of the gate.

## Gate 5 - the AOT commit check

GREP_AOT_RC=1
git log --oneline 18a27e02..HEAD | grep -i aot

No output: no commit in 18a27e02..HEAD (12 commits at d337362b; 14 at the current tip a53335fd - the same
GREP_AOT_RC=1 was re-run there) mentions AOT in its subject or body. This is the expected state, not a
missing deliverable: row 18's AOT work is measurement-only (compile the variant and diff rows), the plan's
commit strategy gives measurement rows no commit, and `STRATA_SYCL_AOT` predates 18a27e02 (last changed by
pre-range commits 8e957cd5 / 9a01b6e3). There is therefore no AOT commit whose `git show --name-only` needs
checking; the scope half was run over every commit that does exist in the range instead:

- at d337362b, `git show --name-only --format=` for all 12 range commits lists only `sycl/` paths
  (1-12 files each; 27 distinct files in `git diff --name-only 18a27e02..HEAD`, all `sycl/`);
- the two commits that landed during this review are the [Docs] ones named above, touching
  sycl/docs/sycl-dsv41-measured-report.md and docs/SYCL_DSV41_PLAN.md + docs/SYCL_DSV41_PLAN2.md - no product
  code outside `sycl/` anywhere in the range.

## RC summary

| Gate | RC | Meaning |
| --- | --- | --- |
| AOT configure | AOT_CONFIGURE_RC=0 | cache names STRATA_SYCL_AOT=bmg-g21, icpx 2026.2 |
| AOT build | AOT_BUILD_RC=0 | [260/260], grep -c FAILED = 0 |
| positive ds4_parity | PARITY_RC=0 | Q2_K/Q4_K/Q5_K gemv rows pass within one BF16 ulp |
| six closure invocations | NEG_BAD_SCALE_RC=1, NEG_BAD_SCALE_GU_RC=1, NEG_CORRUPT_RC=1, NEG_CORRUPT_SCALE_RC=1, NEG_CORRUPT_LDX_RC=1, BLAS_SELFTEST_RC=0 | identical to todo 1's baseline RCs and NO MATCH counts |
| rows diff (AOT) | DIFF_RC=1 | 6 baseline rows changed, 6 rows added (full diff above) |
| rows diff (JIT reference) | JIT_DIFF_RC=1 | additions only; 175/175 baseline rows byte-identical |
| ctest | SYCL_CTEST_RC=0 | 2/2, ds4_parity and ds4_blas_parity |
| literal ctest path | LITERAL_CTEST_RC=1 | build-sycl-dsv41/sycl does not exist under cmake -S sycl |
| AOT commit in range | GREP_AOT_RC=1 | none - row 18's AOT work is measurement-only |

## Findings

1. The AOT half of the closure is confirmed independently: the `-DSTRATA_SYCL_AOT=bmg-g21` variant configures
   and compiles at the reviewed source state (AOT_CONFIGURE_RC=0, AOT_BUILD_RC=0, [260/260], zero FAILED),
   with the cache naming the variant and the exe switch ON. The REJECT review's blocker "AOT never compiled at
   review time" no longer applies.
2. The byte-identity half is not true as written at HEAD. On the prescribed AOT build the six closure rows
   diff against todo 1's baseline with DIFF_RC=1: six added row-14 GEMV rows plus six set-tile tally rows that
   moved exact <-> within-one-BF16-ulp (outside counts identical). Full-set byte-identity against todo 1's
   baseline is in fact unattainable at HEAD by design: the harness gained the three `Q*_K gemv` rows per
   ds4_parity run in 4e003d34 (row 14), after todo 1's baseline was recorded.
3. No numerical regression is visible inside the one-BF16-ulp band, and the executor lanes' closure receipts
   are genuine: the baseline extraction reproduces their SHA256 byte-for-byte, their run-side files survive
   identical with a 0-byte diff, and a JIT build reproduces all 175 baseline rows byte-identically. The six
   tally shifts are attributable to AOT vs JIT device codegen in the set-tile kernels, not to a code change in
   any baseline row (JIT evidence above).
4. The negative controls remain loud and unchanged: same five non-zero RCs, same NO MATCH outside counts
   (27/29/63/192/256), same selftest OK, ctest 2/2.
5. Row 18's AOT work is measurement-only (GREP_AOT_RC=1; no AOT commit in the range), and the literal plan
   ctest path `build-sycl-dsv41/sycl` does not exist under `cmake -S sycl` (the tests register at the build
   root; the root-path gate passes).

To make rows 18/F2 honest as written, either re-scope the row text to the measured statement - the JIT rows
are byte-identical to todo 1's baseline modulo the row-14 GEMV rows, and the AOT build reclassifies six
set-tile tallies within one BF16 ulp with identical outside counts - or re-record a baseline at HEAD in the
AOT configuration and diff against that. The closure was real; the literal claim is stale.

## Cleanup

CLEANUP_RC=0
rm -rf /tmp/r18v_build

The node binary is not on PATH on this machine (PROBE_RC=0 lists only the bundled Bun runtime), so the
existsSync proof ran in this session's JS runtime against `require("node:fs")` (Node-API compatible):

```
require("node:fs").existsSync("/tmp/r18v_build") === false
```

Scratch row/log files created by this review under /tmp/r18v_* were removed (SCRATCH_CLEANUP_RC=0,
`ls /tmp/r18v_*` exits 2). The row-16 lanes' /tmp/r16v4_build and /tmp/r16v2_build were only read and were not
removed - they belong to earlier rows.

VERDICT FAIL
