# Gate Review: Row 16 (attempt d) - audit note at every remaining dpct advisory block

Reviewer: omo-native-gate-reviewer (st_01a1791a)
Date: 2026-10-27
Commit under review: ce6218ee2642eaaba29be0735c4a8e45c27d11f2 (branch sycl/dsv41-sycl-support)
Build identity: working tree HEAD d337362b, which contains ce6218ee (verified `git merge-base --is-ancestor ce6218ee HEAD` -> 0)
Hardware: Intel Arc Pro B60, device 0, SYCL_BE=PI_LEVEL_ZERO, JIT (no AOT flag), icx/icpx from oneapi 2026.2

## Recommendation

APPROVE

## RC values (un-piped)

SYCL_BUILD_RC=0
SYCL_CTEST_RC=0
RUN_RC=0
CLEANUP_RC=0 (rm -rf /tmp/r16v6_build; existsSync === false, see below)

Literal-path RCs, recorded because the row's command paths do not match this configure:
SYCL_CTEST_RC_LITERAL=1 ("Failed to change working directory to /tmp/r16v6_build/sycl: No such file or directory")
RUN_RC_LITERAL=127 ("bash: line 1: /tmp/r16v6_build/sycl/strata-dsv41: No such file or directory")

VERDICT confirmed

## OriginalIntent

Row 16 of `.omo/plans/dsv41-sycl-dpct-and-xpu-test.md`: add a one-line accepted note (class + reason) to
every remaining dpct advisory block in `sycl/src/dsv41`, closing the IS-1 shortfall that F4's fidelity audit
recorded (zero inline notes at DPCT1093's 107 blocks at `28bd2aaa`).

## DesiredOutcome

Every dpct advisory block in `sycl/src/dsv41` carries an adjacent one-line note naming its DPCT class and the
acceptance reason, with no code behavior change, and the change still builds, passes the ds4 tests, and runs on
the card. The previous attempt (`row16b-gate-review.md`) REJECTED on exactly one gap: build, ctest and the card
run were never executed.

## UserOutcomeReview

MET. The shipped artifact satisfies row 16's substance criterion and the executed gates that attempt b could not
run. From the user's side: a reader opening any dpct advisory block in the port now finds, directly after it, a
line stating which advisory it is and why the port accepts it, and the engine still builds, still passes both ds4
parity tests, and still answers the gate prompt on one card. The only mismatch is the row's command paths
(`/tmp/r16v6_build/sycl/...`), which do not exist under `cmake -S sycl -B <build>`; the equivalent invocations at
the real build root pass, and row 14's gate used the same build-root form.

## Executed gates

### 1. SYCL_BUILD_RC=0

Command line as run (incremental re-run after the full configure+build of the same target dir):

SYCL_BUILD_RC=0
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; TMPDIR=/home/maxious/tmpbuild cmake --build /tmp/r16v6_build'

The full command 1 (configure + build) ran first and is logged at `/tmp/r16v6_build.log`:
`SYCL_BUILD_RC=0`, `grep -c FAILED` = 0, last target `[53/53] Linking CXX executable strata`.
The re-run is logged at `/tmp/r16v6_build3.log`: `SYCL_BUILD_RC=0`, FAILED lines = 0, last target `[53/53]`.
All targets reached: 53 executables present at the build root, including `ds4_parity` (16 MB),
`ds4_blas_parity` (664 KB) and `strata-dsv41` (37,892,880 bytes).

Note: this tree relinks all 53 targets on every `cmake --build` invocation (the configure step marks the link
targets dirty; it is environmental, not caused by the comment-only change). No compile or link FAILED line
appeared in either run.

### 2. SYCL_CTEST_RC=0

SYCL_CTEST_RC=0
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ctest --test-dir /tmp/r16v6_build -R ds4 --output-on-failure'

Output (`/tmp/r16v6_ctest.log`), naming both required tests:

    Test project /tmp/r16v6_build
        Start 1: ds4_parity
    1/2 Test #1: ds4_parity .......................   Passed    0.28 sec
        Start 2: ds4_blas_parity
    2/2 Test #2: ds4_blas_parity ..................   Passed    0.34 sec

    100% tests passed, 0 tests failed out of 2

The row's literal form (`--test-dir /tmp/r16v6_build/sycl`) exits 1 on a nonexistent directory only; `enable_testing()`
and the two `add_test` calls belong to the `sycl` source directory, so with `-S sycl` the test dir is the build root.
Row 14's gate used the same build-root form (`cd /tmp/r14v_build && ctest -R ds4`,
`.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row14-gate-review.md` lines 78-96).

### 3. RUN_RC=0

Gate card free before the run: `pgrep -c strata-dsv41` = 0.

RUN_RC=0
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO /tmp/r16v6_build/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 --gpus 0 --ids @/tmp/q_france.ids --n 8 --max-context 512'

Prompt line quoted verbatim from `/tmp/r16v6_run.log`:

    prompt: 10 tokens in 268 ms (37.32 tok/s, windows of 8192)

Emitted ids: `671,6102,294,8760,344,2619,51119,42499` - the same sequence the manual QA decoded as
"The capital of France is **Paris**." (`artifacts/decoded-answers.txt`). CPU lane 105 of 1680 experts,
decode `8 tokens in 2391 ms (2.93 tok/s)`. One card (`--gpus 0`); no GPU reset reported in the log.

### 4. Cleanup

`rm -rf /tmp/r16v6_build` -> `CLEANUP_RC=0`, and `ls -d /tmp/r16v6_build` ->
"ls: cannot access '/tmp/r16v6_build': No such file or directory".
Proof with `node:fs` (this host has no `node`; `bun -e "const {existsSync}=require('node:fs'); ..."` -> `PROOF_RC=0`):

    existsSync === false

## Static claims reproduced (not re-run beyond direct reads)

- Comment-only shape: `git show ce6218ee --numstat` = 756 insertions / 0 deletions across 12 files under
  `sycl/src/dsv41`; every added line matches `// audit note:` (non-audit-note added lines: 0).
- Class + reason: all 756 added notes carry a DPCT id and a reason clause (`accepted — <reason>`); zero added
  notes without a reason.
- Reasons match the per-class record: the DPCT1065 and DPCT1093 reason text in the notes matches
  `sycl/docs/sycl-dsv41-dpct-triage.md` rows 43 and 44 verbatim.
- Placement: each note sits immediately after the advisory block whose class it names (verified in
  `ds4_model.dp.cpp`: `/* DPCT1001:31 */`, `/* DPCT1000:32 */`, `/* DPCT1009:33 */`, `/* DPCT1093:20 */`,
  `/* DPCT1093:21 */` each followed by a note naming that class).
- Coverage reproduced independently at HEAD with block parsing over `sycl/src/dsv41/*.cpp`: 751 advisory blocks,
  743 carrying the `audit note:` form, 8 carrying a prose note written by earlier todos (todos 2, 3, 4 and row 14),
  0 blocks with no note. Attempt b's script reported 739 blocks / 0 missing; the count differs from its parser, the
  conclusion (no unnoted block) agrees.

## remove-ai-slops overfit/slop pass (run directly, not inherited)

- Excessive or useless tests: none. The commit adds zero test files and zero test lines
  (`git show ce6218ee --name-only` lists only the 12 source files; test/parity path matches: 0).
- Deletion-only tests / tests that merely verify a requested removal: none; nothing was removed.
- Tautological or implementation-mirroring tests: none added.
- Unnecessary production extraction, parsing or normalization: none. The diff adds comments only; no function,
  type, constant, build flag or default appears (`cmake --build` target list unchanged at 53).
- Scope drift: none inside the commit. 756 notes for 751 blocks is the requested shape (one note per block; some
  notes name a second related class, e.g. DPCT1025 alongside DPCT1093 at queue creation).
- Repeated identical reason text (155 DPCT1065 notes, 110 DPCT1093 notes) is inherent to a per-block note and is
  what row 16 asks for; the per-class record in `sycl/docs/sycl-dsv41-dpct-triage.md` still carries the same text.

## programming criteria check (run directly)

- No maintenance burden added: comments cannot change behavior; 0 deletions means no code path moved, renamed or
  re-indented.
- No false confidence: the notes state a reason and, where a follow-up owns the fix, name it (DPCT1093 -> todo 4),
  rather than asserting a fix that does not exist.
- No new abstraction or default appeared; `sycl/CMakeLists.txt` is untouched by ce6218ee.

## Code review report coverage

Row 16 has no separate code-review report artifact in `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/`; the only
prior report is `row16b-gate-review.md` (a gate review, not a code review), which does not show a skill-perspective
or overfit/slop criterion. The two skill checks above were therefore run directly on the diff, tests and production
code rather than inherited from a report. Recorded as an evidence gap, not a blocker: row 16's acceptance criteria
do not require a code-review artifact.

## Blockers

None. Every criterion in row 16 and in this gate's instructions is satisfied with cited evidence above.

## Notes (not blockers)

1. Row 16's literal command paths (`/tmp/r16v6_build/sycl/...`) do not exist under `cmake -S sycl -B <build>`;
   the runbook should say `<build>` and `<build>/strata-dsv41`. F3's runbook gap already records the `<build>/sycl`
   assumption. Evidence: `/tmp/r16v6_ctest_literal.log`, `/tmp/r16v6_run_literal.log`.
2. DPCT1093 is labelled ACTIONABLE in the triage doc while the inline notes say "accepted"; the note text names
   todo 4 as the owner of the device-selection check, so the disposition is recorded honestly. The row's criterion
   is class + reason, which holds.
3. Eight advisory blocks carry earlier todos' prose notes instead of the `audit note:` wording. No block is unnoted.
4. `/home/maxious/tmpbuild` (TMPDIR for command 1) and the gate logs `/tmp/r16v6_build.log`,
   `/tmp/r16v6_build3.log`, `/tmp/r16v6_ctest.log`, `/tmp/r16v6_ctest_literal.log`, `/tmp/r16v6_run.log`,
   `/tmp/r16v6_run_literal.log` remain; row F5 owns scratch cleanup.
5. A concurrent unrelated build of `/dev/shm/r14gate_build` was live during these runs (different directory); it
   did not touch this build or the card run.
6. Pre-existing, outside this row: the loud-spin control exits 0 instead of 1 (`SPIN_RUN_RC` in
   `artifacts/SUMMARY.txt`), and `isa_guard_check` fails on `iq512x_rows` (row 17, needs scope sign-off).

## Checked artifact paths

- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (row 16 line 231, IS-1 row, F4 gap line 239)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/row16b-gate-review.md` (prior REJECT and its blocker)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f4-fidelity.md` (IS-1 shortfall that row 16 closes)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/dsv41-sycl-dpct-and-xpu-test-manual-qa-two-card.md` (commit identity ce6218ee)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/artifacts/SUMMARY.txt`, `artifacts/decoded-answers.txt`
- `/home/maxious/Strata/sycl/docs/sycl-dsv41-dpct-triage.md` (per-class reasons, rows 43-44)
- `/home/maxious/Strata/sycl/src/dsv41/**` (751 advisory blocks, 0 unnoted, at HEAD)
- `/tmp/r16v6_build.log`, `/tmp/r16v6_build3.log`, `/tmp/r16v6_ctest.log`, `/tmp/r16v6_ctest_literal.log`,
  `/tmp/r16v6_run.log`, `/tmp/r16v6_run_literal.log`
- `/tmp/q_france.ids` (51 bytes), pack `/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4`
- `/tmp/r16v6_build` (removed; existsSync === false)

## Evidence gaps

1. No row-16 code-review report exists to confirm a skill-perspective / overfit-slop check; the checks were run
   directly instead (see above).
2. The row's literal invocation paths are wrong for this configure; approval rests on the equivalent build-root
   invocations, which are the same commands row 14's gate used.
3. AOT (`-DSTRATA_SYCL_AOT=bmg-g21`) was not rebuilt for this row; it is row 18's scope and was already closed there.
4. No GPU-reset journalctl check was run for this single-card run; the run log shows no reset and F3's matrix covers
   resets for the same commit.
