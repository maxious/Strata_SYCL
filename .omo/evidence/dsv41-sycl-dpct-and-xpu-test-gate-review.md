# Gate Review: dsv41-sycl-dpct-and-xpu-test — todo 1 (baseline re-record)

- **recommendation:** APPROVE
- **blockers:** none
- **reviewer:** omo-native-gate-reviewer (st_01a1774e), final gate for todo 1
- **date:** 2026-10-26
- **target artifact:** `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-1-dsv41-sycl-dpct-and-xpu-test.log` (5.5 MB)
- **plan todo audited:** `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` todo 1

> This file supersedes the earlier plan-review receipt written by st_01a17718 to this
> same path (that review APPROVEd the plan document itself; this review is the
> todo-1 execution gate and re-verifies the lane's evidence log against the five gates).

## originalIntent

The user wants today's (2026-10-26) SYCL dsv41 baseline re-recorded before any correctness
work: build the opt-in SYCL exe, run the parity harnesses and their negative controls, run the
engine once on one Arc Pro B60, and capture today's prompt/decode lines, the CPU-lane share,
and the emitted ids — as a measurement, not a change (no product edits).

## desiredOutcome

A baseline evidence log that (1) is on branch `dsv41-dpct-w1` at HEAD
`18a27e02fd5d3b63dddf9a5a1863abaf5581f769` with a clean tree, (2) carries green
configure/build/ctest rows with 2/2 ds4 tests and five failing (non-zero) negative controls,
(3) preserves a named manual-QA run artifact quoting the prompt/decode/ids lines verbatim,
(4) shows the lane navigated the four adversarial traps (flag ownership, invalid `xpu-smi -m`,
piped exit codes, post-exit xpu-smi sampling), and (5) leaves a cleanup receipt with the build
tree removed. It must not claim throughput without a completed prompt, and must record the
degenerate 8x`129276` / `CPU lane: 0 of 1680 experts` output as the degenerate baseline it is,
not as a working answer.

## userOutcomeReview

The shipped log satisfies the outcome. Every acceptance-criterion RC is present and at the
required value, the five gates each PASS with an exact evidence line (below), the manual-QA
artifact exists on disk and its quoted lines match the artifact byte-for-byte, and the two
secondary integrity questions resolve cleanly: there is no throughput figure not backed by a
completed prompt, and the degenerate baseline is explicitly labelled degenerate. The worktree
still holds the claimed branch/HEAD with a clean status (independently reproduced, not just
read from the log).

## Gate results (all PASS)

### Gate 1 — Plan reread — PASS
- `GIT_BRANCH_RC=0` — log line 48; output `dsv41-dpct-w1` (lines 44–48).
- `GIT_HEAD_RC=0` — log line 55; output `18a27e02fd5d3b63dddf9a5a1863abaf5581f769` (lines 51–55).
- `STATUS_RC=0` — log line 61; `git status --short` output empty (lines 57–61).
- Independently reproduced against the live worktree (not the log):
  `/home/maxious/Strata-wt/dsv41-dpct-w1` → branch `dsv41-dpct-w1`, HEAD
  `18a27e02fd5d3b63dddf9a5a1863abaf5581f769`, `status --short` empty (rc 0).

### Gate 2 — Automated build/test — PASS
- `CMAKE_CONFIGURE_RC=0` — line 107.
- `CMAKE_BUILD_RC=0` — line 56985.
- `CTEST_RC=0` — line 57005; output `100% tests passed, 0 tests failed out of 2` (ds4_parity, ds4_blas_parity).
- `BLAS_SELFTEST_RC=0` — line 57204; output `MATCH: 50937/51242 ...` and `ds4_blas_parity OK`.
- Negative controls (all non-zero, each preceded by its own `NO MATCH: N values outside`):
  - `NEG_BAD_SCALE_RC=1` — line 57035 (`ds4_parity --bad-scale`, `NO MATCH: 27`).
  - `NEG_BAD_SCALE_GU_RC=1` — line 57065 (`ds4_parity --bad-scale-gu`, `NO MATCH: 29`).
  - `NEG_CORRUPT_RC=1` — line 57100 (`ds4_blas_parity --corrupt`, `NO MATCH: 63`).
  - `NEG_CORRUPT_SCALE_RC=1` — line 57135 (`ds4_blas_parity --corrupt-scale`, `NO MATCH: 192`).
  - `NEG_CORRUPT_LDX_RC=1` — line 57170 (`ds4_blas_parity --corrupt-ldx`, `NO MATCH: 256`).
- `GREP_RESET_MARKERS_RC=1` — line 57258; command `grep -ci 'reset' /tmp/dpct_run/run1.log`
  (line 57254) printed count `0`; grep's exit 1 for zero matches is the PASS value here.

### Gate 3 — Named Manual-QA artifact — PASS
- `/tmp/dpct_run/run1.log` exists on disk (2041 bytes, verified) and its full content is also
  inlined in the log via `cat` (lines 57260–57278).
- `MANUAL-QA-CHANNEL` block (lines 57281–57289) quotes verbatim:
  - prompt line: `prompt: 1 tokens in 353 ms (2.83 tok/s, windows of 8192)` (matches run1.log exactly).
  - decode line: `decode: 8 tokens in 376 ms (18.61 tok/s); CPU lane: 0 of 1680 experts; the cache moved 0 (0 in all)` (matches exactly).
  - emitted ids line: `129276,129276,129276,129276,129276,129276,129276,129276,` — this is the
    ids portion of the physical line, which in run1.log continues with `dspark: ...`; the lane
    discloses this ("the same physical line continues with the dspark stats because the stdout
    newline was still buffered at process exit"). Accurate, not a misquote.
- `RUN_RC=0` — line 57225 (un-piped; engine output redirected to run1.log).

### Gate 4 — Adversarial classes probed — PASS
- (a) Flag-ownership trap — risk note line 57523: `--bad-scale / --bad-scale-gu belong to
  ds4_parity (which silently ignores unknown args), --corrupt / --corrupt-scale / --corrupt-ldx
  to ds4_blas_parity. All five controls were run against their real binaries and are non-zero`.
  Verified against source: `ds4_blas_parity.cpp` `main` parses `--corrupt/--corrupt-scale/--corrupt-ldx`
  (lines ~227–231); `ds4_parity.cpp` header documents `--bad-scale / --bad-scale-gu` (lines 8–13).
  The five controls were indeed run across the two correct binaries (see Gate 2 lines).
- (b) `xpu-smi -m` invalid subcommand — `XPU_SMI_PROBE_RC=2` (line 56952) with the NOTE
  (lines 56987–56991) naming `xpu-smi v2.0 ... has no -m subcommand`, then the correct forms
  used: `xpu-smi --query-gpu=memory.used,memory.total --format=csv --id=0/1`
  (`XPU_SMI_MEM0_RC=0`, `XPU_SMI_MEM1_RC=0`, lines 57231/57237) and `xpu-smi --list-gpus`
  (`XPU_SMI_LIST_RC=0`, line 57243).
- (c) Piped-exit-code trap — log convention lines 19–21 ("the UN-PIPED exit code, captured as
  `$?` immediately after the command"; "No measured command is ever placed in a shell pipe;
  redirection (>) does not affect `$?`"); the three `grep -c` commands are run un-piped and
  their exit-1-for-zero-matches is recorded as the pass value (lines 57244–57258) with DoneClaim
  annotations `count 0 (zero matches)` / `count 0 (no xe reset markers)` / `count 0 (supplementary)`.
- (d) Post-exit xpu-smi readings labelled unverified — `MEM-TIMING-ANALYSIS` block (lines 57290–57297)
  and DoneClaim risk #1: "xpu-smi sampled after process exit. Its 254.578125 / 246.140625 MiB
  values are the idle cards. ... Whether xpu-smi would have shown the allocation in-window is
  unverified." The loaded-state memory is attributed to the run's own line
  (`weights 5.86 GB, experts 14.89 GB ... 596 MiB free`), not to the xpu-smi samples.

### Gate 5 — Cleanup receipt — PASS
- `RM_RC=0` — line 57306, for `rm -rf /tmp/dpct_w1_build` (line 57304).
- `LS_RC=2` — line 57311, for `ls /tmp/dpct_w1_build` (line 57309); output
  `ls: cannot access '/tmp/dpct_w1_build': No such file or directory`.
- Verified on disk: `/tmp/dpct_w1_build` absent (`ls` exits 2).

## Throughput-claim check

No throughput claim is present that is not backed by a completed prompt. The only tok/s figures
in the log are `2.83 tok/s` (prompt) and `18.61 tok/s` (decode), both from the completed run
whose `prompt:` line is present (run1.log lines 10/14; log lines 57269/57273 and their verbatim
re-quotes). No stale prior-run figures (`16.04`, `18.21`, `384 ms`, `62 ms`) appear anywhere in
the log. No throughput number is published from a prompt-less run.

## Degenerate-baseline honesty

Recorded honestly. DoneClaim risk #2 (line 57521): "The 8 decoded tokens all carry the same id
129276 (degenerate output for this one-token prompt). The ids are captured exactly as printed."
The log header (line 1) states the todo is a "baseline re-record (measurement only, no product
changes)"; the quoted decode line carries `CPU lane: 0 of 1680 experts`; nothing presents the
8x`129276` output as a working answer.

## Skill-perspective check (remove-ai-slops / programming)

The `remove-ai-slops` and `programming` skills are not installed in this session (only
`bun-1-4` is present), so their criteria are applied directly:

- No diff and no new tests exist to slop-review: `changed_files` is empty in the DoneClaim,
  `git status --short` is empty at start (line 61) and end (line 57302 / reproduced live), and
  the negative controls are the pre-existing `ds4_parity`/`ds4_blas_parity` harnesses run
  as-is, not tests written by this lane. Therefore there are no excessive/tautological/
  implementation-mirroring/deletion-only tests and no added production extraction, parsing, or
  normalization to flag.
- No scope drift or maintenance burden: the worktree stayed clean and only the `.omo/evidence/`
  log was produced. The one non-command prose block (DoneClaim risks) adds no code.

## checked artifact paths

- `/home/maxious/Strata/.omo/plans/dsv41-sycl-dpct-and-xpu-test.md` (todo 1 acceptance text, full read)
- `/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-1-dsv41-sycl-dpct-and-xpu-test.log` (5,517,108 bytes; grepped for every `*_RC=` and read the gate-critical sections in full)
- `/tmp/dpct_run/run1.log` (exists; full read; byte-compared to the MANUAL-QA-CHANNEL quotes)
- `/tmp/dpct_w1_xpusmi_probe.txt` (exists; 2270 bytes; confirms `'-m' is not a valid subcommand` + usage text)
- `/tmp/dpct_w1_doneclaim.json` (exists; full read of the inline copy in the log)
- `/tmp/dpct_w1_build` (absent; `ls` exits 2, reproducing `LS_RC=2`)
- `/home/maxious/Strata-wt/dsv41-dpct-w1` (live worktree: branch/HEAD/status reproduced)
- `sycl/src/dsv41/ds4_blas_parity.cpp` (flag ownership, lines ~227–231) and
  `sycl/src/dsv41/ds4_parity.cpp` (header lines 8–13) — read to confirm the flag mapping

## exact evidence gaps

None. No success criterion requires an artifact that is missing: the build tree is
intentionally deleted (with a receipt), the run artifact and probe file are retained on disk,
and every RC the plan requires is present at its required value.

## Advisory notes (non-blocking)

1. The MANUAL-QA "emitted ids line" quote is the ids prefix of the physical line (which
   continues into `dspark:` stats on the same line). The lane discloses this in both the
   channel block and the DoneClaim; a strict "full physical line" reader should read it as the
   ids portion, which is what the gate asks for.
2. The NOTE at line 56991 says the `xpu-smi -m` block "recorded rc 2 and its usage text", but
   the inline OUTPUT block is empty because the command redirected to
   `/tmp/dpct_w1_xpusmi_probe.txt`; the usage text is in that referenced file, not inline.
   The rc 2 is inline, and the gate only requires catching the invalid subcommand and using the
   correct forms, so this is wording, not a gap.
3. The log demonstrates the piped-exit-code catch by convention plus correct grep-`-c`
   exit-code handling rather than by the literal phrase "piped-exit-code trap"; this satisfies
   the gate's intent.
