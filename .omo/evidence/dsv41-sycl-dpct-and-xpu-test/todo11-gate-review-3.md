# Todo 11 gate review 3 — final re-gate after the test-claim split

Date: 2026-10-26 (UTC). Branch `sycl/dsv41-sycl-support`.
HEAD `aa09dc0707f20d492b595bc42fc0f6fcfa421f1d` (four docs commits: `d337362b` measured report,
`74110114` pack-identity correction, `a53335fd` docs re-tag, `aa09dc07` test-claim split).
Reviewer: independent final re-gate of todo 11 after
`.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo11-gate-review-2.md`, whose only defect was report line 58's
unqualified default-suites-pass claim. No product file was edited, nothing was pushed, no monitor was used.
Every check below runs against primary evidence — the raw ctest artifacts under
`regression-gates-artifacts/`, `regression-gates.md`, the pack manifest `maya-ds4/tensors.txt`, `du -b` of the
pack's GGUF files, `lspci -nn`, the git object DB and this session's own JS kernel — not against lane prose.

Convention: every `*_RC` line below sits directly above the command whose exit code it is. Each value is the
un-piped exit code captured as `$?` immediately after its command (never through a pipe). Where a `grep -c`
check expects no match, the printed count is 0 and the exit code is 1; the line says so. GB figures are
decimal bytes/1e9 at two decimals, the report's own convention. The `node:fs` check runs in this session's
JS kernel (the `node` CLI is absent on this machine); kernel cells have no exit code, so its booleans are
printed verbatim.

## Check 1 — report line 58 against the regression gates

The corrected bullet (report lines 56-61):

REPORT_BLOCK_RC=0
CMD: sed -n '56,61p' sycl/docs/sycl-dsv41-measured-report.md

```
## The checks that must pass

- Passed: both ds4 parity tests, `ds4_parity` and `ds4_blas_parity`, pass on the Linux SYCL build (2 of
  2). Failed: the default build's test suite has one pre-existing failure, `isa_guard_check` on
  `iq512x_rows` (1 of 57). Not run: the CUDA and HIP builds were not run because no CUDA or HIP hardware
  is present on this machine.
```

STATES_GREP_RC=0
CMD: grep -n 'Passed:\|Failed:\|Not run:' sycl/docs/sycl-dsv41-measured-report.md

Line 58 carries `Passed:` (ds4_parity and ds4_blas_parity, Linux SYCL build, 2 of 2), line 59 carries
`Failed:` (default build's test suite, pre-existing `isa_guard_check`), line 60 carries `iq512x_rows`
(1 of 57) and `Not run:` (CUDA and HIP, no hardware). All three states the caller names are present and
match the required wording.

The old unqualified clause is gone (expected: count 0, exit 1):

OLD_CLAUSE_RC=1
CMD: grep -c 'The default test suites' sycl/docs/sycl-dsv41-measured-report.md

SUITES_PLURAL_RC=1 (count 0: the plural "suites" appears nowhere in the report)
CMD: grep -c 'suites' sycl/docs/sycl-dsv41-measured-report.md

Every remaining pass-wording in the report (enumerated to prove no unqualified suite-pass claim survives):

PASS_MENTIONS_RC=0
CMD: grep -niE 'suites?|pass' sycl/docs/sycl-dsv41-measured-report.md

Three hits only: `:56` the heading "The checks that must pass" (what the gates must do, not a claim that
anything passes), `:58` and `:59` the qualified three-state bullet itself (which names the two tests, the
build, 2 of 2, and immediately states the failure and the not-run states). No line claims that a suite
passes without qualification.

Cross-check against `regression-gates.md`:

DEF_RC_LINE_RC=0
CMD: grep -n 'DEF_CTEST_RC=8' .omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates.md

SYCL_RC_LINE_RC=0
CMD: grep -n 'SYCL_CTEST_RC=0' .omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates.md

DEF_SUMMARY_RC=0
CMD: grep -n 'and nothing else\|98% tests passed\|tests failed out of 57' .omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates.md

(`:109` `DEF_CTEST_RC=8`; `:118-119` "Failing tests (complete list): `55 - isa_guard_check (Failed)` and
nothing else. The run is `98% tests passed, 1 tests failed out of 57`".)

ISA_NAME_RC=0
CMD: grep -n 'isa_guard_check' .omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates.md

IQ512_RC=0
CMD: grep -n 'iq512x_rows' .omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates.md

(`:120-121` names the sole failure `isa_guard_check` on `iq512x_rows`, `iq_avx512.cpp.o`, 777 wide
instructions.)

SYCL_NAMES_RC=0
CMD: grep -n 'Test #1: ds4_parity\|Test #2: ds4_blas_parity\|100% tests passed, 0 tests failed out of 2' .omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates.md

(`:158` `SYCL_CTEST_RC=0`; `:164-165` "1/2 Test #1: ds4_parity ... Passed 0.27 sec", "2/2 Test #2:
ds4_blas_parity ... Passed 0.31 sec", "100% tests passed, 0 tests failed out of 2".)

Raw artifacts (independent of any prose):

ARTIFACTS_LS_RC=0
CMD: ls .omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates-artifacts/

DEF_FAILED_CAT_RC=0
CMD: cat .omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates-artifacts/def2-ctest-failed.txt

DEF_LOG_GREP_RC=0
CMD: grep -n 'tests passed\|tests failed\|isa_guard_check' .omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates-artifacts/def2-ctest.log

SYCL_LOG_GREP_RC=0
CMD: grep -n 'Test #\|tests passed\|tests failed' .omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates-artifacts/sycl-ctest.log

The raw logs agree with the report exactly: default suite `55 - isa_guard_check (Failed)` and nothing else,
`98% tests passed, 1 tests failed out of 57` = 1 of 57; SYCL `ctest -R ds4` shows `1/2 Test #1: ds4_parity
... Passed`, `2/2 Test #2: ds4_blas_parity ... Passed`, `100% tests passed, 0 tests failed out of 2` =
2 of 2. The "pre-existing" descriptor is backed by the row-17 record and the clean `origin/main` control:

PREEXIST_NOTE_RC=0
CMD: grep -n 'row 17\|pre-existing' .omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates.md

PREEXIST_CTL_FILES_RC=0
CMD: grep -rn 'CTL_CTEST_RC=8\|FAILDIFF_RC=0' .omo/evidence/ --include='*.md'

(`regression-gates.md:122,193` "open row 17, a `tools/` change outside this plan's scope";
`dsv41-sycl-dpct-and-xpu-test-todo10-gate-review.md:37,60-62` and `f4-fidelity.md:154-155` record the
identical failure on a clean `origin/main` `4b9e05a2` control, `CTL_CTEST_RC=8`, `FAILDIFF_RC=0`.)

Result: the correction holds. Report line 58 now separates passed (ds4 pair, Linux SYCL build, 2 of 2),
failed (default suite's pre-existing `isa_guard_check` on `iq512x_rows`, 1 of 57) and not run (CUDA, HIP,
no hardware), each state matching the primary regression evidence, and no unqualified suite-pass claim
remains anywhere in the report.

## Check 2 — removed strings and pack identity

Removed strings (expected: count 0, exit 1 for all six):

Q4K_QUANTISED_RC=1
CMD: grep -c 'Q4_K quantised' sycl/docs/sycl-dsv41-measured-report.md

GB115_RC=1
CMD: grep -c '115 GB' sycl/docs/sycl-dsv41-measured-report.md

PLAN_Q4K_QUANTISED_RC=1
CMD: grep -c 'Q4_K quantised' docs/SYCL_DSV41_PLAN.md

PLAN_GB115_RC=1
CMD: grep -c '115 GB' docs/SYCL_DSV41_PLAN.md

PLAN2_Q4K_QUANTISED_RC=1
CMD: grep -c 'Q4_K quantised' docs/SYCL_DSV41_PLAN2.md

PLAN2_GB115_RC=1
CMD: grep -c '115 GB' docs/SYCL_DSV41_PLAN2.md

My own census of the pack manifest (row format `t <name> <quant> <file> <offset> <bytes> <dims...>`, so
bytes = field 6):

CENSUS_RC=0
CMD: awk '$1=="t"{cnt[$3]++; bytes[$3]+=$6} END{for (q in bytes) printf "%s\t%d\t%d\t%.4f GB\n", q, cnt[q], bytes[q], bytes[q]/1e9}' /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/tensors.txt

| quant | tensors | bytes | GB |
| --- | --- | --- | --- |
| IQ1_M | 23040 | 59,454,259,200 | 59.4543 |
| IQ2_XXS | 16128 | 49,049,763,840 | 49.0498 |
| IQ2_XS | 6912 | 23,569,367,040 | 23.5694 |
| Q5_K | 524 | 5,329,223,680 | 5.3292 |
| IQ4_NL | 2 | 110,595,290,400 | 110.5953 |
| Q8_0 | 80 | 389,939,200 | 0.3899 |
| BF16 | 71 | 281,182,208 | 0.2812 |
| F32 | 489 | 159,322,560 | 0.1593 |

PERFILE_RC=0
CMD: awk '$1=="t"{fb[$4]+=$6; fc[$4]++} END{for (f in fb) printf "file %s\t%d tensors\t%d bytes\t%.4f GB\n", f, fc[f], fb[f], fb[f]/1e9}' /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/tensors.txt

Files 0-3 (main shards): 39.0699 + 39.3884 + 39.2469 + 20.5278 GB; file 4 (Engram) 110.5953 GB.

TOTAL_RC=0
CMD: awk '$1=="t"{tot+=$6; n++} END{printf "TOTAL %d tensors %d bytes %.4f GB\n", n, tot, tot/1e9}' /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/tensors.txt

TOTAL 47246 tensors 248,828,348,128 bytes 248.8283 GB.

TENSOR_Q4K_COUNT_RC=1 (count 0 = expected: zero Q4_K tensors)
CMD: grep -c ' Q4_K ' /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/tensors.txt

My own `du -b` of the pack's five GGUF files (they live beside `maya-ds4/`, not inside it):

DU_B_RC=0
CMD: du -b /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/*.gguf

DU_BC_RC=0
CMD: du -bc /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/*.gguf

DU_SHARDS_BC_RC=0
CMD: du -bc /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/*-of-00004.gguf

| file | bytes | GB |
| --- | --- | --- |
| ...IQ2_XXS-00001-of-00004.gguf | 39,075,714,624 | 39.0757 |
| ...IQ2_XXS-00002-of-00004.gguf | 39,388,449,504 | 39.3884 |
| ...IQ2_XXS-00003-of-00004.gguf | 39,246,891,616 | 39.2469 |
| ...IQ2_XXS-00004-of-00004.gguf | 20,527,837,312 | 20.5278 |
| ...Engram-IQ4_NL.gguf | 110,595,290,816 | 110.5953 |
| total (`du -bc`) | 248,834,183,872 | 248.8342 |
| main shards (`du -bc` of the four) | 138,238,893,056 | 138.2389 |

REPORT_IDENTITY_RC=0
CMD: grep -n 'IQ1_M' sycl/docs/sycl-dsv41-measured-report.md

Report line 12 reads: "IQ1_M 59.45 GB, IQ2_XXS 49.05 GB, IQ2_XS 23.57 GB, Q5_K 5.33 GB and an IQ4_NL
engram of 110.60 GB, with zero Q4_K tensors and 248.83 GB on disk in total (231.7 GiB; main shards
138.24 GB)." Every figure matches my own census and `du -b` at the report's two-decimal precision:
59.4543 -> 59.45, 49.0498 -> 49.05, 23.5694 -> 23.57, 5.3292 -> 5.33, 110.5953 -> 110.60, zero Q4_K
tensors, 248.8342 -> 248.83 GB total (248,834,183,872 B = 231.73 GiB), 138.2389 -> 138.24 GB main
shards. Both prior corrections still hold.

## Check 3 — docs tags, tok/s lines, hardware block, PLAN2 negation

TAG_RC=0 (count 52, the required >= 1)
CMD: grep -c 'measured on this PC' docs/SYCL_DSV41_PLAN.md

TOKS_LIST_RC=0
CMD: grep -nE '[0-9]+(\.[0-9]+)? ?tok/s' docs/SYCL_DSV41_PLAN.md docs/SYCL_DSV41_PLAN2.md sycl/docs/sycl-dsv41-measured-report.md

Exactly three `tok/s` figures exist in the docs: `docs/SYCL_DSV41_PLAN.md:77` (38.22), `:79` (37.32),
`:81` (10.71). Each sits on the same line as its own completed prompt line
(`prompt: 10 tokens in 262/268/934 ms (... tok/s, windows of ...)`) and carries its measured-on-this-PC
tag and commit.

TAS_LIST_RC=0
CMD: grep -n 'tokens a second' sycl/docs/sycl-dsv41-measured-report.md

The report's own figures (`:22` prompt 7.92, `:23` answer 1.65, `:42` run 1 prompt 19.88, `:44` run 2
prompt 38.09, with the wrapped answer figures on `:43`/`:45`) each sit beside their own completed
"N tokens in M ms" line inside a run the report states finished with exit code 0.

NO_UNMEASURED_TOKS_RC=1 (counts 0, 0, 0 = expected: no tok/s figure lacks a completed-prompt line)
CMD: grep -cP '^(?!.*tokens in).*[0-9]+(?:\.[0-9]+)?\s*tok/s' docs/SYCL_DSV41_PLAN.md docs/SYCL_DSV41_PLAN2.md sycl/docs/sycl-dsv41-measured-report.md

NO_UNMEASURED_TAS_RC=1 (counts 0, 0, 0: the same holds for the report's "tokens a second" figures)
CMD: grep -cP '^(?!.*tokens in).*[0-9]+(?:\.[0-9]+)?\s*tokens a second' docs/SYCL_DSV41_PLAN.md docs/SYCL_DSV41_PLAN2.md sycl/docs/sycl-dsv41-measured-report.md

Hardware verified on this machine myself, not from the docs:

LSPCI_RC=0
CMD: lspci -nn > /tmp/rv3_lspci.txt

LSPCI_E211_COUNT_RC=0 (count 2 = both cards)
CMD: grep -c '8086:e211' /tmp/rv3_lspci.txt

LSPCI_E211_LINES_RC=0
CMD: grep -n '8086:e211' /tmp/rv3_lspci.txt

(`0d:00.0` and `11:00.0`, both `Intel Corporation Battlemage G21 [Arc Pro B60] [8086:e211]`.)

HW_BLOCK_RC=0
CMD: sed -n '30,50p' docs/SYCL_DSV41_PLAN.md

The docs hardware block names "2× Intel Arc Pro B60, device id `8086:e211`, 24 GB of graphics memory
each, on PCIe", "Intel oneAPI 2026.2; `icpx` prints 2026.2.0", and "Level Zero backend; every device run
sets `SYCL_BE=PI_LEVEL_ZERO`" — all three present and matching `lspci -nn`.

HW_ID_RC=0
CMD: grep -n '8086:e211' docs/SYCL_DSV41_PLAN.md

ONEAPI_RC=0
CMD: grep -n 'oneAPI 2026.2' docs/SYCL_DSV41_PLAN.md

BE_TAG_RC=0
CMD: grep -n 'SYCL_BE=PI_LEVEL_ZERO' docs/SYCL_DSV41_PLAN.md

PLAN2 negation — the old global negation is gone (expected: count 0, exit 1):

PLAN2_OLD_NEGATION_RC=1
CMD: grep -c 'tokens-per-second number has been measured' docs/SYCL_DSV41_PLAN2.md

PLAN2_NEG_TOKS_RC=1
CMD: grep -c 'no .* tok/s\|no .*tokens-per-second' docs/SYCL_DSV41_PLAN2.md

PLAN2_NEG_SEARCH_RC=0
CMD: grep -niE 'tok|tokens-per-second|throughput|speed number' docs/SYCL_DSV41_PLAN2.md

The remaining hits are `:58` pointing at the measured figures in PLAN.md, `:202-203` saying Phase D's
first speed numbers are now measured, and `:166`/`:182`/`:203` — scoped "states no throughput/speed
number" statements about subsystems whose timed runs genuinely have not happened. PLAN2 contains no
tok/s negation.

## Check 4 — scope of the four docs commits and the whole range

SHOW_D337_RC=0
CMD: git show --name-only --format= d337362b

SHOW_7411_RC=0
CMD: git show --name-only --format= 74110114

SHOW_A533_RC=0
CMD: git show --name-only --format= a53335fd

SHOW_AA09_RC=0
CMD: git show --name-only --format= aa09dc07

SHOW_ALL_CAT_RC=0
CMD: cat /tmp/rv3_show_d337.txt /tmp/rv3_show_7411.txt /tmp/rv3_show_a533.txt /tmp/rv3_show_aa09.txt

`d337362b`: `sycl/docs/sycl-dsv41-measured-report.md`; `74110114`: the same file; `a53335fd`:
`docs/SYCL_DSV41_PLAN.md`, `docs/SYCL_DSV41_PLAN2.md`; `aa09dc07`: `sycl/docs/sycl-dsv41-measured-report.md`
again — only `sycl/docs/` or `docs/` paths.

S1_OUTSIDE_RC=1 (count 0 = expected)
CMD: grep -cvE '^(sycl|docs)/' /tmp/rv3_show_d337.txt

S2_OUTSIDE_RC=1 (count 0 = expected)
CMD: grep -cvE '^(sycl|docs)/' /tmp/rv3_show_7411.txt

S3_OUTSIDE_RC=1 (count 0 = expected)
CMD: grep -cvE '^(sycl|docs)/' /tmp/rv3_show_a533.txt

S4_OUTSIDE_RC=1 (count 0 = expected)
CMD: grep -cvE '^(sycl|docs)/' /tmp/rv3_show_aa09.txt

LOG_SUBJECTS_RC=0
CMD: git log --format=%s 18a27e02..HEAD > /tmp/rv3_subjects.txt

SUBJECT_COUNT_RC=0
CMD: wc -l /tmp/rv3_subjects.txt

(15 subjects, as the caller states.)

SUBJECTS_CAT_RC=0
CMD: cat /tmp/rv3_subjects.txt

All 15 subjects: four `[Docs] dsv41 SYCL: ...` (split the test claim; tag every claim; correct the pack
identity and the test claim; measured report) and eleven `[Engine] ...`, in landing order.

SUBJECT_VIOLATIONS_RC=1 (count 0 = expected)
CMD: grep -cvE '^\[(Engine|Docs)\] ' /tmp/rv3_subjects.txt

DIFF_RANGE_RC=0
CMD: git diff 18a27e02..HEAD --name-only > /tmp/rv3_range.txt

OMO_BENCH_COUNT_RC=1 (count 0 = expected: no committed `.omo/` or `bench/results` path)
CMD: grep -cE '^(\.omo/|bench/results)' /tmp/rv3_range.txt

RANGE_OUTSIDE_RC=1 (count 0 = expected: every committed path is `sycl/` or `docs/`)
CMD: grep -cvE '^(sycl|docs)/' /tmp/rv3_range.txt

RANGELOG_RC=0
CMD: git log --name-only --format= 18a27e02..HEAD > /tmp/rv3_rangelog.txt

RANGELOG_OMO_BENCH_RC=1 (count 0 = expected: same for every commit's own path list, not just the net diff)
CMD: grep -cE '^(\.omo/|bench/results)' /tmp/rv3_rangelog.txt

RANGELOG_OUTSIDE_RC=1 (count 0 = expected)
CMD: grep -cvE '^((sycl|docs)/|$)' /tmp/rv3_rangelog.txt

STATUS_TRACKED_RC=0 (no output: zero tracked changes)
CMD: git status --porcelain --untracked-files=no

STATUS_RC=0
CMD: git status --porcelain > /tmp/rv3_status.txt

TRACKED_CHANGES_COUNT_RC=1 (count 0 = expected: the full status carries only untracked `??` entries)
CMD: grep -cvE '^\?\?' /tmp/rv3_status.txt

## Check 5 — push receipt

BRANCH_RC=0
CMD: git rev-parse --abbrev-ref HEAD

(`sycl/dsv41-sycl-support`.)

RP_HEAD_RC=0
CMD: git rev-parse HEAD

RP_FORK_RC=0
CMD: git rev-parse fork/sycl/dsv41-sycl-support

Both print `aa09dc0707f20d492b595bc42fc0f6fcfa421f1d`. HEAD equals the fork ref at the gated commit.

LOG4_RC=0
CMD: git log --format='%h %s' -4

(`aa09dc07` test-claim split, `a53335fd` docs re-tag, `74110114` pack-identity correction, `d337362b`
measured report — the four docs commits, newest first.)

## Check 6 — cleanup

WORKTREE_RC=0
CMD: git worktree list

Only `/home/maxious/Strata aa09dc07 [sycl/dsv41-sycl-support]` — the main checkout alone.

NODE_CLI_RC=1 (the `node` CLI is absent, as the caller states)
CMD: command -v node

The `node:fs` check therefore runs in this session's JS kernel (kernel cell, no exit code):

```
const { existsSync } = require('node:fs');
paths = ['/home/maxious/Strata-wt/row11fix', '/home/maxious/Strata-wt/row11tags', '/home/maxious/Strata-wt/row11fix2']
existsSync('/home/maxious/Strata-wt/row11fix') === false
existsSync('/home/maxious/Strata-wt/row11tags') === false
existsSync('/home/maxious/Strata-wt/row11fix2') === false
ALL_THREE_FALSE=true
```

All three scratch worktrees are gone.

## Result

The test-claim split at report line 58 is verified against primary evidence: the passed state (ds4_parity
and ds4_blas_parity, Linux SYCL build, 2 of 2), the failed state (default build suite, pre-existing
isa_guard_check on iq512x_rows, 1 of 57, DEF_CTEST_RC=8) and the not-run state (CUDA and HIP, no hardware)
each match regression-gates.md and the raw ctest artifacts, and no unqualified suite-pass claim remains in
the report. The earlier corrections still hold (both removed-string greps count 0; the pack identity
matches my own tensors.txt census and du -b exactly). Docs tags, tok/s placement, the hardware block
(2x Arc Pro B60 8086:e211 verified with lspci -nn, oneAPI 2026.2, SYCL_BE=PI_LEVEL_ZERO) and PLAN2's
absence of a tok/s negation all hold. Commit scope, the 15 prefixed subjects, the absence of committed
.omo/ or bench/results paths, the clean tracked status, the push receipt (HEAD = fork ref =
aa09dc0707f20d492b595bc42fc0f6fcfa421f1d) and the cleanup checks all pass as specified.

VERDICT confirmed
