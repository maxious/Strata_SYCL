# Todo 11 gate review 2 — re-gate of the measured report, docs re-tag and push

Date: 2026-10-26 (UTC). Branch `sycl/dsv41-sycl-support`. HEAD `a53335fd598e5b9e976d830faa23badc1192a924`
(three docs commits: `d337362b` measured report, `74110114` report correction, `a53335fd` docs re-tag).
Reviewer: independent re-gate of todo 11 after the prior verdict at
`.omo/evidence/dsv41-sycl-dpct-and-xpu-test/todo11-gate-review.md` (which failed on the pack identity and
the CUDA/HIP test claim). No product file was edited, nothing was pushed, no monitor was used. Every check
below runs against primary evidence — the pack manifest `maya-ds4/tensors.txt`, `du -b` of the pack's GGUF
files, `lspci -nn`, `xpu-smi discovery`, the git object DB, `ctest -N` of the SYCL build tree and
`.omo/evidence/dsv41-sycl-dpct-and-xpu-test/regression-gates.md` — not against lane prose.

Convention: every `*_RC` line below sits directly above the command whose exit code it is. Each value is the
un-piped exit code captured as `$?` immediately after its command. Where a `grep -c` check expects no match,
the printed count is 0 and the exit code is 1; the line says so. GB figures are decimal bytes/1e9 at two
decimals, the report's own convention.

Supporting probes (recorded for completeness): tensors.txt row format read first
(`t <name> <quant> <file> <offset> <bytes> <dims...>` — offsets advance by the byte column, so bytes = field 6).

VIEW_RC=0
CMD: sed -n '7,10p;100,102p' /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/tensors.txt

The five GGUF files are not inside `maya-ds4/` (which holds the manifest, engram.bin and tokenizer); the
first du attempt there found no `*.gguf` and is recorded as run:

DU_RC=1
CMD: du -b /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/*.gguf

LS_PARENT_RC=0
CMD: ls -la /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/

FIND_RC=0
CMD: find /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO -maxdepth 2 -name '*.gguf'

## RC1 — report content measured against the pack itself

### 1a. Pack identity: my own census of `maya-ds4/tensors.txt` (47,246 tensor records)

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

Per-file sums (files 0-3 are the main shards, file 4 the Engram):

PERFILE_RC=0
CMD: awk '$1=="t"{fb[$4]+=$6; fc[$4]++} END{for (f in fb) printf "file %s\t%d tensors\t%d bytes\t%.4f GB\n", f, fc[f], fb[f], fb[f]/1e9}' /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/tensors.txt

file 0: 14592 tensors, 39,069,941,760 B (39.0699 GB); file 1: 13824, 39,388,446,720 B (39.3884 GB);
file 2: 13056, 39,246,888,960 B (39.2469 GB); file 3: 5772, 20,527,780,288 B (20.5278 GB);
file 4 (Engram): 2, 110,595,290,400 B (110.5953 GB). Shards sum 138,233,057,648 B (138.2330 GB).

TOTAL_RC=0
CMD: awk '$1=="t"{tot+=$6; n++} END{printf "TOTAL %d tensors %d bytes %.4f GB\n", n, tot, tot/1e9}' /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/tensors.txt

TOTAL 47246 tensors 248,828,348,128 bytes 248.8283 GB.

Zero Q4_K tensors (expected: count 0, exit 1):

TENSOR_Q4K_COUNT_RC=1
CMD: grep -c ' Q4_K ' /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4/tensors.txt

### 1b. On-disk sizes: my own `du -b` of the pack's five GGUF files

DU_B_RC=0
CMD: du -b /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/*.gguf

DU_BC_RC=0
CMD: du -bc /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/*.gguf

| file | bytes | GB |
| --- | --- | --- |
| ...IQ2_XXS-00001-of-00004.gguf | 39,075,714,624 | 39.0757 |
| ...IQ2_XXS-00002-of-00004.gguf | 39,388,449,504 | 39.3884 |
| ...IQ2_XXS-00003-of-00004.gguf | 39,246,891,616 | 39.2469 |
| ...IQ2_XXS-00004-of-00004.gguf | 20,527,837,312 | 20.5278 |
| ...Engram-IQ4_NL.gguf | 110,595,290,816 | 110.5953 |
| total (`du -bc`) | 248,834,183,872 | 248.8342 |

Main shards (four IQ2_XXS files): 138,238,893,056 B = 138.2389 GB → 138.24 GB. Total 248,834,183,872 B =
248.83 GB = 231.7 GiB.

### 1c. Comparison with the report

Report line 12 (the identity line added by the correction):
"The model is DeepSeek V4.1 Flash in the `DeepSeek-V4.1-Flash-1017-GSQ-RCO` pack: IQ1_M 59.45 GB,
IQ2_XXS 49.05 GB, IQ2_XS 23.57 GB, Q5_K 5.33 GB and an IQ4_NL engram of 110.60 GB, with zero Q4_K tensors
and 248.83 GB on disk in total (231.7 GiB; main shards 138.24 GB)."

Every figure matches my own measurement at the report's precision: 59.45, 49.05, 23.57, 5.33, 110.60,
zero Q4_K, 248.83 GB total (248.8342 from du), 231.7 GiB (231.73 computed), 138.24 GB main shards
(138.2389 from du). The prior review's FAIL 1 ("Q4_K quantised, about 115 GB") is corrected. The doc-side
pack section (`docs/SYCL_DSV41_PLAN.md` "Pack identity") carries the same tiers; its per-shard 39.07 comes
from the tensors.txt byte-offset sum (39.0699) while the file-size sum is 39.0757 — a 0.01 GB rounding
wrinkle between two named sources, both summing to 138.24 GB as published. Not a defect.

Removed strings (expected: count 0, exit 1 for both):

Q4K_QUANTISED_RC=1
CMD: grep -c 'Q4_K quantised' sycl/docs/sycl-dsv41-measured-report.md

GB115_RC=1
CMD: grep -c '115 GB' sycl/docs/sycl-dsv41-measured-report.md

CONTEXT_RC=0
CMD: grep -n 'Q4_K\|115' sycl/docs/sycl-dsv41-measured-report.md

The only `Q4_K` mentions left are "zero Q4_K tensors" (line 12) and the parity-shape list (line 59); `115`
occurs nowhere.

### 1d. The test claim (report line 58)

REPORT_TESTCLAIM_RC=0
CMD: grep -n 'default test suites\|CUDA\|HIP\|Linux SYCL build' sycl/docs/sycl-dsv41-measured-report.md

Line 58 reads: "The default test suites and both ds4 parity tests, `ds4_parity` and `ds4_blas_parity`, pass
on the Linux SYCL build. The CUDA and HIP builds were not run because no CUDA or HIP hardware is present
here."

- CUDA/HIP: correct now. The claim no longer says the suites pass on CUDA or HIP, and it states they were
  not run for lack of hardware. This fixes the prior FAIL 2; the supported scope (Linux SYCL only) matches
  `regression-gates.md` and the plan's IS-8.
- The ds4 pair on the Linux SYCL build: backed (`regression-gates.md`, `SYCL_CTEST_RC=0`, 2/2).
- **DEFECT (flagged) — the sentence's first clause claims "The default test suites ... pass" without the
  qualification the evidence requires.** Against `regression-gates.md`, the default suite on the default
  build exits `DEF_CTEST_RC=8` ("98% tests passed, 1 tests failed out of 57"), its sole failure the
  pre-existing `isa_guard_check` on `iq512x_rows` (reproduced on `origin/main` `4b9e05a2`, open row 17).
  A pass claim for the default suites with no mention of that failure contradicts the recorded result.
  The build qualifier "on the Linux SYCL build" does not repair it, because that build does not run the
  default suite at all:

CTESTFILE_LS_RC=0
CMD: ls build-sycl-dsv41/CTestTestfile.cmake

CTEST_N_RC=0
CMD: ctest --test-dir build-sycl-dsv41 -N

(The SYCL tree registers its own 40 tests, `Total Tests: 40`, of which the plan's gate runs only
`ctest -R ds4` — tests #1 and #2. No evidence log records a full 40-test run, and none records any
default-build ctest exiting 0.)

ADD_TEST_COUNT_RC=0
CMD: grep -c 'add_test' sycl/CMakeLists.txt

ADD_TEST_LINES_RC=0
CMD: grep -n 'add_test' sycl/CMakeLists.txt

(18 `add_test` sites in `sycl/CMakeLists.txt`, including the `${p}_parity` loops behind the 40-test
inventory; the default suite's tests come from the root project and its `DEF_CTEST_RC=8` run.)

T11_GREP_RC=1
CMD: grep -nE 'CTEST|ctest|tests passed|isa_guard|DEF_CTEST' .omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-11-dsv41-sycl-dpct-and-xpu-test.log

T11_WC=5 lines, no test run recorded at all in the todo-11 lane's own log.

So under either reading of "the default test suites" the clause is an unbacked pass claim: on the default
build it is contradicted (`DEF_CTEST_RC=8`, sole pre-existing `isa_guard_check` failure on `iq512x_rows`),
and on the named Linux SYCL build the full suite was never run (the gated run is the 2 ds4 tests). Per the
caller's rule, a claim that the default suites pass unqualified is a defect; it is present in the corrected
wording and drives the result below. A conforming sentence would separate what passed (the ds4 pair on the
Linux SYCL build, 2/2), what failed (default suite on the default build: `isa_guard_check` on `iq512x_rows`,
`DEF_CTEST_RC=8`, pre-existing) and what was not run (CUDA, HIP).

## RC2 — docs tags

TAG_RC=0 (count 52, the required >= 1)
CMD: grep -c 'measured on this PC' docs/SYCL_DSV41_PLAN.md

TOKS_LIST_RC=0
CMD: grep -nE '[0-9]+(\.[0-9]+)? ?tok/s' docs/SYCL_DSV41_PLAN.md docs/SYCL_DSV41_PLAN2.md

Exactly three tok/s figures exist in the docs: `docs/SYCL_DSV41_PLAN.md:77` (38.22), `:79` (37.32), `:81`
(10.71). Each sits on the same line as its own completed prompt line
(`prompt: 10 tokens in 262/268/934 ms (... tok/s, windows of ...)`), each tagged with build/date and an
evidence path. Manual window check: every figure's ±3-line window contains the completed `prompt:` line.
The report's own figures are written as "N tokens a second (M tokens in ... ms)", also beside completion.

NO_UNMEASURED_TOKS_RC=1 (counts 0, 0, 0 = expected: no tok/s figure lacks a completed-prompt line)
CMD: grep -cP '^(?!.*tokens in).*[0-9]+(?:\.[0-9]+)?\s*tok/s' docs/SYCL_DSV41_PLAN.md docs/SYCL_DSV41_PLAN2.md sycl/docs/sycl-dsv41-measured-report.md

Hardware block (`docs/SYCL_DSV41_PLAN.md:33-46`): 2× Intel Arc Pro B60, device id `8086:e211`, 24 GB each,
on PCIe; oneAPI 2026.2 (`icpx` 2026.2.0); Level Zero with every device run setting `SYCL_BE=PI_LEVEL_ZERO`.

HW_ID_RC=0
CMD: grep -n '8086:e211' docs/SYCL_DSV41_PLAN.md

BE_TAG_RC=0
CMD: grep -n 'SYCL_BE=PI_LEVEL_ZERO' docs/SYCL_DSV41_PLAN.md

ONEAPI_RC=0
CMD: grep -n 'oneAPI 2026.2' docs/SYCL_DSV41_PLAN.md

ARC_COUNT_RC=0 (count 9)
CMD: grep -c 'Arc Pro B60' docs/SYCL_DSV41_PLAN.md

Hardware verified on this machine myself, not from the docs:

LSPCI_RC=0
CMD: lspci -nn

LSPCI_E211_COUNT_RC=0 (count 2 = both cards)
CMD: grep -c '8086:e211' /tmp/rv_lspci.txt

LSPCI_E211_LINES_RC=0
CMD: grep -n '8086:e211' /tmp/rv_lspci.txt

(`0d:00.0` and `11:00.0`: `Intel Corporation Battlemage G21 [Arc Pro B60] [8086:e211]`.)

XPU_SMI_RC=0
CMD: xpu-smi discovery

(Device 0 and device 1 both `Intel(R) Arc(TM) Pro B60 Graphics`, SOC UUIDs ending `e2118086`, BDFs
`0000:0d:00.0` and `0000:11:00.0`. The docs' hardware block matches the machine.)

`docs/SYCL_DSV41_PLAN2.md` negation check — the old global negation is gone (expected: count 0, exit 1):

PLAN2_OLD_NEGATION_RC=1
CMD: grep -c 'tokens-per-second number has been measured' docs/SYCL_DSV41_PLAN2.md

PLAN2_NODSV41_RC=1
CMD: grep -c 'No dsv41' docs/SYCL_DSV41_PLAN2.md

PLAN2_TOKS_RC=1 (no literal tok/s figure in PLAN2)
CMD: grep -c 'tok/s' docs/SYCL_DSV41_PLAN2.md

PLAN2_NEG_TOKS_RC=1
CMD: grep -c 'no .* tok/s\|no .*tokens-per-second' docs/SYCL_DSV41_PLAN2.md

PLAN2_NEG_SEARCH_RC=0
CMD: grep -niE 'tok|tokens-per-second|throughput|speed number' docs/SYCL_DSV41_PLAN2.md

(The remaining hits are :58 pointing at the measured figures in PLAN.md, :166/:182/:203 — scoped "states no
throughput/speed number" statements about subsystems whose timed runs genuinely have not happened, each
tagged `(not measured yet)` — and :202-203 stating Phase D's first speed numbers are now measured. PLAN2
no longer negates tok/s.)

## RC3 — scope of the three docs commits and the whole range

LOG3_RC=0
CMD: git log --format='%h %s' -3

(`a53335fd` "[Docs] dsv41 SYCL: tag every claim as measured, static, prior art, or not measured yet";
`74110114` "[Docs] dsv41 SYCL: correct the pack identity and the test claim in the measured report";
`d337362b` "[Docs] dsv41 SYCL: measured report for the ported engine".)

SHOW_D337_RC=0
CMD: git show --name-only --format= d337362b

SHOW_7411_RC=0
CMD: git show --name-only --format= 74110114

SHOW_A533_RC=0
CMD: git show --name-only --format= a53335fd

(`d337362b`: `sycl/docs/sycl-dsv41-measured-report.md`; `74110114`: the same file; `a53335fd`:
`docs/SYCL_DSV41_PLAN.md`, `docs/SYCL_DSV41_PLAN2.md` — only `sycl/docs/` or `docs/` paths.)

S1_OUTSIDE_RC=1 (count 0 = expected)
CMD: grep -cvE '^(sycl|docs)/' /tmp/rv_show1.txt

S2_OUTSIDE_RC=1 (count 0 = expected)
CMD: grep -cvE '^(sycl|docs)/' /tmp/rv_show2.txt

S3_OUTSIDE_RC=1 (count 0 = expected)
CMD: grep -cvE '^(sycl|docs)/' /tmp/rv_show3.txt

LOG_SUBJECTS_RC=0
CMD: git log --format=%s 18a27e02..HEAD

SUBJECT_VIOLATIONS_RC=1 (count 0 = expected)
CMD: grep -cvE '^\[(Engine|Docs)\] ' /tmp/rv_subjects.txt

(14 subjects in `18a27e02..HEAD`, every one starting `[Engine]` or `[Docs]`.)

DIFF_RANGE_RC=0
CMD: git diff 18a27e02..HEAD --name-only

OMO_BENCH_COUNT_RC=1 (count 0 = expected: no committed `.omo/` or `bench/results` path)
CMD: grep -cE '^(\.omo/|bench/results)' /tmp/rv_range.txt

RANGE_OUTSIDE_RC=1 (count 0 = expected: every committed path is `sycl/` or `docs/`)
CMD: grep -cvE '^(sycl|docs)/' /tmp/rv_range.txt

STATUS_TRACKED_RC=0 (no output: zero tracked changes)
CMD: git status --porcelain --untracked-files=no

STATUS_RC=0
CMD: git status --porcelain

TRACKED_CHANGES_COUNT_RC=1 (count 0 = expected)
CMD: grep -cvE '^\?\?' /tmp/rv_status.txt

(The full status carries only untracked `??` entries.)

## RC4 — push receipt

RP_HEAD_RC=0
CMD: git rev-parse HEAD

RP_FORK_RC=0
CMD: git rev-parse fork/sycl/dsv41-sycl-support

Both print `a53335fd598e5b9e976d830faa23badc1192a924`. HEAD equals the fork ref at the gated commit.

## RC5 — cleanup

WORKTREE_RC=0
CMD: git worktree list

Only `/home/maxious/Strata a53335fd [sycl/dsv41-sycl-support]` — the main checkout alone.

The `node` CLI is not installed on this machine (recorded as run):

NODE_EXISTS_RC=127
CMD: node -e "const {existsSync}=require('node:fs'); console.log(existsSync('/home/maxious/Strata-wt/row11fix')); console.log(existsSync('/home/maxious/Strata-wt/row11tags'))"

The same `node:fs` check was then run in this session's JS kernel, where `node:fs` resolves:

```
existsSync('/home/maxious/Strata-wt/row11fix') = false
existsSync('/home/maxious/Strata-wt/row11tags') = false
```

Both are `false` as required.

## Result

The prior review's two failures are corrected and verified against primary evidence: the pack identity now
matches my own tensors.txt census and `du -b` exactly (IQ1_M 59.45, IQ2_XXS 49.05, IQ2_XS 23.57, Q5_K 5.33,
IQ4_NL engram 110.60 GB, zero Q4_K, 248.83 GB total, 138.24 GB main shards), and both "Q4_K quantised" and
"115 GB" are gone. The test claim no longer credits CUDA or HIP and names the Linux SYCL build as the
measured one. RC2 (docs tags, hardware block verified with lspci and xpu-smi, PLAN2 negation removed), RC3
(commit and range scope, subjects, clean tracked status), RC4 (push receipt) and RC5 (cleanup) all pass as
specified. One defect remains and is flagged in RC1d: report line 58 claims "The default test suites ...
pass" unqualified — contradicted by `regression-gates.md`'s `DEF_CTEST_RC=8` (sole pre-existing failure
`isa_guard_check` on `iq512x_rows`) on the default build, and unmeasured on the Linux SYCL build, whose
full 40-test suite no evidence log ever ran. An unqualified default-suites-pass claim is a defect, so the
corrections are not fully confirmed.

VERDICT FAIL
