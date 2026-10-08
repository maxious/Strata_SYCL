# benchy v1 on upstream main: the baseline

The reference every improvement is measured against, on this box (2x Arc Pro B60, G21; Ryzen 7 5700X3D, 121.5 GiB;
Ubuntu 26.04.1, kernel 7.3.0-rc1-xe-perf), measured **2026-10-08**.

Engine: real upstream `Niko1221/Strata` main `d5ea713` ("Version 0.1.40.3"), engine reports `0.1.40.3`. Built with
`cmake -S sycl -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release`
with `-DSTRATA_SYCL_AOT=` empty (SPIR-V, JIT); binary md5 `632f455e76d16bb1fd46a76d342008ee`. Re-building this branch's tree
reproduces that md5 exactly, so the harness below and the binary that produced the table are the same artifact.

Config `strata-q2_0.json` (single card): Q2_0, `--expert-cache auto --prefill 4096 --max-context 262144 --kv int8
--stream-experts --vram-reserve-mib 2048 --kv-resident 32768 --spec 2 --spec-min-p 0.5`.
How: `sudo env PATH=$PATH STRATA_NATIVE=1 STRATA_SYCL_BIN=$PWD/build-sycl/strata python3 sycl/tools/perf_matrix.py
--configs strata-q2_0.json` - one run per row, greedy, 256 new tokens, the page cache dropped before every row
(cold). Artifacts kept out of git in `sycl/benchy-results/v1-2026-10-08-upstream-main/` (its `machine.json`
`commit` field names the worktree HEAD at run time, not the engine - the md5 above is the engine's identity).

| Input tokens | PP (tok/s) | TTFT (s) | TG (tok/s) |
|---:|---:|---:|---:|
| 20 | 30.23 | 0.8 | 28.36 |
| 2,185 | 509.98 | 4.5 | 31.69 |
| 8,000 | 680.30 | 12.0 | 16.89 |
| 40,000 | 676.06 | 59.4 | 15.04 |
| 128,000 | 592.61 | 216.2 | 13.66 |
| 256,000 | 494.12 | 518.3 | 13.45 |

Per row: 12,878 expert slots (16.6 GiB) with 11,698 experts (15.1 GiB) in the pinned host mirror, 1,463 slots lent to
the prompt path, peak VRAM 22.2-22.7 GB, 77-89 s load, 38.8 GB read from the SSD at load.

Rows over 2,000 tokens decode **one** token: the v1 long prompt ends the turn immediately under `--stop-eos`, so their
TG is a single token's time. Read PP and TTFT for long-prompt changes.

The **dual** config (`strata-q2_0-dual.json`, `--serve --layer-split auto --split-device 1`) does not run on this
engine: it dies in the latter stage's expert-cache sizing (upstream issue #1054). The branch
`sycl-fix-1054-cross-card-sizing` clears that stage; a second defect on the first window then stops the row. Use the
single-card config for A/Bs here.

`sycl/tools/arm-run.sh` and `sycl/tools/test-dual.sh` run one arm of a single-card A/B and the dual gate; both print
the engine md5 and whether the lever is in the binary before measuring.
