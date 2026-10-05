# Experiment 34 - where a split window's time goes (P0c D1, P0d E2)

## Question

README.sycl.md's D1 says: measure the window before porting anything - is there a host gap between the two
stages? E2 (from the SYCL graph guide) says: host work belongs in a host-task, and each stage wants its own
executable graph so a double-buffered pair can be in flight together. This measures the budget and decides
whether either is worth building.

## Method

`STRATA_DECODE_TIMING=1` on the dual Q2_0 config (2x B60, `--layer-split auto` -> K=22: CUDA0 layers 0-21,
CUDA1 layers 22-47 + the head + the drafter), the 2,185-token prompt, 256 greedy new tokens, cold page cache,
two runs. The instrument prints one line per request with per-window averages.

## The budget, ms per window

| component | run 1 | run 2 | share |
|---|---:|---:|---:|
| **window total** | 53.81 | 54.15 | |
| verify | 49.20 | 49.57 | 91% |
| - GPU-reach wait (the host waiting on the device) | 20.98 | 21.24 | 39% |
| - host, stage | 0.62 | 0.62 | 1.2% |
| - host, per-layer (pool 0.10 + plan 0.04 + actq 0.09 + jobs 0.09 + CPU 0.02) | 0.34 | 0.35 | 0.6% |
| - device execution (the rest of verify) | 27.26 | 27.36 | 51% |
| commit/emit | 1.67 | 1.64 | 3.1% |
| draft | 2.94 | 2.93 | 5.5% |
| decode (host-clocked) | 7,748 ms / 256 tok | 7,797 ms | 33.0 / 32.8 tok/s |

The request ran 144 windows at avg T 1.90 (1.78 tokens/window), decode expert cache hit **100.0%**, CPU experts
0.01 and PCIe 0.00 per layer-window, at K=22 with card 1 holding its 13,179-slot cache.

## D1 - there is no host gap between the stages

The host's own work is **0.96 ms/window (1.8%)**, and it spends **39% of the window waiting on the GPU**
(`GPU-reach wait`); the rest is the device chain. The code agrees: the earlier stage blocks on `cs_->wait()` for
its own window and then calls the next stage's `run()` from the host (`sycl/src/core/verify.cpp:1278` and
`:1395-1397`) - so the stages are host-chained - but the two stages are *dependent* (stage 1 reads stage 0's
residual out of the mapped hand-off buffer), so that chaining costs a launch round trip, not a serialized wait
that could be removed. **D1 closes: the split's decode parity is not a host-wait problem.**

## E2 - both of the guide's rules are under the noise floor here

- **Host work in a host-task.** The ceiling is the 1.8% above, and the work does not vanish when the graph owns
  it, so the real ceiling is smaller than the +-1.5% engine noise floor exp 33 measured. Nothing to build.
- **Both stages' graphs in flight at once.** The stages are strictly dependent, so the only recoverable time is
  the host round trip between stage 0's `cs_->wait()` returning and stage 1's launch - tens of microseconds,
  under 0.2%, also below the floor. Note this is not free ground either: the port's window uses a *bounded*
  device spin, so a stall introduced by touching the ordering becomes a wrong window rather than a hang.

**E2 closes as measured**, with the numbers above rather than a code change.

## What the budget does point at: D2

The host accounts `commit/emit 1.64 + draft 2.93` = **4.57 ms, 8.5% of every window** (`verify -> commit/emit ->
draft`, stamped at `sycl/src/program/generate.cpp:6397`). Part of that pair is *already* overlapped on the device
rather than serial: the last stage's commit is left running so the drafter overlaps it (`verify.cpp:1578`, reached
with `wait = !use_mtp` from `generate.cpp:7314`) - but 8.5% is still the upper bound on what is not, and the far
bigger prize is the 89% of the window that is two *dependent* stages' device time. That is the same ground as the
Hardin22 fork's serial-to-pipelined gain (x1.17 on code, x1.06-x1.30 across its four workloads), whose pipelining
exists to overlap a window with the *next* one. **D2 is the next lever** - and its prerequisite is the
device-ordered stage hand-off E2
declines to build on its own, which is the right way to pay for it.

## Note: why the per-segment table is empty

`STRATA_VERIFY_PROFILE=1` should print the per-segment table (`waitA`/`waitB`/`waitCPU`/`(gap)`) but its stamping
is gated on `G == 1` (`verify.cpp:1354`) and this config runs more than one expert group, so the table came back
`total 0.00 ms/window over 145 windows`. A split also has no per-card GPU timing: `--gpu-stages` is refused with
`--layer-split` (`generate.cpp:1809`). The per-window timing line is the instrument that works on this
configuration, and it reproduced to within 0.6% across the two runs.
