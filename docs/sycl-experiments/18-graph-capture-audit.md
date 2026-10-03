# Experiment 18 - the capture audit, and the A/B that says the graph is not the fault

Date: 2026-10-04. Hardware: **one Intel Arc Pro B60 (Battlemage G21)**, oneAPI 2026.1 (IntelLLVM 2026.1.1),
`syc1/build-b60` (Ninja, Release). Model: **Coder IQ1_M** (`Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001/00002-of-00002.gguf`,
ranked by `data/expert-profile-coder.bin`, `--expert-cache auto`, `--stream-experts`: 9340 of 12288 experts resident in
17.80 GiB, the remaining 2948 mirrored in pinned host memory and read over PCIe).

> **Superseded in part by [experiment 19](19-doorbell-completion-gate.md).** The stall that left this experiment
> without a trustworthy baseline was a **completion-gate bug** in the per-layer wait - `DPCT_CHECK_ERROR` discards
> the value of the expression it is given, so `ext_oneapi_empty()` was never consulted - not the graph and not
> host-mapped-write visibility. It is fixed, and the engine decodes at 17-20 tok/s. The `-48` node drift below is
> resolved there too: it is `STRATA_VERIFY_DEVICE_PLAN=1`, one node per layer. The capture audit and its findings
> stand unchanged.

## Question

README.sycl.md's P2 item: *"oneDNN/MKL calls must not fight SYCL graph capture."* The port captures the verify
window, the commit, and the drafter's round/step/prefill graphs once at load and replays them. A "graph break" here
is not a slow path - a captured `command_graph` replays with the USM pointers it was recorded with, so a break is a
wrong answer or a stalled replay, and both are silent. So: (a) can a break be made loud, and (b) is one happening?

## What was added

`sycl/include/strata/core/graph_audit.hpp` (header-only, no new build inputs) makes three things loud and one of
them fatal:

1. `dpct::experimental::end_recording` RETURNS WITHOUT SETTING `*graph` when the queue is not in its recording map
   (`sycl/include/dpct/graph.hpp`) - a mismatched begin/end hands back a null graph and the caller dereferences it.
   `err == 0 && graph == nullptr` now aborts with the key and the call site.
2. `begin_recording` silently no-ops when the queue is already recording, which would fold a nested capture into
   the outer graph. A second `graph_capture_begin` before an end aborts.
3. A key captured more than once (the exec was dropped and re-recorded) prints a loud line, and aborts under
   `STRATA_GRAPH_STRICT=1`; a node count that moves between captures does the same.

Wired into every capture site: `Verifier::capture` / `capture_commit` (`verify.window<T>`, `verify.commit`),
`MtpDrafter::capture_prefill` / `_dev` / `round` / `step` (`mtp.prefill<T>`, `mtp.round<T>[.coupled]`, ...),
and `session_capture` (`session.layer<L>.pre|post|preA|preB|preP<k>`). `STRATA_GRAPH_AUDIT=1` prints one
machine-readable line per capture plus a summary at exit; the summary counts keys, captures, re-captures and breaks
and needs no env var.

`sycl/src/core/graph_audit_test.cpp` (+ `add_test(graph_audit_test)`) is the audit's own gate. Each negative case
runs in a forked child with stderr on a pipe, because the assertion's contract is an `abort()` plus a named line:

```
PASS: a real capture is accepted
PASS: a null graph with no error aborts
PASS: an overlapping begin aborts
PASS: a re-capture is fatal under STRATA_GRAPH_STRICT
PASS: a moved node count is fatal under STRATA_GRAPH_STRICT
graph_audit_test: all assertions fire
```

`ctest` -> **27/27 passed** (the audit test is #25; exp 17's 26 tests are unchanged). Engine build: exit 0,
34 warnings (the pre-existing unused-`dpct` class).

## The A/B: seven configurations, zero capture breaks

Same prompt (`9707,11,1879,1130,374,279`), greedy, `--spec 4 --mtp <rt> --max-context 4096`, one variable each.

| arm | outcome | wall | captures / breaks |
|---|---|---|---|
| default (warm capture, graphs replayed) | `verify: layer 12 never rang (graph finished)` | 8 s | 23 / **0** |
| `STRATA_VERIFY_EAGER=1` (window run directly, no window graph) | `layer 1 never rang` | 9 s | 17 / **0** |
| `STRATA_VERIFY_NO_HOST=1` (wait for the whole window) | completes | 44 s | 23 / **0** |
| `STRATA_WARM_GRAPHS=0` (capture lazily) | `layer 12 never rang` | 8 s | 2 / **0** |
| `+ --no-prefill-borrow` | `layer 10 never rang` | - | 23 / **0** |
| `+ --no-prefill-borrow --stream-experts` | `layer 12 never rang` | - | 23 / **0** |
| without `--stream-experts` (full 31.64 GiB host arena) | `layer 10 never rang` | - | 23 / **0** |

The failing layer moves between runs (1, 10, 12, 28 on Q2_0) for identical flags - a race, not a fixed wrong node.

**The graph is not the fault.** `STRATA_VERIFY_EAGER=1` runs the *same* `record_window` body on the *same* queue
with no window graph at all (`verify.cpp` selects `record_window(...)` in place of `cs_->ext_oneapi_graph(*exec_[T])`),
and it fails the same way. Every arm reports 0 re-captures and 0 breaks: each key is captured exactly once, every
`end_recording` returned a graph, and no key was recorded twice.

This is INTEL.md's documented platform limit, not a regression introduced here: *"on this platform a kernel's writes
to host-mapped memory are not reliably visible while the graph runs (measured: the ring is seen late or not at all),
so waiting on them per layer fails."* The kernel log agrees there is no device fault during these runs: the 798
`xe ... PAGEFAULT ... SVM_RANGE_NOT_FOUND` lines are a single burst at 00:33:07-00:34:48, before this session, and
no GT reset is logged.

`STRATA_VERIFY_NO_HOST=1` is the documented workaround and it does complete - but its output is a fixed point, not a
completion: `output : 1879 13 1879 13 ...` (16 tokens in 37,984 ms, 0.42 tok/s; the Q2_0 run gave `712 78 13`
cycling, 0.30 tok/s). So **no configuration on this box produced trustworthy tokens**, and a token-parity A/B
(graph on vs off) could not be run - not because the arms agree, but because neither arm is a baseline.

## A measured drift worth chasing: 48 nodes, one per layer

`STRATA_GRAPH_AUDIT=1`, default arm, window graph node counts:

| window | this run | INTEL.md exp 05 | delta |
|---|---|---|---|
| 1 token | 2367 | 2415 | -48 |
| 2 tokens | 2409 | 2457 | -48 |
| 4 tokens | 2493 | 2541 | -48 |
| 6 tokens | 2577 | 2625 | -48 |

Both slopes are exactly 42 nodes/token, and both bases differ by exactly 48 = `g.n_layers`. **Resolved by
experiment 19:** the 48 is `STRATA_VERIFY_DEVICE_PLAN=1`, one device-plan node per layer. With that flag the same
run measures 2415/2457/2499/2541/2583/2625 for 1..6 tokens - an exact match to exp 05's documented figures - so
the drift was a configuration difference, not a lost node. Two goldens now record both configurations
(`sycl/bench/graph-golden/coder-iq1_m.txt` and `...-device-plan.txt`).

Shapes small enough to read whole: `verify.commit` is 158 nodes; the drafter's `mtp.prefill1..6` are 21, 24, 29,
32, 37, 40; `mtp.round1..6` are 87, 90, 95, 98, 103, 106; `mtp.step1..4` are 63 each. A model that stopped
emitting a node, or a step graph that started carrying a round's worth of work, is visible in one line.

## Profilers

VTune 2026.4.0 (build 632893) is installed at `/opt/intel/oneapi/vtune/2026.4/bin64/vtune` (not on `PATH`).
GPU hardware counters were denied by `dev.xe.observation_paranoid=1`; set to 0 for this session with
`sudo sysctl dev.xe.observation_paranoid=0` (runtime only - it does not survive a reboot; `vtune-set-perf-caps.sh`
or a kernel-cmdline `dev.xe.observation_paranoid=0` makes it persistent). unitrace is not installed on the host -
it comes from `sycl/tools/Dockerfile.unitrace`. INTEL.md's `sycl/rank_kernels.py` does not exist in this tree.

## Conclusion

- The capture audit is built, wired into all three capturing units, and every one of its assertions is proven to
  fire by `graph_audit_test`.
- **No SYCL graph break was found.** Across seven configurations the capture layer reported 23 clean, exactly-once
  captures and zero breaks, and the observed failure reproduces with the graphs removed entirely.
- The engine cannot complete a decode round on this box at HEAD: the per-layer doorbell ring is not visible to the
  host, which INTEL.md already documents, and the `STRATA_VERIFY_NO_HOST=1` workaround yields degenerate output.
  That - not the graph - is what blocks a trustworthy token-parity A/B. **Corrected by experiment 19:** the ring
  was visible all along; the completion gate that was supposed to tell "still running" from "finished" never read
  its own condition, so the host gave up 2 ms after any pause. Token parity was measured after the fix.
- Open: the -48 (one per layer) node drift against exp 05's documented window counts.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh && cd sycl/build-b60
ninja strata graph_audit_test && ./graph_audit_test          # 5/5 assertions fire
ctest -R graph_audit_test

SHARD=~/ComfyUI/koboldcpp/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
S2=~/ComfyUI/koboldcpp/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
STRATA_GRAPH_AUDIT=1 ./strata --pack ~/ComfyUI/koboldcpp/pack-coder --native "$SHARD" --ple-gguf "$S2" \
    --expert-profile ../../data/expert-profile-coder.bin --expert-cache auto --stream-experts \
    --prefill 128 --spec 4 --mtp ~/ComfyUI/koboldcpp/mtp/rt \
    --tokens 9707,11,1879,1130,374,279 --max-new 16 --max-context 4096
# the same command with STRATA_VERIFY_EAGER=1 fails at the same wait: the graph is not the fault
```
