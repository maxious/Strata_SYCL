# Experiment 23 - Q6_K MMVQ stall reasons: pipe-bound, not memory-bound

Date: 2026-10-04. One Intel Arc Pro B60 (Battlemage G21), oneAPI 2026.1, `sycl/build-b60`.
Target: `mmvq_bench 2560 2560 4` - the bench, not the engine, because the engine's run degenerates under
instrumentation (exp 20) while the bench cannot: no host/GPU handshake to disturb.

## Question

Experiments 21, 22 and 20 each eliminated a candidate without naming the limiter: the 384 B spill is free (21), the
kernel is not bandwidth-limited (best 281 GB/s of ~608) and not parallelism-starved (10240 rows = 210.2 GB/s) (22),
and the XVE array is stalled or idle 92.3% of GPU-busy time (20). Is it memory latency or the arithmetic?

## The collection

`gpu-hotspots` documents the knobs for this; the two that matter are `gpu-profiling-mode=source-analysis` and
`source-analysis=stall-sampling`, and `computing-tasks-of-interest` restricts collection to one kernel:

```sh
sudo -n vtune -collect gpu-hotspots \
  -knob gpu-profiling-mode=source-analysis \
  -knob source-analysis=stall-sampling \
  -knob computing-tasks-of-interest="*q6k_wide*" \
  -result-dir /tmp/vt/stall -finalization-mode=full -- ./mmvq_bench 2560 2560 4
```

(The bench warms ~300 ms and then runs 400 timed iterations, so there is ~310 ms of kernel work to sample. Report
reads need `sudo`: the result directory is root-owned, and without it VTune returns
`Error: 0x40000006 (Insufficient permissions)`. That was exp 22's empty `-report hotspots` CSVs - a permissions
error, not a missing report.)

## The stall breakdown

`-report summary -format csv` for the wide kernel, 8,061 instances, 0.216 s total, **27 us average against the
bench's own 26.1 us** - the two agree, so the report describes the same kernel the bench times:

| Control | Dist or Acc | Instruction Fetch | **Pipe** | SBID | **Send** | Synchronization | Pixel Shader Order | Other |
|---|---|---|---|---|---|---|---|---|
| 0.2% | 2.1% | 0.1% | **16.8%** | 3.4% | **0.0%** | 0.0% | 0.0% | 0.1% |

22.7% of samples are in a categorized stall and 77.3% are running. Of the stalled samples, **Pipe is 74%**, and the
memory-side categories are empty: `Send` 0.0%, and the only math-dependency category, `Dist or Acc`, 2.1%.

## Verdict

**The Q6_K wide MMVQ is execution-pipe bound, not memory bound.** The kernel is not waiting on its loads; it is
waiting for an execution pipe to accept its work.

That single line retro-explains the whole sequence, which is the useful part:

- exp 21 - removing every spilled byte and 23 instructions changed nothing: the spill was never on the pipe's
  critical path.
- exp 21 - the 256-GRF mode changed nothing: it buys memory-level parallelism the kernel does not need.
- exp 22 - the row sweep did not raise GB/s (10240 rows = 210.2): more concurrent work does not help a pipe-bound
  kernel.
- exp 22 - per-column cost grew (6.55 us/col at 4 columns, 9.75 at 8): more columns mean more Q6_K decode per weight
  byte, i.e. exactly the currency the pipe is short of.
- exp 22 - the cache-policy hypothesis was the wrong shape entirely, and is now moot rather than merely untested.

The lever is therefore **arithmetic per weight byte in the Q6_K decode** - either a lighter formulation of the 6-bit
unpack/scale sequence, or moving the dot onto the XMX units. The port has already priced the second one for the
*expert* path: the fused quantized XMX GEMM measured 0.37x of dequant + oneMKL (exp 00/02), so a DPAS form of this
kernel starts from a known bad precedent and would need a different argument than "use the tensor cores".

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh && cd sycl/build-b60
sudo -n vtune -collect gpu-hotspots -knob gpu-profiling-mode=source-analysis \
  -knob source-analysis=stall-sampling -knob computing-tasks-of-interest="*q6k_wide*" \
  -result-dir /tmp/vt/stall -finalization-mode=full -- ./mmvq_bench 2560 2560 4
sudo -n vtune -report summary -result-dir /tmp/vt/stall -format csv \
  -csv-delimiter comma -report-output /tmp/vt/stall_sum.csv
grep q6k_wide /tmp/vt/stall_sum.csv      # ... 16.8%,3.4%,0.0% ... = Pipe high, Send zero
```
