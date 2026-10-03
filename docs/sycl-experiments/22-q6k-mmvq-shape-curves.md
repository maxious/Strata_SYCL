# Experiment 22 - the Q6_K MMVQ shape curves, and a null cache sweep its own verification caught

Date: 2026-10-04. One Intel Arc Pro B60 (Battlemage G21), oneAPI 2026.1, `sycl/build-b60`.
`mmvq_bench [n_in] [n_out] [ncols]` (defaults 2560 2560 4; it warms up with 20 iterations before timing).

## Question

Experiment 21 ruled the 384 B spill out as the limit: eliminating it changed nothing and the kernel is not
instruction-bound. The remaining cheap hypothesis was the memory access policy (`-cl-load-cache-default=4`,
`-cl-store-cache-default=2` are in its option string) for weights that are streamed once. Second question: is the
205.7 GB/s plateau a property of the *kernel* or of the *shape* the engine uses it at?

## The cache sweep is null, and here is why that is worth recording

`IGC_ExtraOCLOptions="-cl-load-cache-default=N"` for every N in 0..7, and `-cl-store-cache-default` for
0/1/3/4/7, all returned **26.1-26.2 us (205.4-205.7 GB/s)**. Thirteen runs, one number.

That is not a result about the cache. The re-dump check caught it:

```sh
IGC_DumpToCustomDir=/tmp/igc3 IGC_ExtraOCLOptions="-cl-load-cache-default=1" ./mmvq_bench
grep -ao 'cl-load-cache-default=[0-9]*' /tmp/igc3/*q6k*_entry_*.asm   # -> nothing
```

The option never reached the compiler, so all thirteen runs compiled the default kernel. The channel works for
`-ze-*` (`-ze-opt-large-register-file` does move the kernel to 256 GRF) but not for `-cl-load/store-cache-default`
on a SYCL/SPIR-V input. **So the cache policy is untested, not refuted** - and the lesson from exp 21's own memory
note ("validate the option took effect by re-dumping; a wrong name is silently ignored") is what stopped a
thirteen-run coincidence from being written up as a finding.

## The shape curves (the real data)

Columns, `n_in=n_out=2560`:

| cols | time | GB/s of weights | per-column |
|---|---|---|---|
| 1 | 19.1 us | **281.1** | 19.10 us |
| 2 | 21.5 us | 250.0 | 10.75 |
| 4 | 26.2 us | 205.6 | 6.55 |
| 5 | 30.1 us | 178.5 | 6.02 |
| 6 | 35.9 us | 149.7 | 5.98 |
| 7 | 39.4 us | 136.6 | 5.63 |
| 8 | 55.4 us | **97.1** | 6.93 |

Output rows, 4 columns: 640 -> 10.4 us (129.6 GB/s); 2560 -> 26.1 (205.7); 10240 -> 102.3 (**210.2**);
20480 -> 209.5 (205.3).

Read together:

- **Not bandwidth-limited.** The best this kernel does is 281 GB/s against the card's ~608 GB/s, and it is the
  1-column case that reaches it - the multi-column cases are *lower* per weight byte.
- **Not parallelism-starved at the production shape.** Ten thousand rows buys 210.2 GB/s against 2560 rows'
  205.7, so the engine's shape is already at the saturation point for this template.
- **Per-column cost turns superlinear past ~4 columns**: +2.4 us for the second column, ~+2.4/column through 4,
  then +4.9/column at 5-6 and +9.8/column at 7-8. `--spec 4` windows reach 6 tokens, so the engine operates at
  cols 4-6 - past the knee, but not yet in the 8-column collapse.

## The 256-GRF mode does not touch any of it

`-ze-opt-large-register-file` at cols 1/2/4/8: 19.1/21.5/26.1/55.5 us - the same curve to 0.1 us. So the column
behaviour, including the 8-column collapse, is **not register pressure**. (At `NCOLS=8` the same template does
spill 6272 B in 1631 instructions, against 384 B in 985 at `NCOLS=4` - the spilling is a symptom of the unrolling,
not its cause.)

## What this says

The Q6_K MMVQ at the engine's 4-column shape runs at 205.7 GB/s because of what it does **per weight byte**, not
because DRAM is saturated: it decodes 6-bit quants, applies 16 scales per 256 values and unrolls the columns'
worth of that work, and its own arithmetic leaves the array stalled most of the time either way. Prior art agrees
where it can: INTEL.md already recorded the expert dot family as ALU-bound (77% XVE active), and the exp 20
collection says the XVE array is stalled or idle 92.3% of the time the GPU is busy.

Per token, the curve is the mechanism behind a result the port already had end to end: a 6-token window costs
35.9 us against a 4-token window's 26.2 us, i.e. 5.98 vs 6.55 us per token for this kernel - larger windows are
per-token *worse* here, and the draft-policy sweep still chose `--spec 4` because fewer rounds win overall. Both
statements hold; this is the kernel-level half.

## Not obtained

The per-kernel stall breakdown. `vtune -report gpu-hotspots` and `-report gpu-compute-media-hotspots` both export
empty CSVs from the exp 20 result directory (`-report summary` is the only report that populates), so the split
between memory latency and ALU dependency stalls for this kernel is still unmeasured.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh && cd sycl/build-b60
./mmvq_bench 2560 2560 1     # 19.1 us / 281.1 GB/s
./mmvq_bench 2560 2560 4     # 26.2 us / 205.6 GB/s  <- the engine's shape
./mmvq_bench 2560 2560 8     # 55.4 us /  97.1 GB/s
./mmvq_bench 2560 10240 4    # 102.3 us / 210.2 GB/s (rows do not buy bandwidth)
IGC_ExtraOCLOptions="-ze-opt-large-register-file" ./mmvq_bench 2560 2560 8   # 55.5 us: not registers

# a cache-policy option that silently does nothing (the check that caught the null sweep)
rm -rf /tmp/igc3 && mkdir -p /tmp/igc3
IGC_ShaderDumpEnable=1 IGC_ForceIgnoreCaching=1 NEO_CACHE_PERSISTENT=0 SYCL_CACHE_PERSISTENT=0 \
  IGC_DumpToCustomDir=/tmp/igc3 IGC_ExtraOCLOptions="-cl-load-cache-default=1" ./mmvq_bench
grep -ao 'cl-load-cache-default=[0-9]*' /tmp/igc3/*q6k*_entry_*.asm    # empty: the option never applied
```
