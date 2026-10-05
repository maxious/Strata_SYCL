# Experiment 21 - the Q6_K wide MMVQ ISA dump: the 384 B spill is not the lever

Date: 2026-10-04. One Intel Arc Pro B60 (Battlemage G21), oneAPI 2026.1, `sycl/build-b60`.
`mmvq_bench` (`launch_q6k_wide`, the 2560 x 2560 x 4-column case the spec window uses).

## Question

Experiment 20 ranked the top decode kernels with GPU hardware counters (`native_mmvq_q6k_wide_a2` 3.713 s,
`wait_flag_ge_kernel` 2.510 s, `native_gu_port<18,8>` 1.426 s) and recommended attacking the Q6_K MMVQ kernel
because it was the only one carrying a *known defect*: exp 15's IGC dump found it spilling 384 B/work-item while
running at 205.7 GB/s against the card's ~608 GB/s. Is the spill the lever?

## The dump

```sh
export IGC_ShaderDumpEnable=1 IGC_ForceIgnoreCaching=1 NEO_CACHE_PERSISTENT=0 SYCL_CACHE_PERSISTENT=0
export IGC_DumpToCustomDir=/tmp/igc
./mmvq_bench          # 26.1 us, 205.7 GB/s
```

(The cache-busting variables are load-bearing: on a program-cache hit IGC dumps only the `.spv` and never runs the
backend, so there is no `.asm`.) Three kernels came out. The wide one, `native_mmvq_q6k_wide<4>`:

    //.thread_config numGRF=128, numAcc=4, numSWSB=16
    //.RA type GRAPH_COLORING_SPILL_FF_BC_RA
    //.spill size 384
    //.spill GRF est. ref count 15
    //.instCount 985
    grf_count: 128   simd_size: 32   spill_size: 384

**Where the 384 B is.** Three 128-bit vectors are spilled to `Scratch[0x64]`, `Scratch[2x64]`, `Scratch[4x64]`
(plus ~11 f32 scalars), and they materialize as exactly two scratch stores and two matching fills:

    store.ugm.d32x32t (128 B, "spill to offset[4*64]")   ... later ... load.ugm.d32x32t ("fill from ...")
    store.ugm.d32x64t (256 B, "spill to offset[0*64]")   ... later ... load.ugm.d32x64t ("fill from ...")

128 + 256 = 384. That is **4 instructions out of 985** (0.4%), and the kernel has **no loop** - 2 forward `jmpi`
and zero loop branches - so the pair executes once per work-item, not per iteration. It is a single save/restore
across the region where register pressure peaks.

## The test

IGC honours `IGC_ExtraOCLOptions` at JIT time, so the register-file mode can be changed with no rebuild:

```sh
IGC_ExtraOCLOptions="-ze-opt-large-register-file" ./mmvq_bench
```

That moves the kernel to the 256-GRF file and **removes the spill entirely**: `numGRF=256, numAcc=8,
numSWSB=32`, zero `//.spill` lines, `//.instCount 962` (23 fewer instructions). Two runs each:

| register file | q6_k 2560x2560 x4 cols | instruction count | spill |
|---|---|---|---|
| 128 GRF (default) | 26.1 / 26.1 us (205.6 / 205.7 GB/s) | 985 | 384 B |
| 256 GRF | 26.1 / 26.1 us (205.6 / 205.7 GB/s) | 962 | **0 B** |

**Identical.** Removing every spilled byte and 23 instructions changes the time by nothing measurable.

(An intermediate single reading of 27.5 us for the 256-GRF case was noise - the controlled 2x2 above gives 26.1 us
four times. It is recorded here because it is exactly the kind of outlier that would have been read as "the spill
matters, and fixing it costs 5%".)

## Conclusion

- **The 384 B spill is not the lever.** It is 4 instructions in 985, once per work-item, and eliminating it
  entirely (256 GRF, spill 0) leaves the kernel at 26.1 us. exp 20's recommendation to attack this kernel *for its
  spill* is refuted by measurement; the same trap exp 15 recorded ("DPCT1110 does not predict spills") has a second
  form: a real spill is not automatically a real cost.
- The kernel is **not instruction-bound** (2.3% fewer instructions, 0% faster) and not spill-bound. With the
  collection reporting the XVE array stalled or idle 92.3% of the time the GPU is busy, the plateau at ~34% of
  peak bandwidth is a latency problem. The levers that remain are memory-level parallelism and the access/cache
  policy (`-cl-load-cache-default=4`, `-cl-store-cache-default=2` are in the option string), not register pressure.
- The 256-GRF mode is available and free at runtime via `IGC_ExtraOCLOptions`, which makes any future
  register-pressure experiment a one-line A/B with no rebuild.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh && cd sycl/build-b60
./mmvq_bench                                            # 26.1 us / 205.7 GB/s, spill 384 B
IGC_ExtraOCLOptions="-ze-opt-large-register-file" ./mmvq_bench   # 26.1 us, spill 0 B

rm -rf /tmp/igc && mkdir -p /tmp/igc
IGC_ShaderDumpEnable=1 IGC_ForceIgnoreCaching=1 NEO_CACHE_PERSISTENT=0 SYCL_CACHE_PERSISTENT=0 \
  IGC_DumpToCustomDir=/tmp/igc ./mmvq_bench
grep -a -m1 '//.thread_config' /tmp/igc/*_simd*_entry_*.asm     # numGRF=128, numAcc=4, numSWSB=16
grep -a -m1 '//.spill size'    /tmp/igc/*_simd*_entry_*.asm     # 384
grep -an 'store.ugm.*spill'    /tmp/igc/OCL_asm5b0f5ff4ca18f870_simd32_entry_0001.asm
```
