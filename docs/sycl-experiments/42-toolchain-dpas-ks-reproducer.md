# Toolchain reproducer: int8 DPAS results depend on the translation unit, from identical generated code

**Filed from exp 42's engine integration (2026-10-06).** This is a self-contained reproducer, not a port bug report
about our source: the same kernel, compiled from two translation units with the same flags, produces byte-identical
device code and different results.

## Symptom

`sycl/src/kernels/cuda/q6k_dpas.dp.cpp` (exp 42's DPAS int8 Q6_K decode matvec) reduces its per-work-item partials
across the `KS` work-items of a tile through SYCL local memory:

```cpp
sycl::local_accessor<float, 1> red(sycl::range<1>((std::size_t) KS * NC * NT), cgh);
cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((std::size_t) ntiles * KS), sycl::range<1>(KS)),
                 [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
    ...
    float* sh = red.get_multi_ptr<sycl::access::decorated::no>().get();
    for (int i = 0; i < NC * NT; ++i) sh[lid * (NC * NT) + i] = Cf[i];
    it.barrier(sycl::access::fence_space::local_space);
    if (lid != 0) return;
    ... v = sum over k < KS of sh[k * (NC * NT) + m * NT + n] ...
});
```

* With **KS = 1** (no cross-work-item combine) the kernel is **correct**: rel 1.6e-07 against the reference, which is
  the fp32-rounding floor (`q6k_dpas_parity`: 0 failures at ncols 1-8).
* With **KS = 8** the same kernel is **wrong**: rel 0.87 at every ncols - most output rows are zero or garbage.
* The identical source in `sycl/src/kernels/xmx_mmvq_bench.cpp`, at KS = 8, is **correct** (rel 3.44e-03 against an fp64
  reference, matching the shipped dp4a kernel).

## What is ruled out

1. **Generated code.** Dumping both instantiations at NC = 1, KS = 8 with
   `IGC_ShaderDumpEnable=1 IGC_ForceIgnoreCaching=1 NEO_CACHE_PERSISTENT=0 SYCL_CACHE_PERSISTENT=0
   IGC_DumpToCustomDir=<dir>` gives **2,317 .asm lines each**, and a line-by-line diff differs by **one line: the
   symbol-name comment**.
2. **Inputs.** The DPAS B tiles, the Q6_K scales and d that the engine's transform produces are byte-identical to a
   host-built reference of the same layout: 0 differences in 6,553,600 / 409,600 / 25,600 bytes.
3. **Launch geometry.** Both launch `ntiles * KS` groups of `KS` (160 tiles x 8 for a 2,560-row output) with the same
   local accessor size.
4. **Compile flags.** Identical (`-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-default-sub-group-size=32
   -fsycl-device-code-split=per_kernel -fp-model=precise`), verified from `build.ninja`.
5. **Library vs executable.** Compiling the failing TU into the consumer executable instead of the static library does
   not change the result.
6. **Other formulations of the combine.** Local memory fails; a scalar `atomic_ref` is unavailable under
   `[[intel::sycl_explicit_simd]]` ("not supported in ESIMD context"); a two-pass reduce over distinct global partial
   slots fails the same way.

## Reproducer

```sh
source /opt/intel/oneapi/setvars.sh
cd sycl
# KS = 1 (default): correct
ninja -C build-b60 q6k_dpas_parity && ./build-b60/q6k_dpas_parity          # 0 failures, rel ~1.6e-07
# KS = 8: wrong. The width is a compile-time switch, so no source edit is needed:
icpx -fsycl -O3 -DNDEBUG -std=c++20 -DSTRATA_DPAS_KS_N=8 -fsycl-default-sub-group-size=32 \
     -fsycl-device-code-split=per_kernel -I include -I src -I third_party/ggml \
     -c src/kernels/cuda/q6k_dpas.dp.cpp -o /tmp/ks8.o                    # compiles clean
# then link that object ahead of libstrata_kernels.a and run: rel 0.87 at every ncols.
```

Reference run: `sycl/build-b60/q6k_dpas_parity` at 2,560 x 2,560, random Q6_K weights, q8_1 activations, ncols 1-8.

## Environment

| | |
|---|---|
| device | Intel Arc Pro B60 (BMG G21, `ext_intel_matrix` present, 2 GPUs) |
| driver / kernel | xe 1.1.0, Linux 7.3.0-rc1-xe-perf |
| compiler | oneAPI 2026.1 `icpx`, SPIR-V + JIT (`STRATA_SYCL_AOT` empty) |
| dpas header | `sycl/ext/intel/esimd/xmx/dpas.hpp`, `dpas<8, 1, int>` (int8, K = 32, N = 16) |
| build | `-fsycl-device-code-split=per_kernel`, `-fsycl-default-sub-group-size=32`, `-fp-model=precise` |

## What would confirm it for Intel

The smallest case is the local-memory combine alone under `[[intel::sycl_explicit_simd]]`: `KS` work-items write
`KS * NC * NT` floats to a `local_accessor<float, 1>`, barrier, work-item 0 sums each element across the `KS` slots.
The dump shows this sequence emitted correctly, yet the summed values do not match the per-work-item inputs. A
standalone kernel doing only that (no DPAS, no tiles) would be enough to isolate it.

## Update (2026-10-06, later): the local-memory mechanism is exonerated

A 40-line isolation of exactly the failing construct - KS work-items each fill their own slot of a
local_accessor<float, 1>, barrier, work-item 0 sums the KS slots - is **exact** on this device in both shapes:

    A  scalar local memory, explicit_simd    exact (0 wrong)
    C  simd<float,16> local memory, explicit_simd    exact (0 wrong)

(/home/maxious/exp42-harness/shm_probe.cpp.) So the combine mechanism is fine and this is **not** a general
local-memory or barrier defect. The width sweep with the flag shows the failure appears at every KS above 1:

| STRATA_DPAS_KS_N | rel at ncols 1 |
|---:|---:|
| 1 | **1.6e-07 (correct)** |
| 2 | 4.85e-01 |
| 4 | 7.37e-01 |
| 8 | 8.76e-01 |

which narrows it: it needs KS > 1 *in this kernel*, while the same shape works in isolation. The remaining
difference between them is **register pressure** - the real kernel holds the 512-byte DPAS tiles and the DPAS
operands live across the same function where the isolation holds a handful of scalars - so the leading
hypothesis is a spill or scheduling interaction around the local-memory store, not the store itself.

Not yet tried, and the next things to try:

1. `reqd_work_group_size(KS, 1, 1)` with `range<1>(ntiles)` instead of `nd_range` - the classic fix for
   work-group-size-dependent scheduling, and it changes the launch shape the compiler sees.
2. Reducing live state at the barrier: accumulate the per-column DPAS results into a small array first, or
   re-materialise the scale vectors inside the group so fewer registers are live across the store.
3. Read `//.spill size` for the two kernels from the ISA dumps already on disk
   (`/home/maxious/exp42-harness/isa/`) to confirm or refute the pressure hypothesis before changing code.

### Also excluded (2026-10-06, after the isolation)

- **Register pressure / spills.** The ISA dumps carry no spill annotations at all (`grep -c spill` = 0 in both
  kernels), only a scratch-location declaration, so the pressure hypothesis is unsupported as IGC reports it.
- **The one compile-flag difference between the two TUs.** The bench TU is compiled with `-DMKL_ILP64` and
  `strata_kernels` is not; rebuilding the engine TU with `-DCMAKE_CXX_FLAGS=-DMKL_ILP64` at KS = 8 still reports
  rel 8.70e-01, so that flag is not it either.

So the defect needs KS > 1 in this kernel, is not the local-memory or barrier mechanism (exact in isolation), and
survives every source, codegen, launch, flag and formulation difference that has been tried. The reproducer below
is the honest state: a real, small, unexplained failure with the search space already narrowed.

## Diagnosis update (2026-10-06, evening): the defect is the per-work-item PARTIALS, and the kernel is bistable

Two more experiments, both at KS = 8, changed the diagnosis.

**1. Adding an unrelated global store flips the result.** With a debug block that writes a few values to a global
buffer before the combine, `ncols 1` went from rel 8.76e-01 to **rel 1.51e-08** in the same binary, and stayed correct
with the debug branch *not* taken. Nothing about the computation changed - only the code around it.

**2. Removing local memory and the barrier entirely changes nothing.** The combine was rewritten as two kernels with
**no `local_accessor`, no `barrier`, no atomics**: each work-item writes its partial to a distinct global slot and a
second kernel sums them. It still reports **rel 8.70e-01** at KS = 8, and is **exact (rel 1.67e-07, 0 failures) at
KS = 1**.

Taken together these move the fault decisively:

- it is **not the combine** (local memory, barrier, atomics and a barrier-free two-pass all fail identically);
- it is **not local memory or barriers as mechanisms** (the 40-line isolation is exact);
- it is **not codegen** (2,317 identical .asm lines), **not spills** (no annotations), **not a flag** (`-DMKL_ILP64`);
- it therefore lives in **the per-work-item partial**, i.e. in the strided `for (sb = lid; sb < bpr; sb += KS)` loop
  that runs the DPAS - which only executes when a work-group has more than one work-item;
- and it is **bistable under trivial edits** (a global store fixes it; adding a barrier breaks it again), which is a
  scheduling/visibility sensitivity localised to this kernel shape in this binary, not a logic error.

At KS = 1 the loop is `sb = 0..bpr-1` in one work-item and everything is exact, which is why the shipped default is
KS = 1 and why the engine A/B loses: the boost only exists at KS > 1.

### The next experiment, with the oracle already in place

`q6k_dpas_parity` at `-DSTRATA_DPAS_KS_N=8` is a pass/fail oracle that builds in one step. The decisive measurement is
to **write each lid's partial to a distinct global slot (already done in the two-pass) and compare it against the same
partial computed on the host** for the same tile and block set. That splits the remaining space cleanly:

| result | conclusion | what to do |
|---|---|---|
| lid 0 correct, lids 1..KS-1 wrong | the strided block loop is mis-executed with >1 work-item per group | replace the stride with **contiguous chunks** (`sb = lid * chunk + it`), which is a one-line change and keeps the same partial layout |
| all lids wrong | the DPAS operand path is wrong for any multi-work-item group | abandon the K-split; the only remaining parallelism is **8-row tiles** (`N=8` is a legal DPAS execution size, so twice the tiles and still one work-item per tile) |
| all lids correct but y wrong | the second kernel's read/store | trivially fixable |

A cheaper first probe of the same hypothesis, needing no new code: the rel values are almost exactly `1 - 1/KS`
(measured 8.76e-01 / 7.37e-01 / 4.85e-01 against 0.875 / 0.75 / 0.5), i.e. **only one in KS of the output rows is
right**. Whether that one is tile 0 (a launch/grid effect) or every KS-th tile (a store effect) is one `printf` in the
two-pass reduce kernel, and it points straight at the table above.

### What the boost is worth even if this is fixed

From step 1's measurements, at KS = 8 the prize is 0.909 ms per 4-token spec-4 window (~1% of a 45-52 ms round) and
1.6 ms per token at ncols 1 (~4-6%), against ~168 expert-cache slots (~-0.49 tok/s) the tiles cost. So a fix is worth
chasing for **non-speculative decode** (net positive, ~+1.9 tok/s) and is **net negative at spec 4** (~+0.19 vs
-0.49 tok/s). The engine runs spec 4 by default, so the honest answer to "will this path work" is: only if the tiles
can come from somewhere that is not the expert cache, or if decode runs without speculation.
