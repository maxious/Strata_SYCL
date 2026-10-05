# Experiment 37 - the QSA top-k at long context: Xe2 holds 66 keys per thread, so 262K cells stay on the register kernel

**Question** (archive item D3 + E3/F6). The register top-k holds `4 * 1024 * TK_PER` cells, and `TK_PER_MAX` was
**66 under HIP but 33 on every other build**, so the 262,144 context the port ships fell off the register kernel on the
B60 - every key re-read from memory on each radix pass, one block per query, the rest of the card idle. The CUDA side
fixed the same shape with a block cluster (200 -> 22 us at 262K), but `sycl_ext_codeplay_cuda_cluster_group` is
CUDA-only. The Intel routes were the register/GRF knobs and `private_alloca`.

**Verdict. SHIPPED: `TK_PER_MAX = 66` on the SYCL branch** (it was 33). With the shipped `--max-context 262144` the
top-k runs **1.9x-2.4x faster** and the selected ids stay **identical to the reference**; a capacity of 131,072 or
less is unchanged bit for bit.

## Two bugs had to be found before the knob did anything

1. **The bench was never a CMake target** (`qsa_select_bench.cpp` existed; nothing built it). Added, linking
   `strata_kernels` + `strata_prefill`.
2. **`TK_PER_MAX` was dead code on SYCL.** The dispatch computed
   `fit = TK_T * (counted ? TK_PER_MAX : TK_PER)`, and `counted` is the CUDA active-count path - always false on this
   branch. So the capacity dispatch measured the fit against `TK_PER` whatever the build asked for: a `-DSTRATA_TK_PER_MAX=66`
   build first measured **bit-identical to the baseline at every size** (0.257 / 0.353 / 0.541 ms at 135K / 200K / 262K),
   which is what exposed it. On SYCL the capacity dispatch now uses `TK_PER_MAX` - still `TK_PER` unless a build
   overrides it, so the default dispatch is unchanged by the edit alone.

The width is a template argument, so it is fixed at compile time: `-DSTRATA_TK_PER_MAX=<n>` overrides, and the
shipping SYCL default is now 66.

## Measurement

`qsa_select_bench <ctx> <queries> <reps> <capacity>`, one query (the decode shape), best of 3 runs of 200 reps,
`ONEAPI_DEVICE_SELECTOR=level_zero:0`. "identical" is the bench's own check that the dispatch's ids equal
`qsa_block_topk_ref`'s.

**With the shipped 262,144 capacity** (what `--max-context 262144` gives the dispatch):

| ctx (cells) | TK_PER_MAX=33 | TK_PER_MAX=66 | speedup | ids identical |
|---:|---:|---:|---:|---|
| 135,168 | 0.257 ms | **0.138 ms** | 1.86x | 1/1 both |
| 200,000 | 0.353 ms | **0.177 ms** | 1.99x | 1/1 both |
| 262,144 | 0.541 ms | **0.224 ms** | 2.42x | 1/1 both |

**With capacity = ctx** (a deployment sized to its context) - the change is inert where it should be:

| ctx = capacity | 33 | 66 | note |
|---:|---:|---:|---|
| 65,536 | 0.059 ms | 0.059 ms | identical: both take the 33-wide kernel |
| 131,072 | 0.117 ms | 0.117 ms | identical |
| 135,168 | 0.254 ms | **0.136 ms** | 33 no longer fits (33,793 blocks > 33,792); 66 does |

That last row is the whole item in one line: 135,168 cells is exactly the old cliff edge, and the wider fit is the
difference between falling off it and not.

Both builds carry identical compile flags apart from the define (checked in `build.ninja`:
`-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel
-fp-model=precise`), so the difference is the register width and nothing else.

## What this does not buy yet

The kernel is 0.2 ms per call at 262K, and the decode calls it once per token per QSA layer's top-k, so the win is
real but small next to the engine's ~24 ms/token on the Coder. It matters at 262K context, where before this the
per-call cost was 2.4x higher **and** the wide kernel re-read every key per radix pass. The end-to-end confirmation
needs a real prompt past ~135K cells that keeps generating (exp 32's synthetic long prompts stop early), which is the
one measurement this item does not yet have.

`private_alloca` (archive F6) was not needed: the register array is a fixed-size private array whose width is the
template parameter, and 66 was enough to cover the shipped context. A build may still pass a larger
`-DSTRATA_TK_PER_MAX` to see where Xe2 stops fitting.

## Reproduce

```sh
# the baseline tree and the 66-wide tree (same source, one define)
cmake -S sycl -B sycl/build-b60-tk66 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_CXX_FLAGS=-DSTRATA_TK_PER_MAX=66 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_SYCL_PARITY=ON
ninja -C sycl/build-b60-tk66 qsa_select_bench

ONEAPI_DEVICE_SELECTOR=level_zero:0 ./sycl/build-b60/qsa_select_bench 262144 1 200 262144
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./sycl/build-b60-tk66/qsa_select_bench 262144 1 200 262144
```

## Trap this run adds

- **A build-time knob can be dead code and still compile.** `-DSTRATA_TK_PER_MAX=66` changed nothing measurable until
  the dispatch's `fit` was fixed; the tell was that the two builds were bit-identical at every size. When an A/B
  comes back as *exactly* 1.00x everywhere, suspect the knob is not wired, not the hardware.
- **The bench's capacity argument drives the dispatch, not the context.** With `capacity=262144` every row measures
  the long-context deployment, including a small `ctx`. Pass `capacity = ctx` for the sized-to-context deployment.
- **A second build tree needs `CMAKE_C_COMPILER` as well as `CMAKE_CXX_COMPILER`**: without it ggml's C files compile
  with `/usr/bin/cc` and die on `-fp-model=precise` (exp 35's silent-build family of traps, different face).