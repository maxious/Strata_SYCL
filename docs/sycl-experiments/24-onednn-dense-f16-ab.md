# Experiment 24 - the oneDNN avenue (README P0b #1): fused dequant is blocked; dense-F16 A/B is a no-win

## Question

README P0b #1 proposed oneDNN's graph compiler fusing `Dequantize -> MatMul` to attack the prompt's dequant
phase (27-30% of prompt time). Could oneDNN improve the prompt path, and is it even worth linking? Hardware: the
B60 box (4x Arc Pro B60, XMX present), oneDNN 3.11.4 installed, SYCL port had no oneDNN linkage.

## The fused premise is blocked before any A/B

1. **oneDNN's `Dequantize` op reads a standard quantized tensor** (per-channel / per-tensor s8/u8 with a scales
   vector) - verified in the installed header (`oneapi/dnnl/dnnl_graph.hpp`: `data_type s8/u8`, `Dequantize` op,
   `scales` attr). It has **no notion of ggml block formats** (Q2_0, i-quants), which are exactly how Strata's
   expert weights are stored.
2. Reaching a oneDNN-readable quantized form means requantizing to s8 - which exp 13 already measured as the
   killer: 0.065 ms vs our FP16 dequant's 0.014 ms, and lossy (1.0/2.4% gate/down).
3. **llama.cpp does not fuse a block dequant into oneDNN either** - its oneMKL/oneDNN prompt attention "converts
   non-F16 KV to F16 first" (README section 1). It runs a dense FP16 GEMM/SDPA on already-F16 data, exactly what
   Strata already does (`iq_dequant_f16` -> `Gemm::f16`).

So the honest library question is a **dense-F16 A/B**: on already-F16 weights, is oneDNN's dense FP16 matmul
faster than the oneMKL GEMM the prompt path ships (`Gemm::f16`, oneMKL `compute_type f32`)?

## Enable (kept, opt-in): link + probe

- **Link:** `STRATA_SYCL_DNNL=1` (default OFF; base build untouched) does `find_package(dnnl CONFIG)` and links
  `DNNL::dnnl` to the prefill path. `messages: Strata SYCL: linked oneDNN 3.11.4`.
- **Probe** (`onednn_probe`, mirrors llama.cpp #28985's `ggml_sycl_dnnl_detect_optimized_gemm`): asks oneDNN which
  f16 matmul it picks for a small problem via `impl_info_str()`. On the B60: `jit:gemm:any`, i.e. **oneDNN has a
  real JIT (non-reference) GEMM here**, and XMX is present. So oneDNN is alive on this card - the question is only
  whether it is worth using.

## The dense-F16 A/B (`onednn_gemm_bench`)

Methodology, down W[N=2560,K=640], gate/up W[N=1280,K=2560], T = prompt batch, integer grid (|v|<=63, exact in
FP16) plus realistic data (`W ~ N(0,0.02)`, `X ~ N(0,1)`), compared against an fp64 host reference. **Column-major
FF layout is mandatory** (the engine's `gemm` uses `lda=ldb=K, ldc=N`; a row-major buffer silently computes a
different matrix and sent an earlier session chasing fp16-overflow ghosts - see the trap note).

**Numerics: oneDNN and oneMKL are identical.** On the integer grid both are exact (0 error vs fp64 ref); on
realistic data both sit at ~2-6e-7 max relative error and match each other. `impl = jit:gemm:any` throughout.

**Speed (oneDNN/oneMKL ratio, real data, B60):**

| T | down (K=640) | gu (K=2560) |
|---|---|---|
| 16 | 1.38 | 0.78 |
| 32 | 1.24 | 0.73 |
| 64 | 1.01 | 1.01 |
| 128 | 1.08 | 1.19 |
| 256 | 1.14 | 1.16 |
| 512 | 1.00 | 1.76 |
| 1024 | 1.38 | 1.36 |
| 2048 | 1.27 | 1.20 |
| 4096 | 1.21 | 1.08 |

oneDNN is consistently faster at T >= 64 (up to 1.76x at gu T=512) but **loses at tiny T on the gate/up shape**
(0.73-0.78x at T=16-32).

## The engine decides the effective T: per-expert routed batch

`Gemm::f16` is called per expert with `ne = m.cnt[e]` (`sycl/src/prefill/prefill.cpp:3082`), the **routed token
count** for that expert in the chunk - small and tail-heavy (each token increments only the experts it routes to),
not the whole chunk. That is precisely the small-T region where oneDNN's gu GEMM loses. And the dequant phase
(oneDNN cannot touch it - blocked above) dominates the prompt regardless.

## Verdict

A dense-F16 oneDNN substitution at the `Gemm::f16` site is not a clear win on the engine's real call pattern, and
the fused-dequant version that motivated P0b #1 is structurally blocked. **Keep the oneMKL default.** oneDNN stays
linked opt-in (`STRATA_SYCL_DNNL=1`) with `onednn_probe` and `onednn_gemm_bench` in the tree for a future A/B if a
large-batch (dense-experts) prompt path ever lands; the P0b #1 avenue is otherwise closed. Do not reopen without
new measurements (e.g. a dense non-routed prompt variant with T >= 256).

## Trap worth knowing

**Row-major vs column-major for oneMKL/dpct GEMM.** Answering whether the engine's

```cpp
dpct::blas::gemm(h, T, N, T, K, &alpha, W, real_half, K, X, real_half, K, &beta, Y, real_float, N, compute_type::f32);
```

matches a mental "W[k,n], X[k,t]" forces the buffer to be column-major (`W[k,n]` at `k + n*K`, `X[k,t]` at
`k + t*K`, output at `n + t*N`). Fed row-major buffers, oneMKL and a stride-correct oneDNN compute two different
matrices and no fp64 reference (scanned over 8 orientations) matches oneMKL. Verified both paths bite-exactly on
the integer grid only after the layout was fixed.