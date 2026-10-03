# Experiment 08 - MMQ prefill, and why the CUDA port is the wrong path on Intel

(Being filled in the next session, when the native-SYCL MMQ adoption is implemented. This page records the
decision and the scope.)

## The prompt bottleneck it targets

Experiment 07: the B60 prompt path is dequant-then-oneMKL-GEMM bound (dequant 30.1% + gemm down 20.0% +
gemm gate/up 8.5% = 58.6% of prompt time). The prompt writes ~10 MB of FP16 per expert before multiplying.

## What Strata's MMQ path is

`include/strata/prefill/moe_mmq.hpp` defines `strata::prefill::mmq` (built/supported/fits/quantize/Product):
weights stay quantized, activations round to q8_1, products run on int8 tensor cores (dp4a). It reads each
expert once (~1.4-2 MB) instead of dequantizing then GEMMing. The implementation `sycl/src/prefill/moe_mmq.dp.cpp`
is a migration of llama.cpp's **ggml-cuda** `mmq.cuh`; it was never wired into the build (INTEL.md "Not ported
yet": needs llama.cpp's ggml-cuda sources).

## Wrong first step tried, and why it fails

Attempt: add `moe_mmq.dp.cpp` + `ggml_cuda_host.dp.cpp` to the prefill sources with the fetched ggml-cuda
include path and `GGML_COMMON_DECL_SYCL`. Progressive compile errors, each a CUDA artifact:

1. `cuda_fp16.h` not found (ggml-cuda `ggml-common.h`) -> fixed by the vendored SYCL ggml-common + the SYCL
   declaration macro.
2. `cuda_runtime.h` not found (`ggml_cuda_host.dp.cpp`) -> the CUDA port pulls CUDA runtime types deep into the
   build.

**Conclusion:** ggml-cuda's MMQ cannot live behind Strata's prefill on an Intel card without dragging in the CUDA
toolchain. It is the wrong path. (Reverted; the worktree is clean and the committed baseline builds green.)

## The correct direction (user guidance)

The GPU is Intel - no CUDA. llama.cpp already has a **native SYCL** implementation of the same fused-dequant
int8-tensor-core matmul: `ggml/src/ggml-sycl/mmq.cpp` / `mmvq.cpp` / `vecdotq.hpp`, written in SYCL
(`sycl::half2`, `__dpct_inline__`, warp shuffles, no CUDA headers). This is the kernel to adopt behind Strata's
`strata::prefill::mmq` API, not the ggml-cuda one.

## Scope for the adoption (next session)

1. Wrap llama.cpp's `ggml-sycl/mmq.cpp` `mul_mat_q<?>` kernels behind `strata::prefill::mmq` (same
   quantize/Product interface; the int8-Q8_1 activation layout and the per-type `vec_dot` products transfer
   directly).
2. `ggml-sycl/mmq.cpp` is coupled to llama.cpp's `ggml_tensor`/`ggml_backend_sycl_context` dispatch; the port
   must take only the kernel bodies, not ggml's op plumbing (Strata has its own expert gather bounds / Product).
3. Types: q8_0, and the i-quants the Coder's experts use (IQ2_S/IQ4_NL/IQ3_S-style); IQ1_M is listed as not
   covered - the Coder's gate/up are IQ4_NL/IQ2_S, so the supported set should cover the Coder.
4. Gate: on/off behind the existing `STRATA_PREFILL_MMQ=0` env switch (default ON once `mmq::built()` is real),
   so the FP16 oneMKL path stays the fallback and both can be measured.
5. Parity gate: prompt output identical at the documented tolerance; prompt tok/s vs the 571 tok/s baseline.

## Evidence so far

- Prompt is dequant+GEMM bound at 58.6% (experiment 07).
- ggml-cuda MMQ is un-buildable into the SYCL prefill (this page).
- llama.cpp's ggml-sycl `mmq.cpp` is the native-SYCL replacement (source inspected).
- Baseline 571 tok/s (1,280-token prompt, Coder, single B60), worktree clean, `strata_prefill` builds green.