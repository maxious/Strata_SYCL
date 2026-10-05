# Experiment 00 - baseline: build and validate on two Arc Pro B60

Date: 2026-10-03. Hardware: **two Intel Arc Pro B60 (Battlemage G21)**, oneAPI 2026.1 (IntelLLVM 2026.1.1,
icpx at `/opt/intel/oneapi/compiler/2026.1/bin/icpx`), host cmake 4.2.3 / ninja. Built on the host (no Docker
image available here); DPCT/vendored helpers come from `sycl/`.

This is the "before" snapshot every later experiment is measured against. Reference hardware in INTEL.md is a
single B70 (G31); this box is two G21 cards, so absolute numbers differ - the point is the delta each change
makes on the same box, and that validity (byte/bit parity) is preserved.

## Build

Configure + build in `sycl/build-b60` (this checkout's `sycl/CMakeLists.txt`, ggml pinned at 0.24.0 / commit
3cf03257f):

    source /opt/intel/oneapi/setvars.sh
    cmake -S sycl -B sycl/build-b60 -G Ninja -DCMAKE_CXX_COMPILER=icpx
    ninja -C sycl/build-b60 strata           # engine
    ninja -C sycl/build-b60 $(...all parity + bench targets)   # 62/62 linked

Both Level-Zero devices enumerate: `sycl-ls` shows `level_zero:0` and `level_zero:1`, each
"Arc(TM) Pro B60 Graphics", two `opencl:gpu` discrete entries as well. Build ran with
`-fsycl-default-sub-group-size=32`, `-fp-model=precise`, SPIR-V JIT (no `STRATA_SYCL_AOT`); 33-36 warnings, no
errors.

## Validity baseline (parity suite on B60)

23 kernel parity tests run through their `--selftest` (or default) gate; **22 pass**:

pass: dequant_s2, s2_gemv, shared_expert, gr, gdn, s2_gemv_q8, sampler, rope, quantize_act (0 failures over
5 Q8_0 + 4 Q8_K), router_top10 (512 experts x 64 tokens), s_gemv, elementwise, bf16_gemv, s_gemv_q8k, qsa,
kv_q8, kv_stream (q4_0 ring restore identical), kv_q4, cvec, iq_multi (bitwise vs old kernels), native_grouped
(IQ3_S/IQ4_XS bitwise vs v1).

**fail: `s2_expert_grouped_parity` (4 failures)** - the s2 activation path. This is a pre-existing failure
already recorded in INTEL.md's "Still failing" list (the s2 path uses `__fadd_rn` rounding fixed upstream; not
exercised by the Coder model here). Confirmed unrelated to this build.

## Performance baseline on B60

`xmx_gemm_bench` (IQ3_XXS type 18, M=96, gate/up, N=1280, K=2560):

    dequant 0.078 ms + oneMKL 0.043 ms = 0.126 ms (14542 GFLOP/s on the GEMM)
    xmx (joint_matrix) 0.344 ms (1827 GFLOP/s, 0.37x)

Fused quantized XMX GEMM is 0.37x of dequant+oneMKL on the B60 - same conclusion as the B70: XMX is only won
on the FP16 (oneMKL) dense GEMM, not the per-quant fused path.

`q6k_align_bench` (the alignment lever, 2560-col rows):

    q6_k 2560 x 12288 cols1: stride 210 -> 255.8 GB/s | stride 224 -> 274.0 GB/s (1.07x), rel diff 0
    q6_k 6144 x  2560 cols1: stride 210 -> 339.0 GB/s | stride 224 -> 373.8 GB/s (1.10x), rel diff 0
    q6_k 2560 x 248320 cols1: stride 210 -> 267.0 | stride 224 -> 300.6 GB/s (1.13x)

Alignment (two aligned 16-byte loads) is worth ~7-13% on the single-column Q6_K rows; identical outputs.
The wider multi-column cases flip signs run to run (cols2/6 sometimes favor stride 210) - expected, they're
latency-bound.

`mmvq_bench`: q6_k 2560x2560, 4 cols: 26.1 us / 205.7 GB/s, wide-vs-shared rel 5.449e-08 (within
tolerance).

## Notes for later experiments

- XMX paths in the engine are gated only by env flags (`STRATA_PROMPT_ATTN_XMX=1`, etc.), never by a device
  capability probe. llama.cpp gates with `gpu_has_xmx()` = `dev.has(sycl::aspect::ext_intel_matrix)`. This is
  the gap Experiment 1 closes.
- The parity suite (except the s2 path) is the validity gate every optimization must stay behind on this box.