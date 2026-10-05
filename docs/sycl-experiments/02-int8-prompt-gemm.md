# Experiment 02 - INT8 prompt expert GEMM on XMX (headroom measurement)

## Question

README.sycl.md P0 item 3 / INTEL.md planned item 6: *"INT8 prompt GEMMs: experts dequantized to INT8, oneMKL/oneDNN
INT8 on XMX (half the dequant bytes, 2x rate)."* The engine's prompt path (`sycl/src/prefill/gemm.dp.cpp`)
dequantizes quantized expert weights to FP16 (`dequant_f16`), then runs `oneapi::mkl::gemm` (half x half -> float)
on XMX. Does switching the GEMM to INT8 actually buy anything, given INTEL.md's claim that the prompt path is
*dequant-bound, not product-bound*?

Reference: llama.cpp keeps its flash-attention XMX GEMM on F16 (converting K/V to F16 first, `fattn-mkl.cpp`);
its expert/prefill weights also decompose to F16. So llama.cpp's SYCL does **not** run quantized expert weights as
INT8 GEMM either. This experiment measures whether Strata should diverge.

## Change

New `sycl/src/kernels/int8_gemm_bench.cpp` (target `int8_gemm_bench`, registered in `sycl/CMakeLists.txt`):
times `oneapi::mkl::gemm` FP16 vs INT8 (`dpct::blas::gemm`, `real_half` vs `real_int8`, F32 output) at the exact
engine expert-GEMM shapes (down: W[N=2560,K=640]; gate/up: W[N=1280,K=2560]) across prompt batch sizes T. No
engine source changed; this is the measurement the README requires before building a quantize-to-int8 pipeline.

## Benchmark: FP16 vs INT8 GEMM on the B60 (oneAPI 2026.1, warm clocks)

| shape | T | fp16 | int8 | int8/fp16 |
|---|---|---|---|---|
| down N=2560 K=640 | 96 | 0.014 ms (23.2 TFLOP/s) | 0.009 ms (35.3 TFLOP/s) | **1.52x** |
| gu N=1280 K=2560 | 96 | 0.043 ms (14.7 TFLOP/s) | 0.019 ms (32.3 TFLOP/s) | **2.20x** |
| down N=2560 K=640 | 8 | 0.009 ms (2.82 TFLOP/s) | 0.009 ms (2.90 TFLOP/s) | 1.03x |
| gu N=1280 K=2560 | 8 | 0.012 ms (4.46 TFLOP/s) | 0.010 ms (5.36 TFLOP/s) | 1.20x |

The gain is only at the full prompt batch (T=96, a full 4096-token prefill chunk): 1.5-2.2x. At small batches
(T=8, e.g. a streaming ring or a short chunk) INT8 is within noise (1.0-1.2x) - these are launch/latency-bound.

## Validity: INT8 GEMM is bit-exact

The benchmark uses **integer-valued operands on a shared grid** (`v in [-63,63]`, exact in both fp16 and int8; K
products fit exactly in an fp32 accumulation), so a correct INT8 GEMM must match the FP16 GEMM to within exactness:

    int8-vs-fp16 exactness (same int grid): max err 0 mean err 0, 0/245760 outside 1% (nonfinite 0)

All four shapes: **max err 0**. oneMKL's INT8 GEMM primitive is sound on this card.

## Why the pipeline is still not the next lever

The GEMM headroom is real, but it does not reach the engine's bottleneck on this card:

1. **The prompt path is dequant-bound, not product-bound.** From the Experiment 00 / `xmx_gemm_bench` run at the
   same shapes: dequant 0.078 ms vs the FP16 GEMM 0.043 ms (gu M=96) - the dequant is ~1.8x the GEMM. Halving the
   GEMM to 0.019 ms still leaves ~0.078 ms of dequant: the total only drops ~0.024 ms on a 0.126 ms path. Only an
   INT8 dequant (half the written bytes) would bite, and that is a separate kernel with its own cost.
2. **oneMKL's INT8 GEMM here cannot carry per-block dequant scales.** Quantized weights have a distinct `d` scale
   per block along K; an INT8 expert GEMM must apply those per-K-block scales, but the dpct `real_int8 -> real_float`
   GEMM takes no offset/scaling operands (and llama.cpp solves this by keeping weights on F16). Applying the scales
   would need either the fused kernel that already exists or a grouped post-pass - both of which are the
   `xmx_gemm_iq` shape, and that is already measured **0.34-0.37x** (slower) than dequant + oneMKL (Experiment 01,
   INTEL.md's "parked" list).
3. **The evidence agrees with the two XMX lessons in README.sycl.md.** XMX is for fat, dense, static-size work
   (here: only the full prefill batch, not the streaming/T=small path), and layout/dequant bytes beat product units
   for the memory-bound prompt GEMMs.

## Conclusion

Measured, bit-exact: **oneMKL INT8 GEMM is 1.5-2.2x faster than FP16 at full prompt batch (T=96) on the B60**, and
0/245760 outputs off on a shared integer grid. But it does not move the engine's binding cost, which is the dequant
(0.078 ms vs 0.043 ms GEMM), and a correct INT8 expert GEMM needs per-block scales that the exposed oneMKL overload
cannot carry - collapsing onto the fused `xmx_gemm_iq` path already measured slower (0.37x). Recommend **parking
the INT8 prompt GEMM item** (README P0 item 3) with this measurement, unless an INT8 dequant kernel (half the write
bytes) is built and measured against `dequant_f16` first - that is the part that would actually hit the dequant-bound
cost. No engine code changed in this experiment.