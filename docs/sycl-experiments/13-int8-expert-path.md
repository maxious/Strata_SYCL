# Experiment 13 - the INT8 prompt expert GEMM, built end to end (refuted)

## Question

Exp 02 measured the INT8 oneMKL GEMM alone and found 1.5-2.2x at the full prompt batch, then parked it: "the
GEMM headroom is real, but it does not reach the engine's bottleneck on this card" - the prompt path is
*dequant-bound*, and for the i-quants the dequant is LUT-bound (exp 10, exp 11).

**Q2_0 reopens it.** `dequant_bench` measures Q2_0's dequant at **419-473 GB/s (bandwidth-bound)** against
IQ4_NL 68-75, IQ2_XS 192, IQ2_S 183 GB/s (LUT-bound), and Q2_0 is the size the port now tells people to take.
The earlier conclusion was drawn on the Coder's IQ1_M/IQ4_NL; on Q2_0 the dequant that gated the INT8 GEMM is
~6x cheaper. So: **once the whole expert is built the INT8 way - not just the GEMM - does it pay?** Exp 02
measured a GEMM; this measures an expert.

## Change

The full path, in the tree, default off:

| file | what |
|---|---|
| `sycl/src/kernels/cuda/iq_kernels.dp.cpp` | `iq_quant_gu_i8` / `iq_quant_i8`: Q2_0 (ggml type 42, `block_q2_0 = {f16 d; uint8_t qs[16]}` per 64 values, `w = d*(code-1)`) -> int8 with one scale per output row, in both `iq_dequant_gu_f16`'s interleaved gate/up layout and the flat down layout |
| `sycl/src/prefill/kernels.dp.cpp` | `quantize_act_i8` (FP16 activation -> int8 + per-row scale), `scale_rows_i8` (the epilogue that applies both rows' scales) |
| `include/strata/prefill/gemm.hpp`, `sycl/src/prefill/gemm.dp.cpp` | `Gemm::int8` - `dpct::blas::gemm` with `real_int8` operands and an `real_float` result (the exact integer sums; K=2560 of \|v\|<=127 is 41e6 < 2^31) |
| `sycl/src/prefill/prefill.cpp` | the FP16 expert branch takes this pipeline when `STRATA_PREFILL_INT8=1` and the pack's gate/up and down are both Q2_0 |
| `sycl/src/kernels/int8_path_bench.cpp` | the whole expert both ways on a **real expert from the shard**, fidelity and per-phase time together |

The scale contract is the standard dynamic W8A8 one: `sW[n] = max|W[n,:]|/127`, `sX[t] = max|X[t,:]|/127`, so
`Y[t,n] = sX[t]*sW[n]*(Xq.Wq^T)`. INT8 is not exact here and cannot be: **one scale per row cannot carry Q2_0's
per-64 block scales**, which is where the fidelity below comes from.

## Measured - speed (B60, one real Q2_0 expert, layers 0/1/20/47, `int8_path_bench <shard> <layer> <T>`)

The whole expert, FP16 dequant+oneMKL against INT8:

| T (the expert's token batch) | FP16 | INT8 | INT8/FP16 |
|---|---|---|---|
| 96 | 0.052 ms | 0.122 ms | **0.42x** |
| 256 | 0.060 ms | 0.146 ms | 0.41x |
| 640 | 0.145 ms | 0.270 ms | 0.54x |
| 1280 | 0.261 ms | 0.451 ms | 0.58x |
| 2560 | 0.489 ms | 0.765 ms | 0.64x |

The INT8 GEMM itself is the win exp 02 measured - **gate/up 2.11x at T=96 and 1.87x at T=640**, down 1.65x at
T=640 (0.99x at T=96, where both are 0.010 ms and the timer is at its resolution). The expert is 0.42x. The
phase breakdown (T=96 / T=640) says why:

| phase | FP16 | INT8 |
|---|---|---|
| weight dequant -> requant | 0.014 / 0.014 ms | **0.065 / 0.064 ms** |
| activation quantize (2 calls) | - | 0.025 / 0.031 ms |
| gate/up GEMM | 0.022 / 0.056 ms | 0.010 / 0.030 ms |
| down GEMM | 0.010 / 0.028 ms | 0.010 / 0.017 ms |
| rescaling epilogue (2 calls) | - | 0.009 / 0.049 ms |

## Measured - fidelity

Against the FP16 path, which is **exact** for Q2_0 (its values are exact in FP16), median relative error per
output, four layers:

| output | rel-mean | \|rel\| median | p90 |
|---|---|---|---|
| gate/up | 1.0% | 1.0% | 6.3-6.7% |
| down | 2.4-2.7% | 2.4-2.6% | 15.1-16.6% |

Down is worse because its reduction is short (K=640) and its input is the SwiGLU output, which is itself
quantized to int8 per row before the second GEMM.

## Measured - in the engine

The Q2_0 pack end to end on the B60, 5-token prompt, warm, `--prefill 128`, two runs each:

| run | FP16 (default) | `STRATA_PREFILL_INT8=1` |
|---|---|---|
| 1 | 397.0 ms (12.6 tok/s) | 501.7 ms (10.0 tok/s) |
| 2 | 395.8 ms (12.6 tok/s) | 523.3 ms (9.6 tok/s) |

Same direction as the bench - **1.29x slower** - so the wiring is right and the loss is real. (A 5-token prompt
is too short to see the expert GEMM in isolation; most of it is fixed cost and expert streaming. The kernel-level
bench above is the measurement; this is the corroboration that both paths run and which one is slower.)

## Why it fails

Three compounding reasons, in order of size:

1. **Requantizing to INT8 costs more than dequantizing to FP16** (0.065 ms vs 0.014 ms, 4.6x). "Half the dequant
   bytes" counts the *write* only: int8 writes half the bytes, but the per-row scale needs the row's max, so the
   kernel reads the codes **twice** (a max scan then the write) and reduces across the row - where
   `iq_dequant_f16` makes one pass and needs no scale at all.
2. **The batch is far too small for a GEMM-rate win.** A routed expert sees `ne ~ T/51` tokens, so the two GEMMs
   are ~0.032 ms of a ~0.051 ms expert at T=96. At T=96 the INT8 pipeline's non-GEMM work (0.065 requant + 0.025
   act-quant + 0.009 epilogue) is 0.085 ms more than the FP16 dequant it replaces, against a **0.012 ms** GEMM
   saving. At T=2560 the ratio only improves to 0.64x because the epilogue (which scales `T*1280` and `T*2560`
   floats) grows with T too.
3. **The fidelity cost is not optional.** Per-row int8 cannot represent Q2_0's per-64 block scales; the only
   structures oneMKL's int8 GEMM accepts are per-tensor or (via a separate epilogue) per-row. Keeping the block
   scales would mean splitting K into 64-value runs - 40 GEMMs per expert.

## Conclusion

**Parked, opt-in, default off.** `STRATA_PREFILL_INT8=1` selects it; the FP16 dequant+oneMKL path is unchanged
and stays the default. This closes exp 02's open question the other way: the INT8 GEMM's 1.5-2.2x is real, and
it still loses once the requantization that has to feed it is counted. It also corrects the "Prefer Q2_0" note in
README.sycl.md, which claimed Q2_0 "keeps the INT8 GEMM win live" - it does not; **Q2_0 stays the pick because
its dequant is cheap, not because an INT8 GEMM pays.**

What that leaves of P0 is the *dequant-speed* half, and only for the i-quants: `dequant_bench` says Q2_0's is
already bandwidth-bound (419-473 GB/s of the card's ~608), while IQ4_NL reads 12% and IQ3_XXS/IQ3_S 30% of card
bandwidth. That is a Strata-side kernel project on the types the port does *not* recommend - so it no longer
justifies repacking anyone onto Q2_0 for a GEMM reason.

## Fix (2026-10-06): the prompt path's counters did not count the int8 buffers

The two hand-kept counts that size the borrowed chunk - `moe_set_bytes` for the region and `bytes_needed` for the
whole loan, which the block above them documents as "the same `take` sequence `init` uses" - omitted every buffer
`carve` takes when `STRATA_PREFILL_INT8=1`: `Xq8`, `Hq8`, `sx`, `sh`, and `Xs`/`Hh`, which the int8 branch takes
even when MMQ covers every layer. They are 96 KB a token at this artifact's K=10 / N=2560 (221 MB at a 2,304-token
chunk, 394 MB at 4,096). The region is `max(gdn, qsa, moe)`, so what the loan was short by is the part above the
old max - 70 MB at chunk 2,304, 131 MB at 4,096 - against the count's 8 MiB slack.

Measured on the B60, Q2_0 pack, 256K config (`--max-context 262144 --vram-reserve-mib 2048 --prefill 4096`, warm,
one run each; the before build is d672fe2, the after one is the same tree plus this fix):

| build | prompt | loan | result |
|---|---|---|---|
| before | 2,185 tokens | 1024 slots (1.32 GiB) | `prefill: device buffers for a chunk of 2304 tokens do not fit` |
| after | 2,185 tokens | 1077 slots (1.39 GiB) | prefill 2,184 tokens in 2 chunks, 327.3 tok/s; decode 32.1 tok/s |
| after | 8,000 tokens | 1680 slots (2.16 GiB) | prefill 7,999 tokens in 3 chunks, 530.3 tok/s; decode 35.6 tok/s |

The loan is a function of the chunk alone: the same config without the int8 path borrows 1585 slots (2.04 GiB)
for a 4,096-token chunk, so the 1,680-slot loan above is the added terms.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh
cd sycl/build-b60 && ninja int8_path_bench strata
SHARD=~/ComfyUI/koboldcpp/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
./int8_path_bench "$SHARD" 0 96 100     # fidelity + per-phase time, layer 0
STRATA_PREFILL_INT8=1 ./strata --pack pack/full --native "$SHARD" --spec 2 --prefill 128 \
    --tokens 9707,11,1879,1130,374,279 --max-new 4 --max-context 2048
```
