# Experiment 25 - oneMKL dense-FP16 GEMM at the decode batch, against native_mmvq (README P0b #2 + #3)

## Question

README P0b #2 asked whether a oneMKL FP16 GEMM at **M = 1..6** - the decode/verify window - beats `native_mmvq`,
and P0b #3 named the one untried XMX-for-decode version: **dequantize once into a persistent FP16 copy and GEMM on
XMX**, cost = 2x dense weights resident in VRAM. Exp 23 left the matrix units idle while the Q6_K decode MMVQ is
PIPE-bound, so both were "one measurement decides it". Hardware: the B60 box, oneMKL fp16 f32-accumulate.

## What was measured (`decode_xmx_gemm_bench`)

2560x2560 Q6_K dense grid, persistent FP16 weight copy `W[k,n]` resident (column-major `lda=K`), real activations.
Each path timed back-to-back in the same process, warmed ~300 ms; the GEMM path is checked against an fp64 host
reference (max rel err ~1.3-1.6e-7, ok). `native_mmvq` uses the exact `native_quantize_q8_1` + `native_q6_k_mmvq`
calls of `mmvq_bench` (its standalone curve, run in the same session: 19.1/21.5/26.1/35.9/55.4 us, matches).

| ncols | native_mmvq (us) | oneMKL fp16 GEMM (us) | native/gemm | GB/s (GEMM) |
|---|---|---|---|---|
| 1 | 21.4 | 39.9 | 0.54x | 135 |
| 2 | 23.9 | 38.0 | 0.63x | 142 |
| 4 | 29.7 | 34.4 | 0.86x | 156 |
| 6 | 40.9 | 30.8 | 1.33x | 174 |
| 8 | 62.4 | 26.1 | 2.39x | 206 |

## The engine's call pattern decides it

`native_mmvq` is called at **ncols = 1** every layer (`layer.cpp:154`, the dense projection for one token - the
overwhelming majority of decode time) and at **ncols = T** in the drafter (`mtp.cpp`, T up to 6 with `--spec 4`).
The GEMM *loses at every shape the engine actually runs*: at ncols=1..4 it is 1.2-1.9x SLOWER (0.54-0.86x), the
XMX launch/kernel overhead dwarfing a single-column matvec; it only crosses over at the fat drafter window
(ncols>=6), which is a fraction of decode time and needs a persistent 2x-VRAM copy (13.1 MB vs the packed 5.4 MB)
plus the ~12 us one-time materialization (dequant_bench: 5.4 MB at Q2_0's 419-473 GB/s). Even the "win" at ncols=6
(1.33x) does not repay the residency + materialization.

## Verdict

**P0b #2 and #3 are closed: dequant-then-XMX-GEMM is a no-win for the decode matvec.** The pipe-bound `native_mmvq`
stays the decode path; the matrix units were idle for a reason - XMX's backend cost only repays at GEMM-shaped
batch, which decode is not. The decode lever remains P0 #2's perishable work: avenue 1 (a lighter Q6_K unpack/scale
sequence - ops per weight byte on the bottleneck pipe) and avenue 2 (the NCOLS>=5 unroll -> column loop). The
persistent-FP16 approach is parked, not refuted for *other* shapes: oneMKL remains the fast prompt GEMM (exp 07/13)
and the dense-F16 comparison for any future large-batch dense (non-routed) prompt path lives in exp 24's table.

Keep `decode_xmx_gemm_bench` in the tree (built under the PARITY bench set, links oneMKL directly); it is the
canonical re-measure for P0 #2 if a decode-slimming change ever competes a dense GEMM against the MMVQ.

## Trap worth knowing

Tiny-M GEMM timing is dominated by launch/backend overhead, so GFLOP/s is a misleading lens at M<=8 (4.0 TFLOP/s at
ncols=8 yet it is the *winner*); report the per-call **us**. And the FP16 copy must be explicitly persistent - a
per-call dequant-to-FP16 before the GEMM adds back the 5.4 MB read + fill every call (the cold-vs-warm arithmetic
that P0b #2 already flagged).