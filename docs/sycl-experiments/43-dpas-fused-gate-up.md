# Experiment 43 - DPAS fused gate+up: the fusion is real, the target shape is not what the design assumed (measured 2026-10-05)

**Status: measured, bench-only.** A fused DPAS gate+up kernel (two weight streams, one activation read, the GLU in the
epilogue) beats the two-projection + `swiglu_kernel` sequence at **every ncols 1-8 on both shapes measured**
(1.05-1.71x, largest at decode width). It is **not** an engine win yet, for two measured reasons: the bench baseline is
two *DPAS Q6_K* projections while the engine's shared expert runs S-form `native_mmvq`, and **only 26 of 96 gate/up
tensors are Q6_K**. Nothing is wired into the engine.

## Question

llama.cpp PR 29864's second kernel computes `up * act(gate)` for both FFN weights in one launch, reading the
activations once and applying the GLU in the epilogue. This port already fuses gate+up in two places, so the honest
question is which of them an XMX gate+up would move. The design recommended building the **shared expert** first
(dense, quantized, currently unfused: `ffn_gate_shexp` / `ffn_up_shexp` + a separate `swiglu_kernel`).

## What was built

`sycl/src/kernels/xmx_gateup_bench.cpp` (582 lines, CMake target `xmx_gateup_bench`), a sibling of exp 42's bench so
exp 42's published plateau stays untouched:

- exp 42's single-projection kernel copied verbatim (`xmx_dpas_q6k_kernel`, lines 50-124) - it is the **self-check**:
  in the same process it reproduces exp 42's 17.0-17.3 us at 2560x2560 (published 17.3) and 13.2 us at 640 (published
  13.5).
- the fused kernel `xmx_dpas_q6k_fused_kernel` (156-264) with a `STORE_GLU` dispatch wrapper (266-291): two weight
  tiles, two accumulator sets, the activation read once, `silu(gate) * up` applied before the store.
- the port's `swiglu_kernel` reproduced as `swiglu_f64` (300-312) as the baseline's second phase.

## Correctness (all three values, fp64 host reference, tolerance declared before the run)

Tolerance: gate < 1e-2, up < 1e-2 (exp 42's DPAS plateau is 3.4e-03 rel-RMS, so ~3x headroom), product < 2e-2.
Observed at 2560x640, all ncols: gate 3.4-4.1e-03, up 4.3-5.0e-03, **fused product identical to the baseline product
to every printed digit** (4.5-5.8e-03). The gate and up projections are read out of the **fused arm's own
accumulators** (a second `STORE_GLU=false` instantiation sharing the exact DPAS accumulation with the timed arm), so a
product error cannot hide behind an input error.

**The design's "float32 epilogue will not be bitwise" is true and irrelevant: the float32-vs-fp64 `silu` gap is
5e-08 to 9e-08 relative**, five orders of magnitude below the int8 dot's 3.4e-03. That number is worth having - it means
the epilogue does not need an fp64 path.

## Measured (per-call us, one process, warm 300 ms, 400 reps, `ZE_AFFINITY_MASK=1`)

**2560 x 640 - the shared expert's real gate/up shape**

| ncols | 1 | 2 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| fused | 16.1 | 21.9 | 30.6 | 43.1 | 53.1 |
| unfused (2 proj + swiglu) | 27.6 | 35.3 | 45.6 | 56.1 | 67.0 |
| win | **1.71x** | 1.61x | 1.49x | 1.30x | 1.26x |

**2560 x 2560 (square)**

| ncols | 1 | 2 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| fused | 24.5 | 30.5 | 41.4 | 62.5 | 77.8 |
| unfused | 35.5 | 43.5 | 56.1 | 70.3 | 81.8 |
| win | **1.45x** | 1.42x | 1.35x | 1.12x | 1.05x |

Independently re-verified at 2560x256 (a third shape): 1.77x at ncols 1 falling to 1.22x at ncols 8, all arms within
tolerance. The trend - biggest win at decode width, shrinking toward 8 - is what the mechanism predicts (one launch,
one activation read, no elementwise pass).

## Two measured facts that keep this from being an engine win

1. **The shared expert's `n_ff` is 640, not 10240.** Read from the GGUF header
   (`Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf`): `blk.N.ffn_gate_shexp.weight dims (2560, 640)` and
   `ffn_down_shexp dims (640, 2560)` in every layer. So the target is the **small** shape - and at small shapes exp 42
   found the DPAS kernel *loses* to the shipped kernels (0.47x at 640x2560). The 1.26-1.71x above is the **fusion
   lever** (one launch vs three), not the DPAS lever, and the engine's current shared-expert cost is S-form
   `native_mmvq` x2 + swiglu, which this bench does not measure.
2. **Only 27% of gate/up tensors are Q6_K.** Across the 96 gate+up tensors: Q4_K 30, IQ4_XS 28, **Q6_K 26** (13 gate +
   13 up), Q5_K 12. A Q6_K-only wiring would leave **73%** of the shared expert's gate/up untouched - the design's own
   step-8 warning, confirmed with a count rather than an assumption. (IQ4_NL is the *down* projection only.)

Still open from the design: the 256- vs 128-GRF spill gate needs an ISA dump, and the reorder traits must cover
Q4_K/Q5_K/IQ4_XS before any of this is visible in the engine.

## Verdict

- "Beats the two-projection + swiglu sequence at ncols 1-8" - **yes, at the bench level, 1.05-1.77x**, on every shape
  and width measured.
- "No win -> parked" - **not applicable; there is a win.** But it is a win on a *baseline that does not exist in the
  engine*, on the one shape where the DPAS lever is weakest, for 27% of the tensors. **The honest next step is the
  non-Q6_K reorder traits, then an engine A/B** - not a default flip.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh
ninja -C sycl/build-b60 xmx_gateup_bench
ZE_AFFINITY_MASK=1 ./sycl/build-b60/xmx_gateup_bench 640 2560        # the shared expert's real shape
ZE_AFFINITY_MASK=1 ./sycl/build-b60/xmx_gateup_bench 2560 2560
```

Harness and logs: `/home/maxious/exp43-harness/`; independent re-verification: `/home/maxious/exp43-verify.log`.
`ctest`: 30/30.

**A trap this experiment hit, worth keeping:** the first run had no device selector and landed on the card another
measurement was using. The self-check caught it - the exp 42 arm read 56.9 us instead of 17.3 - and the numbers above
are from a re-run pinned to the idle card with `ZE_AFFINITY_MASK=1`. Nothing here rests on the contended run.