# Experiment 26 - the Q6_K wide MMVQ NCOLS unroll vs a runtime column loop (README P0 #2 avenue 2)

## Question

The README's P0 #2 avenue 2: the dense Q6_K decode MMVQ unrolls all NCOLS columns into one work-item, and the
register blow-up is a symptom - NCOLS=8 spills 6272 B in 1631 instructions vs 384 B / 985 at NCOLS=4 (exp 22), and
the per-added-column-pair cost roughly doubles (35.9 -> 55.4 us at 6 -> 8). Would a small *column loop* instead of
the full unroll fix the blow-up? Hardware: the B60 box, the engine's `native_mmvq_q6k_wide_a2` kernel.

## What was built (`native_mmvq.dp.cpp`)

`native_mmvq_q6k_wide_loop_kernel`: the a2 aligned-load Q6_K wide kernel with the activation load *fused into* the
dot and NCOLS a runtime loop (`#pragma unroll 1`), so only one column's activation (u0/u1/ds) is live at a time;
acc[NCOLS] stays live (each column accumulates independently). Opt-in behind `STRATA_MMVQ_LOOP=1` (default off),
applied only for `NCOLS >= 5` so 1-4 keep the unrolled kernel. Correctness: mmvq_bench's `wide vs shared` line is
bit-identical to the unrolled kernel at every ncols (same `first -87393.8`, rel 5.45-5.50e-08).

## Measured (mmvq_bench 2560x2560, two runs each)

| ncols | unrolled (us) | loop (us) | unroll/loop |
|---|---|---|---|
| 1 | 19.2 / 19.1 | 19.1 / 19.1 | 1.00x |
| 2 | 21.5 / 21.5 | 21.5 / 21.5 | 1.00x |
| 3 | 24.7 / 24.7 | 24.7 / 24.8 | 1.00x |
| 4 | 26.1 / 26.1 | 26.1 / 26.2 | 1.00x |
| 5 | 30.1 / 30.1 | 30.6 / 30.6 | 0.98x |
| 6 | 35.9 / 35.9 | 33.5 / 33.5 | 1.07x |
| 7 | 39.4 / 39.4 | 37.5 / 37.5 | 1.05x |
| 8 | 55.4 / 55.5 | 40.4 / 40.4 | 1.37x |

The superlinear 7 -> 8 jump that motivated the item collapses: unrolled goes 39.4 -> 55.5 us (+41%), loop 37.5 ->
40.4 us (+8%). The mechanism (register pressure, exp 22's 6272 B spill) is confirmed by the flattened slope; the
ncols=1-4 unroll is untouched by the NCOLS>=5 gate.

## The engine call pattern decides the value

Every layer decodes at ncols=1 (`layer.cpp:154`) - first curve row, identical 1.00x, so the loop does nothing for
the primary decode. The drafter (`mtp.cpp`, T up to 6 with `--spec 4`) is where it pays: ncols=6 is 1.07x faster
and ncols=8 (larger spec) 1.37x. ncols=5 is a 0.98x regression. So this is a *drafter-window* lever, opt-in, not
the primary decode lever.

## Verdict

**Avenue 2 is validated but not a net decode lever.** The column loop does fix the NCOLS>=6 register blow-up
(1.07x at 6, 1.37x at 8, superlinear tail flattened) at no cost to ncols 1-4, and stays opt-in
(`STRATA_MMVQ_LOOP=1`, default off) because ncols=5 is 0.98x. The engine's dominant ncols=1 decode is unchanged, so
the primary P0 #2 decode lever remains avenue 1 (a lighter Q6_K unpack/scale sequence - ops on the bottleneck pipe
per weight byte). Keep the loop kernel in the tree as the canonical opt-in for a large-batch drafter; if a spec
with a wider MTP window (T>=6) matters, flip it on.

## Traps worth knowing

- The opt-in gate must default **off** (`getenv(...) != nullptr && ...`, not the `q6k_a2` default-on pattern) or the
  whole engine silently runs the loop variant.
- A template-constant NCOLS loop bound alone does NOT force a runtime loop - the compiler still unrolls it; the
  `#pragma unroll 1` is what makes it a real loop and drops the simultaneous column registers.
- The loop variant is only a win where the unroll was spilling (NCOLS>=6); per-NCOLS dispatch (the existing switch
  in `native_q6_k_mmvq`) is how the hybrid keeps 1-5 (unroll) and 6-8 (loop).