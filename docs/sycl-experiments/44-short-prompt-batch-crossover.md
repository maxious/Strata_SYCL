# Experiment 44 - the short-prompt batch: where the XMX matvec stops beating dequant + FP16 GEMM (design, not yet run)

**Status: design.** No number below is new; the prompt-path figures are cited from exp 07/09/25 and the chunk
defaults from `prefill.cpp`. This experiment exists because llama.cpp PR 29864 raises its matvec threshold from 8 to
80 columns for XMX weights, and this port has no equivalent call site to compare against.

## Question

The PR dispatches `mul_mat_vec_q` on dense weights when `src1->ne[1]` (tokens in the batch) is 9-80, where the
default threshold is 8. `src1->ne[1]` is the token count of one decode call, so that window is a **short prompt**, the
**tail chunk** of a longer prompt, and - in a server - **continuous batching** (N concurrent sequences). The last
case does not exist here: `docs/DETAILS.md:581` states this port "does not execute requests concurrently".

This port routes **every** prompt chunk through dequant + oneMKL FP16 GEMM (`gemm.native` / `gemm.bf16`), whatever
its width. Widths 9-80 do occur - the first chunk is 256 (`STRATA_PREFILL_FIRST`, default 256), a short prompt is a
single chunk of its length, and `--prefill` chunking leaves small tails - and they take the GEMM path today.

**At what prompt-batch width M does an XMX int8 matvec stop beating dequant + FP16 GEMM?**

## Prior evidence

- **exp 25** (2560x2560, per-call us): the oneMKL FP16 GEMM loses to the native dp4a matvec at ncols 1-4
  (0.54/0.63/0.86x) and crosses only at 6-8 (1.33/2.39x). The GEMM is overhead-dominated at small M.
- **exp 07** (Coder IQ1_M, 1,280-token prompt): dequant 760 ms (30.1%) + gemm down 504 ms (20.0%) + gemm gate/up
  216 ms (8.5%) = 58.6% of the prompt; the path is **dequant-bound**, not product-bound.
- The dequant is per-weight and amortized over the chunk's tokens, so at M=9-80 it is amortized over 30-300x fewer
  tokens than exp 07's 1,280. A matvec that never materializes FP16 is the PR's argument, and this window is where
  it is strongest.
- **exp 09** is the counterweight: a batched i-quant matvec was a ~6x regression at 1,280 tokens (17,540 ms vs
  2,858 ms), because the matvec shape cannot amortize across rows the way a GEMM can. The crossover is real, and the
  matvec must not be pushed past it.

## Proposed change

None in production until a crossover is measured. The experiment is a **measurement**:

1. In-process, the cheap half: run exp 42's `xmx_mmvq_bench` over a wider M sweep - 1..8 as in exp 42, then
   9/16/32/64/80 - with the same four arms plus the oneMKL **FP16** GEMM arm (the current prompt path's kernel).
   One bench serves exp 42 and exp 44; the only new work is the M sweep and the FP16 arm.
2. Engine side, the decisive half: `STRATA_NATIVE=1 STRATA_PREFILL_TIMING=1` at 9-80-token prompts, and at
   `STRATA_PREFILL_FIRST=32/64/80` on a longer prompt, single B60, one run each - the phases say whether the dense
   projections are even a measurable share at those widths.
3. Only if a crossover M* lands in (8, 80]: a dispatch that sends prompt chunks below M* to the XMX matvec, behind
   `STRATA_*` first, then a short-prompt tok/s A/B before any default.

## Gates and traps

- The bench must use the engine's real dense shapes, not only the square (exp 22/36).
- A short prompt is a **single chunk**, and exp 38's trap applies in reverse: do not instrument a chunking or
  stage-overlap question on a one-chunk run. Here the chunk *is* the subject, so state the chunk size in every row.
- Per-call us, not GB/s (exp 27).
- The type scope is exp 42's: the PR's kernel covers k-quants + Q8_0, so this only applies to the dense matrices of
  those forms - not to the Coder's i-quant experts.

## Success criteria

- A measured M* in (8, 80] where the DPAS matvec wins below and the FP16 GEMM wins above -> the window is real and
  the dispatch is worth wiring (opt-in first).
- Crossover at M <= 8 -> null: the GEMM path stays for every prompt chunk, and exp 25's result simply extends.
- Either way, record that the **upside is bounded**: the expert path is 58.6% of prompt time (exp 07) and is not
  touched by this experiment, so even a real M* is a TTFT / short-prompt lever, not a prompt-throughput lever.
