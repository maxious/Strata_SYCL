# Experiment 45 - QSA block selection on XMX: the one prompt phase that scales with context, not tokens (design, not yet run)

**Status: design.** No number below is new. The share figures are exp 38's; the kernel structure is the CUDA
tensor-core scorer already in the tree. This is INTEL.md's open item 4, and it is the only XMX item in the plan that
is neither parked nor blocked.

## Question

The QSA select scores every (query, pooled block) pair:

    score(q, b) = sum over the 4 indexer heads of relu(q_h . k_b),   K = IDX_DIM = 128

(`qsa_select.dp.cpp`: `IDX_DIM = 128, IDX_HEADS = 4, R = 4`; the block count is `cells / 4`). On SYCL it runs the
**warp kernel** (`block_scores_kernel`: one work-item per (query, block), a K=128 dot per head). The two tensor-core
selectors live in the same file and are disabled on SYCL:

- the TF32 `mma.sync` scorer sits under `#elif 0   // SYCL: inline PTX (mma/ldmatrix/cp.async) - the XMX port is
  pending; see tools/fixups.py`, so `STRATA_SEL_SM80 = 0` and `qsa_block_scores_tc` refuses the device;
- the gfx12 bf16 WMMA scorer is `__gfx1200__`-guarded.

**Does an XMX (DPAS) select beat the warp kernel at the shapes the prompt path actually runs, and does it keep the
selections?**

## Why this is the one XMX item that is not parked

- **Expert dots on DPAS are parked** for a structural reason llama.cpp PR 29864 does not change: an expert receives
  1-8 rows, and the three `joint_matrix` versions behind `STRATA_EXPERT_XMX=1` were 1.4x and 2-3x slower than dp4a
  (INTEL.md).
- **Decode XMX is parked** on exp 25 (a persistent-FP16 GEMM, 0.54-0.86x at the engine's real ncols).
- **The select is a real GEMM**: 16+ queries per tile, K=128, and a column count of `context/4`. That is the shape
  XMX exists for. It is also the **only prompt phase whose work grows with the context rather than with the
  tokens**: at a fixed chunk, dequant and the GEMMs are constant while the block count rises. The port ships
  `--max-context 262144`.

## Prior evidence

- **exp 38** (dual B60, 8,000-token prompt, 7 chunks, `STRATA_PREFILL_TIMING=1`): `qsa select` is **0.2-4.1% of each
  stage timeline**. That is the *low-context* end - 8,000 tokens is 2,048 blocks; the shipped 262,144 context is
  65,536 blocks, 32x more, with the token-scaled phases unchanged.
- **exp 38's trap**: a single-chunk prompt (2,185 tokens with `--prefill 4096`) reported select as **96.7%** of a
  stage timeline. That was an artifact of a one-chunk run. A select measurement needs a prompt several times
  `--prefill`.
- **exp 37's trap**: in `qsa_select_bench` the *capacity* argument drives the dispatch and the buffers while the
  *context* drives the work. Set `capacity_cells == context` or the rows measure the wrong dispatch.
- The two CUDA scorers are the reference design: `TC_QT=16` queries/CTA, `TC_NB=32` blocks/tile, `TC_ITER=4`, 3xTF32
  (hi*hi + hi*lo + lo*hi); the gfx12 one is a 3-way bf16 split. **Neither is bitwise against the warp kernel** (the
  summation order differs), so the contract is "the same selections", not identical scores.

## Proposed change

A third scorer, XMX/DPAS, added as a bench arm first - never wired before it is measured:

- keep the 4 indexer heads **separate** in the accumulator (the relu is per head, so the heads cannot be folded into
  one K reduction) and sum `relu(acc_h)` at the end, exactly as the CUDA kernels do;
- precision: Xe2 DPAS takes bf16 and tf32; use the gfx12 3-way split (hi/mid/lo) or 3xTF32 for FP32-level accuracy;
- the product is **triangular**: only blocks `< n_bid(query)` are scored, and the tail block `n_bid` (the `dead`
  key, +1e9) is scored by the warp kernel's own code. A tile kernel must early-exit on `hi_nbid` and leave the tail
  to the warp kernel - the CUDA design does both;
- start from `joint_matrix` (already used by the port's IQ prompt kernels in `iq_kernels.dp.cpp`) rather than raw
  ESIMD DPAS; the PR's `xmx::dpas` is the fallback if `joint_matrix`'s tiling does not fit.

## Instrument

`qsa_select_bench [context] [queries] [reps] [capacity_cells]` already times the warp and tensor-core scorers, checks
the score error against an FP64 host reference, and prints `selections identical N/M, cells differing X%`. On SYCL it
currently prints `tensor-core scorer not available on this device`. Add the XMX scorer in the `qsa_block_scores_tc`
slot and sweep **context 8,192 / 32,768 / 131,072 / 262,144 x queries 256 / 1,024 / 4,096** (the chunk sizes), with
`capacity_cells == context`.

**Measure the share before building the kernel.** The engine-side number that decides everything is the select's
share of a stage timeline at long context, on a prompt several times `--prefill` (exp 38's method). If the share is
still a few percent at 262K, a kernel win cannot reach the engine and this closes as a measurement.

## Success criteria

- The select's share **grows materially with context** on a real multi-chunk prompt (the hypothesis), and the XMX
  scorer beats the warp scorer by more than the noise floor with selections identical (or a near-tie count inside the
  bench's own rule, `err_new <= max(4 * err_old, 1e-6 * scale)`) -> opt-in `STRATA_*`, then a **long-context** prompt
  tok/s A/B. A short-prompt A/B would show nothing, for the same reason exp 38's one-chunk probe lied.
- The share stays small at 262K -> record it and keep the warp kernel; INTEL item 4 closes as "the product grows, the
  share does not".

## Honest ceiling

exp 38 puts the select at 2.7-4.1% of a chunk at 8,000 tokens. Even if the XMX kernel removed the whole phase, the
prompt gain is bounded by that share at that context - which is why the first deliverable here is the long-context
share, not the kernel. This is a "measure, then maybe build" experiment, and the doc should stay honest about that.
