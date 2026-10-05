# Experiment 43 - DPAS fused gate+up for the shared expert (design, not yet run)

**Status: design.** No number below is new; the profile figures are cited from exp 20/33 and the kernel comments.
This is the companion to exp 42: the same DPAS mechanism, applied to a gate+up that is **already fused in this
port** - the experiment is the engine, not the fusion.

## Question

llama.cpp PR 29864's second kernel computes `up * act(gate)` for both FFN weights in one launch, reads the
activations once, and applies SWIGLU/GEGLU in the epilogue (up to 16 columns). Its point is not the fusion - llama.cpp
already fused gate+up for the reorder path - it is doing the fusion **on XMX**.

This port already fuses gate+up in two places, so the honest question is which of them an XMX gate+up would move:

| gated FFN in this port | layout | state | measured |
|---|---|---|---|
| `native_gu_port<TG,LN>` (routed experts, grouped) | S-form quantized | **already one launch** (gate+up outputs) | 1.426 s in exp 20's decode profile, 2nd-largest kernel; ALU-bound, 77% XVE active at 58 of 608 GB/s (exp 33) |
| `fused_gr_read_multi` (dense GR block) | BF16, 2 B/elem | **already fused**, multi-column (exp 06: default fastest at every width) | - |
| shared expert (`ffn_gate_shexp` / `ffn_up_shexp`) | S-form quantized | two projections + a separate `swiglu_kernel` | kernel comment cites the block at 0.8883 ms, 26.7% of its block - a stale comment, re-measure before quoting |

So there are two candidate targets, and they answer different questions:

- **A - the shared expert.** Dense (n_embd -> n_ff), quantized, currently unfused: the exact shape of the PR's
  kernel. Smaller cost, but the kernel is a near-copy of exp 42's plus the epilogue.
- **B - `native_gu_port`.** The highest-value target - measured the 2nd-largest decode kernel and ALU-bound, which is
  precisely the "move the dot to the matrix pipe" lever - but it is a *grouped* kernel (each expert's token rows
  gathered), so it needs the expert-grouped activation layout as well.

**Recommendation: build A first** (it is exp 42's kernel plus an epilogue and a second weight stream), and extend to
B only if A's win is real. Do not start with B.

## Proposed change (target A)

- A fused variant of exp 42's kernel: two weight streams (`vx` = up, `vg` = gate), two accumulator sets, one
  activation read, the GLU applied to `up * act(gate)` in the epilogue before the store - the PR's `xmx_mul_mat`
  with `FUSED=true` and `xmx_glu_act`.
- Both weight sets must be in the reorder layout; the shared expert's S-form (S2/S4/S8, IQ4_NL codebook) is broader
  than exp 42's Q6_K, so the reorder/unpack traits must cover the forms the pack actually uses. **Count them first**
  (`sycl/tools/gguf_count_dense_usage.py`) - a lever can be real and invisible end to end because the model barely
  uses the type (the skill's step 8).
- Gate and up in this model are separate tensors (`ffn_gate_shexp`, `ffn_up_shexp`), so the fused kernel is a real
  fusion, not a re-labeling of an existing call.

## Instrument

- Extend exp 42's `xmx_mmvq_bench` with a fused arm: gate+up weights, the GLU epilogue, and the reference
  two-GEMV + `swiglu_kernel` sequence as the baseline arm. Per-call us at ncols 1-8, warm, best of 3.
- Correctness on all three values: the gate projection, the up projection, and their product vs an fp64 reference
  (the product hides an error in either input, so check the inputs too).
- Engine side only if the bench wins: the shared expert runs every token, so measure with the decode profile
  (exp 20's method) or `STRATA_PREFILL_TIMING` for the prompt path, and pin everything the arms must share.

## Gates and traps

- The PR's fused kernel keeps **two** sets of weights and scales and always takes the 256-GRF file; a 128-GRF build
  spills. Verify the register file the way exp 21 did (ISA dump, not the source).
- Activation columns must satisfy the same `ncols % QK_K == 0` / 64-byte alignment precondition; the shared expert's
  n_ff may not be a multiple of the tile - check the tail path.
- The epilogue must match the reference's order: this port's `swiglu_kernel` computes `silu(gate) * up` in float64
  internally for parity (the kernel comment records that a float32 exp differs in the last bits). A DPAS epilogue in
  float32 will not be bit-identical - define the tolerance, and do not "fix" it by loosening the reference.

## Success criteria

- Beats the two-projection + swiglu sequence at ncols 1-8 -> opt-in `STRATA_*`, then the engine A/B; if the shared
  expert is a measured slice of decode time, consider target B next.
- No win -> parked, and the record should say the fusion was never the cost (as exp 06 found for the GR variants).
- Note the earlier error this doc corrects: the drafter's `fc_embedding`/`fc_hidden` are two **sequential** FC
  layers, not a gate/up pair - there is nothing to fuse there, and any doc that says otherwise is wrong.
