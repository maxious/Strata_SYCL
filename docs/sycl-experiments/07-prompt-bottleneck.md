# Experiment 07 - prompt path: where the B60's prefill time goes

## Question

The objective's headline is the ~350+ tok/s XMX prefill. INTEL.md's B70 reads a 2,184-token prompt at ~780
tok/s. Where does the single B60's prompt time actually go, and is any README lever (INT8 GEMM, prefill chunk)
still open there?

## Measurement (Coder IQ1_M, single B60, serve, 1,280-token prompt, `STRATA_PREFILL_TIMING`)

    prompt 1280 tokens read in 2522 ms -> 507-571 tok/s (warm; 571.5 with --prefill auto)

GPU timeline 2,522 ms by phase:

    dequant         760 ms (30.1%)   <-- largest
    gemm down       504 ms (20.0%)
    gemm gate/up    216 ms ( 8.5%)
    host grouping   179 ms ( 7.1%)
    gdn             153 ms ( 6.1%)
    qsa attn        144 ms ( 5.7%)
    hc read         125 ms ( 5.0%)
    qsa proj         78 ms ( 3.1%)   ... router+shared 51, combine 41, gather 44, gdn recurrence 88, gdn out proj 86

**dequant + gemm down + gemm gate/up = 1,480 ms = 58.6%** of the prompt. The prompt is bound by the
dequant-then-oneMKL-GEMM expert path, exactly as INTEL.md's XMX note says ("the prompt path is bound by the
dequant that feeds them, not by the products").

Per-expert dequant (xmx_gemm_bench; the Coder's i-quants): type 21 (IQ3_S) 0.036 ms, type 20 (IQ4_NL) 0.087 ms,
type 42 0.018 ms - the write-bound FP16 dequant is the throughput limit.

## Levers tested

- **INT8 prompt GEMM** (README P0 item 3 / INTEL.md planned #6): the GEMMs are oneMKL-XMX FP16; Experiment 02
  measured oneMKL INT8 GEMM 1.5-2.2x faster at full batch but found the pipeline dequant-bound and that the
  per-block scale correction after oneMKL INT8 GEMM (no offset support in the exposed overload) collapses onto the
  fused path already measured slower (0.37x). Confirmed here: the dequant (not the GEMM) is the 30% + 20% pair's
  larger half, so INT8 dequant (half the written bytes) is the real prize, but it is a full new-kernel + GEMM +
  scale-correction rewrite - parked with the Experiment 02 measurement.
- **Prefill chunk size** (measured this experiment): auto 571.5, 1024 576.9, 4096 538.2, 8192 491.7 tok/s. The
  default `auto` chunk is already optimal; larger buffers add no benefit at 1,280 tokens (they fit the prompt in
  the first place). No config lever.

## Validity

- Timings are the engine's own `STRATA_PREFILL_TIMING` marks (GPU timeline 2,522 ms vs wall 2,523 ms - the phases
  sum to the round time, so the split is complete and measured, not estimated).
- tok/s stable across runs (571.5 with auto; 507-577 across chunk sweep).
- Draft acceptance 0/9 on the synthetic prompt; the decode tail (10.8 tok/s) is not the subject.

## Conclusion

The B60 prompt path is **dequant + oneMKL-GEMM bound at 58.6%** (dequant 30% is the single largest phase). The
README/P1 prompt lever is real but its two candidate implementations are both already ruled out or blocked: the
INT8 GEMM pipeline is dequant-bound and scale-handling-blocked (Experiment 02), and the prefill chunk default is
already optimal (measured here). The genuinely-open prize is an **INT8 dequant that writes half the bytes** the
30%+20% dequant+GEMM consumes - a new dequant kernel + oneMKL INT8 GEMM + per-row scale correction, sized as a
standalone Experiment 8 with the dequant rate as the gate.

No engine change in Experiment 7 beyond the measurements. Next: either the INT8-dequant rewrite (Experiment 8,
large) or accepting that the prompt path is at its tuned F16-dequant + oneMKL-GEMM frontier.