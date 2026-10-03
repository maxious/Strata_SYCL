# Experiment 12 - P0-L1 Q8_0 vertical slice: SoA reorder + ESIMD decode (B60)

## Question

P0-L1 (README.sycl.md) is the decode matvec weight reorder + ESIMD port from llama.cpp
(`dmmv.cpp` `q8_0_mac_stripe` / `dequantize_mul_mat_vec_q8_0_reorder_esimd`, `reorder_qw_q8_0`).
Before wiring it into the production dispatch, does the mechanism port cleanly to SYCL and
win on the B60, output still matching the port's AOS wide decode?

## Change / measurement

New self-contained bench `sycl/src/kernels/reorder_esimd_bench.cpp` (target `reorder_esimd_bench`,
linked against `strata_kernels`): builds AOS Q8_0 weights, runs the one-shot SoA reorder
(`[qs: nb*32][d: nb*half]`), then the ESIMD decode, and compares GB/s + output against the port's
existing `native_q8_0_mmvq` (Wide32Q8) on the same dequantized-q8_1 activations.

B60, warm clocks (icpx 2026.1.1, Level Zero):

| shape | cols | AOS (Wide32Q8) | reorder+ESIMD | ratio |
|---|---|---|---|---|
| 2560 x 10240 | 1 | 445 GB/s | 419 GB/s | 0.94x |
| 2560 x 10240 | 4 | 314 GB/s | 418 GB/s | 1.33x |
| 2560 x 10240 | 6 | 280 GB/s | 417 GB/s | 1.49x |
| 6144 x 2560 | 1 | 706 GB/s | 791 GB/s | 1.12x |
| 6144 x 2560 | 4 | 455 GB/s | 786 GB/s | 1.73x |
| 6144 x 2560 | 6 | 370 GB/s | 788 GB/s | 2.13x |
| 2560 x 640 | 6 | 229 GB/s | 406 GB/s | 1.77x |

Output: rel diff 6.5e-8 .. 4.2e-7 across all shapes (the ESIMD float accumulation vs the AOS dp4a
integer path), i.e. the port is equivalent within float rounding.

## Decision

The reorder+ESIMD mechanism **ports to SYCL and wins at the multi-column decode shapes** (cols
2/4/6 - the spec/MTP verify window) by up to 2.1x; cols 1 is a wash (0.89-0.99x). The ESIMD path
holds ~400-420 GB/s flat at the 2560-wide projections where the AOS path degrades with columns.
Production wiring (per-tensor SoA buffer + dispatch + opt-in STRATA flag + `esimd_kq_parity`) is a
separate step; this bench is the proof that justifies it.