# Experiment 12 - P0-L1 Q8_0 vertical slice: SoA reorder + ESIMD decode (B60)

## Question

P0-L1 (README.sycl.md) is the decode matvec weight reorder + ESIMD port from llama.cpp
(`dmmv.cpp` `q8_0_mac_stripe` / `dequantize_mul_mat_vec_q8_0_reorder_esimd`, `reorder_qw_q8_0`).
Before wiring it into the production dispatch, does the mechanism port cleanly to SYCL and win
on the B60, output still matching the port's AOS wide decode?

## Change / measurement

New self-contained bench `sycl/src/kernels/reorder_esimd_bench.cpp` (target `reorder_esimd_bench`,
linked against `strata_kernels`): builds AOS Q8_0 weights, runs the one-shot SoA reorder
(`[qs: nb*32][d: nb*half]`), then the ESIMD decode, and compares GB/s + output against the port's
existing `native_q8_0_mmvq` (Wide32Q8) on the same dequantized-q8_1 activations.

The ESIMD DMMV is SINGLE-COLUMN (llama.cpp runs it per token position), so the multi-column cost is
measured by running it once per column; the AOS Wide32Q8 kernel is column-vectorized (one launch serves
all `nc` columns). Same total work, same dequantized activations, same shapes.

B60, warm clocks (icpx 2026.1.1, Level Zero):

| shape | cols | AOS (Wide32Q8) | reorder+ESIMD (per col) | ratio |
|---|---|---|---|---|
| 2560 x 10240 | 1 | 445 GB/s | 419 GB/s | 0.94x |
| 2560 x 10240 | 2 | 408 GB/s | 209 GB/s | 0.51x |
| 2560 x 10240 | 4 | 314 GB/s | 105 GB/s | 0.33x |
| 2560 x 10240 | 6 | 280 GB/s | 70 GB/s | 0.25x |
| 6144 x 2560 | 1 | 706 GB/s | 803 GB/s | 1.14x |
| 6144 x 2560 | 2 | 631 GB/s | 400 GB/s | 0.63x |
| 6144 x 2560 | 4 | 456 GB/s | 197 GB/s | 0.43x |
| 2560 x 640 | 1 | 404 GB/s | 403 GB/s | 1.00x |
| 2560 x 640 | 4 | 271 GB/s | 106 GB/s | 0.39x |

Output: rel diff 3.9e-7 .. 4.8e-7 across all shapes (the ESIMD float accumulation vs the AOS dp4a
integer path), i.e. the port is equivalent within float rounding.

## Decision

The mechanism **ports cleanly and is correct** (bit-equivalent within float rounding, `--selftest` green),
but it does NOT out-perform the port's existing AOS Wide32Q8 path on the B60:

- Single column is a wash: 0.88-1.14x (faster only on the tall 6144x2560 projection, slower on the
  2560-row projections).
- Multi-column is much slower (0.25-0.63x): the ESIMD DMMV is single-column and must be re-launched per
  column, losing the column-vectorization the AOS Wide kernels get for free.

llama.cpp's reorder+ESIMD wins only where its OWN AOS baseline is slow; Strata's Wide32Q8 is already
column-vectorized and leaves no headroom for this port. **Production dispatch is NOT wired** - doing so
would regress the multi-column (spec/MTP verify) decode. The port and its correctness gate
(`reorder_esimd_bench --selftest`) stay as a reference; P0-L1's reorder+ESIMD is parked as not-a-win
on B60 unless a measured shape shows otherwise. Do not reopen without new measurements.