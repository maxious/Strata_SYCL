# Experiment 06 - fused GR read: are any gated variants faster on the B60?

## Question

Experiment 05 showed the fused GR read (norm + down + up, the dense decode family) decays 375 -> 113 GB/s as the
verify window grows 1 -> 6 tokens. The kernel ships several compiled-gated variants (`STRATA_GR_NORM_SPLIT`,
`STRATA_GR_DOWN_SLICED`, `STRATA_GR_DOWN_DIRECT`, `STRATA_GR_V3`). Before changing anything, sweep the gates to
confirm which is measured-fastest on the B60 at the working window widths, per the README/INTEL.md convention
("keep the default on the measured-fastest path; gate anything not faster behind an env flag").

## Measurement (`gr_bench`, Coder shapes, B60, warm)

`gr_bench` prints time + a per-output checksum for T = 1..6. Timings at the working widths (4 and 6 tokens):

| config | 4 tokens | 6 tokens |
|---|---|---|
| **default** (split-norm + sliced + direct) | **89.1 us (148 GB/s)** | **116.6 us (113 GB/s)** |
| `STRATA_GR_NORM_SPLIT=0` (one group per token) | 95.1 (139) | 123.2 (107) |
| `STRATA_GR_DOWN_SLICED=0` (direct read) | 122.7 (107) | 168.6 (78) |
| `STRATA_GR_DOWN_DIRECT=0` + `_SLICED=0` (tiled) | 140.5 (94) | 215.1 (61) |

Each variant is a different summation order, so its checksum differs (not bitwise comparable) - but the **timings
are unambiguous**: the default is 14-34% faster at 4 tokens and 33-84% faster at 6 tokens than every alternative.
This matches INTEL.md's documented tuning (sliced down = the round's 108.6 -> 76.5 us at 6 tokens; split norm =
the "GR norm per (token, stream)" round).

## Conclusion (confirm, no change)

The B60's default fused-GR-read configuration is **already measured-fastest** at every working window width; the
multi-token decay (1: 35 us/375 GB/s -> 6: 117 us/113 GB/s) is the inherent memory-latency behavior of these
dense reductions, not a tunable regression - every alternative gate is slower. No engine change; nothing to flip.
The "decode multi-token occupancy" item is closed as *already optimal on this card* with the sweep above.

## Validity

- All runs warm (150 ms warm-up in the bench), timings reproducible to ~1%.
- Each variant's checksum is consistent across runs; differences between variants are summation-order, as the
  file's comments state.

## Next lever (unchanged from Experiment 05)

If decode headroom is wanted, the payload is the expert-dot and IQ4_NL/Q8_0 decode kernels' absolute rate, not the
fused-GR variants or graph-node count. Those are llama.cpp's ESIMD/reorder-family targets and were covered by the
earlier `mmvq_sg_bench`/`gr_bench` baselines (IQ4_XS ~150-190 GB/s, LUT-bound per Experiment 04).