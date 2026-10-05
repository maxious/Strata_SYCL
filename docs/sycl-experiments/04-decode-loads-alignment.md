# Experiment 04 - aligned loads for the still-misaligned decode types (IQ4_XS / IQ4_NL / Q8_0)

## Question

README.sycl.md P1 and INTEL.md's "still misaligned" list: IQ4_XS (136 B, 8-aligned), IQ4_NL (18 B) and Q8_0
(34 B) decode matvecs stream well below the card's capability while the aligned Q6_K path reached 400+ GB/s after
the `load16_a2` two-aligned-loads-and-shift fix (Q6_K 150 -> 407 GB/s, decode +9 tok/s, 2026-09-30). Does the
same `load16_a2` treatment move those three types off the misaligned list?

Reference: llama.cpp's newest decode work (`ggml-sycl`, commit `392ded654` "Q8_0 DMMV ESIMD and MMVQ wide load",
and the reorder-ESIMD series) is the same family of layout move.

## Survey (what the code actually does)

- **Q8_0 (34 B, 2-aligned) and IQ4_NL (18 B, 2-aligned)** already use `load16_a2` in the port's wide-32 kernels
  (`native_mmvq.dp.cpp` `WideQ8_0`/`WideIQ4NL`, the `STRATA_MMVQ_WIDE_32` path, since upstream 0.1.35's decode
  round; INTEL.md's "still misaligned" note for them is stale - those were the 2026-10-01 "wide 16-byte-load
  kernels for Q8_0 and IQ4_NL" rows). So they are **not** open.
- **IQ4_XS** (136 B, 8-aligned) is the only genuinely-open one: `WideIQ4XS::load` reads its 16 weight bytes as
  two 8-byte `int2` loads (`b->qs + 16*g`, 8-byte aligned). This is the actual candidate.

## Change tried

Replace IQ4_XS's two 8-byte `int2` loads with one `load16_a2(b->qs + 16*g)` (a shifted single aligned 16-byte
load, the Q6_K-winning helper), then feed the same four 32-bit dwords to the `iq4_lut4` codebook path. This is
the direct analog of the Q6_K fix.

## Benchmark (B60, `mmvq_sg_bench`, SIMD32; identical checksums both variants)

IQ4_XS, the shapes the Coder's decode kernels use:

| shape | baseline (two int2) | load16_a2 | delta |
|---|---|---|---|
| 2560 x 10240, 2 cols | 78.1 us (178.3 GB/s) | 84.2 us (165.3 GB/s) | -7% |
| 2560 x 10240, 4 cols | 93.6 (148.7) | 103.4 (134.7) | -9% |
| 2560 x 12288, 2 cols | 95.2 (175.6) | 102.9 (162.4) | -7% |
| 6144 x 2560, 2 cols | 44.3 (188.6) | 48.8 (171.2) | -9% |
| 2560 x 640, 2 cols | 7.1 (122.8) | 7.3 (120.0) | -2% |
| 2560 x 2560, 2 cols | 20.9 (166.9) | 22.7 (153.7) | -8% |

The `load16_a2` variant is **consistently 2-9% slower** at every shape and column count. SIMD16
(`STRATA_MMVQ_SG=16`) gives the same result, so the effect is not sub-group-width dependent.

## Why it loses (unlike Q6_K)

IQ4_XS's wide kernel is **not** bound by unaligned-load splitting the way Q6_K was. Its sixteen weight bytes feed
an `iq4_lut4` codebook LUT (4 nibbles -> codebook bytes in registers); the extra ALU (LUT) work, not the load
split, caps it at ~150-190 GB/s. `load16_a2` adds a shift+extract for every load and buys nothing - the opposite
of Q6_K, which is pure saturation-bound byte streaming where alignment was the whole story.

## Validity

Both variants produce **identical checksums** on every shape and column count (`sum 3.076753e+08 ... 2.187518e+08`,
matching bit-for-bit across the two load schemes), so the slower result is not a correctness artifact - it is a
genuine (small) regression.

## Conclusion (negative result, revert)

The IQ4_XS `load16_a2` change is **reverted** - it is 2-9% slower with no benefit and identical output. The
README/INTEL.md "still misaligned" item for IQ4_XS is closed as *not the bottleneck*: IQ4_XS is LUT/ALU-bound,
not load-alignment-bound, at these working loads, and Q8_0/IQ4_NL were already on `load16_a2`. The remaining
IQ4_XS headroom (79-189 GB/s vs the aligned types' 230-400+) would need a different lever (larger rows per
sub-group, vectorized codebook gather), not aligned loads. This matches llama.cpp's own guidance that decode
matvecs win from layout/reorder work, not from load alignment alone for the codebook-based i-quants.

INTEL.md's "Still misaligned" note should be narrowed to the interpretation above (Q8_0/IQ4_NL are done; IQ4_XS
is LUT-bound, not misaligned-load-bound).