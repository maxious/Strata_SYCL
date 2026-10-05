# Experiment 27 - the Q6_K decode's unpack is the pipe cost: pre-unpacked byte decode is ~2x (README P0 #2 avenue 1)

## Question

README P0 #2 avenue 1: the dense Q6_K decode MMVQ is PIPE-bound (exp 23: Pipe 16.8%, Send 0.0%) and the lever is "a
lighter formulation of the 6-bit unpack/scale sequence" - ops on the bottleneck pipe per weight byte, NOT instruction
count (exp 21 removed 2.3% instr for 0%). How much does the in-kernel 6-bit unpack actually cost?

## The ISA establishes the unpack dominates the pipe

ISA dump of `native_mmvq_q6k_wide_a2` (the deployed kernel, ncols=4, 985 instr): **dp4a is only 32 of 985**
instructions, and on this BMG card dp4a runs on the **same ALU int pipe** as everything else (`dp4a ... // ALU pipe:
int`). The pipe work is dominated by the unpack/gather: bfn 74, xor 65, shl/and/shr/or ~63, plus ~120 per-byte movs
building the vi0/vi1 packed-byte vectors. Q6_K packs ql (4-bit low nibbles) and qh (2-bit high bits) with
mismatched byte alignment, which forces the per-byte gather into the dp4a operands. That unpack is ~4x the dp4a
count, all on the same pipe.

## The decisive test: time the no-unpack ceiling

`q6k_preunpack_bench` times the SAME decode shape (2560x2560, engine's ncols, same q8_1 activations) with the
shipped Q8_0 `wide32` kernel - signed bytes, load + dp4a only, no bit-unpack - against `native_mmvq_q6k`. A
pre-unpacked Q6_K (one-time transform to `{d' = d*scale, 32 int8}` per 32-subblock) decodes through exactly that
kernel, so the Q8_0 timing IS the achievable pre-unpacked decode. Two runs each:

| ncols | Q6_K unpack-in-kernel (us) | Q8_0 byte no-unpack (us) | Q8_0 GB/s | ratio |
|---|---|---|---|---|
| 1 | 19.1 / 19.1 | 9.3 / 9.3 | 749 | 2.06x |
| 2 | 21.7 / 21.7 | 11.1 / 11.1 | 625 | 1.95x |
| 4 | 26.3 / 26.3 | 16.5 / 16.5 | 422 | 1.59x |
| 6 | 36.1 / 36.1 | 20.9 / 20.9 | 333 | 1.73x |
| 8 | 55.3 / 55.4 | 25.9 / 26.0 | 268 | 2.13x |

**~2x at the engine's primary decode (ncols=1: 19.1 -> 9.3 us), 1.6-2.1x across the curve**, stable over two runs.
Memory is not the limit: the byte path reads 272 B per 256 weights against Q6_K's 210 (1.30x), yet hits 749 GB/s
and the kernel showed Send 0.0% stalls (exp 23), so bandwidth has ~2x headroom.

## Verdict

**Avenue 1 is VALIDATED - the Q6_K 6-bit unpack/gather is the decode's pipe cost, and the lighter formulation is to
not unpack per token.** A one-time pre-unpack of Q6_K rows to a signed-byte layout (per-32 `d' = d * scale`, decoded
by the existing load+d4a path) roughly halves the dense decode matvec. This is NOT the parked memory-reorder
(exp 12/22): that reordered for contiguous loads and had no stall to remove; this removes the ALU-pipe unpack that
exp 23 named as the lever. Cost: a persistent ~1.30x weight buffer (6.97 MB vs 5.4 MB for 2560x2560) and a one-time
per-weight-load transform.

**Wiring is the next step and is worth it**: a pre-unpack kernel at weight load, a persistent signed-byte buffer per
GpuStage, and routing the dense Q6_K decode (`layer.cpp:154`, `mtp.cpp`) to the byte path, with the Q6_K parity
checked (the pre-unpacked path must reproduce the packed path's output within float tolerance). `q6k_preunpack_bench`
stays in the tree as the regression measure.

## Traps worth knowing

- The Q8_0 `wide32` kernel is the correct ceiling ONLY because a pre-unpacked Q6_K maps 1:1 onto it (same `d * scale
  * dot(qs_w, qs_a)` per 32-subblock); it is not the same as dequant-to-FP16 + oneMKL (exp 25's no-win) - this keeps
  the native dp4a kernel and only removes the per-token bit-unpack.
- Report per-call **us**, not GB/s, for the comparison: the byte path's GB/s is inflated by its 1.30x bigger buffer.
- Data-independence holds (dequant cost is data-independent), so random Q8_0/Q6_K rows time the real kernels.