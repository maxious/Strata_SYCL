# Experiment 42 - DPAS int8 MMVQ on reordered Q6_K: the decode dot on the matrix pipe (design, not yet run)

**Status: design.** No number below is new. Every measurement is cited from a closed experiment (21-28); the
proposal exists to test a mechanism exp 25 did not try. It reopens the README's parked "XMX for decode" row, and
that is deliberate - the parked verdict rests on a *dequant-to-FP16 GEMM*, not on a DPAS int8 matvec on quantized
weights.

## Question

The dense Q6_K decode MMVQ is ALU-pipe-bound. exp 23 put Pipe at 16.8% with Send 0.0%, and exp 27's ISA dump found
the 6-bit unpack is roughly 4x the dp4a count on that same pipe (`dp4a` is 32 of 985 instructions). Two levers have
been tried against it and neither moved the *dot* off the pipe:

- **exp 27** removed the per-token unpack with a one-time pre-unpack to signed bytes -> ~2x at kernel level
  (19.1 -> 9.3 us at ncols=1), but the dot is still `dp4a` on the ALU pipe. It ships opt-in (`STRATA_MMVQ_PREUNPACK=1`)
  and loses end to end on VRAM (exp 28/36: +2.53 GiB, 9,478 -> 8,152 expert-cache slots, 12.4% decode loss on the
  Coder; at matched slots the arms are within 1.0%).
- **exp 25** reached XMX by materializing FP16 and calling oneMKL, and lost at every ncols the engine runs
  (0.54-0.86x at ncols 1-4), crossing only at ncols>=6 while costing 2x dense VRAM plus ~12 us of materialization.

[llama.cpp PR 29864](https://github.com/ggml-org/llama.cpp/pull/29864) takes the third route: keep the weights
quantized in a reordered layout and feed **DPAS int8** directly - a thread owns 16 weight rows, the q8_1 activation
columns are the A operand, and the per-group scales are applied on the vector units after each DPAS. The dot leaves
the ALU pipe entirely. **Does that beat the shipped dp4a kernel - and, more tellingly, exp 27's no-unpack byte
ceiling - at the engine's real ncols 1-8?**

## Prior evidence (B60, 2560x2560 Q6_K, per-call us)

| ncols | 1 | 2 | 4 | 6 | 8 | source |
|---|---:|---:|---:|---:|---:|---|
| shipped AOS dp4a `native_q6_k_mmvq` | 19.1 | 21.7 | 26.3 | 36.1 | 55.3 | exp 27 |
| pre-unpack byte dp4a (the no-unpack ceiling) | 9.3 | 11.1 | 16.5 | 20.9 | 25.9 | exp 27 |
| oneMKL FP16 GEMM (materialized) | 39.9 | 38.0 | 34.4 | 30.8 | 26.1 | exp 25 |
| shipped / GEMM | 0.54x | 0.63x | 0.86x | 1.33x | 2.39x | exp 25 |

The engine decodes at ncols=1 (`layer.cpp:154`); the drafter and the verify window run 2-8 (`kVerifyMaxT = 8`), so
1-8 is the whole relevant range. The PR's 9-80 range is a llama.cpp short-prompt path and has no call site here
(see exp 44).

## Proposed change

A new bench-only kernel, not a production dispatch:

- `sycl/src/kernels/xmx_mmvq_bench.cpp`, a CMake target in the parity bench set (the `decode_xmx_gemm_bench`
  pattern), linking ESIMD.
- One ESIMD DPAS kernel for **Q6_K only** (the PR's `xmx_traits_q6_k`): reorder layout
  `[ql: nb*128][qh: nb*64][scales int8: nb*16][d: nb*2]`, 16 rows/thread, 2D block loads, `xmx_bytes_sub` unpack,
  one DPAS per group with the two 16-wide scales applied after it.
- The reorder is a **one-time transform at weight load**, exactly like the existing Q6U pre-unpack - it must never
  be inside the timed region.
- `grf_size` 128 for narrow tiles and 256 for wide ones (the PR's `xmx_grf`); confirm it applied by re-dumping the
  ISA (`IGC_ShaderDumpEnable=1`), or fall back to the runtime `IGC_ExtraOCLOptions=-ze-opt-large-register-file`
  route exp 21 used.

## Instrument

Single-process `xmx_mmvq_bench`, same shape and activations as `mmvq_bench`/`q6k_preunpack_bench`, four arms in one
process so the ratio is same-run:

1. shipped AOS dp4a `native_q6_k_mmvq` - the self-check (must reproduce the exp 22/27 plateau: ~19.1 us / ~281 GB/s
   at ncols=1, 26.1 us / 205.7 GB/s at ncols=4);
2. pre-unpacked byte dp4a (exp 27's ceiling);
3. DPAS int8 on the reordered layout - the candidate;
4. oneMKL int8 GEMM on the same layout - a control that proves the win is DPAS, not "any int8 GEMM".

Report **per-call us, not GB/s** (exp 27's trap: the byte path's GB/s is inflated by its 1.30x bigger buffer).
Warm ~300 ms, ~400 iterations, best of 3. Sweep ncols 1/2/3/4/5/6/7/8, then the Coder IQ1_M's real dense shapes
(2560->512/640/10240/12288, 6144->2560) - the square alone is not enough (exp 22/36).

## Gates before any ratio means anything

- Correctness vs an fp64 host reference, max rel err ~1e-7 (exp 25's standard); a NaN at any row count that is not
  a multiple of 4 was the PR author's own first bug - check non-multiple-of-16 row counts explicitly.
- The PR's `xmx_supported` precondition: `ncols % QK_K == 0` and a 64-byte-aligned activation base, else the
  short-row variant.
- B60 is BMG G21, which the PR's arch check lists; the port's `gpu_has_xmx` gate already exists.
- The pre-unpack arm and the DPAS arm must be compared at **the same resident VRAM** - the reorder costs the same
  ~1.30x dense buffer class as Q6U, and exp 36 is the proof that a kernel win can be a 12.4% engine loss from that
  cost alone.

## Success criteria

- **Beats arm 2 at ncols 1-4** -> the matrix pipe removes the pipe bottleneck and this is a real new lever; wire
  opt-in behind `STRATA_*` and run the engine A/B (matched expert-cache slots, `--layer-split` pinned) *before* any
  default flip.
- **Matches arm 2 only** -> parked: the bottleneck was the unpack, not the dot unit, and the cheaper Q6U path wins.
- **Loses to arm 1** -> parked with exp 12/25, and the PR's win on llama.cpp does not transfer because this port's
  AOS wide kernel is already column-vectorized (exp 12's finding).

## Traps carried in from the earlier XMX work

- Tiny-M GEMM timing is launch/backend-overhead dominated: report us, not GFLOP/s (exp 25 fabricated 0.3-4.0
  TFLOP/s at M=1..8).
- A "clean" build can be stale: verify the binary before trusting an arm (exp 33).
- `--expert-cache N` is a hint the sizer exceeds; read each arm's reported slot count, not the flag (exp 36).
