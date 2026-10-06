# Experiment 42 - DPAS int8 MMVQ on reordered Q6_K: the decode dot on the matrix pipe (measured 2026-10-05)

**Status: measured, bench-only.** The candidate is built and verified (`sycl/src/kernels/xmx_mmvq_bench.cpp`, CMake
target `xmx_mmvq_bench`). It beats the shipped kernel by 1.7-2.0x on the model's *real* dense shapes and loses on the
small square. Nothing in the engine calls it; the engine A/B and the opt-in `STRATA_*` wiring are the next step and are
**not** done, so no default may move.

## Question

The dense Q6_K decode MMVQ is ALU-pipe-bound (exp 23: Pipe 16.8%, Send 0.0%; exp 27's ISA dump: `dp4a` is 32 of 985
instructions against a 6-bit unpack roughly 4x its size on the same pipe). exp 27 removed the per-token unpack with a
one-time pre-unpack to signed bytes and got ~2x, but the dot stayed `dp4a` on the ALU pipe. exp 25 reached XMX by
materializing FP16 for oneMKL and lost at every ncols the engine runs. llama.cpp PR 29864 takes the third route: keep
the weights in a reordered layout and feed **DPAS int8** directly. **Does that beat the shipped dp4a kernel - and exp 27's
no-unpack ceiling - at the engine's ncols 1-8?**

## What was built, and the two toolchain facts that shaped it

`xmx_mmvq_bench` runs three arms in one process at the engine's shapes:

| arm | what it is | bytes / 256 weights |
|---|---|---:|
| 1 `native_q6_k_mmvq` | shipped AOS 6-bit unpack + `dp4a` | 210 |
| 2 `native_mmvq_q6k_unpacked` | exp 27's one-time pre-unpack to signed bytes, still `dp4a` | 272 |
| 3 `xmx_dpas_q6k` | this experiment: 6-bit unpack in registers, dot on DPAS int8 | 210 (layout only) |

Two facts had to be established before any ratio meant anything, both by standalone probes:

1. **int8 DPAS on this device is K=32, N in {8,16}, repeat count (M) 1..8** - `dpas.hpp` static-asserts
   `SystolicDepth == 8`, and for s8 that fixes `K = 8 * 4 = 32`. Q6_K keeps one scale per **16** weights, so a 32-deep
   DPAS spans two different scales: the kernel issues **two DPAS per group with the unused half of B zeroed**, which is
   what "zeros in B drop the other half of K" means upstream. Measured cost of that masked pair: dropping the second
   DPAS (`XMX_ONE_DPAS=1`, numerically wrong by construction) moves ncols 1 from 17.3 to 16.5 us - **the mask is not the
   cost**.
2. **The M>1 repeat count does not accept the documented A layout.** With M=1 and B in K-major VNNI order
   (element (k,n) at dword `(k/4)*16+n`, byte `k%4`), `dpas<8,1,int>` matches a host dot product exactly for every
   random case (`dpas_vnni` probe). At M=2,4,8 with A laid out row-major (M rows x K bytes, the form
   `xmx_int8_bench`'s `joint_matrix` path uses and validates) the same probe matches **only 16 of RC*16 lanes - one
   row** - and neither a K-major interleave of A nor a transposed result index fixes it. The kernel therefore issues
   **one M=1 DPAS per column** with the unpacked B tile held in registers across columns; that is verified correct, and
   it costs nothing extra in matrix-pipe work (M=1 wastes none of it).

`esimd::load_2d(..., Transposed=true)` does **not** deliver a usable 16-row x 8-dword tile in this toolchain (a
row-encoded probe returns a layout that is neither `(kd,n)` nor `(n,kd)`, and repeated every 16 elements), so arm 3 does
not use it. Instead the one-time reorder writes the DPAS B order directly - per (16-row tile, block, 32-wide group) a
512-byte tile with element (k,n) at dword `(k/4)*16+n` - which makes each group **one contiguous 512-byte load** with no
in-register transpose. The cost is that the buffer holds bytes rather than 6-bit codes: 272 B per 256 weights, the same
class as arm 2, so arms 2 and 3 differ in the dot and not in resident bytes. (The layout-only reorder the design assumed
would have been free is not: VNNI needs materialized bytes.)

Two ESIMD traps cost real time and are worth recording: `select<16,16>(off)` on a per-row scale block returns
something that is not that strided slice (row 0 came out exact and every other row wrong - the same signature as a real
layout bug), and a ternary between two `simd` values (`one ? simd(0) : dpas(...)`) silently changed results. Both are
silent-wrong-answer, not compile-error.

## Correctness gate

Arm 3 is checked against an fp64 host reference (llama.cpp's `dequantize_row_q6_K` on the host, 48 rows x every column)
and must land at the *same* rel-RMS as the two shipped arms - it does, to all printed digits, at every ncols 1-8 and every
shape below. The bench exits non-zero otherwise (`XMX_CMP=1` prints ref/AOS/pre-unpack/DPAS side by side). An earlier
build that read the wrong dump offset reported 95/4096 tile matches and looked like a layout bug; the shipped gate is
the rel-RMS line, not the dump.

## Measured (B60, one process, warm 300 ms, 400 reps, best arm per row not needed - all within the +-1.5% floor)

### The square the old experiments used (2560 x 2560)

| ncols | 1 | 2 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| arm 1 AOS `dp4a` (us) | 19.2 | 21.7 | 26.3 | 36.7 | 54.9 |
| arm 2 pre-unpack (us) | 9.3 | 11.4 | 16.7 | 21.2 | 26.4 |
| arm 3 DPAS (us) | 17.3 | 20.5 | 27.4 | 34.0 | 40.1 |
| DPAS vs AOS | 1.11x | 1.06x | 0.96x | 1.08x | 1.37x |
| **DPAS vs pre-unpack** | **0.54x** | 0.55x | 0.61x | 0.62x | 0.66x |

Arms 1 and 2 reproduce exp 22/27's published plateau (19.2/21.7/26.3/36.7/54.9 and 9.3/11.4/16.7/21.2/26.4), so the
self-check passes. On this shape the DPAS arm **loses to the no-unpack ceiling by ~1.8x**.

### The model's real dense shapes - where it wins

Q6_K dense projections in the Coder IQ1_M are 2560->12288, 2560->10240 (down) and 6144->2560 (up), plus small
2560->640/512. Per-call us, arm 3 vs the two others:

| shape (n_out x n_in) | ncols | AOS | pre-unpack | DPAS | vs AOS | vs pre-unpack |
|---|---:|---:|---:|---:|---:|---:|
| 12288 x 2560 | 1 | 100.5 | 94.1 | **50.8** | **1.98x** | **1.85x** |
| 12288 x 2560 | 8 | 258.0 | 156.5 | 150.8 | 1.71x | 1.04x |
| 10240 x 2560 | 1 | 77.9 | 77.1 | **44.6** | **1.75x** | **1.73x** |
| 10240 x 2560 | 8 | 212.7 | 129.5 | 121.8 | 1.75x | 1.06x |
| 2560 x 6144 | 1 | 37.9 | 29.2 | **22.0** | **1.72x** | **1.33x** |
| 2560 x 6144 | 8 | 105.7 | 60.3 | 53.8 | 1.97x | 1.12x |
| 640 x 2560 | 1 | 6.3 | 4.1 | 13.5 | 0.47x | 0.30x |
| 640 x 2560 | 8 | 20.1 | 9.3 | 33.4 | 0.60x | 0.28x |
| 2560 x 2560 | 1 | 19.2 | 9.3 | 17.3 | 1.11x | 0.54x |

All rows passed the correctness gate ("all arms within tolerance" in each run's log).

## Verdict

**A real new lever, and the square is the wrong place to look for it.** The design's success criterion was "beats arm 2
at ncols 1-4"; at the shapes that dominate the model's dense Q6_K work it does, by 1.33-1.85x at decode width and
1.04-1.12x at the verify window's 8. It loses on the small shapes (640 out, the 2560 square) where 16 rows per work-item
and one 512-byte load per group cannot fill the machine - the same "bench the square" trap exp 22 and exp 36 document,
now with the sign flipped.

What this says mechanistically: exp 23's "the decode dot is ALU-pipe-bound" is true of the shipped kernel and **not** a
property of the decode dot. Moving the dot to DPAS removes that bottleneck, and at 12288x2560 the arm is 1.98x the
shipped kernel - a bigger factor than the 2.05x the pre-unpack bought on the square, and it buys it *while also* removing
the in-kernel unpack.

What is **not** established, and must be before any default moves:

- **No engine A/B.** The bench is where the design's instrument lives, but exp 28/36's lesson is that a kernel win can be
  an engine loss: arm 3's buffer is 272 B per 256 weights, the same 1.30x class that cost the pre-unpack 2.53 GiB and
  12.4% end to end. Wiring it opt-in behind `STRATA_Q6K_DPAS=1` with a parallel-buffer registration like
  `native_mmvq_register_q6k_preunpack`, then an engine A/B at **matched expert-cache slots** (read the reported slot
  count, not the `--expert-cache` hint) with `--layer-split` pinned, is the required next step.
- **The one-time transform cost is not measured here** (this bench builds the tile buffer on the host). The engine would
  do it on device at load; it must be bounded before shipping.
- **The M>1 repeat count is unusable**, so every column costs its own DPAS and the tile is 16 rows wide. Both are
  probably why the small shapes lose; N=8 tiles or 8-row tiles with two tiles in flight are the obvious next attempt.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh
ninja -C sycl/build-b60 xmx_mmvq_bench
./sycl/build-b60/xmx_mmvq_bench                 # 2560x2560 square
./sycl/build-b60/xmx_mmvq_bench 12288 2560 200  # the shape that wins
XMX_CMP=1 ./sycl/build-b60/xmx_mmvq_bench 256 256 2   # ref / AOS / pre-unpack / DPAS side by side
XMX_ONE_DPAS=1 ./sycl/build-b60/xmx_mmvq_bench        # masked-half DPAS cost (wrong results on purpose)
```

Probe files used to establish the operand layouts (kept outside the repo, `/home/maxious/exp42-harness/`):
`dpas_vnni.cpp` (M=1 exact, M>1 mismatch), `dpas_map.cpp`/`dpas_ramp.cpp` (one-hot and ramp index maps),
`load2d_layout.cpp` (the 2D transposed load's layout).
## Integration status (2026-10-06): step 1 measured, step 2 wired, step 3 blocked by a correctness failure

### Step 1 - the prize, measured

**Which tensors exist.** The Coder IQ1_M has **128 of its 300 dense decode tensors in Q6_K** (29ssm_out, 22 attn_qkv,
14 attn_gate, 13 ffn_up_shexp, 11 attn_output, 11 attn_k, 10 attn_v, 5 attn_q); the Q2_0 pack has only **12 of 301**.
The shapes, and what the DPAS arm does to each at ncols 4 (the spec-4 window):

| Coder projection | shape (n_in x n_out) | count | AOS | DPAS | net per window |
|---|---|---:|---:|---:|---:|
| attn_q | 2560 x 12288 | 5 | 127.1 | 96.2 | +0.154 ms |
| attn_qkv | 2560 x 10240 | 22 | 102.7 | 81.6 | +0.464 ms |
| attn_gate | 2560 x 6144 | 14 | 62.1 | 60.3 | +0.025 ms |
| ssm_out + attn_output | 6144 x 2560 | 40 | 50.7 | 34.6 | +0.644 ms |
| ffn_up_shexp | 2560 x 640 | 13 | ~13 | 22.1 | **-0.118 ms** |
| attn_k + attn_v | 2560 x 512 | 21 | 9.9 | 22.3 | **-0.260 ms** |

**Net 0.909 ms saved per 4-token window** (1.288 won, 0.379 lost) against a 45-52 ms round - about **1% of decode**.
At **ncols 1** the same tensors are worth 1.6 ms per token, about **4.3%**.

**What the bytes cost.** The four winning shapes are 1,118 MB packed, so the VNNI tiles add **+330 MB**. The auto-sizer
turns VRAM into expert-cache slots at the measured **0.51 slots/MiB** (exp 27's calibration), so that is **~168 slots**.
What 168 slots are worth, measured on this box with the Coder at 256 generated tokens:

| expert cache | VRAM | decode |
|---:|---:|---:|
| 7,815 slots | 14.9 GiB | 26.79 tok/s |
| 6,519 slots | 12.40 GiB | 23.08 tok/s |
| 3,903 slots | 7.44 GiB | 4.89 tok/s |

That curve is steep: ~0.0029 tok/s per slot in the 6.5k-7.8k region, so **168 slots is about -0.49 tok/s**.

**So the gate's answer is configuration-dependent, and it is negative where the engine runs today.** At spec 4 the prize
is ~+0.19 tok/s against ~-0.49 tok/s of slots: **net negative**. At ncols 1 the prize is ~+1.15 tok/s against the same
-0.49: **net positive**. Nothing flips a default until step 3 measures both arms on the real engine; the wiring is opt-in
for exactly that reason.

### Step 2 - wired, opt-in, default off

`sycl/src/kernels/cuda/q6k_dpas.dp.cpp` (its own translation unit) holds the one-time AOS -> DPAS-order transform, the
tile registry and the decode matvec; `native_mmvq.dp.cpp` keeps only the dispatch, which fires when
`native_mmvq_q6k_dpas_lookup` finds a registered pointer. `native_dense.cpp` builds the tiles at weight load when
**STRATA_Q6K_DPAS=1**, only for tensors whose packed size clears **STRATA_Q6K_DPAS_MIN_BYTES** (default 8 MiB, i.e.
the shapes the bench measured as wins - the 1.34 MB and 1.08 MB ones lose and are never converted). Default off;
`STRATA_Q6K_DPAS=0` is the opt-out. `ctest` is 30/30 with the wiring in the tree.

### Step 3 - blocked: the wired path does not reproduce the shipped kernel

`sycl/src/kernels/q6k_dpas_parity.cpp` runs the wired path against `native_q6_k_mmvq` at ncols 1-8. It reports
**rel 0.87** - the same kernel source in the bench harness (`xmx_mmvq_bench`) is at **3.44e-03**, matching the shipped
arms. What is established, in order:

1. The kernel **runs** and reads its inputs (a debug store of `tiles[0] + scales[0] + d[0]` and of the first
   activation byte came back non-zero).
2. The transform's **tiles are byte-identical** to a host-built reference of the same layout: 0 differences in 6,553,600
   bytes. Scales and `d` are byte-identical too, **after fixing a real bug this hunt exposed**: the transform wrote the
   16 scales and `d` only for **row 0 of each 16-row tile** (first mismatch at offset 160 = row 10), while the kernel
   reads all 16 rows - which is exactly the symptom "row 0 correct, rows 1-15 zero" that the first runs showed.
3. With that fixed the residual is 0.87, so the remaining defect is in the wired kernel or its launch, not the
   transform and not the data. Copying the bench kernel **verbatim** into the engine TU, and compiling that TU into the
   consumer executable instead of the static library, both failed to change it.

The parity binary is therefore **not** a ctest - registering it would put a red test in the suite for a path that is
opt-in and off. Run it by hand before enabling STRATA_Q6K_DPAS. **Next step:** with the transform proven byte-exact,
the remaining suspect is the launch geometry or the local-memory reduction in the wired kernel (the bench uses
`bpr >= 8 -> KS = 8`; the engine picks KS from the same rule, and 2560 gives bpr = 10, so two of the eight
work-items take two blocks and six take one - a reduction bug there would corrupt *most* rows while leaving row 0 of
each tile intact, which is what the numbers show).
