# Experiment 28 - the pre-unpacked Q6_K decode, wired into the engine (README P0 #2 avenue 1)

## What was built

The exp-27 headroom (~2x) is now a real opt-in engine path. `STRATA_MMVQ_PREUNPACK=1` pre-unpacks each Q6_K
tensor to a signed-byte `Q6U` layout once at weight load, and routes the dense decode through the no-bit-unpack
kernel:

- `Q6UBlock { float d0, d1; int8_t qs[32] }` - 32 consecutive signed weight bytes with the two per-16 scales Q6_K
  keeps, `d0/d1` fp32 so they match the packed kernel's `dsc0/dsc1` exactly.
- `native_q6k_preunpack` - one-time device transform, Q6KBlock array -> Q6UBlock array, transcribed from
  `dequant.hpp`'s `dequantize_q6_K` (the per-16 scale mapping and the ql/qh nibble/high-bit decode).
- `native_mmvq_q6k_unpacked` - the no-bit-unpack decode: `Wide32Q6U` mirrors the Q8_0 `wide32` kernel
  (`load16_a2` + `dp4a4` + per-16 scale), so the decode lands at the measured ~2x ceiling.
- Routing: `native_q6_k_mmvq` (and so the generic `native_mmvq(14)` used by layer.cpp:154, verify.cpp, the head,
  shared-expert and PLE) checks an opt-in side-table - packed device ptr -> pre-unpacked ptr - and runs the Q6U
  path when registered and `native_mmvq_set_q6k_preunpack(true)`. Default OFF (no env): behavior is bit-identical
  to before.
- Load hook: `native_dense.cpp` publishes, for each Q6_K tensor with the flag on, an unpacked buffer (kept alive in
  `weights_`) and registers packed->unpacked once, then enables routing.

## Correctness (q6k_preunpack_parity, new ctest)

Packed oracle (`native_q6_k_mmvq`, trust) vs the Q6U decode vs the routed path, random Q6_K:

| ncols | max-rel oracle-vs-unpack | oracle-vs-routed |
|---|---|---|
| 1 | 1.02e-07 | 1.02e-07 |
| 4 | 1.43e-07 | 1.43e-07 |
| 6 | 1.58e-07 | 1.58e-07 |
| 8 | 1.56e-07 | 1.56e-07 |

The transform and the routing reproduce the packed path within fp32 summation order (the scales are fp32-exact;
only the dp4a accumulate order differs). Registered as `q6k_preunpack_parity`; the README's "wide vs shared"-style
gate.

## Speed (q6k_preunpack_bench, B60, deployed Q6U path vs packed Q6_K)

| ncols | Q6_K unpack (us) | Q6K-unpacked routed (us) | ratio | Q8_0 ceiling (us) |
|---|---|---|---|---|
| 1 | 19.1 | 9.4 | 2.05x | 9.3 |
| 2 | 21.7 | 11.4 | 1.91x | 11.2 |
| 4 | 26.3 | 16.7 | 1.58x | 16.5 |
| 6 | 36.1 | 21.2 | 1.70x | 20.9 |
| 8 | 55.4 | 26.3 | 2.10x | 26.0 |

The deployed path matches the Q8_0 ceiling (1.58-2.10x; 2.05x at the engine's primary decode ncols=1).

## Default-behavior regression check

`mmvq_bench` with no env is unchanged: cols 1/4/8 = 19.1/26.1/55.5 us (same curve), so the packed path is
bit-identical for everyone who does not set `STRATA_MMVQ_PREUNPACK`.

## Still to verify on a real run

The end-to-end decode tok/s and output parity over a real model with `STRATA_MMVQ_PREUNPACK=1` need the runtime /
bench box (this dev box builds and runs the kernel-level gates but not the full engine run). The cost is a
persistent ~1.25x weight buffer per Q6_K tensor (36 -> 40 bytes per 32 values). The pre-unpack runs once at load
(outside graph capture); the routed decode is captured normally.

## Traps worth knowing

- Q6_K scales per-16, so the pre-unpacked block needs TWO fp32 scales per 32 (d0/d1); a single-d Q8_0-style block
  would not reproduce the packed output (the exp-27 Q8_0 number was the ceiling proxy, not the faithful layout).
- The routing is opt-in via the side-table AND the `native_mmvq_set_q6k_preunpack(true)` flag; the engine enables it
  only if at least one Q6_K tensor was actually pre-unpacked, so a build without the env or with no Q6_K layers
  leaves the default path untouched.
- The pre-unpack launch must round its global range up to the work-group size (a fixed-size local against a
  tiny tensor underflows on this device).