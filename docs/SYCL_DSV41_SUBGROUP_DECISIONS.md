# SYCL sub-group sizes for the DeepSeek engine (dsv41) kernels

Every `dpct::permute_sub_group_by_xor` call in the SYCL port, one row each, with the sub-group size that call
needs and what the B60 evidence says about it. The 40 sites are the ones the Wave 1 audit listed
(`.omo/evidence/dsv41-sycl-round2b/w1-audit.md`). The sizes come from a probe on this PC's Intel Arc Pro B60
(`.omo/evidence/ulw-research-sg16-round2/sg-probe.log`) and from the ahead-of-time build
(`.omo/evidence/dsv41-sycl-round2b/w1-aot.log`). Everything here was measured on this PC, 2026-10-29, unless a
row says `(static reading)`.

The audit's kernel names and launch lines were read from an earlier pass. Each row below re-read the file at
`849a7a5a`, so the kernel class is the function that holds the call at HEAD and the launch lines are HEAD's. Where
the audit disagreed, the row says so.

## Probe evidence

The numbers below are copied verbatim from the probe logs in `.omo/evidence/ulw-research-sg16-round2/`, run on
this PC on 2026-10-29: `sg-probe.log` for the sizes, `aot-probe.log` for the negative controls.

The hardware is an Intel(R) Arc(TM) Pro B60 Graphics, Battlemage G21, which is `bmg-g21` in the AOT device target,
under oneAPI 2026.2 with `icpx` 2026.2.0 (2026.2.0.20261001). The probe saw four device entries, driver versions
26.35.39758.10 and 1.17.39758+10, and all four report the same sizes.

The verdict block in `sg-probe.log`:

```
sg-default=16, supported=[16, 32], jit_rc=0, aot_rc=0, reqd16_rc=0, reqd32_rc=0
```

The negative controls, recorded in `aot-probe.log`:

```
SUPPORTING_EVIDENCE neg_aot_compile=1 reqd8_aot=compile_error reqd64_aot=compile_error neg_jit_run=kernel_not_supported(13)
VERDICT aot_compile=0 reqd32_aot=ok reqd16_aot=ok jit=ok
```

What each value proves:

- `sg-default=16`. A kernel with no pin of its own gets 16 lanes.
- `supported=[16, 32]`. These are the only two sizes the card accepts. The log writes them with a space after the
  comma, and that is kept here.
- `jit_rc=0`, `aot_rc=0`, `reqd16_rc=0`, `reqd32_rc=0`. A kernel pinned at 16 and a kernel pinned at 32 each built
  and ran on the card, under JIT and under AOT for `spir64`. Neither was silently skipped: the readback verified
  every element was written.
- `neg_aot_compile=1`. A source that pins 8 and 64 fails the AOT build with exit 1 and produces no binary. `ocloc`
  names the offending kernel, and the two sizes give two different messages: "Kernel compiled with required
  subgroup size 8, which is unsupported on this platform" for 8, "Unsupported required sub group size" for 64.
- `reqd8_aot=compile_error`, `reqd64_aot=compile_error`. Both sizes are refused at compile time under AOT, not at
  link time and not at run time. The whole translation unit fails.
- `neg_jit_run=kernel_not_supported(13)`. Under JIT the same source compiles fine and throws when the kernel is
  submitted, with errc 13. `exception.hpp` in 2026.2 names 13 `kernel_not_supported` and 12
  `feature_not_supported`, and `SYCLCategory::message()` returns the generic "SYCL Error" for both, so the number
  is the only thing that tells them apart. The message is "Sub-group size N is not supported on the device".

A bad pin is caught at build time under AOT and only at submission under JIT. AOT is the stricter path, which is
why a pin outside {16, 32} cannot ship.

One note on the plan against the logs: checkbox 3.2 lists all ten fields as coming from `sg-probe.log`, but that
log's verdict line carries only the six above. The four negative-control fields are recorded in `aot-probe.log`,
in the same evidence directory. The logs are the authoritative record, so the values here are theirs.

## The rule

A permute swaps a value with the lane `mask` away. A lane only swaps inside its own sub group, so a mask of 16
needs a sub group at least 32 wide. The device default on this card is 16 (`sg-default=16` in `sg-probe.log`), so
a kernel that permutes at mask 16 must ask for 32 with `[[sycl::reqd_sub_group_size(32)]]`. Left alone it gets 16,
the mask-16 step swaps a lane with itself, and the sum loses the other half of the block. Nothing fails: the
answer is quietly wrong.

Sizes the card takes: 16 and 32. 8 and 64 are refused at build time (`compile_error` in `aot-probe.log`), and 64 is
refused again by `ocloc` in the AOT build (`build_kernel_failure` in `w1-aot.log`). A pin outside `{16, 32}`
cannot ship.

## The table

`reqd Size` is the size the site needs. `none` means the site needs no pin of its own. `Device Verified` names the
log: `B60 run` means a kernel that reaches this call ran on the card in that log, `compile-only` means the file
built and no run of that kernel is in this round's logs.

| Kernel Class | File | Line | reqd Size | Rationale | Device Verified |
| --- | --- | --- | --- | --- | --- |
| ds4_tq tile_dot | ds4_tq.dp.cpp | 291 | 32 | Mask 16, one step of the pair swap. All four tq wrappers are pinned `[[sycl::reqd_sub_group_size(32)]]`: `gate_up_wrapper` at `ds4_tq.dp.cpp:642`, `down_wrapper` at :657, `set_wrapper` at :671, `mv_wrapper` at :685. Those are the pin lines at the integration tip 7bd2f17d (`git show 7bd2f17d:sycl/src/dsv41/ds4_tq.dp.cpp | grep -n reqd_sub_group_size` -> 642, 657, 671, 685); `:695`, `:708`, `:721`, `:734` are the four `parallel_for` launch lines, not pins. This wave-2 worktree carries no tq pin yet (`grep -n reqd_sub_group_size sycl/src/dsv41/ds4_tq.dp.cpp` -> rc 1); checkbox 1.2 brings the four pins here. | compile-only (w1-aot.log) |
| ds4_tq tile_dot | ds4_tq.dp.cpp | 294 | 32 | Mask 16, the second operand of the same step. Covered by the same four pins - `reqd(32)` at `ds4_tq.dp.cpp:642`, :657, :671 and :685, one per wrapper (`gate_up_wrapper`, `down_wrapper`, `set_wrapper`, `mv_wrapper`). | compile-only (w1-aot.log) |
| ds4_iq warp_sum | ds4_iq.dp.cpp | 654 | 32 | Masks 16..1. The launches pin 32 at HEAD: `mv_kernel` (:2354, :2369, :2384), `light_q8_kernel` (:2317), `light_kernel_rows` (:2259, :2277, :2293). | B60 run (w1-u-parity.log) |
| ds4_iq lanes_sum | ds4_iq.dp.cpp | 694 | 32 | Masks 16/RW down to 1. `kGuRW` is 1, so the gate and up path reaches mask 16 and needs 32; `kDnRW` is 2, so the down path stops at 8. Both launches pin 32 anyway: `gate_up_kernel` (:1867, :1892), `down_kernel` (:1927, :1950). | compile-only (w1-aot.log) |
| ds4_iq light_q8_kernel | ds4_iq.dp.cpp | 1587 | 32 | Masks 16..1, the Q8 scale's max. The launch pins 32 at HEAD (:2317). | compile-only (w1-aot.log) |
| ds4_kernels finish_kernel | ds4_kernels.dp.cpp | 172 | none | Mask 1, the rope pair swap. Covered by the default 16. The launches (:2596, :2619) pin 32 for the other step in this kernel. | compile-only (w1-aot.log) |
| ds4_kernels finish_kernel | ds4_kernels.dp.cpp | 194 | 32 | Masks `g/2 .. 1` with `g` 32 for an F8 group, so mask 16 is reached. The launches pin 32 at HEAD (:2596, :2619). The audit read this as mask 4 (static reading, re-checked). | compile-only (w1-aot.log) |
| ds4_kernels finish_kernel | ds4_kernels.dp.cpp | 221 | 32 | Same loop in the packed path, mask 16 for an F8 group. Covered by the same pins (:2596, :2619). | compile-only (w1-aot.log) |
| ds4_kernels hc_split_body | ds4_kernels.dp.cpp | 321 | none | Mask 1, a row sum. Covered by the default 16. The audit read this site as `route_kernel` mask 16 (static reading, re-checked). | compile-only (w1-aot.log) |
| ds4_kernels hc_split_body | ds4_kernels.dp.cpp | 323 | none | Mask 2, the same row sum. | compile-only (w1-aot.log) |
| ds4_kernels hc_split_body | ds4_kernels.dp.cpp | 328 | none | Mask 4, a column sum. | compile-only (w1-aot.log) |
| ds4_kernels hc_split_body | ds4_kernels.dp.cpp | 330 | none | Mask 8, the same column sum. 8 fits inside the default 16. | compile-only (w1-aot.log) |
| ds4_kernels hc_split_body | ds4_kernels.dp.cpp | 335 | none | Mask 1, the row max. | compile-only (w1-aot.log) |
| ds4_kernels hc_split_body | ds4_kernels.dp.cpp | 338 | none | Mask 2, the same row max. `hc_mix_split_kernel` (:2664) and `hc_split_kernel` (:2682) pin 32 for the block step. | compile-only (w1-aot.log) |
| ds4_kernels route_kernel | ds4_kernels.dp.cpp | 619 | 32 | Masks 16..1, the best logit of a pass. The launch pins 32 at HEAD (:2861). | compile-only (w1-aot.log) |
| ds4_kernels route_kernel | ds4_kernels.dp.cpp | 621 | 32 | Masks 16..1, the index of that logit. Covered by the same pin. | compile-only (w1-aot.log) |
| ds4_kernels ext_to_ring_kernel | ds4_kernels.dp.cpp | 1146 | 32 | Masks 16..1, the magnitude max of a warp's 32 values. The launch pins 32 at HEAD (:3055). The audit read this as `argmax_share_kernel` (static reading, re-checked). | compile-only (w1-aot.log) |
| ds4_kernels argmax_share_kernel | ds4_kernels.dp.cpp | 1279 | 32 | Masks 16..1, the row's best value. The launch pins 32 at HEAD (:3177). | compile-only (w1-aot.log) |
| ds4_kernels argmax_share_kernel | ds4_kernels.dp.cpp | 1281 | 32 | Masks 16..1, its index. Covered by the same pin. | compile-only (w1-aot.log) |
| ds4_kernels route_publish_kernel | ds4_kernels.dp.cpp | 1514 | 32 | Masks 16..1, the best score of a pass. The launch pins 32 at HEAD (:2424). | compile-only (w1-aot.log) |
| ds4_kernels route_publish_kernel | ds4_kernels.dp.cpp | 1516 | 32 | Masks 16..1, its index. Covered by the same pin. | compile-only (w1-aot.log) |
| ds4_kernels route_ahead_kernel | ds4_kernels.dp.cpp | 1599 | 32 | Masks 16..1, the best score of a pass. The launch pins 32 at HEAD (:2487). | compile-only (w1-aot.log) |
| ds4_kernels route_ahead_kernel | ds4_kernels.dp.cpp | 1601 | 32 | Masks 16..1, its index. Covered by the same pin. | compile-only (w1-aot.log) |
| ds4_gemv warp_sum | ds4_gemv.dp.cpp | 24 | 32 | Masks 16..1. The three wrappers pin 32 at HEAD (:218, :237, :253), tile path and scalar fallback alike. | B60 run (w1-u-blas-parity.log) |
| ds4_numerics warp_sum | ds4_numerics.dp.hpp | 52 | 32 | Masks 16..1. Every launch that runs a kernel taking it pins 32: `ds4_attn_hc` (:642, :662, :708, :794, :844), `ds4_kernels` (:2861, :3177), `ds4_gemv` (:218, :237, :253). | B60 run (w1-u-parity.log) |
| ds4_numerics warp_max | ds4_numerics.dp.hpp | 58 | 32 | Masks 16..1. The sparse attention launches pin 32 at HEAD (:708, :794, :844). | B60 run (w1-u-parity.log) |
| ds4_gguf warp_sum | ds4_gguf.dp.cpp | 72 | 32 | Masks 16..1. The launches pin 32 at HEAD (:1628, :1640, :1667, :1691, :1699, :1705, :1882, :1912, :1946, :1974). | compile-only (w1-b5-ninja.log) |
| ds4_gguf q8_quant_kernel | ds4_gguf.dp.cpp | 312 | 32 | Masks 16..1, the Q8 scale's max. The `q8_quant` launch pins 32 at HEAD (:328). The audit names launch :1524 for this row; that launch is `mx_set_wrapper`, which has no permute (static reading, re-checked). | compile-only (w1-b5-ninja.log) |
| ds4_attn_split wsum | ds4_attn_split.dp.hpp | 37 | 32 | Masks 16..1. The launcher pins 32 at HEAD (:295). | compile-only (w1-b5-ninja.log) |
| ds4_attn_split wmax | ds4_attn_split.dp.hpp | 44 | 32 | Masks 16..1, the same pass's max. Covered by the same pin (:295). | compile-only (w1-b5-ninja.log) |
| ds4_softmax block_reduce | ds4_softmax.dp.cpp | 32 | none | Masks follow the live sub-group width (`sgw/2 .. 1`). `softmax_wrapper` pins 32 at HEAD (:105), so the derived masks stay inside it. A pin here would change the order of the adds and the BF16 bits that come out. | B60 run (w1-u-blas-parity.log) |
| ds4_softmax block_reduce | ds4_softmax.dp.cpp | 43 | none | Same reduction, the second operand. | B60 run (w1-u-blas-parity.log) |
| ds4_vision block_reduce | ds4_vision.dp.cpp | 77 | 32 | Masks 16..1, the warp step. The launches pin 32 at HEAD: `softmax_kernel` (:733), `rmsnorm_kernel` (:670, :781, :836). The audit calls this site `softmax_tile` and names launch :272, which is a GEMM call (static reading, re-checked). | compile-only (f3tc-build-aot.txt) |
| ds4_vision block_reduce | ds4_vision.dp.cpp | 102 | 32 | Masks 16..1, the step over the per-warp partials. Covered by the same pins. | compile-only (f3tc-build-aot.txt) |
| vendor/exl3 had128 | vendor/exl3.dp.cpp | 254 | 32 | Masks 1, 2, 4, 8, 16 - the 128-point fold. The launches pin 32 at HEAD (:1631, :1654, :1676, :1698, :1721, :1752, :1795, :1835, :2488, :2506). The audit names launch :353, which is a tile loader (static reading, re-checked). | compile-only (f3tc-build-aot.txt) |
| vendor/exl3_legacy had128 | vendor/exl3_legacy.dp.cpp | 161 | 32 | Masks 1..16, the same fold. The launches pin 32 at HEAD (:1088, :1124, :1153, :1181, :1253, :1303). | compile-only (f3tc-build-aot.txt) |
| vendor/exl3_legacy block_gemv | vendor/exl3_legacy.dp.cpp | 272 | none | Mask 1, a 4-lane group sum. Covered by the default 16. | compile-only (f3tc-build-aot.txt) |
| vendor/exl3_legacy block_gemv | vendor/exl3_legacy.dp.cpp | 275 | none | Mask 2, the same group sum. | compile-only (f3tc-build-aot.txt) |
| vendor/exl3_legacy block_gemv_n | vendor/exl3_legacy.dp.cpp | 380 | none | Mask 1, the multi-row version of that sum. Covered by the default 16. | compile-only (f3tc-build-aot.txt) |
| vendor/exl3_legacy block_gemv_n | vendor/exl3_legacy.dp.cpp | 383 | none | Mask 2, the same sum. | compile-only (f3tc-build-aot.txt) |

## Why 32 for the xor-16 kernels

The 27 rows that ask for 32 are warp reductions: `warp_sum`, `warp_max`, `lanes_sum`, the Hadamard fold, the
best-of-a-pass sites, the Q8 scale's max, the attention and vision tiles. Each walks masks 16, 8, 4, 2, 1 so 32
lanes fold into lane 0. The mask-16 step is the one that needs a 32-wide sub group: with the default 16 it pairs a
lane with itself, so lane 0 keeps half the block and the row's answer is short by that half. Pinning 32 makes every
step in the walk valid. Pinning 16 would be wrong, because 16 is the size that cannot hold the mask.

## Why 16 for the XMX and joint_matrix kernels

Five sites ask for 16 and none of them permutes: `ds4_gguf.dp.cpp:1476`, `ds4_iq.dp.cpp:1266` and :1494, and the
two launches in `xmx_global_pack_probe.cpp` (:43, :51). `joint_matrix` on Xe2 tiles at 8x16x16, which needs a sub
group of 16 lanes, and the tile's accumulator layout is defined for that width. A 32-wide sub group would split the
tile across two sub groups and change the result. The probe covers both sizes on this card: a `reqd(16)` kernel and
a `reqd(32)` kernel each built and ran on the B60, JIT and AOT (`sg-probe.log`, and `aot-probe.log` for the
`bmg-g21` artifact). So 16 is for the matrix tiles and 32 is for the reductions; no kernel needs both.

## Sites that need no pin

The 13 rows marked `none` permute at mask 8 or below, or read the sub-group width at run time and walk half of it
down to 1 (`ds4_softmax.dp.cpp`'s `block_reduce`). A mask of 8 fits inside the device default of 16, so pinning
these would add a constraint for no gain. In `block_reduce` a pin would be worse: the masks are derived from the
live width, so forcing 32 changes the order of the adds and the BF16 bits that come out. The softmax wrapper
already pins 32 at HEAD (:105), which is what keeps the derived masks inside the sub group.

## What the Device Verified column is, and what it is not

Six rows ran on the card. `w1-u-blas-parity.log` ran the `gemv` and `softmax` kernels against a CPU reference,
tile path and scalar fallback. Every one of those rows reports 0 values outside one BF16 ulp, and the suite prints
`MATCH: 50937/51242 values equal to the CPU reference` with `BLAS_SELFTEST_RC=0`. The corrupt-input controls in the
same log exit 1 on purpose (`BLAS_CORRUPT_RC=1`, `BLAS_CORRUPT_LDX_RC=1`), which is what shows the harness can
fail. `w1-u-parity.log` ran the `iq` GEMV rows
(`Q2_K`, `Q4_K`, `Q5_K`) and the attention rows (`sparse_attn`, `hc_pre_norm`, `hc_mix_scale`) with 0 values
outside one BF16 ulp. The flagged build ran the same rows the same way (`w1-parity-flagged.log`,
`w1-blas-parity-flagged.log`).

The other 34 rows are compile-only. `w1-aot.log` built `ds4_tq.dp.cpp`, `ds4_iq.dp.cpp` and `ds4_kernels.dp.cpp`
ahead of time for `bmg-g21` with the pins in place, exit 0, and its negative control shows a `reqd(64)` pin is
rejected by `ocloc`. `w1-b5-ninja.log` built `ds4_gguf.dp.cpp`, `ds4_softmax.dp.cpp`, `ds4_gemv.dp.cpp`,
`ds4_attn_hc.dp.cpp` (which includes `ds4_attn_split.dp.hpp`) and `ds4_iq.dp.cpp` for the parity targets, exit 0.
`f3tc-build-aot.txt` built the `strata-dsv41` target, the only target that compiles `ds4_kernels.dp.cpp`,
`ds4_vision.dp.cpp`, `vendor/exl3.dp.cpp` and `vendor/exl3_legacy.dp.cpp`; it reports 0 errors and `Build
succeeded`.

No row here is claimed as device verified for the two `ds4_tq` gap rows. The `tq` harness needs a model file and
was not run in this round's logs, so those rows stay compile-only until a run lands.  The four tq wrappers are
pinned `[[sycl::reqd_sub_group_size(32)]]` at `ds4_tq.dp.cpp:642`, :657, :671 and :685 - `gate_up_wrapper`,
`down_wrapper`, `set_wrapper` and `mv_wrapper`, one pin each - at the integration tip 7bd2f17d.  This branch has
no tq pin yet: `grep -n reqd_sub_group_size sycl/src/dsv41/ds4_tq.dp.cpp` -> rc 1.  No run of a pinned tq kernel
is in this round's logs either, so the pinned rows are compile-only as well.

## The staged opt-in switch (`STRATA_XMX_F16_STAGE`)

`STRATA_XMX_F16_STAGE=1` is a runtime switch for the IQ4_XS set tiles in `ds4_iq.dp.cpp`. Set it before the program
starts. With it on, the gate/up tiles split each activation row into its F16 value and the bits the F16 rounding
dropped, and run the two multiplies into the same F32 accumulator. It is off by default, so the normal launch is the
one this branch shipped before. It does nothing unless the build carries the staged kernel, and only the AOT build
for `bmg-g21` carries it (`STRATA_XMX_SG16=ON`); the JIT build has none.  The gate that turns it on is
`sycl/CMakeLists.txt:40-41`, which fires for an AOT device starting with `bmg` or `lnl`; `bmg-g21` is the
device this PC has, and it is the only build whose staged instantiation was run.  The down rows arrive as
F16 bits, so their residual is zero and the switch leaves them as they were.

Measured on this PC's Intel Arc Pro B60 on 2026-10-29, with the bmg-g21 AOT artifact of this branch and
`ds4_parity --only iq_set_tc`: gate/up goes from 1792/8192 bits exact to 7936/8192 at `n_embd=256 n_ff=256`, and
from 2048/8192 to 8192/8192 at `n_embd=512 n_ff=256`.  Both legs report 0 values outside (rel 2.0e-4, floor 1.0e-3),
and the down rows keep their default counts, 1024/8192 and 2048/16384
(`w2-parity-aot-default.log`, `w2-parity-aot-staged.log`).  The controls still fail on purpose: `--bad-scale-gu`
lands 32 values outside on the 256-wide row and 31 on the 512-wide one, and `--bad-scale` lands 32 on each down row
(`w2-parity-aot-bad-scale-gu-staged.log`, `w2-parity-aot-bad-scale-staged.log`).

One note on the harness verdict: `ds4_parity` exits 1 in both builds, `NO MATCH: 3 values outside one BF16 ulp`, in
the `stageA` sets at M = 16 and M = 17. Those are the XMX `reqd(16)` tiles, not a sub-group reduction, and the same
3 values fail at HEAD with no pins at all (`w1-parity-head.log`, `w1-u-parity.log`). Checkbox 1.4 keeps that row.
