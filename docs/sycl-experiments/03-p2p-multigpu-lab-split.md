# Experiment 03 - P2P device-to-device (multi-GPU layer split) on two B60

## Question

The objective asks for P2P multi-GPU support. Strata's layer split (`docs/MULTI_GPU.md`) runs consecutive layer
ranges on different cards and hands the verify window from one card to the next once per window (a few hundred
KB). On CUDA/HIP that hand-off goes through pinned host RAM and is deliberately built so that *"no NVLink or
peer-to-peer access is needed"*. llama.cpp's SYCL multi-GPU is a device-carve: `ONEAPI_DEVICE_SELECTOR` /
`--split-mode`, `--main-gpu`. This experiment measures the P2P transfer between the two B60s directly, checks
that the engine's device model surfaces both cards, and gets a real dual-GPU layer-split run to work.

Reference: llama.cpp SYCL enumerates devices by Level-Zero ordinal, picks `--main-gpu` by ordinal, and splits
layers across them the same way (host-visible device buffers, no special P2P enablement). Its per-window
transfer is the same class as ours.

## Change / measurement

New `sycl/src/kernels/p2p_bench.cpp` (target `p2p_bench`, registered in `sycl/CMakeLists.txt`): selects the two
Level-Zero GPUs, allocates USM device memory on each, times `sycl::memcpy` D0->D1 and D1->D0 directly, then
round-trips and verifies every byte. This is the exact primitive the layer-split hand-off runs.

Plus a **real dual-B60 split bug fix** (see below): `stage_room` in the port's `sycl/src/program/generate.cpp`
had its `cudaMemGetInfo` call dropped by the migration, so every later split stage priced 0 GiB free and the
split crashed. Restored `dpct::get_current_device().get_memory_info(fb, tb)`.

## P2P benchmark (B60, oneAPI 2026.1, warm clocks)

    dev0=Intel(R) Arc(TM) Pro B60 Graphics
    dev1=Intel(R) Arc(TM) Pro B60 Graphics
    D0->D1 0.034 ms (262144 bytes) 7.7 GB/s | D1->D0 0.034 ms 7.7 GB/s
    round-trip check: 0 mismatches
    D0->D1 0.123 ms (1048576 bytes) 8.5 GB/s | D1->D0 0.123 ms 8.6 GB/s
    round-trip check: 0 mismatches

Device-to-device copy is available in both directions at **7.7-8.6 GB/s**, symmetric, with **0 byte errors**
(the PCIe Gen3 x8 class transfer; both cards share the platform fabric). At the split's ~256 KiB
hand-off size a direct copy costs **~34 us** - far below a decode round, so the hand-off is not the decode
bottleneck.

## Device enumeration (the ordinal mapping the split needs)

`sycl::device::get_devices()` / dpct on this box:

    dpct device_count = 5
      dev 0: Intel(R) Arc(TM) Pro B60 Graphics (gpu=1)    <- Level-Zero card 1
      dev 1: Intel(R) Arc(TM) Pro B60 Graphics (gpu=1)    <- Level-Zero card 2
      dev 2: AMD Ryzen 7 5700X3D 8-Core Processor (gpu=0) <- CPU
      dev 3 / 4: Intel(R) Arc(TM) Pro B60 Graphics (gpu=1) <- the same two cards via OpenCL

`--split-device 1` maps to the second Level-Zero B60. The OpenCL duplicates (dev 3/4) query 0 GiB free via
`get_memory_info` (`ext_intel_free_memory` unsupported there) - not the path the engine's split uses, but it
means a wrong device ordinal would read 0.

## The bug found and fixed (how the split actually runs on two B60s)

The `--serve --layer-split` run reached the second card and crashed:

    strata generate: layer split, CUDA1 expert cache: no room
    PLEASE submit a bug report ... (segfault, exit 139)

`STRATA_TRACE_SPLIT` instrumentation showed the port's `stage_room` for device 1 reported `fb = 0` - the
`cudaMemGetInfo` had been **dropped during migration** (the DPCT1106 comment was empty and `fb` stayed 0; every
other `get_memory_info` site in the file kept the call). So every split stage after the first priced 0 free and
"no room" -> segfault (the no-room path crashes in the Intel runtime instead of the engine's return 1).

Fix in `sycl/src/program/generate.cpp` `stage_room`:

    dpct::get_current_device().get_memory_info(fb, tb);   // restored: migration dropped the cudaMemGetInfo

After the fix, with a fixed layer split (K=24) and `--split-device 1`, both cards configure:

    strata trace: stage_room dev 1: fb 19.66 GiB reserve 2560 MiB -> room 17.16 GiB
    strata generate: layer split: CUDA1 runs layers 24-47, expert cache 6144 slots (12.35 GiB), 6144 of its 6144 profiled pairs; slot 0 verified
    strata generate: layer split: CUDA0 runs layers 0-23
    strata generate: token graph hit path: 12288 resident experts, decided on the device
    strata generate: session is up (engine 0.1.35-sycl)

Both B60s now hold the split: card 0 runs layers 0-23, card 1 runs layers 24-47 (12,288 resident experts across
the pair), and the split search (auto) also runs (it chose K=4 with the bug; K=24 fixed is the sane manual
placement for this model).

Serve needed the MTP draft layer; the benchmark section below shows the full fetch/pack/rt pipeline built and the
split run to completion.

## Dual-GPU layer-split benchmark (Coder IQ1_M on both B60s, warm, reproducible)

The split needs `serve` mode, which requires the MTP draft layer. That pipeline was built from the BF16
checkpoint's 31 `mtp.*` tensors: `mtp_fetch` (4.9 GB, verify 0), `mtp_pack.py --experts q2_0`
(`mtp-q2_0.gguf`, 0.889 GB), `mtp_rt.py` (`rt/`: experts.bin 707 MB + dense.bin 116 MB). MTP loads at 793 MiB of
VRAM.

Both paths measured via `serve` (`GEN 16` on a 5-token prompt, spec 2, 2 runs each, warm clocks):

| path | decode | prompt | drafts accepted |
|---|---|---|---|
| single B60 | 21.5 / 25.4 tok/s | 39.8 / 39.5 tok/s | 6 of 9 |
| **dual B60 split 24** (layers 0-23 card 0, 24-47 card 1) | **35.5 / 35.4 tok/s** | 49.0 / 43.8 tok/s | 6 of 9 |

**The dual-GPU layer split decodes ~1.4-1.65x faster** (35.5 vs 25.4 tok/s best-of) and reads prompts faster
(49 vs 39.8 tok/s) on the Coder IQ1_M. Both cards hold a 6144-slot / 12.35 GiB expert cache = 12,288 resident
experts total, one window hand-off (P2P 7.7-8.6 GB/s, ~34 us/256 KiB) per verify window. Same output tokens
(`T 271 2 220 17 ...`, drafts accepted 6 of 9 both paths), so the split is decode-parity-fast, not a different
answer.

A cold split run measured 14.2 tok/s initially - the GPU clock/graph-capture ramp (INTEL.md's "warm the clocks"
rule). Warm runs are the number to keep.

### Q2_0, expert profile resident: decode parity, the win is the prompt (2026-10-04)

The 1.4x decode is the Coder IQ1_M's number, not the split's in general. Q2_0 on the same two B60s, both paths
measured in serve (one GEN request each, cold page cache, 256 greedy new tokens, the real 2,185-token v1 long
prompt), gives:

| path | TG (tok/s) | PP (tok/s) | TTFT (s) | resident experts | decode cache hit rate | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| single B60 | 32.0 | 445.3 | 4.9 | 12,996 (16.7 GiB) | 95.5% | 22.2 GB |
| dual B60 split (K=22) | 33.1 | 576.4 | 3.8 | 24,492 (31.5 GiB) | 100.0% | 41.9 GB |

**On Q2_0 the split's decode is parity (1.03x) and its win is the prompt (1.29x).** The hit rate says why: the
single card's hottest-first profile already serves 95.5% of the routed experts from VRAM, so the dual config's
near-2x resident experts add ~4.5% of hits and ~3% of TG. Decode at this config is not expert-bandwidth-bound,
while prefill is - the same benchy widens the prompt gap with length (PP 668 -> 945 tok/s at 40,000, 583 -> 938
at 128,000, 485 -> 794 at 256,000), and TTFT follows it (60.1 -> 42.5 s, 219.8 -> 136.7 s, 528.7 -> 322.8 s).
Those long-prompt PP figures are the benchy's single one-shot against dual serve (the 2,185 pair above is
mode-matched); the modes differ ~2% on PP (single serve 445 vs single one-shot 455 at 2,185), well below the gap.

So the split's decode gain depends on whether one card's decode work is the constraint: it is on the IQ1_M
(1.4x), it is not on Q2_0. What would move Q2_0's decode is overlapping the cards instead of alternating them
per verify window (README.sycl.md's P0 item on the Hardin22 fork).

## Validity

- P2P round-trip bytes are **identical in both directions** (0 mismatches at 256 KiB and 1 MiB), the bit-exactness
  the split's "hand-off is bit-exact" requires.
- The two devices are distinct physical cards (identical names, distinct Level-Zero handles), independently
  allocatable and queryable.
- The split bug fix is verified by the trace: `stage_room dev 1` now prices 17.16 GiB of room instead of 0, and
  both cards configure their caches and come up as a valid session.
- The dual-GPU split is reproducible (35.5 / 35.4 tok/s), same acceptance (6 of 9), same output tokens as
  single-GPU.

## Conclusion

P2P device-to-device transfer between the two B60s works through the SYCL runtime at 7.7-8.6 GB/s with exact
content. A real dual-B60 layer split now works and decodes faster than one card: after finding and fixing a
one-line migration bug (dropped `get_memory_info` in `stage_room`, which made every second-card split stage
report "no room" and crash), the split configures both cards (CUDA0 layers 0-23, CUDA1 layers 24-47, 12,288
resident experts) and, with the MTP draft layer, decodes at 35.5 tok/s vs 25.4 single-card on the Coder IQ1_M
(~1.4x) with identical outputs. P2P transfer and the split hand-off are both validated. **The decode gain is
model-dependent**: 1.4x on the IQ1_M, decode parity on Q2_0, whose single-card decode is already expert-resident
(95.5% hit rate) so the split's second card pays for itself in prefill instead (Q2_0 section above).
