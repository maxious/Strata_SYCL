# Experiment 36 - the Q6_K/Q5_K pre-unpack, end to end: the kernel wins, the engine loses

**Question.** exps 28/30 wired the byte pre-unpack and made it **default on**, with kernel numbers of2.05x
(Q6_K) and 1.29x (Q5_K) at ncols=1. The one measurement never taken was the engine-level A/B, which is the only
number that decides whether default-on was right.

**Verdict. DEFAULT OFF.** The pre-unpack is a **12.4% decode loss** end to end on the Coder IQ1_M - the model whose
dense decode is Q6_K x128, the case the change was built for. The kernel is genuinely faster; the loss is the
2.53 GiB of extra dense VRAM, which on a 32 GB card is 1,300 expert-cache slots, and the slots are what the decode
actually runs on. `STRATA_MMVQ_PREUNPACK=1` now opts in.

Measured on: Arc Pro B60 (32 GB), engine 0.1.38-sycl, Coder IQ1_M shards (dense decode Q6_K x128, Q5_K x35),
single card, `--spec 4 --prefill 128 --max-context 4096 --expert-cache auto`, 512-token prompt (the v1 long prompt's
first 512 ids), `--max-new 256 --greedy`, arms alternating on one binary. The config decodes deterministically
(100% expert-cache hit rate), so token identity is a usable gate.

## The engine numbers

| run | expert slots | pre-unpack OFF (packed) | ON (pre-unpacked) | delta |
|---|---:|---:|---:|---:|
| rep1 | auto -> 9,340 / 8,025 | **40.81 tok/s** | 35.60 tok/s | -12.8% |
| rep2 | auto -> 9,340 / 8,025 | **40.84 tok/s** | 35.70 tok/s | -12.6% |
| rep3 | auto -> 9,340 / 8,025 | **40.88 tok/s** | 35.73 tok/s | -12.6% |
| timing runs (128 tok) | 9,478 / 8,152 | **42.20 / 42.22** | 37.95 / 38.07 | -10.1% |
| **slot-matched, rep1** | 8,334 vs 8,152 | **38.39 tok/s** | 38.03 tok/s | -0.9% |
| **slot-matched, rep2** | 8,334 vs 8,152 | **38.44 tok/s** | 38.03 tok/s | -1.1% |

Output is token-identical between arms in every run.

## Why: the pre-unpack is priced in expert slots

The load log is unambiguous:

```
arm OFF: 300 native projection matrices, 2018.88 MiB of weights
         expert cache auto: 18.73 GiB free -> 7179 slots; settled 9340 slots, 17.80 GiB
         2948 experts missing from VRAM mirrored in pinned host memory (5.62 GiB)
arm ON:  463 native projection matrices, 2018.88 MiB of weights   <- +163 = 128 Q6_K + 35 Q5_K copies
         expert cache auto: 16.20 GiB free -> 6160 slots; settled 8025 slots, 15.27 GiB
         4263 experts missing from VRAM mirrored in pinned host memory (8.15 GiB)
```

+163 matrices and 2.53 GiB less free VRAM, and the auto-sizer converts every GiB it does not get into expert
residency. The decode streams the difference over PCIe every token: the prompt path pulled **2,972 experts with the
pre-unpack off and 5,533 with it on**, and the mirror grows 5.62 -> 8.15 GiB.

**The slot-matched arms settle it.** `--expert-cache N` is a *hint* the auto-sizer exceeds (a hint of 8,025 landed
9,478 slots on the packed arm), so the packed arm was given a 6,400 hint to land on the pre-unpack arm's 8,152:

| slots | packed | pre-unpacked |
|---:|---:|---:|
| ~8,200-8,350 (matched) | 38.39 / 38.44 tok/s | 38.03 / 38.03 tok/s |
| ~9,400 vs 8,150 (auto) | 42.20 tok/s | 38.03 tok/s |

At matched slots the arms are within 1%, and the residual is itself explained by the last182 slots. So the decode
kernel is worth about what the bench says, and the whole regression was the cache.

## The kernel really is faster - on the shapes the engine decodes

The bench had only been run on 2560x2560. The Coder's actual dense Q6_K shapes (read out of the GGUF headers with
`gguf_count_dense_usage.py`'s parser) are (2560, 512) x21, (2560, 640) x26, (2560, 6144) x14, (2560, 10240) x22,
(2560, 12288) x5 and (6144, 2560) x40. `q6k_preunpack_bench` on each, 200 reps:

| shape (n_in x n_out) | 1 col | 2 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| 2560 x 512 | 1.53x | 1.78x | 1.73x | 2.08x | 2.27x |
| 2560 x 640 | 1.57x | 1.72x | 1.67x | 1.96x | 2.16x |
| 2560 x 6144 | 1.52x | 1.49x | 1.38x | 1.50x | 1.95x |
| 2560 x 10240 | **1.04x** | **1.05x** | **1.08x** | 1.23x | 1.66x |
| 6144 x 2560 | 1.28x | 1.26x | 1.23x | 1.54x | 1.74x |

The 2.05x headline is the square shape; on the **largest** and most numerous shape (2560 x 10240, 22 tensors) the
win at the engine's ncols=1 is 1.04x. So even without the VRAM effect the aggregate decode win is modest.

## Decision

- **Default off.** `native_dense.cpp`'s `q6k_preunpack_enabled()` now needs an explicit
  `STRATA_MMVQ_PREUNPACK=1`. Parity gates (`q6k_preunpack_parity`, `q5k_preunpack_parity`) stay registered; ctest is
  29/29.
- **Where it would still win:** a card whose expert cache is not the binding constraint (all profiled experts
  resident anyway), or a model whose dense decode is mostly the 2560x512/2560x640/6144x2560 shapes. Neither is the
  configuration the port ships on, so opt-in is the honest default.
- **What would change it:** shrinking the pre-unpacked buffer. The scales are fp32 in a 40-byte block (32 weight
  bytes + 2 fp32); storing `d0`/`d1` as bf16 or fp16 and keeping the per-block layout 32-byte aligned would cut the
  2.53 GiB roughly in half, and a bf16 scale pair is within the ~1e-7 the parity test already allows. That is a
  small, testable change - not a guess - and it is the only route by which this lever pays on a 32 GB card.

## Reproduce

```sh
# engine A/B, arms alternating on one binary (the slots differ; that is the point)
STRATA_MMVQ_PREUNPACK=$arm ./strata --pack ~/ComfyUI/koboldcpp/pack-coder \
  --native ~/ComfyUI/koboldcpp/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --ple-gguf ~/ComfyUI/koboldcpp/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf \
  --expert-profile data/expert-profile-coder.bin --expert-cache auto --stream-experts \
  --prefill 128 --spec 4 --mtp ~/ComfyUI/koboldcpp/mtp/rt --max-context 4096 \
  --tokens-file <512 ids> --max-new 256 --greedy --stats

# the slot-matched arms: the packed arm takes a 6,400 hint, the pre-unpack arm 8,025
# (the hint is a floor the auto-sizer exceeds; verify with the "expert cache N slots" line)

# the kernel side, on the shapes the model actually decodes
./q6k_preunpack_bench <n_in> <n_out> 200# 2560 512 / 2560 640 / 2560 6144 / 2560 10240 / 6144 2560
```

## Traps this run adds

- **A default-on change needs its engine A/B before it ships, not after.** A 2x kernel number is not a decode
  number: the same kernel that wins 1.5-2x in isolation was worth -12.4% in the engine because of what its memory
  costs elsewhere.
- **`--expert-cache N` is a hint, not a pin.** Asking for 8,025 slots produced 9,478. Any A/B that intends to hold
  the cache fixed must read the achieved `expert cache N slots` line out of the log, not trust the flag.
- **The auto layer split is unstable on Q2_0 dual**: two runs of the same config chose K=22 and K=24 (and 11,264 vs
  12,288 slots), which alone moved PP 582 -> 350 tok/s. Pin `--layer-split K` before any dual A/B.
- **GPU Hotspots did not capture this run** (`GPU utilization is low`, 0.188s elapsed): two collections of a
  one-shot engine run reported 0% GPU time, so in-situ kernel attribution stayed unavailable and the attribution
  above came from the engine's own load log and `STRATA_DECODE_TIMING` (host cost identical at 1.2 ms/round, so
  the difference is on the device/PCIe side).