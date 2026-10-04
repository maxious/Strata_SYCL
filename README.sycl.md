# Strata on SYCL: the seven open paths

**Environment first.** Anything that builds or runs the SYCL port needs the oneAPI environment initialized:
`source /opt/intel/oneapi/setvars.sh` (older installs: `setenv.sh`). `sycl/tools/build.sh` does this itself, but a
handwritten build or run must too - without it `icpx`/oneMKL are not on `PATH`/`LD_LIBRARY_PATH` and no ICD is
registered, so the build misses the MKL headers and a built binary fails with `No device of requested type
available`.

This page lists the only work still worth doing on the SYCL port, ranked. Everything else has been measured and
closed: the full plan, with every experiment and its numbers, is archived at
[docs/sycl-experiments/README.sycl.full-archive.md](docs/sycl-experiments/README.sycl.full-archive.md), the
run-by-run log is [docs/INTEL.md](docs/INTEL.md), and the read-outs are `docs/sycl-experiments/NN-*.md`. Read INTEL.md
before changing anything in `sycl/`: it records every trap that ended a run.

Where the numbers come from: **Arc Pro B60 (32 GB)** and 2x B60 for the split, Q2_0 and the Coder IQ1_M, at
October 2026. The port's decode is 78.2 tok/s against llama.cpp's 23-25 on the same card, and its prompt reads up to
1,117 tok/s at 40K, so the remaining headroom is not in catching llama.cpp - it is in the specific places below.

Two rules that every item inherits:

- **Noise floor +-1.5% TG** on the dual Q2_0 decode config (exp 33). Anything smaller is not a result.
- **Verify the binary is fresh.** `build.sh` used to exit rc=1 silently with a stale `build.log` (fixed in
  `f5c84a2`; exp 33's engine arms were the same binary). Check `strings <bin> | grep <new string>`, or
  `ninja -n <target>`, before believing an A/B.

---

## 1. The Q6_K/Q5_K pre-unpack end to end - MEASURED (exp 36): DEFAULT OFF, it costs 12.4% of decode

**Done, and it reversed the default.** The byte pre-unpack (Q6_K -> `Q6U` signed-byte blocks, Q5_K -> `Q5U`,
loaded once at weight load) is faster in isolation on every shape the model actually decodes - 1.04-2.27x at
ncols=1..8, verified against the Coder IQ1_M's real dense shapes, not just the bench's square one. End to end it is a
**12.4% decode loss** (42.2 -> 38.0 tok/s, three alternating reps, token-identical output).

The mechanism is memory, not arithmetic: the pre-unpacked copies are **+163 matrices / 2.53 GiB** of dense VRAM, and
on a 32 GB card that is **1,300 expert-cache slots** (9,478 -> 8,152, RAM mirror 5.37 -> 7.90 GiB), which the decode
streams over PCIe every token. At **matched slot counts** the arms are within 1% (38.4/38.4 packed vs 38.0/38.0
pre-unpacked), so the kernel is worth what the bench says and the regression was the cache all along.

`native_dense.cpp` now defaults it **off**; `STRATA_MMVQ_PREUNPACK=1` opts in, where a card whose expert cache is not
the binding constraint may still win. The route that could make it pay on a 32 GB card is shrinking the block: the
scales are fp32 in a 40-byte block, so bf16/fp16 scales would halve the 2.53 GiB at the same ~1e-7 parity. Full
numbers, the slot-matched arms and the new traps (`--expert-cache` is a hint, not a pin; the Q2_0 dual auto-split
picks K=22 or K=24 between runs; GPU Hotspots reports 0% GPU time on a one-shot engine run) are in
[docs/sycl-experiments/36-preunpack-end-to-end.md](docs/sycl-experiments/36-preunpack-end-to-end.md).

## 2. The QSA top-k at long context (archive D3 + E3/F6) - DONE (exp 37): SHIPPED, `TK_PER_MAX = 66` on SYCL

Xe2's GRF holds 66 keys per thread at 1,024 threads exactly as RDNA's VGPRs do, and the port was leaving that on
the table: `TK_PER_MAX` was 33 on every non-HIP build, so the shipped 262,144 context fell off the register kernel
onto the wide one (every key re-read per radix pass). **Now 66 on the SYCL branch**, with the shipped capacity:

| ctx (cells), capacity 262,144 | 33 (was) | 66 (now) | speedup | ids identical |
|---:|---:|---:|---:|---|
| 135,168 | 0.257 ms | **0.138 ms** | 1.86x | 1/1 |
| 200,000 | 0.353 ms | **0.177 ms** | 1.99x | 1/1 |
| 262,144 | 0.541 ms | **0.224 ms** | 2.42x | 1/1 |

A deployment sized to its context (capacity = ctx) is **unchanged** at 65,536 and 131,072 cells (0.059/0.117 ms,
identical), because both widths take the 33-wide kernel there; 135,168 is exactly the old cliff edge. Compile flags
were verified identical apart from the define. `qsa_select_bench` is now a CMake target, `private_alloca` was not
needed, and `-DSTRATA_TK_PER_MAX=<n>` still overrides. ctest 29/29.

Two dead-code traps are in the write-up: `-DSTRATA_TK_PER_MAX=66` changed nothing until the dispatch's `fit` was
fixed (`fit = TK_T * (counted ? TK_PER_MAX : TK_PER)`, and `counted` is the CUDA active-count path, always false on
SYCL - an A/B that returns exactly 1.00x everywhere means the knob is not wired), and the bench's *capacity*
argument drives the dispatch, not the context. Full numbers:
[docs/sycl-experiments/37-qsa-topk-tk-per-max.md](docs/sycl-experiments/37-qsa-topk-tk-per-max.md).

**Still owed**: the end-to-end confirmation on a real prompt past ~135K cells that keeps generating - exp 32's
synthetic long prompts stop early.

## 3. Conversation parking on a split - MEASURED (exp 41): IMPLEMENTED, a switch is 1.9x-2.9x faster

The refusal is gone: a split now parks one image per stage (`SavedStage`), each captured and restored on the card
that owns those layers, with the drafter's image on the last stage. Two prefix-disjoint 4,000-token conversations,
alternating on 2x B60 (`--conversation-cache-mib 0` vs `8192`, split auto -> K=22):

| request | no parking | with parking | |
|---|---:|---:|---|
| A, 1st visit | 7367.6 ms | 7396.6 ms | full prefill either way |
| B, 1st visit | 6478.7 ms | 6627.6 ms | full prefill either way |
| **A, return** | 7021.7 ms | **3684.4 ms** | 2185 tokens reused -> **1.91x** |
| **B, return** | 6435.6 ms | **2184.9 ms** | 3278 tokens reused -> **2.95x** |

Park 108-141 ms (298-417 MB snapshot), restore **32-36 ms**. Not the ~200x the raw numbers suggest, because a
restore only brought back a *checkpoint prefix* (2,185 / 3,278 of 4,000 tokens) and the rest was re-read - the copy
leaves the bottleneck but does not remove it. ExTV's 0.4-0.5 s restores the whole conversation, not a prefix.

**Open: one generated token differs** (request 4, position 2: `21966` vs `3575`, the other 11 identical; requests 1-3
are bit-identical). The obvious rounding explanation is unsupported - request 3 had a different reuse split and
matched exactly - so it is recorded as undiagnosed. `STRATA_SNAPSHOT_VERIFY=1` is the next step. **Also open:**
`retain()` still recycles only the primary's K/V, so later stages re-copy in full on each park (`reused_kv_bytes=0`),
which costs every conversation *rewrite* - regenerate, branch, or a future compaction - not just a switch.

Four defects lived in this item, all in the new code and all found by reading the engine's own refusal rather than
by a test: an empty `live.ids` that restored stale pooled rows; the `stage_parts` guard that survived removing the
refusal; stripping `stage_parts` to satisfy it, which broke the checkpoint restore (SIGSEGV *after* a 30.9 ms
restore); and a test that measured a truncation rather than a switch. Numbers and traps:
[docs/sycl-experiments/41-split-conversation-parking.md](docs/sycl-experiments/41-split-conversation-parking.md).

## 4. Adjacent prompt-chunk overlap on 2x B60 - MEASURED (exp 38): CLOSED, the stages already overlap

**1.25x of concurrency, measured.** On the 8,000-token prompt the per-stage GPU timelines sum to **18.66 s against a
14.94 s wall** - the chunks are already software-pipelined across the two stages (`prefill.cpp` runs the next stage's
chunk on a `std::future` while this one runs). The missing 0.75x is not idle time waiting for a neighbour: stage 1
reads stage 0's residual from the mapped hand-off buffer, so the wall is bounded by the same dependent chain exp 34
found on the decode side. **There is no adjacent-chunk overlap left to build.**

Two side findings worth keeping. The prompt's real bound is the **expert stream**, not the split: two chunks show
wall ~2x their GPU timeline, which is `--stream-experts` pulling blobs over PCIe plus host grouping. And the shipped
`TK_PER_MAX=66` is **neutral on the prompt** (535.5 vs 543.1 tok/s for 33 - 1.4%, inside the noise floor), which is the
regression check exp 37 owed.

**The trap that cost the most time**: with `--prefill 4096` a 2,185-token prompt is a **single chunk**, so D4's premise
does not exist at that size, and the phase shares are garbage - a first probe put 96.7% of stage 1's timeline in one
`qsa select` interval. At 8,000 tokens (7 chunk-timelines) select is 2.7-4.1%. Any probe of this item needs a prompt
several times the chunk size. Write-up:
[docs/sycl-experiments/38-prompt-stage-overlap.md](docs/sycl-experiments/38-prompt-stage-overlap.md).

## 5. `kernel_args_restrict` on the MMVQ - MEASURED (exp 40): NULL, because the port already has it by hand

The attribute is real and it **compiles and links** on icpx 2026.1 (verified with a standalone kernel - the opposite of
item 7's cache controls). Applied to the two Q6_K wide launches behind `-DSTRATA_MMVQ_RESTRICT_ATTR=1`, it changes
**nothing**: `mmvq_bench` is bit-identical at every ncols (19.1 / 21.5 / 26.1 / 55.3 us at ncols 1 / 2 / 4 / 8), while the
object md5 differs - so the attribute was applied and simply buys no time. The reason is that
`native_mmvq_q6k_wide_kernel` **already declares `__restrict__` on w and x** (88 occurrences in the file, 0 uses of the
attribute in the tree), and the kernel is pipe-bound rather than alias-bound (exp 23). Parked, default off; the knob
stays in the source for a kernel that is *not* hand-annotated, and it is an unchecked assertion, so it must not be
sprayed over kernels that accumulate in place.

## 6. Peer access for the peer expert tier - MEASURED (exp 40): SUPPORTED both ways, and already in use

`can_access_peer` is **true in both directions** on the B60 pair, and `enable_peer_access` followed by the identical
copy is **1.00x** (7.7-7.8 GB/s both ways, round-trip clean): the plain `sycl::memcpy` between the two contexts is
*already* on the peer path. That corrects exp 03's "silent no-op" reading - the hand-off's 34 us is not a missing peer
path, and 7.7 GB/s is above the pair's own 13.8 GB/s host-to-device figure, so the copy is peer-class. The peer expert
tier stays a **plumbing** change, not a bandwidth project. Details in
[docs/sycl-experiments/40-mmvq-restrict-and-p2p.md](docs/sycl-experiments/40-mmvq-restrict-and-p2p.md).

## 7. The MMVQ cache policy - MEASURED (exp 39): CLOSED, this toolchain cannot express it

exp 22's sweep was null because `-cl-load/store-cache-default` never reached the compiler. The in-source route
(`sycl_ext_intel_cache_controls`, a `read_hint` property on the launch) is also unusable on **icpx 2026.1**: every
spelling - the spec's two-entry L1/L2+L3 form, a single-entry `read_hint`, and `annotated_ptr` - **compiles and then
fails at the SPIR-V device link** with `InvalidLlvmModule: CacheControlLoadINTEL requires exactly 2 extra operands`,
once per kernel. A~12-line standalone kernel reproduces it, so this is the compiler, not the port.

The knob stays in the source (`-DSTRATA_MMVQ_CACHE_MODE`, default 0 = no hints, the shipping path) so a fixed compiler
can sweep it later; modes 1-3 do not link here. **exp 22's null is now permanent rather than untested.** Note the
contrast with item 2: register-width control (`TK_PER_MAX`, a template argument) links fine on the same compiler, so
this is specific to the cache-control intrinsics, not to build-time kernel tuning. Write-up:
[docs/sycl-experiments/39-mmvq-cache-policy-toolchain.md](docs/sycl-experiments/39-mmvq-cache-policy-toolchain.md).

---

## Parked, and why - do not reopen without new evidence

| Parked | Verdict |
|---|---|
| Weight reorder + ESIMD for decode matvecs (archive L1) | exp 12: a wash at 1 column, 0.25-0.63x slower multi-column. Q6_K's stalls are Pipe-bound, so there is no memory stall for a reorder to remove (exp 23). |
| XMX for decode (GEMM, fused quantized, expert dots, `xmx_gemm_iq`) | exp 25: a persistent-FP16 dense GEMM is 0.54-0.86x at the engine's real ncols=1..4 and only crosses at ncols>=6, which does not repay 2x dense VRAM. The matrix units idle because XMX's backend cost repays only at GEMM-shaped batch. |
| INT8 prompt expert path | exp 13: the GEMM is 1.9-2.1x but the whole expert is 0.42-0.64x (requantization 0.065 ms against the FP16 dequant's 0.014) and 1.0-2.4% lossy. Opt-in `STRATA_PREFILL_INT8=1`, default off. |
| Faster i-quant dequant feeding oneMKL | exp 31: bit-identical register-table hoist, still ~75 GB/s. Codebook-*select*-bound, not a memory gather. Resist-optimization. |
| oneDNN: `Dequantize`+`MatMul` fusion | exp 24: structurally blocked - `Dequantize` reads s8/u8, not ggml block formats, and there is no SDPA op kind. oneDNN stays linked opt-in (`STRATA_SYCL_DNNL=1`). |
| oneDNN fused-XMX SDPA, oneMKL GEMM prompt attention, MKL-FA coalescing (archive P1) | depends on the fused paths above; Strata's prompt attention is sparse/gather-bound QSA, so llama.cpp's dense FA work does not map. Revisit only if a future dense GEMM path lands. |
| Pipelined dual-GPU windows (archive D2) | exp 34/35: the ceiling is ~1.13x, not 1.8x - window n+1's tokens come from window n's `outv`, so stage 0 cannot start early. `STRATA_DRAFT_EARLY` (the fork's launch-the-draft-first) measured null: the 2.9 ms moved buckets rather than disappearing. Real work against a bounded device spin that turns a stall into a wrong window; not worth 1.13x. |
| Per-card dense weights on a split (archive D6) | the dual decode win is nil - its cache already hits 100.0% over 24,492 slots. The slot gain is for the single card (95.5%) and the prompt. |
| Each card keeps only its own layers, card ordering, `STRATA_DF_PDL`, P-core/E-core pools | checked, not ours to port - no E-cores, identical cards, no sm_90 equivalent, and the last stage already runs head + draft. |
| `usm_device_read_only`, cooperative-group prefetch (archive F1/F2) | exp 33: 1.000x on the same kernel over plain vs read-only USM, 0 differing results; and no kernel is latency-bound to prefetch into. |
| Per-submission overhead knobs (archive E4/F4), profiling tags (F8), `virtual_mem` (E6), `register_host_memory` (E5) | cleanup or probes. `virtual_mem` pays only if a feature needs grow/remap, and our arena is RAM-bounded, not address-space-bounded; E5's PCIe reads of a mapped file are likely slower than today's pread-into-pinned. |
| `fp_control`, `dot_accumulate` (F3/F7) | parity/perf items with no measured target: dp4a is 32 of 985 instructions in the pipe-bound case, and no profile shows denormal traffic. |
| GDN key-head `cp.async` (exp 14), the Q6_K 384 B spill (exp 21), decode-round fusion (exp 05), fused GR variants (exp 06) | measured at or below the noise floor: 0.6% of prompt time, free, kernel-bound, already confirmed. |
| Host-pinned / dev2dev memcpy for the split (archive P0 host-mem item) | exp 03: the hand-off already uses `malloc_host` and costs 34 us; the 1.4x decode gain comes from the layer-split distribution that already ships. |

Two P2 hygiene items are still open and are worth an afternoon each, not a project: **load-test the IQ3_S
Flash-Next shards** end to end (`python3 sycl/setup_intel.py --model IQ3_S`) so the largest supported size stops
being "not yet load-tested", and **add `xmx_gemm_bench` + a parity step to `sycl/tools/build.sh`** so an XMX
regression fails the build the way any other kernel regression does.

## How to verify any of these

1. **Parity first.** The relevant `*_parity` test(s) must pass byte-for-byte or within their documented tolerance.
   Not "the output looks right" - compare tokens, and for native packs never compare only timings across
   `--prefill-until`. `ctest` is 27 tests, 100% (exp 17).
2. **Window the clocks.** Warm the GPU first; a cold 5 ms run measures the ramp. Kernel numbers come from
   `mmvq_bench`, `native_expert_parity NATIVE_BENCH=1`, `q6k_align_bench`, `q6k_preunpack_bench`,
   `q5k_preunpack_bench`, `xmx_gemm_bench`, `p2p_bench`, `gdn_rec_bench`, and `qsa_select_bench` (item 2, once built).
3. **Log it in INTEL.md** with the two-run measurement and an "output identical / near-tie" note, in the style of
   the existing speed tables. No number without the hardware it was measured on.
4. **Keep the default on the measured-fastest path**, behind a `STRATA_*=0` revert flag whenever it is not
   strictly faster.
5. **Capture changes go through the audit**: `graph_audit_test` plus the node goldens
   (`STRATA_GRAPH_GOLDEN=sycl/bench/graph-golden/<config>.txt`). Remember exp 19: `DPCT_CHECK_ERROR` discards its
   expression's value, which is how a 2 ms per-layer abandonment hid for a run - and exp 19 also found the eager
   path is not token-equivalent to the graph path, so every parity check must run the graph path.
6. **Before optimizing a kernel, get its stall reasons.** `vtune -collect gpu-hotspots -knob
   gpu-profiling-mode=source-analysis -knob source-analysis=stall-sampling -knob
   computing-tasks-of-interest="*<kernel>*"` then `-report summary -format csv`, run with `sudo` (the result
   directory is root-owned), after `sudo sysctl -w dev.xe.observation_paranoid=0` or the counters do not land.
   Profile a **bench**, not the engine - the engine's handshake degenerates under instrumentation. `Pipe` vs
   `Send` is the whole question for a matvec: exp 23 needed one collection to stop us optimizing a memory problem
   that was not there.