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

## 1. Confirm the Q6_K/Q5_K pre-unpack end to end - a missing measurement, not an idea

The byte pre-unpack is **wired and default on** (`STRATA_MMVQ_PREUNPACK=0` opts out): Q6_K goes to `Q6U`
signed-byte blocks, Q5_K to `Q5U {dsc, mn1, qs[32]}`, both loaded once at weight load and decoded through the
no-bit-unpack kernel. Parity is `~1e-7` (`q6k_preunpack_parity`, `q5k_preunpack_parity`, all shapes PASS) and the
kernel-level numbers are real: **2.05x at ncols=1** for Q6_K (19.1 -> 9.3 us), 1.58-2.10x across the curve, 1.29x
for Q5_K. What is missing is the engine-level confirmation on a real model, which is why the item is still open.

It matters because this kernel is **~25% of decode GPU time** (`native_mmvq_q6k_wide_a2`, 3.713 s of 15.141 s on a
256-token decode, exp 20) and it is **pipe-bound, not memory-bound** (exp 23: Pipe 16.8%, Send 0.0%; exp 22: 281 GB/s
of a 608 GB/s card, flat from 1 to 20,480 rows). So the 2x is arithmetic removed per weight byte, and it should
survive into the engine.

**Do**: two runs each of the real-model decode, `--layer-split` off and on, `STRATA_MMVQ_PREUNPACK=1/0`. Keep
`q6k_preunpack_bench` / `q5k_preunpack_bench` as the regression measure. **Win looks like**: TG up by roughly the
kernel ratio, output token-identical, an INTEL.md speed-table row and a checked box in section 2 of the archive.

## 2. The QSA top-k at long context (archive D3 + E3/F6) - a shipped capability that degrades on Intel

The register top-k holds `4 * 1024 * TK_PER` cells, and `TK_PER_MAX` is **66 under HIP (270,336 cells) but 33 on
every other build (135,168 cells)**. So the 262K context we ship stays register-resident on AMD and falls to the
wide kernel on the Intel card - every key re-read from memory on each radix pass, one block per query, the rest of
the card idle. The CUDA side replaced that shape with a block cluster: **200 -> 22 us per call at 262K**
(`decode_cluster_parity --bench`; 58 -> 18 at 128K), and ExTV's patch 01 is upstream's PR #603 - the same fix
arrived at independently for NVIDIA's 64-register fit.

`sycl_ext_codeplay_cuda_cluster_group` is CUDA-only, so the Intel routes are the register/GRF knobs
(`sycl_ext_intel_maximum_registers`, `sycl_ext_intel_grf_size`) and `sycl_ext_oneapi_private_alloca` for a
run-time-sized private array. The port uses none of them.

**Do**: `sycl/src/kernels/qsa_select_bench.cpp` exists and is **not yet a CMake target** - build it first. Measure
the top-k at 135K / 200K / 262K cells on the B60, then retry with the GRF/register hints and `private_alloca` before
writing a new kernel. **Only shows on a real prompt past ~135K cells that keeps generating** - exp 32's synthetic
long prompts stop early, so use a real document.

## 3. Conversation parking on a split (archive D5) - the split's largest user-visible cost

Ours refuses: "layer-split parking is not supported" (`sycl/src/core/conversation_state.cpp`). ExTV's patch 07
parks the main card's session and copies back only the stage's layers, only the cells written since the last copy:
switching between two conversations (30K and 15K) went from **20-48 s to 0.4-0.5 s**, with 0.8-1.7 GB snapshots
per 30K tokens. It needs no new kernel - only the state copy the port already does for a stage.

**Do**: two parked conversations alternating on 2x B60, switch timed with the conversation cache on and off.
**Win looks like**: switch time in seconds, cache hit rate unchanged, output token-identical.

## 4. Do the two stages overlap on adjacent prompt chunks? (archive D4) - first answer is free

Decode parity on 2x B60 is explained (the cards alternate a window at a time; exp 34 shows the window is 89% device
time with no host gap to remove). The prompt path has the same one-chunk-at-a-time shape, so adjacent chunks could
overlap exactly as the windows do.

**Do**: read `STRATA_DECODE_TIMING` / `--stats` over a prompt and look for stage 0 idle while stage 1 runs. **A win
looks like**: prompt tok/s toward 1.8-2x of single (445 -> ~800 at 2,185), which is also the 256K prompt's 322.8 s
TTFT. If the stages already overlap, the 1.29x *is* the prompt's own two-stage dependency - record that number and
close the item.

## 5. `sycl_ext_oneapi_kernel_args_restrict` (archive F5) - one flag, broad codegen

A supported extension that puts `__restrict__` on every kernel argument in the translation unit. The port sets
restrict by hand only where it thought about it. No source change per kernel.

**Do**: build with it and A/B `mmvq_bench` at ncols=1..4 first, then the engine decode at 2,185 tokens. A codegen
change this broad wants the bench, not just the engine. Confirm the option reached the compiler by re-dumping the
IGC ISA - IGC ignores an unknown option silently (the exp 22 lesson).

## 6. Peer access for the peer expert tier (archive E1) - not for the layer hand-off

The port has **zero** `ext_oneapi_can_access_peer` / `enable_peer_access` calls. Two candidate payoffs, and they are
not equal:

- The layer-split hand-off is **34 us per window** against a 53.8 ms window - not worth it (exp 03).
- The **peer expert tier** (`--peer-device`, which cannot be combined with `--layer-split`) copies every expert
  batch through pinned host buffers (`sycl/src/core/remote_experts.cpp`, "small enough to copy as one pinned buffer
  per layer"). Real expert bytes cross the host every window, and `p2p_bench` measures today's bounce at only
  **7.7-8.6 GB/s**. This is the case `enable_peer_access` exists for.

**Do**: on 2x B60, `can_access_peer(dev1, access_supported)`, then a 256 KiB-1 MiB device-to-device copy against
the current bounce. Note the Level Zero caveat the reference e2e test names:
`SYCL_UR_L0_RESTRICT_USM_RESIDENCY_TO_P2P`. **Win looks like**: the transfer time dropping with the tier's output
unchanged in the request log.

## 7. The MMVQ cache policy, which is untested rather than refuted

exp 22 swept `-cl-load/store-cache-default` 13 ways and got a null - but the option **never reached the compiler**
on a SYCL/SPIR-V input, so nothing was measured. The port currently sets cache hints by hand. If a kernel profile
ever shows a non-zero `Send` share, this becomes a real lever; until then it is a 30-minute experiment, not a
project.

**Do**: set the flag **in the source** (`-Xs` or kernel attributes), re-dump the ISA to confirm it applied, and
re-run `mmvq_bench` across ncols=1..8. Either it moves or it closes for good - both are useful.

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