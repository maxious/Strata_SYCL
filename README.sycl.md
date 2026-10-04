# Strata on SYCL: how we compare to llama.cpp and what to do next

**Environment first.** Anything that builds or runs the SYCL port needs the oneAPI environment initialized:
`source /opt/intel/oneapi/setvars.sh` (older installs: `setenv.sh`). `sycl/tools/build.sh` does this itself, but a
handwritten build or run must too - without it `icpx`/oneMKL are not on `PATH`/`LD_LIBRARY_PATH` and no ICD is
registered, so the build misses the MKL headers and a built binary fails with `No device of requested type
available`. (This is the first thing that bit us rebuilding `int8_gemm_bench`; see docs/INTEL.md "The engine
itself on Intel".)

This page is the working list for the **SYCL port** of Strata's engine (`sycl/`). Its goal is one thing: make
the port's engine as fast as the CUDA original on Intel cards, using the SYCL backend of
[llama.cpp](https://github.com/ggml-org/llama.cpp) - especially its XMX work - as the reference point. It states
what each implementation does, where ours stands against it (with the measured numbers), and a ranked todo list.

The story so far lives in [docs/INTEL.md](docs/INTEL.md), which is kept up to date run by run. This file is the
**plan**; INTEL.md is the **log**. Read INTEL.md before changing anything in `sycl/`: it records every trap that
ended a run (the persistent-JIT segfault, the driver livelock past VRAM, the Level Zero v1/v2 event semantics,
the 32-lane sub-group requirement, the fp32 divide rounding rule) and the flags that gate each path.

The reference checkout analysed here is `~/ComfyUI/koboldcpp/llama.cpp` (the modern SYCL backend: ESIMD
reordered matvecs, oneDNN fused-XMX SDPA, oneMKL XMX prompt attention, `SYCL_USE_XMX` mmq tiles, and a
runtime `gpu_has_xmx()` probe).

---

## 1. What llama.cpp's SYCL backend does, in the places that matter

These are the reference mechanisms, each with the file that holds it.

| mechanism | where | what it is |
|---|---|---|
| XMX detection at run time | `ggml-sycl/common.cpp` -> `gpu_has_xmx()` (`sycl::aspect::ext_intel_matrix`) | a runtime device probe that gates the XMX-specific kernels |
| XMX GEMM / fused SDPA | `ggml-sycl/fattn-onednn.hpp`, `fattn-mkl.cpp` | flash attention through oneDNN's fused-XMX SDPA; prompt attention through oneMKL XMX GEMM (converts non-F16 KV to F16 first, so every cache type benefits) |
| mmq on `SYCL_USE_XMX` | `ggml-sycl/mmq.cpp` (`MMQ_*_AMPERE` tile sizes switch under `SYCL_USE_XMX`) | smaller XMX-friendly tile shapes for the quantized matmul kernels |
| ESIMD reordered matvec | `ggml-sycl/esimd.hpp`, `dmmv.cpp`, `reorder` | `sycl::ext::intel::esimd` kernels over a **reordered** weight layout (`reorder_qw_*`) that make the decode matvecs run at memory rate |
| reorder feature | `ggml-sycl/ggml-sycl.cpp` (`should_reorder_tensor`, `reorder_q8_1_soa`) | weights are reordered once at load so the decode matvec can stream contiguous bytes; the 2026-04/05 NEWS entry credits it for the Q4_K/Q5_K/Q6_K/Q8_0 gains |
| runtime dispatch ladder | `ggml-sycl/fattn.cpp` (`best_fattn_kernel`) | ONEDNN -> MKL -> VEC -> TILE chosen per-call by head dim, batch size, cache type, GQA ratio; XMX only where it is actually faster |
| arch-aware kernel choice | `ggml-sycl/sycl_hw.hpp` (`gpu_arch::intel_gpu_bmg_g31`), mmq.cpp taps | BMG (Xe2) gets different kernels from ACM (Xe-HPG); the build also predefines sizes per arch |

Two bigger lessons from the llama.cpp log that matter to us:

- **XMX is only for the fat, dense, static-size work.** llama.cpp uses XMX for prompt SDPA (GEMM-shaped) and for
  the mmq prefill tiles, and it gates it behind a runtime probe plus a per-call shape check. It does **not** use
  XMX for the per-token decode matvecs, which are too small and row-varying; those get ESIMD over a reordered
  layout instead. Our port reached the same conclusion independently (see below) - XMX is not the decode lever.
- **Layout work beats XMX product work for memory-bound kernels.** llama.cpp's decode gains came from reordering
  weights so the matvec reads contiguous bytes, not from adding matrix units. On our B70 the dense decode
  kernels stream 100-280 GB/s of a 608 GB/s card; the alignment fix alone took a Q6_K row 150 -> 407 GB/s and
  moved decode 44.9 -> 54 tok/s. Same family of win.

---

## 2. Where Strata's port stands against that (measured)

Port status compressed from INTEL.md and `sycl/`. Reference hardware throughout: **Arc Pro B70 (32 GB)**,
Coder IQ1_M unless noted.

**Already there, matching or beating llama.cpp:**

- Decode **78.2 tok/s** (19-token prompt) vs llama.cpp's **23-25 tok/s** on the same card/model - the port's
  engines and the MTP draft layer are the reason, not XMX.
- Prompt reading up to **1,117 tok/s** at 40K vs llama.cpp ~150-424 tok/s.
- XMX prompt attention is **written and opt-in** (`STRATA_PROMPT_ATTN_XMX=1`, `qsa_prompt_attn_xmx.dp.cpp`) and
  correct (`qsa_prompt_attn_parity`, `kv_hybrid_parity` pass), but - same as llama.cpp's philosophy - it is only
  taken where it wins, which today is nowhere: the port's FP32 prompt attention is faster (~2-3x). 
- oneMKL FP16 GEMM on XMX already feeds the prompt path at **30-60 TFLOP/s** (`xmx_gemm_bench`).
- A fused dequant + XMX GEMM straight from quantized rows exists (`xmx_gemm_iq`) and is correct but 4-5x slower
  than dequant + oneMKL; opt-in, default off.

**Behind / missing, where llama.cpp has something we do not:**

- **No runtime XMX probe.** llama.cpp calls `gpu_has_xmx()`; our port keys its XMX paths off a compile-time/env
  flag (`STRATA_*_XMX=1`). We should detect the matrix aspect once at startup and only advertise XMX paths on
  cards that have it.
- **No oneDNN fused-XMX SDPA.** llama.cpp's prompt attention has a fused oneDNN SDPA path; our prompt attention
  is our own kernel (XMX and FP32 variants). Worth an A/B on the B70.
- **No weight reorder + ESIMD fast path for decode matvecs.** llama.cpp reorders K-quants once (`reorder_qw_*`,
  ESIMD `mac_pair`) so decode matvecs read contiguous bytes. Our decode kernels are hand-tuned FP32 (`s_gemv`,
  `s2_gemv_fast`, `native_mmvq`, wide Q6_K/Q4_K/...), and the *misaligned IQ4_XS / IQ4_NL / Q8_0* loads are still
  listed as open in INTEL.md. llama.cpp's reorder-then-ESIMD is the pattern to steal here.
- **Fewer arch-specific tiles.** llama.cpp bifurcates ACM vs BMG; our port fixed 32-lane sub-groups globally
  (`-fsycl-default-sub-group-size=32`) and keeps SIMD16 variants opt-in (`STRATA_MMVQ_SG`, measured no net gain
  in-engine). Worth one more look now that we know BMG's native width: llama.cpp's decode kernels are written
  around the native 16 and get the register-per-thread benefit.
- **INT8 prompt GEMM on XMX** - **built, measured, parked** (INTEL.md planned item 6; `int8_path_bench`,
  `Gemm::int8`, `iq_quant_*_i8`, opt-in `STRATA_PREFILL_INT8=1`).  The INT8 oneMKL GEMM is genuinely 1.9-2.1x the
  FP16 GEMM at the gate/up shape, but the whole expert comes out 0.42-0.64x: requantizing Q2_0 to INT8 costs 4.6x
  what the FP16 dequant does (0.065 vs 0.014 ms - two passes a row plus a per-row reduce), and the activation
  quantizer and the rescaling epilogue add four more small kernels over the FP16 path's two.  It is also lossy:
  one scale per row cannot carry Q2_0's per-64 block scales, so the output carries 1.0% (gate/up) / 2.4% (down)
  median relative error against an FP16 path that is exact.  Default off.

---

**New since the last revision (2026-10-04; B60 with VTune 2026.4 and the IGC ISA dump).**

- **The decode stall that had no explanation was a completion gate, not the graph.** `verify.cpp`'s per-layer
  wait asked `DPCT_CHECK_ERROR(cs_->ext_oneapi_empty())` for a yes/no; that macro evaluates its expression as a
  *statement and discards its value*, so `q != 1` was always true and the host abandoned a layer ~2 ms after any
  pause while reporting "graph finished". One statement fixed it: decode **17.4 tok/s** (16 tokens) and
  **20.2-20.7** (32 tokens). The capture layer is clean (23 exactly-once captures, 0 breaks, node counts pinned by
  two goldens in `sycl/bench/graph-golden/`) and the stall reproduced with the graphs removed entirely, so the
  graph was never the fault.
- **GPU Hotspots works with hardware counters** once `dev.xe.observation_paranoid=0`. On a 256-token decode: GPU
  time 15.141 s of 29.9 s elapsed, **XVE array stalled/idle 92.3%** of GPU-busy time, top kernels
  `native_mmvq_q6k_wide_a2` 3.713 s, `wait_flag_ge_kernel` (the handshake spin) 2.510 s,
  `native_gu_port<18,8>` 1.426 s. Collection perturbs the handshake, so the ranking is trustworthy and the token
  stream under instrumentation is not.
- **The top kernel is pipe-bound, not memory-bound.** `mmvq_bench` under stall sampling: **Pipe 16.8% of samples
  against Send 0.0%** (Send is the memory stall). Its 384 B spill is measurably free (256 GRF removes it; 26.1 us
  either way - exp 21), it is not bandwidth-limited (281 GB/s at 1 column, of ~608 peak) and not
  parallelism-starved (20480 rows: 205.3 GB/s).
- **The port still links no oneDNN.** oneDNN **3.11.4** is installed (`/opt/intel/oneapi/dnnl/2026.0/`,
  `libdnnl.so.3.11`, a CMake package dir, SYCL in the same library) and `grep dnnl sycl/CMakeLists.txt` is empty.
  See the new library tier in the todo.

---

## 3. Models in the checkout that run on Strata

The port is validated against these GGUF shards, all under the llama.cpp checkout's parent folder
(`~/ComfyUI/koboldcpp/`). Only the Qwen3.8-Flash-Next family is a Strata model; the rest are what the reference
llama.cpp is being run against and are **not** packable by Strata today.

| file | size | Strata? | note |
|---|---|---|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001/02-of-02.gguf` | 51.1 / 26.8 GB | **yes** - IQ3_S size | 64 GB-RAM size; the port has the IQ3_S expert kernels. Not yet load-tested (this is the largest supported size). |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-02.gguf` | 26.8 GB | yes (shard 2) | byte-identical shard-2 pattern to the other sizes. |
| `Qwen3.8-27B-UQ8_/Q3_K_M ...` | 29.3 / 12.4 GB | no | a different, non-Strata model line. Foreign to the port's engines. |
| `gemma-3-12b-it-heretic-Q4_K_M(.gguf)` + `_mmproj` | 6.8 + 0.8 GB | no | non-Strata architecture. |
| `qwen3vl_32b_...-nvfp4.safetensors` | 14.6 GB | no | safetensors, not GGUF; no Strata path. |

**Prefer Q2_0 for the SYCL port.** On the Arc, the Q2_0 quant is the favorable size for the prompt path, and the
choice is measured, not aesthetic. The prompt bottleneck is the weight dequant feeding oneMKL, and Q2_0's dequant
is bandwidth-bound where the i-quants are LUT-bound: `dequant_bench` on the B60 measures **Q2_0 at 419-473 GB/s vs
IQ4_NL 68-75, IQ2_XS 192, IQ2_S 183 GB/s** (gate/up 1280x2560 and down 2560x640). The prompt GEMM itself also
prefers the narrow type: `int8_gemm_bench` measures **INT8 oneMKL GEMM 1.4-2.2x faster than FP16, bit-exact** at the
expert shapes (down 1.4-1.7x, gate/up 1.9-2.2x at T>=96). So the i-quants pay the LUT-bound dequant that made "the
dequant the real prize" (experiments 02/11) while Q2_0's dequant is ~6x cheaper.  **The narrower GEMM does not
follow, though (exp 13):** built and measured on real Q2_0 rows, the INT8 expert path is 0.42-0.64x of the FP16
one - the per-expert requantization to INT8 (0.065 ms vs the FP16 dequant's 0.014) plus the extra activation and
epilogue kernels cost more than the 1.9-2.1x GEMM saves at the per-expert batch a routed expert actually sees -
and it is lossy by 1.0% (gate/up) / 2.4% (down).  Q2_0 stays the pick because its dequant is cheap, not because
an INT8 GEMM pays.  When the picker offers a size, take **Q2_0** for an Arc.

The **IQ3_S Flash-Next shards are the test target**: they are the biggest supported size, exercises the IQ3_S
expert kernels and the largest host-mirror share, and would validate the INT8 XMX prompt path on a heavier model
than the Coder. Concretely: `python3 sycl/setup_intel.py --model IQ3_S` after building, then the 2,184-token
decode target.

---

## 4. Todo list

Priority order based on payoff on the B70. Ranked for *headroom* (where we already measured ourselves below the
card's capability) and *adoptability* (how directly llama.cpp's code maps onto ours).

### P0 - the biggest measured perf levers (decode first, then prompt)

- [~] **L1 decode matvec: weight reorder + ESIMD, ported from `dmmv.cpp`/`esimd.hpp`.** Ported and correct on
  the B60 (`reorder_esimd_bench`, exp 12, committed): the SoA reorder + `q8_0_mac_stripe` ESIMD decode matches
  the AOS path within float rounding (`--selftest`/esimd_kq_parity green). But the honest measurement is NOT
  a win: single column is a wash (0.88-1.14x), and multi-column (the spec/MTP verify window) is 0.25-0.63x
  SLOWER because the ESIMD DMMV is single-column and must be re-launched per column, losing the AOS wide
  kernels' column-vectorization. NOT wired into production (it would regress). Parked as not-a-win; do not
  reopen without new measurements. (The earlier "2.13x" note was a bench bug - it ran the single-column
  kernel once against nc columns.)
  - exp 22/23 (2026-10-04) add the half this item was missing: for the Q6_K wide kernel the target is **not**
    memory. Its stalls are Pipe 16.8% against Send 0.0%, and its best rate is 281 GB/s at 1 column - so a reorder
    has no stall to remove there, and the 44.9 -> 54 tok/s alignment win cited above was a *different*, alignment-
    bound case. Keep this item for the types INTEL.md still lists as misaligned; do not expect it to move Q6_K.

- [ ] **The dense decode MMVQ (Q6_K/Q8_0/Q5_K wide): ~25% of decode GPU time and PIPE-bound.** Exp 20 puts
  `native_mmvq_q6k_wide_a2` at **3.713 s of 15.141 s** GPU time on a 256-token decode; exp 23 puts its stalls at
  **Pipe 16.8% vs Send 0.0%**; exp 22 gives the cost curve the engine lives on:

  | `mmvq_bench` cols (2560x2560, 4 cols = the engine's shape) | 1 | 2 | 4 | 6 | 8 |
  |---|---|---|---|---|---|
  | us / GB/s of weights | 19.1 / 281 | 21.5 / 250 | 26.2 / 206 | 35.9 / 150 | 55.4 / 97 |

  The engine calls it at **ncols = 1** (`layer.cpp:154`, every layer's dense projection, x quantized for one
  token) and at **ncols = T** (`mtp.cpp`, the drafter's window, T up to 6 with `--spec 4`). Avenues, cheapest
  first:
  1. **Decode slimming on the pipe. MEASURED (exp 27): VALIDATED - pre-unpack the 6-bit weights once, ~2x.** The
     Q6_K unpack/gather is the pipe cost: dp4a is only 32 of 985 instr and runs on the same ALU int pipe as the
     bfn/xor/shl/mov unpack (~4x its count), forced by ql/qh's mismatched byte alignment. Timing the no-unpack
     ceiling (signed-byte weights via the Q8_0 `wide32` kernel) against `native_mmvq_q6k` is ~2x at ncols=1
     (19.1 -> 9.3 us, the engine's primary decode) and 1.6-2.1x across the curve, memory not the limit (Send 0%,
     byte path hits 749 GB/s). The lighter formulation is to NOT unpack per token: pre-unpack Q6_K rows to a
     signed-byte `{d'=d*scale, 32 int8}` layout once at weight load, then decode with the existing load+dp4a path.
     Cost: a persistent ~1.25x weight buffer (fp32 scales). WIRED and DEFAULT ON (exp 28/29): the dense decode
     (`native_mmvq(14)`, covering layer.cpp:154 + verify/head/expert/PLE) pre-unpacks each Q6_K tensor once at load
     (`native_q6k_preunpack` -> `Q6U` signed-byte blocks) through the no-bit-unpack kernel. Correctness-gated
     (`q6k_preunpack_parity` ~1e-7 vs the packed path); `STRATA_MMVQ_PREUNPACK=0` opts out and forces the packed
     path. End-to-end decode over a real model still to be confirmed on the runtime/bench box. `q6k_preunpack_bench`
     is the regression measure.
  2. **The `NCOLS>=5` unroll. MEASURED (exp 26): the column loop fixes the blow-up but it is a drafter-window lever,
     not the primary decode.** A runtime column loop (activation fused into the dot, `#pragma unroll 1`) behind
     `STRATA_MMVQ_LOOP=1` (default off, applied for NCOLS>=5) flattens the superlinear tail - ncols 6/7/8 go
     35.9/39.4/55.5 -> 33.5/37.5/40.4 us (1.07x/1.05x/1.37x), the 7->8 jump drops from +41% to +8%, bit-identical
     output - while the ncols=1-4 unroll is untouched (1.00x). ncols=5 is a 0.98x regression, and the engine's
     dominant ncols=1 decode is unchanged, so the primary lever stays avenue 1. Keep opt-in for a wide-MTP drafter.
  3. **XMX / oneMKL** - CLOSED by exp 25 (see the library tier below): a persistent-FP16 dense GEMM loses the
     engine's real decode ncols (0.54-0.86x at ncols=1..4), so XMX stays a prompt-only lever.
  Do not redo: the 384 B spill (exp 21: free) and the cache policy (exp 22: the sweep was null because the option
  never reached the compiler, so it is untested rather than refuted).
- [~] **GEMM-shaped INT8 prompt dequant path** (the open prompt lever; re-scoped from the parked MMQ item, exp
  07 / 09 / 10 / 11). llama.cpp disables SYCL i-quant MMQ and has no SYCL i-quant prompt GEMM, and its SYCL
  dequant kernels are not on any prompt path (exp 10/11). Strata's `iq_dequant_f16` reads only 12-30% of card
  bandwidth for the i-quants (ALU/LUT-bound, not write-bound), so the ~30% dequant phase has headroom. A
  faster Strata-side dequant (wider per-work-item chunks, fewer table lookups per value) feeding the accepted
  dequant+oneMKL FP16 path (571.7 tok/s baseline) is the one open prompt lever. Acceptance: prompt tok/s up
  at 2,184 and 8,000 tokens with output identical, INTEL.md speed-table row.
  **Outcome (exp 13): the INT8 half is closed; the dequant-speed half is the one that remains.** The INT8 expert
  GEMM is implemented end to end (`Gemm::int8`, `iq_quant_gu_i8`/`iq_quant_i8`, `quantize_act_i8`, `scale_rows_i8`,
  wired behind `STRATA_PREFILL_INT8=1`) and measured on real Q2_0 rows: 0.42-0.64x of the FP16 expert path and
  1-2.4% lossy, so the dequant+oneMKL FP16 path stands. What is left of this item is the faster *dequant* feeding
  FP16 - and `dequant_bench` says Q2_0's is already bandwidth-bound (419-473 GB/s), so the headroom is in the
  i-quants, not in the size the port tells people to pick.
  **Outcome (exp 31): the i-quant dequant half is measured-bounded - resist-optimization.** A register-table hoist of
  `d*codebook` (removing the per-value convert/mul) is bit-identical (check=unchanged) but NULL (~75 GB/s unchanged):
  the i-quant dequant is codebook-*select*-bound over the compile-time int8 codebook (the "LUT" is a register
  select-tree, not a memory gather), so neither that nor a wider-chunk / `iq4nl_lut4`-style bucket rewrite offers real
  headroom. Q2_0 (bandwidth-bound at ~472) stays the prompt dequant pick; the i-quant dequant half is parked as
  resist-optimization (confirms exp 04/11).
- [~] **Host-pinned memory for host-to-device (#26789) and dev2dev memcpy by SYCL API (#24476/#26234/#27550
  P2P).** REFUTED-AS-GAIN in research (exp 03 + close read of the blob-fused weights): Strata runs each
  GpuStage in its own SYCL context, where raw peer-USM `memcpy` is a silent no-op (llama.cpp's own warning,
  ggml-sycl.cpp:7189), the hand-off already uses host-pinned `malloc_host` (generate.cpp:4816), it costs only
  ~34 us / 256 KiB (not the decode bottleneck), and the 1.4x decode gain comes from the layer-split compute
  distribution that already ships. Direct dev2dev would need a single-context rebuild that breaks per-stage
  isolation, to recover a ~34 us hand-off. No code; drop.
- [ ] **Dual-GPU: take the pipelined-window and per-card-weights optimisations from
  [Hardin22/Strata-DualGPU](https://github.com/Hardin22/Strata-DualGPU) (docs/DUAL_GPU.md).** A maintained CUDA
  fork of Strata 0.1.38 for two cards that turned a 29 -> 143 tok/s decode on a 5080 + 4060 Ti by making the cards
  overlap instead of alternate. Its changes, ranked by what they would move on the SYCL split (which already
  runs the 1.4x layer-split distribution on 2x B60): 1) **pipelined windows** - run window K+1 on stage 0 while
  stage 1 verifies K (teacher-forced MTP chain, a guarded spec launch, DeltaNet state rollback on reject) - the
  single biggest lever in that fork (x1.17/x1.09/x1.06/x1.30 over serial); 2) **each card keeps only its own
  layers' weights**, freeing VRAM for the expert cache (`--no-trim-stage-weights` against; upstream's #559/#639);
  3) **server orders the cards fastest-last** (the last stage runs the head and the draft), and the auto split
  is priced by the pipelined slower stage, not the sum; 4) **resident-RAM experts across a split** with
  `--adapt-async` (swaps never stop a window); 5) fewer host waits - device-side stage hand-off flags, one
  CUDA graph for the draft chain, commits batched. None is ported to SYCL yet; each needs the SYCL split's own
  parity-first verification (INTEL.md "How to verify any of these"): a real-model 256-token decode on 2x B60,
  accept/exit-time unchanged, before calling it a win. The current SYCL split (exp 03/dual-B60) keeps every
  dense weight on every card and lets the cards alternate; this is the order to close that gap. **Re-reviewed
  2026-10-04 against two more forks, with the first measurement named for each change: P0c below.**
- [x] **Q8_0/Q8_1 wide-load + DMMV ESIMD (#29186), Q2_K/Q5_K reordered ESIMD (#27490/#26376). MEASURED (exp 30): the
  byte pre-unpack is extended to Q5_K; the other K-quants are parked with verdicts.** An analysis run counted the
  real GGUF shards (sycl/bench/reports/p05/usage.md): the Flash-Next dense decode is Q6_K x128 (shipped), Q4_K x47,
  Q5_K x35, IQ4_NL x47, IQ4_XS x42, Q8_0 x1. Implemented: **Q5_K** pre-unpack to `Q5U {dsc= d*sc, mn1= mn*m, code5 qs[32]}`
  (min-offset analog of Q6_K), default-on like Q6_K via the shared registry/`STRATA_MMVQ_PREUNPACK`; `q5k_preunpack_parity`
  ~1e-7 (all shapes PASS) and `q5k_preunpack_bench` measures **1.29x at ncols=1, 1.12-1.29x across**. Parked: **Q4_K**
  (cheap nibble unpack + irreducible min ones-dp4a -> ~1.2x best, not worth it), **Q3_K** (~2x-plausible but 0 tensors
  in the Flash-Next target), **Q2_0 / IQ4_XS** (low value / LUT-bound, exp 04/11), **Q8_0** (already on the byte wide32
  path, the llama.cpp ESIMD reorder measured not-a-win in exp 12).

### P0b - library avenues: XMX, oneMKL, oneDNN (oneDNN linked opt-in; the fuse premise measured no-win, exp 24)

Ranked by how directly the library replaces work the port currently hand-writes.

- [x] **oneDNN: `Dequantize` -> `MatMul` as one graph, for the dequant-bound prompt path. MEASURED (exp 24): the
  fusion premise is blocked; oneDNN stays linked opt-in only.** Verified on this box:
  oneDNN **3.11.4**, `libdnnl.so.3.11`, a CMake package dir, SYCL in the same library, and the port now links it
  behind `STRATA_SYCL_DNNL=1` (default off). Its graph API's op kinds include **`MatMul`, `Dequantize`, `DynamicDequantize`, `Quantize`, `RMSNorm`,
  `GroupNorm`, `LayerNorm`, `SoftMax`, `Reorder`** - and **no SDPA and no grouped-matmul op**, so llama.cpp's
  "fused-XMX SDPA" is a *composed* MatMul/SoftMax/MatMul subgraph that oneDNN's graph compiler fuses, not a
  primitive we can call. The fits, best first:
  1. **`Dequantize` + `MatMul` fused by the graph compiler.** This is the direct attack on P0's dequant half
     (dequant is 27-30% of the prompt) and it is **not** what `xmx_gemm_iq` tried - that was a hand-written
     `joint_matrix` kernel, 4-5x slower. Letting oneDNN fuse the dequant into the XMX GEMM is the thing
     llama.cpp's fattn-onednn actually relies on. Acceptance: prompt tok/s at 2,184 / 8,000 tokens with output
     identical, against the 571.7 tok/s dequant+oneMKL baseline.
  2. **`MatMul` with `Quantize`/`Dequantize` for the parked INT8 prompt path** (P0): the per-row requantization
     that killed exp 13 (0.065 ms against the FP16 dequant's 0.014) is a library reorder/quantize job, not
     necessarily a hand-written kernel.
  3. **`RMSNorm`/`SoftMax` fusion and `Reorder`** for the P1 fusion items and the L1 reorder item respectively.
  Every one of these is subject to the **P2 caveat below**: a oneDNN/MKL call must not fight graph capture, so
  each A/B runs with `STRATA_WARM_GRAPHS` on AND off, and each stays opt-in until it is measured faster.
  **Outcome (exp 24, B60): the `Dequantize`+`MatMul` fusion is structurally blocked** - oneDNN's `Dequantize` reads
  only standard s8/u8 tensors, not ggml block formats (Q2_0/i-quants), and llama.cpp never fuses a block dequant
  into oneDNN (it converts to F16 first). The honest dense-F16 A/B (`onednn_gemm_bench`) is numerically a wash vs
  oneMKL (~2-6e-7 against an fp64 ref, exact on an integer grid) and faster only at large batch (gu T>=64, up to
  1.76x), while the engine's real call is per-expert routed `ne` (small, tail-heavy) where oneDNN loses (gu
  T=16-32: 0.73-0.78x). OneMKL stays default; oneDNN is linked opt-in (`STRATA_SYCL_DNNL=1`) with `onednn_probe`
  (reports `jit:gemm:any` on the B60) parked for a future large-batch dense path. Full numbers plus the
  column-major-layout trap: docs/sycl-experiments/24.
- [x] **oneMKL: the FP16 XMX GEMM is the prompt path already; the untested shape is the decode batch. MEASURED (exp
  25): the decode shape is a no-win.** `xmx_gemm_bench` measures 30-60 TFLOP/s at prompt shapes and `int8_gemm_bench`
  measures INT8 at 1.4-2.2x FP16. The one missing measurement - a oneMKL GEMM at **M=1..6** against `native_mmvq` -
  is now made (`decode_xmx_gemm_bench`): with a persistent FP16 copy it is *slower at every ncols the engine runs*
  (ncols=1..4: 39.9/38.0/34.4 us vs native 21.4/23.9/29.7, i.e. 0.54-0.86x) and only crosses over at the fat drafter
  window (ncols=6: 1.33x, ncols=8: 2.39x), which costs 2x dense VRAM plus the ~12 us materialization. The launch/
  backend overhead of a single-column XMX GEMM is exactly what P0b #2's traffic arithmetic predicted.
- [x] **XMX for the dense decode matvec: the untried version is dequant-then-GEMM. MEASURED (exp 25): no-win at the
  engine's decode ncols.** Every refuted XMX result here is *fused quantized* (`xmx_gemm_iq` 4-5x slower; the expert
  dots 1.4-3x; `qsa_prompt_attn_xmx` ~3x), while the one that wins is FP16 dense GEMM through oneMKL (30-60
  TFLOP/s). The one candidate exp 23 motivated - "dequantize once into a persistent FP16 copy and GEMM on XMX" -
  closes poorly: at the engine's real decode call (ncols=1 every layer, ncols=T up to 6 in the drafter) the GEMM is
  1.2-1.9x slower (ncols=1..4), and its fat-window cross-over (ncols>=6) does not repay the 2x dense-VRAM residency
  (`decode_xmx_gemm_bench`, exp 25). The matrix units were idle because XMX's backend cost only repays at
  GEMM-shaped batch, which decode is not. Parked with the other fused-XMX losers.

### P0c - dual-GPU: the tests behind the P0 fork item (reviewed 2026-10-04)

Three forks were read against our split: [anon761/strata](https://github.com/anon761/strata) (a sync fork on a
0.1.32 base whose only dual-GPU content is the `SECOND_GPU.md` doc we already ship - nothing to take),
[ExTV/strata-5090-4070](https://github.com/ExTV/strata-5090-4070) (a 5090 + 4070 Ti SUPER in a chipset x1 slot
on Hardin22's fork plus ten patches, every change measured) and
[Hardin22/Strata-DualGPU](https://github.com/Hardin22/Strata-DualGPU) (`docs/DUAL_GPU.md`, the source of most of
it). The target is our own gap: on Q2_0 the two B60s **decode at parity** (32.0 -> 33.1 tok/s) and prefill 1.29x
(445 -> 576 tok/s) because the cards alternate a window at a time (exp 03 + the 2026-10-04 matrix). Each item is a
theory about that gap with its first measurement named. Nothing here is called a win before the SYCL split's own
parity-first verification (INTEL.md "How to verify any of these").

- [ ] **D1 - measure the window before porting anything: is there a host gap between the two stages?** The fork
  launches both stage graphs at once and lets a **flag in mapped pinned memory** order them (`STRATA_XSTAGE=0`
  reverts to host-synchronizing each stage); commits are queued per stream and waited once per request
  (`STRATA_COMMIT_ASYNC=0` reverts); the draft round is one graph. Neither switch is in our tree, but
  `STRATA_DECODE_TIMING=1` **is** (in `src/` and `sycl/`) and prints the per-window host and per-stage GPU timings,
  and `STRATA_COMMIT_SYNC` (`src/core/verify.cpp`) is the existing async-commit gate. **Test**: Q2_0 at 2,185
  tokens, 256 greedy, 2x B60, `STRATA_DECODE_TIMING=1`; read the timeline for a host wait on stage 0 before stage
  1 launches and for the per-stage GPU busy time. **A win looks like**: the gap leaves the timeline and TG rises.
  If the timeline is already back to back, the parity is not a host-wait problem and this closes for the cost of
  one run.
- [ ] **D2 - pipelined windows, from behind D1** (P0's item 1: the fork's x1.17/x1.09/x1.06/x1.30, default on for
  exactly two GPUs). Our alternating window is the shape it attacks, and our split is balanced (K=22 of 48), so
  the x1.17 *code* case is the comparable one - not the x1.30 lookup case our synthetic prompts do not have. One
  design point to carry over: **the auto split must be priced by the pipelined slower stage, not the sum of the
  stages** (the fork's stock search put 2 of 48 layers on the small card and decoded at 29 tok/s; repriced it
  picks K=19 against 158 for a hand-tuned K=20). Our search picked K=4 under the exp-03 bug and we hand-fixed
  K=22/24, so the search has an item of its own here. The fork's parity bar is ours too: `STRATA_IQ_MT_MIN=1
  --adapt-every 0` made serial and pipelined write the same text in 45 of 45 requests. **Test**: Q2_0 at 2,185 with
  `--spec 4`, serial vs pipelined: TG, accept rate, rollback count. **Expected**: ~1.1x on decode (the stages stay
  dependent), not 2x.
- [ ] **D3 - at 262,144 context our QSA top-k is past the register kernel's fit on the B60** (a decode lever none
  of the other items touch). The register kernel holds `4 * 1024 * TK_PER` cells and `TK_PER_MAX` is **66 under HIP
  (270,336 cells) but 33 on every other build (135,168 cells)**, so the 262K context we ship stays
  register-resident on AMD and falls to the wide kernel - every key re-read from memory on each radix pass, one
  block per query, the rest of the card idle - on our Intel card. The CUDA side replaced that shape with a block
  cluster: **200 -> 22 us per call at 262K** (`decode_cluster_parity --bench`; 58 -> 18 at 128K), and ExTV's patch
  01 is upstream's PR #603 - the same fix arrived at independently for NVIDIA's 64-register fit. **Test**:
  `sycl/src/kernels/qsa_select_bench.cpp` already exists and is not yet a CMake target - build it, measure the
  top-k at 135K / 200K / 262K cells on the B60, and try a higher `TK_PER_MAX` for the Intel branch (Xe2's register
  file is what lets the HIP branch hold 66) before writing a new kernel. **Only shows on a real prompt past ~135K
  cells that keeps generating** - exp 32's synthetic long prompts stop, so use a real document.
- [ ] **D4 - prefill is 1.29x, not 2x: do the two stages overlap on adjacent prompt chunks?** Decode parity has
  the alternating-window explanation; the prompt path has the same shape one chunk at a time (stage 0's layers,
  then stage 1's), so adjacent chunks could overlap exactly as D2's windows do. **Test**: read
  `STRATA_DECODE_TIMING` / `--stats` over a prompt for a stage 0 idle while stage 1 runs. **A win looks like**:
  prompt tok/s toward 1.8-2x of single (445 -> ~800 at 2,185), which is also the 256K prompt's 322.8 s TTFT. If the
  stages already overlap, the 1.29x is the prompt's own two-stage dependency and this item closes with that number
  recorded.
- [ ] **D5 - conversation parking is refused with a split** (not decode, but the split's largest user-visible
  cost once a chat holds two long conversations). ExTV's patch 07 parks the main card's session and copies the
  stage's layers back on restore, only the cells written since the last copy: switching between two conversations
  (30K and 15K) went from **20-48 s to 0.4-0.5 s**, with 0.8-1.7 GB snapshots per 30K tokens. Ours returns
  "layer-split parking is not supported" (`sycl/src/core/conversation_state.cpp`). **Test**: two parked
  conversations alternating on 2x B60, switch timed with the conversation cache on and off; it needs no new
  kernel, only the state copy the port already does for a stage.
- [ ] **D6 - each card keeps only its own layers' dense weights** (P0's item 2; upstream #559/#639). **On our
  Q2_0 the dual decode win is nil** - its cache already hits 100.0% over 24,492 slots - so the slot gain is for the
  **single** card (12,996 slots, 95.5% hit) and for the prompt, which is where the fork measured it (IQ2_XS code
  162 -> 172, 32K prompt 1,899 -> 2,201; IQ3_XXS prompt 1,006 -> 1,978 once the last ~2,400 SSD-read experts
  entered VRAM). **Test**: measure the per-card dense footprint and the slots gained on 2x B60, then re-run the
  matrix's hit rate and PP at 2,185 and 32K.
- **Checked, not ours to port.** `--pread-expert-blobs` (ExTV patch 08): `sycl/src/core/gguf_expert_source.cpp`
  already reads expert slices with a vectored `pread` - one call for many slices - so the page-fault storm it fixes
  is not ours. `--resident-experts` across a split (fork item 1): already ours; the dual run's log shows the RAM
  copy at 13,312 slots / 17.1 GiB. P-core/E-core pool policy (fork item 7): no E-cores on a 5700X3D, and the
  default is already "every physical core minus the host's" (7 of 8), so a `--pool-workers` sweep is a constant,
  not a lever. Programmatic dependent launch (fork item 5, sm_90 `STRATA_DF_PDL`) has no SYCL equivalent. Card
  order fastest-last (fork item 2) has no counterpart on two identical cards - what matters is that the last stage
  runs the head and the draft, which our split already does (`mtp.bind(last_st ...)`, `generate.cpp`). The fork's
  remaining dense-kernel items (side streams, batched post-ops, Q4_0 KV) belong with the kernel items above, not
  here.

### P1 - attention A/Bs and fusion (mostly gated on P0, or low measured headroom)

- [ ] **oneDNN fused-XMX SDPA A/B for prompt attention.** llama.cpp has it (`fattn-onednn.hpp`); we have FP32 and
  our own XMX kernel, and our XMX is ~2-3x slower than FP32. Note from the P0b reconnaissance: this oneDNN has
  **no SDPA op kind**, so the A/B means composing MatMul/SoftMax/MatMul in a graph and letting its compiler fuse
  it (which is what llama.cpp's header does) - budget accordingly. Compare against the FP32 fallback at
  8K/40K/128K, with `STRATA_WARM_GRAPHS` on AND off. If it wins, keep it behind the same probe-and-verify gate.
- [ ] **prompt-attention GEMM A/B (the #25025/#25222 lever, re-scoped).** llama.cpp runs prompt attention as
  oneMKL XMX GEMM; Strata's prompt attention is sparse/gather-bound QSA (each query selects its own cells), so a
  direct oneMKL-GEMM flash-attention copy does not map (NOTE: the "~350+ tok/s prefill" branch note is llama.cpp's
  number; Strata's own prompt already reads ~570-1,100 tok/s on the dequant+oneMKL FP16 path). The GEMM part of
  the prompt IS the GEMM-shaped INT8 dequant target (P0); revisit attention only after that lands.
- [ ] **MKL-FA softmax load coalescing (#28918) and large-register-file FA vec kernels (#29062).** llama.cpp's
  fattn-mkl.cpp coalesced the softmax loads (one work-item per row was the bottleneck) and fattn-vec.hpp grew a
  large-register path for D=512 heads. Strata's prompt attention is sparse QSA (not dense FA), so this applies
  only to whatever GEMM attention the P0 INT8 dequant path enables; revisit after it lands.
- [ ] **Fuse mul_mat(gate)+mul_mat(up)+GLU for the dense FFN (#26779), rms_norm+mul+add residual chains
  (#27610), fused UNARY(silu/...)+MUL (#26411).** llama.cpp keeps shaving cross-kernel round-trips; each matches a
  stride in our dense projections (GR down/up, norms). INTEL.md's graph-node item already prices ~5 us/node.
  Low priority (exp 05: kernel-bound).

### P2 - structural / hygiene

- [x] **Read-side blockage: oneDNN/MKL calls must not fight SYCL graph capture.** llama.cpp's fattn.cpp notes MKL
  GEMM is incompatible with graph capture replay. Our prompt path captures window graphs; a oneDNN/MKL SDPA
  experiment must check it does not break `STRATA_WARM_GRAPHS`. Documented in INTEL.md (the read-side note under
  the graph-node item): any MKL SDPA A/B must run with `STRATA_WARM_GRAPHS` on AND off so a victory is not an
  artifact of the SDPA node escaping captured-graph replay.
- [ ] **Model matrix: load-test the IQ3_S Flash-Next shards** (the two shards above) end to end through
  `sycl/setup_intel.py --model IQ3_S`, and add the per-shard checksum / expected sizes to the parity fixtures so
  `IQ3_S` stops being "not yet load-tested" in this table. Acceptance: an IQ3_S row in INTEL.md's measured table.
- [ ] **Match llama.cpp's automated XMX CI gate.** llama.cpp gates XMX paths behind tests (e.g. `topk-moe.cpp`,
  flash-attn self-tests). Add a `xmx_gemm_bench` + parity step to `sycl/tools/build.sh` so an XMX change that
  regresses the quantized GEMM fails the build like any other kernel.

### Experiment log (2026-10-03, B60) - what each measured and decided

Reports live in `docs/sycl-experiments/`; these are the read-outs that set the priorities above.

| # | experiment | result | decision |
|---|---|---|---|
| 00 | baseline build + 22/23 parity + benches | dual-B60 (G21) toolchain works | reference set |
| 01 | XMX runtime probe (`gpu_has_xmx`) | ported, parity green | IMPLEMENTED (commit) |
| 02 | INT8 vs FP16 oneMKL GEMM | 1.5-2.2x at full batch, bit-exact, dequant-bound | GEMM-only parked; dequant is the real prize |
| 03 | P2P + layer split | 7.7-8.6 GB/s D2D, `stage_room` fix, split 35.5 vs 25.4 tok/s | IMPLEMENTED (commit like `stage_room` fix) |
| 04 | misaligned decode loads | IQ4_XS aligned-load 2-9% slower | parked (LUT-bound) |
| 05 | decode-round node fusion | 52 ms/2,541 nodes, kernel-bound | parked |
| 06 | fused-GR gated variants | default 14-84% faster | confirmed; no change |
| 07 | prompt bottleneck | dequant+GEMM = 58.6%; 571 tok/s @1,280 tok | **MMQ = the target** |
| 08 | CUDA MMQ wiring | fails on `cuda_runtime.h` | wrong path; SYCL i-quant kernels are decode matvecs only |
| 09 | prompt-batched i-quant MMQ (mmvq port) | parity-exact, builds, but ~6x SLOWER at prefill (73 vs 571.7 tok/s) | parked opt-in; FP16 dequant+oneMKL stays default |
| 10 | GEMM-shaped INT8 path research | llama.cpp has NO SYCL i-quant GEMM (MMQ off; reorder-MMVQ = Q1_0..Q6_K only); the GEMM path is CUDA-only (mmq-load-tiles.cuh), tensor-core-tuned; B60 dp4a = 0.24-0.37x oneMKL FP16 | frontier accepted: dequant+oneMKL FP16 (571.7 tok/s) is it; no code |
| 11 | dequant-phase A/B | Strata iq_dequant_f16 = 12-60% of card bw (ALU/LUT-bound, headroom exists) but llama.cpp's SYCL dequant is NOT on any prompt path (zero callers) | moot for llama.cpp port; faster dequant is a Strata-side kernel project |
| [13](sycl-experiments/13-int8-expert-path.md) | INT8 prompt expert GEMM, built and measured (`int8_path_bench`, B60, one real Q2_0 expert) | the int8 GEMM itself is 1.9-2.1x (gate/up) / 1.0-1.7x (down), but the whole expert path is **0.42-0.64x**: requantizing Q2_0 to int8 costs 0.065 ms against the FP16 dequant's 0.014, and act-quant + epilogue add 4 kernels; output error 1.0% gate/up, 2.4% down (FP16 is exact).  In the engine (5-token prompt, warm, `strata --prefill 128`, two runs each) the int8 path takes 501.7 / 523.3 ms against FP16's 397.0 / 395.8 | PARKED: opt-in `STRATA_PREFILL_INT8=1`, default off, FP16 dequant+oneMKL stays |
| [14](sycl-experiments/14-dpct-sweep.md) | DPCT migration-marker sweep (1,151 markers, 33 codes): the GDN cp.async lead (DPCT1053) and the barrier audit (DPCT1118) | GDN: **`gdn_rec_kh_kernel` is 6.0-7.3x SLOWER than the default** on the B60 (`gdn_rec_bench`), and the phase it targets is 0.6% of prompt time (`gdn recurrence` 127 of 22,271 ms; `gemm down` alone is 50.2%) - a perfect 1.4x would be 0.16% end to end.  Barriers: **0 of 187 group calls sit under a thread-dependent guard**, so all 75 DPCT1118 markers are conservative false positives (barriers are siblings of the `if`s; early returns test `get_group`, not the local id; `continue` precedes the *next* iteration's barrier).  ctest 25/27 with both failures documented pre-existing | PARKED (no code): do not port cp.async to the GDN key-head kernel; DPCT1118 retired as an audit. Next from the sweep: DPCT1110 (register pressure), DPCT1098 (`__ldg`) |
| [15](sycl-experiments/15-dpct-sweep-2.md) | DPCT sweep round 2: register pressure (DPCT1110, 46 markers) and `__ldg` (DPCT1098, 42) | **DPCT1110 does not predict spills**: with the IGC ISA dump (`//.spill size`, all kernels `numGRF=128`), the flagged `native_mmvq_multi_kernel`, `s_gemv` and `sampler_greedy/one_block/split_merge` kernels spill **0 B**, while the **unflagged** `native_mmvq_q6k_wide` spills 384 B - so "fix the 46 flagged kernels" aims at the wrong list.  The one flagged spiller is `sampler_split_part_kernel` at **2176 B**, and it runs once per *token* (~1 MB of logits to scan ≈ 5 us against a 12.8 ms token).  `__ldg` has no Xe counterpart: every one of the 42 sites is a deref of an already-`const __restrict__` chain, and Xe's L1 is unified | PARKED (no code) for both; the sampler spill is a bounded follow-up if a sampler bench is ever wanted |
| [17](sycl-experiments/17-parity-fixes.md) | the two failing parity tests: `s2_expert_grouped_parity` and `ple_parity` | **Neither was a kernel bug.** `s2_expert_grouped_parity`: the grouped entry point's two kernels are the *same expression* (`chunk_dot`'s `dw*dx*(s-hx)` vs the transposed `dw*dx*(chunk_s-dh.y())`) compiled differently, so they differ by float order - **measured worst 1.5e-07 of the row scale**, and the fp16 packed output is bitwise identical; the old assertion demanded bit-identity, which the kernel's own header says is impossible.  Now: per-hit cases stay byte-for-byte, the grouped cases validate **both** runs against the double-precision reference (worst 9.7e-08, 0 rows outside tolerance each) and bound old-vs-new at 1e-5.  `ple_parity`: the port registered it BARE through the generic loop (no `--in`/`--out`, working directory the build dir), so it looked for `bench/micro/ple_{in,out}.bin` under `sycl/build-b60/` and could never find the capture on any machine; upstream passes the paths and pins the source tree | FIXED: ctest **100%, 26 tests** (was 25/27); `ple_parity` is registered only where its gitignored CUDA-side capture exists, and CMake says so at configure time - never skipped-as-passed |
| [18](sycl-experiments/18-graph-capture-audit.md) | capture audit + the "is the graph at fault" A/B | 23 exactly-once captures, **0 breaks**, in every arm; the stall reproduced with the graphs removed | audit kept (`graph_audit_test`, node goldens); the stall was the gate, fixed in 19 |
| [19](sycl-experiments/19-doorbell-completion-gate.md) | the doorbell completion gate | `DPCT_CHECK_ERROR` **discards its expression's value**, so `q != 1` was always true and the host abandoned each layer after 2 ms while reporting "graph finished" | **FIXED**: decode 17.4 tok/s (16 tokens) / 20.2-20.7 (32), coherent output, ctest 27/27 |
| [20](sycl-experiments/20-followups-eager-and-vtune.md) | VTune GPU Hotspots + where the eager divergence lives | counters land (XVE stalled/idle 92.3%); top kernels 3.713 / 2.510 / 1.426 s; collection perturbs the handshake and degenerates the token stream; eager differs from the graph path from token 1, and is **identical** to it once the handshake is off | profile = kernel ranking only; the divergence localizes to the per-layer handshake |
| [21](sycl-experiments/21-q6k-mmvq-isa-dump.md) | the Q6_K MMVQ ISA dump | the 384 B spill is 4 of 985 instructions with **no loop**; `IGC_ExtraOCLOptions=-ze-opt-large-register-file` removes it entirely (spill 0, 962 instructions) and 26.1 us is unchanged, 4 runs of 4 | the spill is free; do not chase it |
| [22](sycl-experiments/22-q6k-mmvq-shape-curves.md) | Q6_K MMVQ shape curves | cols 1/2/4/6/8 = 19.1/21.5/26.2/35.9/55.4 us (281 -> 97 GB/s); rows 2560 -> 20480 = 205.7 -> 205.3 GB/s; 256 GRF identical at every point; a `-cl-load-cache-default` sweep was **null** (the option never reached the compiler) | not bandwidth- or parallelism-bound; the cache policy is untested, not refuted |
| [23](sycl-experiments/23-q6k-mmvq-stall-reasons.md) | Q6_K MMVQ stall reasons | 8,061 instances, 27 us average (matches the bench); stalls **Pipe 16.8%, Send 0.0%**, Dist or Acc 2.1% | **PIPE-bound, not memory-bound** - the lever is arithmetic per weight byte |
| [24](sycl-experiments/24-onednn-dense-f16-ab.md) | oneDNN dense-F16 A/B (reframed P0b #1) | fusion blocked: oneDNN's `Dequantize` reads s8/u8, not ggml blocks; dense F16 is numerics-identical to oneMKL, faster only at large batch (gu T>=64, 1.76x), slower where the engine's per-expert routed `ne` lives (T=16-32, 0.73-0.78x) | oneMKL default; oneDNN linked opt-in `STRATA_SYCL_DNNL=1` with `onednn_probe`+`onednn_gemm_bench` |
| [25](sycl-experiments/25-decode-xmx-gemm.md) | oneMKL dense-FP16 GEMM at decode batch vs native_mmvq (P0b #2+#3) | GEMM loses the engine's real decode ncols (ncols=1..4: 0.54-0.86x) and only crosses at the fat drafter window (ncols=6: 1.33x, ncols=8: 2.39x); persistent-FP16 costs 2x dense VRAM + ~12 us materialization | dequant-then-XMX-GEMM is a no-win for decode; `native_mmvq` stays; P0b #2+#3 closed, P0 #2 XMX avenue closed |
| [26](sycl-experiments/26-q6k-ncols-loop.md) | Q6_K wide MMVQ NCOLS unroll vs runtime column loop (P0 #2 avenue 2) | loop flattens the register-blow-up tail: ncols 6/7/8 35.9/39.4/55.5 -> 33.5/37.5/40.4 us (1.07x/1.05x/1.37x, 7->8 jump +41%->+8%, bit-identical); ncols 1-4 unchanged, ncols 5 is 0.98x | validated but opt-in (`STRATA_MMVQ_LOOP=1`, NCOLS>=5); a drafter-window lever, not the primary decode - avenue 1 stays the P0 #2 lever |
| [27](sycl-experiments/27-q6k-preunpack-decode.md) | Q6_K decode pre-unpack ceiling (P0 #2 avenue 1) | the 6-bit unpack/gather IS the pipe cost (dp4a 32 of 985 instr, same ALU pipe); a signed-byte no-unpack decode (Q8_0 wide32) is 2.06x at ncols=1 (19.1->9.3 us) and 1.6-2.1x across the curve, stable | VALIDATED: pre-unpack Q6_K once to signed bytes, decode with load+dp4a; wiring is the next step (persistent ~1.30x buffer) |
| [28](sycl-experiments/28-q6k-preunpack-engine.md) | pre-unpacked Q6_K decode wired (P0 #2 avenue 1) | Q6_K pre-unpacked to `Q6U` signed-byte blocks once + routes `native_mmvq(14)` (layer.cpp:154, verify/head/expert/PLE) through the no-bit-unpack kernel; parity ~1e-7 vs packed (q6k_preunpack_parity all shapes PASS); deployed path 2.05x at ncols=1, 1.58-2.10x across | WIRED; now DEFAULT ON (`STRATA_MMVQ_PREUNPACK=0` opts out to the packed path); persistent ~1.25x weight buffer; end-to-end decode to confirm on the runtime box |
| [30](sycl-experiments/30-q5k-preunpack.md) | pre-unpacked Q5_K decode, default-on (P0 #5) | Q5_K pre-unpacked to `Q5U {dsc,mn1,code5 qs[32]}` once + routes `native_mmvq(13)` through the no-bit-unpack decode (min-term ones-dp4a; the float-sum ds[1] is not usable); parity ~1e-7 (q5k_preunpack_parity all shapes PASS); 1.29x at ncols=1, 1.12-1.29x across | WIRED DEFAULT ON; Q4_K/Q3_K/Q2_0/IQ4_XS parked with verdicts (Q8_0 already byte); analysis reports in sycl/bench/reports/p05/ |
| [31](sycl-experiments/31-iq4nl-dequant-null.md) | i-quant dequant speed half (P0 #3, measured null) | baseline IQ4_NL 75 / IQ4_XS 76 / IQ3_XXS 87 GB/s (~4-6x below Q2_0's 472); a register-table hoist of `d*codebook` is bit-identical (checksum unchanged) but NULL (~75 GB/s) - the i-quant dequant is codebook-select-bound (constexpr int8 codebook = register select-tree, not a memory gather) | P0 #3's dequant half parked as resist-optimization; Q2_0 stays the prompt pick |


### Mined from the llama.cpp ggml-sycl git history (2026-10-03)

`git log --oneline -- ggml/src/ggml-sycl` of `~/ComfyUI/koboldcpp/llama.cpp` (master, head containing
#28985/#29062/#29186). The P0/P1 items above were chosen from these; the ones not yet actionable are logged here so none is lost:

| llama.cpp change | file | note |
|---|---|---|
| oneMKL GEMM flash attention (#25025) / oneDNN XMX SDPA (#25222), non-F16 KV (#25874) | fattn-mkl.cpp, fattn-onednn.* | the `~350+ tok/s` XMX-prefill branch's core; P1 item |
| coalesce MKL-FA softmax loads (#28918) | fattn-mkl.cpp | P1 item |
| large register file, D=512 FA vec (#29062) | fattn-vec.hpp | P1 item |
| probe oneDNN once at device init; drop slow oneDNN ref matmul/fattn (#28985) | ggml-sycl.cpp, common.hpp | dispatch hygiene; pairs with our gpu_has_xmx probe |
| Q8_0 wide-load + DMMV ESIMD (#29186); Q2_K reorder ESIMD (#27490); Q5_K ESIMD (#26376) | dmmv.cpp, mmvq.cpp, vecdotq.hpp | P1 decode item |
| P2P dev2dev memcpy (#24476, #26234, #27550 "enhance api to support peer-to-peer copy") | (memcpy path) | matches Experiment 03's B60 measurement; P1 item |
| reduce tensor allreduce sync with pinned host buffers (#29604) | | multi-GPU hand-off sync reduction |
| MTP: remove padding and multiple D2D copies (#24086); deterministic D2D (#26145) | | our MTP draft layer's copies |
| fused MoE / fused top-k (#25217, #25221, fused top-k MoE) | moe, topk-moe | router + top-10 item |
| split long rows in TOP_K vs one work-group/row (#27847); radix select for top_k (#28670) | topk-radix.cpp, topk-moe | our router_top10 kernel |
| fuse rms_norm+mul+add (#27610); fused UNARY(silu...)+MUL (#26411); fuse gate+up+GLU (#26779) | norm.cpp, element_wise.cpp | P1 fusion item |
| Q4_K multi-column MMVQ redundant-work cut (#27062); native subgroup size for K-quant DMMV (#21700) | mmvq.cpp, dmmv.cpp | decode tuning |
| bind F16 KV in place for oneDNN SDPA (#27468); TILE for quantized KV decode on BMG (#26689) | fattn-common.hpp | KV + attention geometry |
| host pinned mem for host-to-device (#26789); count free GPU mem (#27968); alloc >= 19.3 GB fix (#28953) | | memory-path items |
| MEMTRACE by allocation site (#27631) | memtrace.* | wiring for xmx_gemm_bench/parity CI |

Not relevant to Strata's engine (noted so they are not re-researched): conv/ssm/image op ports (a separate
backend), bf16/fp16 op type widening, clean-dup/revert churn, CI/build fixes.

### Not on the list (tried and parked - do not reopen without new evidence)

- Fused **dequant + XMX GEMM** (`xmx_gemm_iq`): 4-5x slower than dequant + oneMKL. INTEL.md: parked, opt-in.
- **Dequant-then-XMX dense GEMM for the decode matvec** (exp 25, `decode_xmx_gemm_bench`): with a persistent-FP16
  weight copy, oneMKL's dense GEMM is *slower* than `native_mmvq` at the engine's real decode ncols (0.54-0.86x at
  ncols=1..4) and only crosses over at MTP/fat window ncols>=6, which costs 2x dense VRAM + ~12 us materialization.
  The matrix units are idle on decode because XMX's backend cost only repays at GEMM-shaped batch. INTEL.md: the
  decode lever is P0 #2's pipe-arithmetic slimming and the NCOLS-unroll-to-loop, not a GEMM.

- **XMX for per-token decode (expert dots, INT8 DPAS)**: three `joint_matrix` versions were 1.4x and 2-3x slower
  than the dp4a path, one hung the GPU. INTEL.md: parked. llama.cpp's own design (ESIMD reorder for decode,
  XMX only for fat GEMM/SDPA) agrees; only revisit if a weight reorder gives decode a big enough target.
- **Grouped (batch-gather) prompt attention on XMX**: measured the union of 8 positions is 3-5x one position's
  cells with only 12% shared; arithmetic cost exceeds the gather saved. INTEL.md: not built.
- **The GDN key-head kernel / its `cp.async` pipeline** (`gdn_rec_kh_kernel`, DPCT1053, exp 14): the `cp.async`
  staging is disabled on SYCL, but the kernel itself is 6.0-7.3x slower than the default column kernel on the
  B60 (26.5 vs 3.6 ms at T=2048, `gdn_rec_bench`), and the phase it would speed up is 0.6% of prompt time. Do
  not port the staging to this kernel; a different GDN kernel would be a new measurement, not a reopen.
- **The Q6_K wide MMVQ's 384 B spill** (exp 21): 4 instructions of 985, once per work-item, and forcing 256 GRF
  removes every spilled byte for **26.1 us either way**. Measured free; do not reopen.
- **Cache-policy tuning of the MMVQ through `IGC_ExtraOCLOptions`** (exp 22): the 13-run sweep was null because
  `-cl-load/store-cache-default` never reached the compiler on a SYCL/SPIR-V input (verified by re-dumping). It is
  *untested*, not refuted - and if retried it must be set in the source (`-Xs`/attributes at build), because IGC
  ignores an unknown option silently. Always confirm an option applied by re-dumping.

---

## 5. How to verify any of these

Every P0/P1 item has the same checklist, from INTEL.md's "Trap worth knowing":

1. **Parity first.** The relevant `*_parity` test(s) must pass byte-for-byte or within the documented tolerance.
   It is not enough that the output "looks right" - compare tokens, and for native packs never compare only
   timings across `--prefill-until`.
2. **Window the clocks.** Warm the GPU first; a cold 5 ms run measures the ramp. Use `mmvq_bench`,
   `native_expert_parity NATIVE_BENCH=1`, `q6k_align_bench`, `xmx_gemm_bench` for kernel-level numbers.
3. **Log the change in INTEL.md** with the two-run-both measurement and an "output identical / near-tie" note, in
   the style of the existing speed tables. No number in this file without the hardware it was measured on.
4. **Keep it opt-in with a flag when it is not strictly faster** (`STRATA_*=0` to revert, like every previous
   round), and keep the default on the measured-fastest path.

5. **Capture changes go through the audit.** Run `graph_audit_test` (it proves all six of its assertions fire,
   including that `DPCT_CHECK_ERROR` discards its expression's value) and check the node goldens
   (`STRATA_GRAPH_GOLDEN=sycl/bench/graph-golden/<config>.txt`, `STRATA_GRAPH_GOLDEN_WRITE` to refresh): a node
   that stops being recorded shows up as `GOLDEN MISMATCH` / `MISSING` instead of silence. `STRATA_GRAPH_AUDIT=1`
   prints a line per capture, and `STRATA_GRAPH_STRICT=1` makes a re-capture or a moved count fatal.
6. **Before optimizing a kernel, get its stall reasons.** `vtune -collect gpu-hotspots -knob
   gpu-profiling-mode=source-analysis -knob source-analysis=stall-sampling -knob
   computing-tasks-of-interest="*<kernel>*"` then `-report summary -format csv` (run the report with `sudo`: the
   result directory is root-owned). Profile a **bench**, not the engine - the engine's handshake degenerates under
   instrumentation. `Pipe` vs `Send` is the whole question for a matvec: 23 needed one collection to stop the
   port from optimizing a memory problem that was not there.

The parity tests that still need machine-local inputs (`iq_parity`, whose fixtures `tools/iq_fixture.py` now
generates into the build tree, and `ple_parity`, whose ggml capture is a gitignored CUDA-side artifact) and the
XMX prompt-attention test were gated in INTEL.md; keep that list accurate as the IQ3_S model gets added.
`ple_parity` must never be re-added to the bare parity loop - it needs its `--in`/`--out` and the tree as its
working directory. As of 2026-10-04 `ctest` is **27 tests, 100%**.