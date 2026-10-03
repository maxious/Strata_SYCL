# Strata on SYCL: how we compare to llama.cpp and what to do next

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
- **INT8 prompt GEMM on XMX** (INTEL.md planned item 6, not built): llama.cpp's MKL path converts KV to F16
  before XMX GEMM; the comparable move for our GEMM-fed prompt path is an INT8/oneMKL-INT8 expert GEMM (half the
  dequant bytes, up to 2x rate).

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

The **IQ3_S Flash-Next shards are the test target**: they are the biggest supported size, exercises the IQ3_S
expert kernels and the largest host-mirror share, and would validate the INT8 XMX prompt path on a heavier model
than the Coder. Concretely: `python3 sycl/setup_intel.py --model IQ3_S` after building, then the 2,184-token
decode target.

---

## 4. Todo list

Priority order based on payoff on the B70. Ranked for *headroom* (where we already measured ourselves below the
card's capability) and *adoptability* (how directly llama.cpp's code maps onto ours).

### P0 - adopt llama.cpp's proven patterns

- [ ] **L1 decode matvec: weight reorder + ESIMD, ported from `dmmv.cpp`/`esimd.hpp`.** The single biggest open
  gap. INTEL.md already lists the misaligned IQ4_XS / IQ4_NL / Q8_0 loads as stuck around ~100-280 GB/s while the
  aligned types hit 400+. llama.cpp's `reorder_qw_*` + ESIMD `mac_pair` is exactly the layout move that wins
  there. Acceptance: parity-clean (`s2_gemv_q8k_parity`, `iq_parity`, `native_expert_parity`), and the wide
  decode kernels move off the "still misaligned" list in INTEL.md.
- [ ] **Runtime XMX probe and dispatch.** Port `gpu_has_xmx()` (`sycl::aspect::ext_intel_matrix`) into
  `sycl/include/strata/sycl_queue.hpp` / device init; advertise XMX paths only when the aspect exists, instead of
  only env flags. Acceptance: XMX kernels auto-off on a non-matrix card, on by default where `xmx_gemm_bench`
  shows a win.
- [ ] **INT8 prompt expert GEMM via oneMKL INT8 on XMX** (INTEL.md planned item 6). Halve the dequant bytes the
  prompt path feeds the GEMM. Measure on the IQ3_S model, where the prompt path dequant+GEMM dominates.
  Acceptance: prompt tok/s up at 2,184 and 8,000 tokens with output identical (or the +- near-tie
  documented, as INTEL.md does).

### P1 - close the measured headroom

- [ ] **oneMKL-GEMM flash attention (XMX prompt attention via GEMM), the #25025/#25222 pattern.** llama.cpp's
  fattn-mkl.cpp (#25025 2026) runs prompt attention as oneMKL XMX GEMM (K/V converted to F16 first so every cache
  type benefits); our port's FP32 prompt attention is currently the fast path and our joint_matrix attention is
  ~2-3x slower. An A/B against L2 (below) is the way to claim the XMX prefill win the branch note "~350+ tok/s".
  Acceptance: prompt tok/s up at 8K/40K/128K with outputs identical at the documented tolerance (kv_hybrid_parity,
  qsa_prompt_attn_parity).

- [ ] **oneDNN fused-XMX SDPA A/B for prompt attention.** llama.cpp has it (`fattn-onednn.hpp`); we have FP32 and
  our own XMX kernel, and our XMX is ~2-3x slower than FP32. One experiment: route prompt attention through
  oneDNN's fused SDPA and compare against the FP32 fallback at 8K/40K/128K. If it wins, keep it behind the same
  probe-and-verify gate.
- [ ] **Fix the remaining misaligned decode loads (IQ4_XS 136 B/8-aligned, IQ4_NL 18 B, Q8_0 34 B).** INTEL.md's
  `load16_a2` fix worked for Q6_K (150 -> 407 GB/s); extend the same two-aligned-loads-and-shift idea to the
  three types still listed as open. Acceptance: each type's bench (e.g. `mmvq_sg_bench`) crosses the aligned
  band, outputs identical.
- [ ] **Re-test SIMD16 (native Xe2 width) decode kernels now that llama.cpp uses them as the default.**
  INTEL.md measured SIMD16 as a no-net-gain (`STRATA_MMVQ_SG`), but llama.cpp ships native-16 kernels on BMG.
  Re-bench at the current engine state; if it stays flat, close the item with the measurement as INTEL.md already
  did.
- [ ] **MKL-FA softmax load coalescing (#28918) and large-register-file FA vec kernels (#29062).** llama.cpp's
  fattn-mkl.cpp coalesced the softmax loads (one work-item per row was the bottleneck) and fattn-vec.hpp grew a
  large-register path for D=512 heads. Our prompt attention completes its softmax the same way; port the
  coalesced-read softmax and check reg-file sizing on the B60's 512-head caps.
- [ ] **Q8_0/Q8_1 wide-load + DMMV ESIMD (#29186), Q2_K/Q5_K reordered ESIMD (#27490/#26376).** llama.cpp's newest
  decode work adds wide-load MMVQ and ESIMD DMMV for Q8_0, and completes reordered-ESIMD for the K-quants. Map
  onto our still-misaligned decode kernels: Q8_0 (`s_gemv_q8k`, `text` mmvq) is the MTP/dense-projections type.
  Acceptance: by-type bench steps off the misaligned list, outputs identical.
- [ ] **Fuse mul_mat(gate)+mul_mat(up)+GLU for the dense FFN (#26779), rms_norm+mul+add residual chains
  (#27610), fused UNARY(silu/...)+MUL (#26411).** llama.cpp keeps shaving cross-kernel round-trips; each matches a
  stride in our dense projections (GR down/up, norms). INTEL.md's graph-node item already prices ~5 us/node.
- [ ] **Host-pinned memory for host-to-device (#26789) and dev2dev memcpy by SYCL API (#24476/#26234/#27550
  P2P).** llama.cpp moved host access to pinned buffers and added a device-to-device (P2P) copy path - the exact
  transfer Experiment 03 measured on both B60s at 7.7-8.6 GB/s. Pipe the layer-split window hand-off through the
  native dev2dev path instead of host staging and measure against the host round trip.

### P2 - structural / hygiene

- [ ] **Read-side blockage: oneDNN/MKL calls must not fight SYCL graph capture.** llama.cpp's fattn.cpp notes MKL
  GEMM is incompatible with graph capture replay. Our prompt path captures window graphs; a oneDNN/MKL SDPA
  experiment must check it does not break `STRATA_WARM_GRAPHS`. Document the interaction in INTEL.md.
- [ ] **Model matrix: load-test the IQ3_S Flash-Next shards** (the two shards above) end to end through
  `sycl/setup_intel.py --model IQ3_S`, and add the per-shard checksum / expected sizes to the parity fixtures so
  `IQ3_S` stops being "not yet load-tested" in this table. Acceptance: an IQ3_S row in INTEL.md's measured table.
- [ ] **Match llama.cpp's automated XMX CI gate.** llama.cpp gates XMX paths behind tests (e.g. `topk-moe.cpp`,
  flash-attn self-tests). Add a `xmx_gemm_bench` + parity step to `sycl/tools/build.sh` so an XMX change that
  regresses the quantized GEMM fails the build like any other kernel.

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
- **XMX for per-token decode (expert dots, INT8 DPAS)**: three `joint_matrix` versions were 1.4x and 2-3x slower
  than the dp4a path, one hung the GPU. INTEL.md: parked. llama.cpp's own design (ESIMD reorder for decode,
  XMX only for fat GEMM/SDPA) agrees; only revisit if a weight reorder gives decode a big enough target.
- **Grouped (batch-gather) prompt attention on XMX**: measured the union of 8 positions is 3-5x one position's
  cells with only 12% shared; arithmetic cost exceeds the gather saved. INTEL.md: not built.

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

The three parity tests that still need fixtures (`iq_parity`, `ple_parity`) and the XMX prompt-attention test
were gated in INTEL.md; keep that "fixtures needed" list accurate as the IQ3_S model gets added.