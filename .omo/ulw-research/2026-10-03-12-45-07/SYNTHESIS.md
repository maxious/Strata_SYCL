# ULW-Research Synthesis: llama.cpp SYCL → Strata P0 lane plans

Members + lanes: 4 (reorder-decode, int8-dequant, p2p-memory, wide-load) · Waves: 1 (all 4 lanes + lead skeptic pass) · Excursions: 0 · Sources: 1 codebase (llama.cpp checkout HEAD cb7934c52) + 1 repo (Strata) + exp docs · Verifications: 9 file:line re-reads by lead (all landed on exact lines), clone-confirmed checkout · Debate rounds: 4 (lead-as-skeptic; ultrabrain lane unspawnable, no category model) · Elapsed: ~15 min

## Executive summary
Investigated the actual llama.cpp SYCL backend (`~/ComfyUI/koboldcpp/llama.cpp`, HEAD cb7934c52) against Strata's four P0 lanes. Result: **two lanes are genuine code wins (P0-L1 reorder+ESIMD, P0-2 dequant levers), one is dropped as refuted-as-gain (P0-3 dev2dev), and one is re-prioritized as mostly-already-done with one cheap A/B and one deferred type gap (P0-4)**. All claims verified against the real checkout; no mechanism was reported from memory.

## Findings by theme

### P0-L1 — decode reorder + ESIMD: the real, on-by-default win (ACTIONABLE)
llama.cpp's decode speedup is a **one-shot SoA weight reorder at load time** (`reorder_qw_q*`, ggml-sycl.cpp:4251-4654) plus **ESIMD `mac_pair` kernels** (`dequantize_mul_mat_vec_reorder_esimd<T>`, dmmv.cpp:1876; Q8_0 `q8_0_mac_stripe`, dmmv.cpp:2012) that stream contiguous reordered bytes. It's on the **decode** path (single/small-batch, `ne[1]<=8`), not prompt (XMX/MMQ is prompt, `ne[1]>8`) — and it's on **by default** (`GGML_SYCL_ENABLE_ESIMD`/`GGML_SYCL_ENABLE_OPT` default 1). Full port map in `plan-P0-L1-reorder-esimd.md`. [Source: ggml-sycl.cpp, dmmv.cpp, esimd.hpp — verified]

### P0-2 — INT8 prompt dequant: Strata-side kernel project (ACTIONABLE, RE-SCOPED)
CONFIRMED: llama.cpp SYCL has **no i-quant prompt GEMM worth porting** (`supports_mmq()` false, ggml-sycl.cpp:4084; mmq.cpp zero i-quant kernels; the i-quant GEMM is CUDA-only/tensor-core, ggml-cuda/mmq-load-tiles.cuh). llama.cpp's SYCL i-quant dequant kernels serve convert/get_rows only, not the prompt path. So the win is a **Strata-only faster dequant** (4 levers: halve LUT via `get_int_from_table_16`, register-block output, wider 16×16 chunks, amortize scales). See `plan-P0-2-int8-dequant.md`. [Source: ggml-sycl.cpp, mmq.cpp, mmvq.cpp, mmq-load-tiles.cuh, Strata iq_kernels.dp.cpp — verified]

### P0-3 — dev2dev / pinned memory: DROPPED (refuted-as-gain)
The README's "pipe the hand-off through native dev2dev" is refuted: Strata runs each GpuStage in its **own SYCL context**, where raw peer-USM memcpy is a **silent no-op** (llama.cpp's own warning, ggml-sycl.cpp:7189), the hand-off already uses host-pinned `malloc_host` (generate.cpp:4816), it costs only **~34 µs / 256 KiB** (exp-03) — not the bottleneck — and the 1.4× decode gain comes from layer-split compute distribution that already ships. See `plan-P0-3-p2p-memory.md`. [Source: ggml-sycl.cpp, exp03, Strata generate.cpp — verified]

### P0-4 — wide-load / Q2_K/Q5_K ESIMD: PARTIAL (mostly done, one A/B)
Strata **already has** the wide MMVQ kernels (Q8_0, IQ4_NL, Q4_K, Q5_K, Q6_K, IQ4_XS — native_mmvq.dp.cpp). The new stuff is Q8_0's `sycl::int4` wide-load (cheap A/B vs `load16_a2`), and Q2_K — the only genuinely missing dense type but with **no Strata model user today**. ESIMD DMMV infra is Intel-only and folds into P0-L1. See `plan-P0-4-wide-load-esimd.md`. [Source: vecdotq.hpp, dmmv.cpp, esimd.hpp, Strata native_mmvq.dp.cpp — verified]

## Codebase findings (absolute paths + line refs)
See the four plan files: `plan-P0-{L1,2,3,4}-*.md` in this directory. Each lead claim carries its exact `file:line`, all 9 load-bearing ones re-read and confirmed by the lead this run.

## Sources (ranked)
1. `~/ComfyUI/koboldcpp/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` — dispatch hub (supports_mmq, reorder gating, mul_mat ladder). Primary. HEAD cb7934c52.
2. `.../ggml-sycl/dmmv.cpp` — the ESIMD decode kernels (generic skeleton + Q8_0 + per-type launchers).
3. `.../ggml-sycl/esimd.hpp` — `esimd_reorder_q_traits` (Q2_K..Q6_K) + helpers.
4. `.../ggml-sycl/vecdotq.hpp` — reorder_vec_dot_* (wide Q8_0, Q2_K, Q5_K).
5. `.../ggml-sycl/mmvq.cpp` / `quantize.hpp` — reorder MMVQ + quantize_and_reorder_q8_1_soa.
6. `.../ggml-cuda/mmq-load-tiles.cuh` + `mmq.cu` — the CUDA-only i-quant GEMM (proves absence in SYCL).
7. Strata `sycl/src/program/generate.cpp` — separate-context-per-stage (2665-2680), host-pinned hand-off (4816); `sycl/src/kernels/cuda/iq_kernels.dp.cpp` (2298); `native_mmvq.dp.cpp`.
8. Strata `docs/sycl-experiments/03,04,07,09,10,11-*.md` — prior measurement grounding.

## Verified claims
| claim | verdict | evidence |
|---|---|---|
| llama.cpp decode reorder+ESIMD on decode path, on-by-default | CONFIRMED | ggml-sycl.cpp:4882-4893 + env defaults; dmmv.cpp:1876/2012/2046 |
| no SYCL i-quant prompt GEMM; CUDA-only | CONFIRMED | ggml-sycl.cpp:4084; mmq.cpp zero iq; mmq-load-tiles.cuh |
| llama.cpp SYCL i-quant dequant not on prompt path | CONFIRMED | convert.cpp:656/746 callers (dmmv/fattn/fallback) |
| P0-3 dev2dev premise refuted-as-gain | REFUTED-GAIN | generate.cpp:2665-2680,4816; ggml-sycl.cpp:7189; exp03 (34µs, gain from split) |
| Q2_K missing in Strata but no user | CONFIRMED | native_mmvq.hpp/README/INTEL grep empty |

## Epistemic instrumentation
- intent-diff: IT1 (decode=reorder/ESIMD, not XMX) → true; IT2 (no prompt i-quant GEMM) → true; IT3 (dev2dev maps/portable) → **violated** (refuted-as-gain); IT4 (P0-4 distinct+portable) → partial (mostly done, Q2_K latent).
- claim-graph: C-P0L1-1/2 supported; C-P0-2-1/2/3 supported; C-P0-3-1 refuted-as-gain, C-P0-3-2/3 supported; C-P0-4 part re-prioritized. All code claims backed by primary source; convergence explicit.

## Debate record
4 rounds; verdicts in `debate-log.md`. No claim entered the synthesis without the lead re-reading its primary source.

## Contradictions
- README.sycl.md P0-3 premise ("pipe through native dev2dev") vs Strata's own architecture (separate contexts make it a no-op + not the bottleneck). Resolution: refuted-as-gain, recommend DROP. Recorded, evidence-backed.

## Gaps
- ESIMD portability to Strata's B60/oneAPI toolchain (does the build accept `<sycl/ext/intel/esimd.hpp>`) is unverified — it needs a compile probe before full ESIMD infra investment (P0-L1 risk).
- Q2_K value depends on a future model entering the matrix; today latent.

## Expansion trace
- Wave 1: 4 lanes → all surfaced concrete EXPAND leads; every lead was folded into a plan (act-parity, ESIMD-infra-probe, int4 A/B, Q8_0/or Q2_K gating). No silent drops.
- Convergence: single-wave, codebase-only brief; 4/4 axes covered by dedicated lanes; no unchecked leads remain (all routed into plan scope/defer decisions).