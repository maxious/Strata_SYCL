# P0-3 plan — host-pinned + dev2dev P2P memory

Verdict: **DROP / RE-SCOPE — the premise is refuted as a gain to unlock** (skeptic-confirmed). The README's "pipe the window hand-off through the native dev2dev path instead of host staging" has **no measured gain** to recover.

## Findings (primary source, verified)
- llama.cpp host-pinned = `sycl::malloc_host` (ggml-sycl.cpp:1626, common.cpp:34); dev2dev P2P = `queue.memcpy` + `ext_oneapi_enable_peer_access` gated by `ext_oneapi_can_access_peer` (ggml-sycl.cpp:822-828), batch-capped via `GGML_SYCL_PEER_MAX_BATCH_SIZE=128` (presets.hpp:69); DEV2DEV_MEMCPY modes SYCL/L0/FORWARD (common.hpp:125-128); cross-device allreduce stages through pinned host (ggml_sycl_comm_exchange, ggml-sycl.cpp:7063-7082).
- **llama.cpp's own comment (ggml-sycl.cpp:7189): across separate SYCL contexts "a raw peer-USM q->memcpy would be a silent no-op"** — so it stages through host.
- Strata runs **each GpuStage in its own SYCL context** (`OnDevice on(st.dev)` → alloc on its own in-order queue; generate.cpp:2655-2680). Direct dev2dev P2P is therefore a silent no-op here too; grep of all Strata `sycl/src` + `sycl/include` finds **no** `ext_oneapi_enable_peer`/peer_access call — only `malloc_host` (hand-off generate.cpp:4816, layer/half staging 2111/3152/3762) and `malloc_shared` inside dpct's blas_utils.
- exp-03 measured the hand-off at **~34 µs / 256 KiB (7.7-8.6 GB/s)** — far below the decode round, **not the bottleneck**. The 1.4× decode gain (21.5-25.4 → 35.5 tok/s) comes from the layer-split **compute distribution** (halving per-GPU layers), which already ships.

## Why the premise fails
Direct dev2dev P2P would require **restructuring Strata to a single shared SYCL context across devices** — precisely what the per-stage isolation the split relies on removes — to recover a ~34 µs hand-off that is not the decode bottleneck. Not worth it.

## Recommended action
- **Drop P0-3 as a perf lane** (no code). Optionally adopt llama.cpp's `DEV2DEV_MEMCPY_FORWARD` host-staged framing as *hardening* documentation, but Strata already effectively does the host-staged hand-off.
- If future multi-GPU work needs real P2P, the prerequisite is a single-context redesign (out of P0 scope), and it would only matter once the hand-off becomes the bottleneck (it is not today).

## Acceptance for re-scope
- No perf code required. Update README.sycl.md to remove/annotate P0-3's dev2dev framing (mark refuted-as-gain, cite this session + exp-03). Criterion: doc reflects that the hand-off is not the bottleneck and the 1.4× gain is already shipped via layer split.