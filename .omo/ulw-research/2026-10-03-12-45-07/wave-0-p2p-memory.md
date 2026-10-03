# Wave 0 — P0-3 P2P/memory return (complete, st_01a101ce)

## Findings (file:line cited)
- Host-pinned: sycl::malloc_host at ggml-sycl.cpp:1626 & common.cpp:34. env GGML_SYCL_ENABLE_HOST_PINNED_MEM(default 1).
- Dev2dev direct P2P: q_dst.memcpy + ext_oneapi_enable_peer_access at ggml-sycl.cpp:822-828, gated by ext_oneapi_can_access_peer; set_peer_access at 3341-3380 gated by GGML_SYCL_PEER_MAX_BATCH_SIZE=128 (presets.hpp:69).
- DEV2DEV_MEMCPY modes enum (SYCL/L0/FORWARD) common.hpp:125-128.
- Cross-device allreduce ggml_sycl_comm_exchange (7063-7082): pinned host staging; l7189 comment "separate SYCL contexts: a raw peer-USM q->memcpy would be a silent no-op".
- No git commits in range referencing #26789/#24476/#26234/#27550.

## Strata mapping
- layer split: generate.cpp:717-740 GpuStage; stage_room 2475-2493; hand-off alloc 4801-4824 (cudaHostAllocMapped -> sycl::malloc_host mapped, portable); verify.hpp:112-119 set_stage + handoff_floats (hc*n_embd + n_embd + hc). set_next chain generate.cpp:4858.
- Strata has SEPARATE SYCL context per stage (generate.cpp:2668 OnDevice). Direct P2P IMPOSSIBLE (l7189 no-op). Hand-off MUST stage through host.

## CONTESTED-CLAIM: P0-3 premise partially REFUTED
- direct dev2dev memcpy CANNOT apply to Strata w/o single-context restructure (breaks per-stage isolation).
- exp-03: hand-off ~34us/256KiB, 7.7-8.6 GB/s, NOT the decode bottleneck.
- 1.4x decode gain (21.5-25.4 -> 35.5 tok/s) comes from layer-split COMPUTE distribution (halving per-GPU layers), NOT P2P.
- => "pipe window hand-off through native dev2dev" has NO MEASURED GAIN to unlock; host-staged path is forced and adequate.
- IMPLICATION for P0-3: likely PARK/RE-SCOPE. Either (a) adopt llama.cpp's DEV2DEV_MEMCPY_FORWARD host-staged framing (already what Strata does) or (b) single-context restructure for a 34us gain = NOT worth it. VERDICT TBD in debate.
