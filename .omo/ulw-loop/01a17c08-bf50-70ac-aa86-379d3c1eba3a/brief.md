# ULW-Research Brief: joint_matrix trellis GEMV for SYCL decode on dsv41

## Analysis
Core question: which joint_matrix / trellis GEMV implementations can be ported into Strata's dsv41 SYCL decode path (2x Intel Arc Pro B60, oneAPI 2026.2) to raise decode throughput, and what is the concrete porting path for each source?

Axes (8 orthogonal):
1. strata-gemv-current - Strata's current SYCL decode GEMV kernels and CUDA twin; where a trellis/joint_matrix path would slot in. Repo /home/maxious/Strata, sycl/src/dsv41 + src/dsv41. Owner: strata-gemv-current (deep-low).
2. ggml-sycl-joint-matrix - upstream ggml-org/llama.cpp SYCL backend joint_matrix/trellis GEMV code + PR #29864. Owner: ggml-sycl-joint-matrix (deep-low, browser).
3. valarauca-xpu-pr - valarauca/llama.cpp PR #1 contents and whether they are superseded by upstream. Owner: valarauca-xpu-pr (deep-low, browser).
4. vllm-xpu-kernels - vllm-project/vllm-xpu-kernels decode kernels incl. PR #644 Int4 GEMV for decode-sized grouped GEMMs. Owner: vllm-xpu-kernels (deep-low, browser).
5. dkhaldi-jm-viz - dkhaldi/sycl_joint_matrix_kernels sources + yiliu30.github.io/viz/joint_matrix_debug.html layout visualization. Owner: dkhaldi-jm-viz (deep-low, browser).
6. onemkl-onednn-capability - oneMKL SYCL gemv/matmul device support and oneDNN XPU matmul/gemm capability on this machine (/opt/intel/oneapi/mkl, /opt/intel/oneapi/dnnl libdnnl 3.13). Owner: onemkl-onednn-capability (deep-low).
7. intel-jm-spec-battlemage - Intel joint_matrix API spec + whether Battlemage BMG-G21 (Arc Pro B60) exposes XMX / ext_intel_matrix in oneAPI 2026.2. Owner: intel-jm-spec-battlemage (unspecified-high).
8. skeptic - attack lane: decode is memory-bandwidth-bound, so does joint_matrix compute help at all; XMX presence on Battlemage; perf claims; source independence. Owner: skeptic (ultrabrain).

Codebase relevant: yes (Strata repo, local oneAPI install) - External: yes (GitHub repos/PRs, Intel docs) - Collection: browser - Verification likely: yes (compile/run probes on B60 locally) - X/social signal: no (x_search unavailable, recorded here).

Scale: 8 axes; territories: local repo + local oneAPI + GitHub + Intel docs; target report ~2000 words inline HTML. Precision demand: HIGH - a wrong "XMX works on Arc Pro B60" claim sends the port down a dead path; every perf number must be MEASURED or cited. Lifecycle: single team (research + debate in place via skeptic), refinement-debate team only if contested claims remain at convergence.

Debate need: (a) XMX/joint_matrix availability on Battlemage G21; (b) whether trellis GEMV beats Strata's current scalar GEMV at decode batch 1; (c) oneMKL gemv support on Intel GPUs; (d) PR #29864 relevance to dsv41 quant formats; (e) whether Strata's existing XMX tile path (ds4_iq.dp.cpp set_tc_kernel) already covers this. Attacker: skeptic (ultrabrain).

## Expected truths (seed intent-diff.md)
- E1 Strata SYCL decode today runs scalar gemv kernels (ds4_gemv.dp.cpp gemv_f16/gemv_f16_rows/gemv_f32) plus an XMX tile path in ds4_iq.dp.cpp; measured decode 1.65 tok/s one card, 12.08 tok/s two cards (sycl/docs/sycl-dsv41-measured-report.md).
- E2 upstream llama.cpp SYCL backend has a joint_matrix trellis GEMV path (mmv_trellis) with a non-XMX fallback.
- E3 PR #29864 exists and its actual diff must be read, not guessed.
- E4 dkhaldi/sycl_joint_matrix_kernels provides joint_matrix GEMM kernels usable as a reference pattern.
- E5 vllm-xpu-kernels PR #644 implements Int4 decode GEMV for decode-sized grouped GEMMs on Xe2 with stated timings.
- E6 oneMKL SYCL gemv/matmul device support is documented in the oneMKL device support matrix.
- E7 oneDNN XPU engine supports matmul for decode-sized shapes and dnnl_sycl.hpp is usable from Strata's build.
- E8 Battlemage XMX presence decides whether joint_matrix porting is feasible at all on this hardware.

## Deliverable
lane: no-format (answered_by: request - "Summarize what you find for implementation", a chat-thread answer)
state: new (answered_by: default)
formats: html inline edition shown via show_html_page (answered_by: request)
destination: this chat thread (answered_by: request)
audience: engineer implementing the SYCL port (answered_by: request)
template: none - clean analyst register (answered_by: default)
format description: implementation-ready summary: per-source findings, porting path, feasibility verdict, priorities (answered_by: request)
Reference compare: not required (no visual source reproduced)

## Fidelity
as of: pending (answered_by: pending)
meaning check: on (answered_by: default)
source floor: standard (answered_by: default)
derived numbers: allowed (answered_by: default)
languages: report en; sources en; quotes original (answered_by: default)
identifier classes: version strings, package names, commit ids, PR numbers (answered_by: default)
fallback: block-quote (answered_by: default)
asked: as of
