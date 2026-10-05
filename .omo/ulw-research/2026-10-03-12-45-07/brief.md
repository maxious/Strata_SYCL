# ULW-Research Brief: llama.cpp SYCL → P0 lane plans

<analysis>
Core question: What in llama.cpp's SYCL backend (in `~/ComfyUI/koboldcpp/llama.cpp`, HEAD cb7934c52) maps onto Strata's four P0 lanes from README.sycl.md, as implemented concrete reference code Strata can port — so each P0 lane gets a coded, cited plan?
Axes (4, one per P0 lane):
- L1/decode: llama.cpp's reorder + ESIMD decode matvecs (`dmmv.cpp`, `esimd.hpp`, `reorder_qw_*`, `mmvq.cpp`) — exact mechanism, code, acceptance mapping to Strata's s2_gemv/iq_parity/native_expert_parity.
- INT8 prompt dequant: llama.cpp's SYCL quant/dequant kernels and whether any prompt-path i-quant kernels exist (Exp 10/11 say none; verify precisely which dequant/quant files exist, their callers, and what a faster Strata-side dequant can lift).
- P2P / memory: llama.cpp's host-pinned mem + dev2dev memcpy SYCL API paths (#26789, #24476/#26234/#27550) — exact code, call sites, and how it maps to Strata's exp-03 window hand-off.
- Q8_0/Q8_1 wide-load + DMMV ESIMD + Q2_K/Q5_K reorder (#29186, #27490, #26376): the newest decode work, code and mapping to Strata's remaining dense types.
Codebase relevant: yes (all local). External: low. Browsing: no. Verification likely: yes (grep/read the actual checkout). X/social: no.
Scale: 4 axes, 1 source territory (local checkout), target ~ a per-lane plan doc + synthesis. Precision demand: high — these plans feed real kernel ports on the B70; a wrong mechanism costs a wasted port.
→ lifecycle: single research team (4 axis owners + skeptic), then my synthesis.
Debate need: contested claims = "which llama.cpp calls are actually on the decode/prompt paths" (callers vs dead code), "is dequant really absent from llama.cpp's prompt path", and which mechanism maps cleanly onto Strata. Skeptic attacks these.
</analysis>

## Axes (one owning member per P0 lane + a cross-cutting skeptic)

1. `reorder-decode` — P0-L1: llama.cpp reorder + ESIMD decode matvec. Owner: deep-low.
2. `int8-dequant` — P0-2: biggest open prompt lever (decode of dequant phased). Owner: deep-low.
3. `p2p-memory` — P0-3: host-pinned + dev2dev memcpy paths. Owner: deep-low.
4. `wide-load` — P0-4: Q8/Q2/Q5 wide-load + reorder ESIMD follow-on. Owner: deep-low.
5. `skeptic` — cross-critiques all four mechanisms + their Strata mapping claims. Owner: ultrabrain.

## Expected truths (seed intent-diff.md)

- IT1: llama.cpp's decode matvec speedups come from weight reorder (reorder_qw_*) + ESIMD kernels, not XMX.
- IT2: llama.cpp has NO SYCL i-quant GEMM/prompt-path dequant that beats Strata's dequant+oneMKL FP16 path (Exp 10/11).
- IT3: llama.cpp's dev2dev/host-pinned memcpy is a real, portable SYCL-API-callable mechanism that maps to Strata's exp-03 hand-off.
- IT4: llama.cpp's Q8_0/Q2_K/Q5_K wide-load/reorder work is distinct code from the L1 reorder and portable to Strata's remaining dense types.

## Deliverable

- Lane: template-strict (a plan doc per P0 lane + one synthesis). Formats: markdown plan + synthesis. Destination: repo `docs/sycl-experiments/` / handoff to future ulw runs. Audience: the Strata SYCL maintainer (self). answered_by: user-directed via prior turn.
</analysis>