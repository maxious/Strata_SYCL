# Claim graph
| claim_id | statement | provenance | risk tier | status |
|---|---|---|---|---|
| C1 | XMX sg16 compiles and runs on the B60 with icpx 2026.2 (headers present; bench builds and runs) | F34,F36,F35,F37 | verified-claims | PASS |
| C2 | Decode batch M=1 is bandwidth-bound: joint_matrix f16 moves 6.55MB in 22.2us (~295 GB/s), int8 halves bytes to 11.1us | F39,F40 | verified-claims | PASS (derived from lead runs) |
| C3 | Strata decode GEMV is scalar on CUDA and SYCL; joint_matrix serves expert set tiles only | F25,F26,F27 | verified-claims | PASS |
| C4 | Upstream merged decode-side XMX path is ESIMD DPAS int8 multi-column MMV (PR #29864), Xe2-only, reorder layout, 1-80 cols | F2,F3,F4 | verified-claims | PASS |
| C5 | Upstream joint_matrix path is grouped MoE GEMM (PR #29245), IQ + K-quants, KSPLIT=4 | F5 | verified-claims | PASS |
| C6 | dkhaldi supplies the joint_matrix API cookbook: load/mad/store, load_checked/store_checked, prefetch hint L1, joint_matrix_apply, ext_intel_packed VNNI, M=1 reduce variant | F6,F7,F8 | verified-claims | PASS |
| C7 | vllm PR #644 shows a pure-SYCL sg16 int4 GEMV beats CUTLASS grouped GEMM at rows<=8 (56->18us gate_up, 34->14us down, +14% decode) with device-side expert scan and no host sync | F12,F13,Q4 | verified-claims (author-run, B70) | PASS as author-reported |
| C8 | oneDNN has no GPU GEMV primitive; M=1 routes through systolic matmul with M-unroll floor 16; grouped_micro_gemm is the only is_gemv_ path | F15,F16,F19 | verified-claims | PASS |
| C9 | oneMKL gemv is dense float/double/complex only - no quantized decode surface | F30,F38 | verified-claims | PASS |
| C10 | valarauca PR #1 is the richest XMX integration reference (FA joint_matrix default, MoE device routing, oneDNN reorder, fused GLU decode, graph replay) but is an open fork on B70, not upstream | F9,F10,Q3 | verified-claims | PASS |
| C11 | 'trellis' is Strata's ds4_tq.cu AMD WMMA path; absent from llama.cpp | F28,Q1 | verified-claims (second observer: R1b L4 zero joint_matrix/trellis hits in fattn*) | PASS |
| C12 | Strata's own ESIMD int8 expert kernels already exist outside dsv41 (esimd_moe.dp.cpp etc.) with I8_ESIMD=AUTO | F23 | verified-claims | PASS |
| C13 | The ds4_iq barrier defect is fixed (a8ced727) | F29 | verified-claims (commit re-read from git log) | PASS |
