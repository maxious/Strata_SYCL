# Sources ledger - joint_matrix trellis GEMV for SYCL decode on dsv41

Pinned revisions; snapshot hashes are in sources/<id>/meta.json.

| id | source | revision | provenance | what it answers |
|----|--------|----------|------------|----------------|
| S1 | command: cat /tmp/w5_jm_research/xmx/run_m1.log | local sg16 build, one Arc Pro B60 | primary | M=1 decode GEMV timings |
| S2 | command: cat /tmp/w5_jm_research/xmx/run.log | local sg16 build, one Arc Pro B60 | primary | M=160 batch timings |
| S3 | command: cat .omo/evidence/dsv41-sycl-dpct-and-xpu-test/dsv41-sycl-dpct-and-xpu-test-manual-qa-two-card.md | repo QA record, build ce6218ee | primary | end-to-end prompt/decode tok/s |
| S4 | https://raw.githubusercontent.com/ggml-org/llama.cpp/d81d0d8fb95e52a66992bcfc212b1891b4d2f94a/ggml/src/ggml-sycl/fused-gemm.cpp | d81d0d8f (PR #29245 merge) | primary | fused-gemm trellis constants and structure |
| S5 | https://github.com/vllm-project/vllm-xpu-kernels/pull/644 | head 4eab928e | primary | MoE int4 GEMV kernel and author-reported runs |
| S6 | https://yiliu30.github.io/viz/joint_matrix_debug.html | live page | primary | Battlemage joint_matrix support decided by the libsycl runtime |
| S7 | command: grep -c gemv /opt/intel/oneapi/mkl/latest/include/oneapi/mkl.hpp | oneMKL 2026.2 install | primary | dense-header GEMV surface |
| S9 | command: grep -c gemv /opt/intel/oneapi/mkl/latest/include/oneapi/mkl/blas/usm_decls.hpp | oneMKL 2026.2 install | primary | USM GEMV declaration count |
| S10 | https://raw.githubusercontent.com/ggml-org/llama.cpp/d81d0d8fb95e52a66992bcfc212b1891b4d2f94a/ggml/src/ggml-sycl/fused-gemm.hpp | d81d0d8f | primary | fused-gemm shape limits |

## Provenance lines

[S1] command: cat /tmp/w5_jm_research/xmx/run_m1.log
[S2] command: cat /tmp/w5_jm_research/xmx/run.log
[S3] command: cat .omo/evidence/dsv41-sycl-dpct-and-xpu-test/dsv41-sycl-dpct-and-xpu-test-manual-qa-two-card.md
[S4] https://raw.githubusercontent.com/ggml-org/llama.cpp/d81d0d8fb95e52a66992bcfc212b1891b4d2f94a/ggml/src/ggml-sycl/fused-gemm.cpp
[S5] https://github.com/vllm-project/vllm-xpu-kernels/pull/644
[S6] https://yiliu30.github.io/viz/joint_matrix_debug.html
[S7] command: grep -c gemv /opt/intel/oneapi/mkl/latest/include/oneapi/mkl.hpp
[S9] command: grep -c gemv /opt/intel/oneapi/mkl/latest/include/oneapi/mkl/blas/usm_decls.hpp
[S10] https://raw.githubusercontent.com/ggml-org/llama.cpp/d81d0d8fb95e52a66992bcfc212b1891b4d2f94a/ggml/src/ggml-sycl/fused-gemm.hpp

Named by the brief but not snapshotted: dkhaldi/sycl_joint_matrix_kernels (5ad2f900; the guessed raw path 404s, the repo was read through the librarian lane instead), valarauca/llama.cpp PR #1 (142336ad, legacy joint_matrix API), the Intel optimization-guide-gpu joint_matrix page (documentation pointer), and ~/llvm/sycl (local checkout, probed by command, not snapshot).
