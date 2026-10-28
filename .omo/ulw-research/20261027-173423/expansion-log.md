# Expansion log
Wave 0 leads queued: PR29864 content (R1); B60 XMX presence (W1, m7, skeptic); trellis format applicability to q4_K/q6_K/IQ (m1, skeptic); oneMKL gemv XPU support (m6); oneDNN XPU gemv vs matmul (R3, m6); dkhaldi API patterns (m5, R2); vllm PR644 kernel mechanism (m4).
CORRECTION (lead, self-verified): recalled memory note claimed joint_matrix tile kernels in ds4_gemv.dp.cpp and ds4_softmax.dp.cpp; grep -c says 0 in both — real sites are ds4_gguf.dp.cpp (16) and ds4_iq.dp.cpp (13). Decode GEMV path has no joint_matrix today.
LEAD: does dpct.hpp (dpcpp-ct 2025.3) supply the legacy joint_matrix API that lets ds4_gguf compile? — WHY: decides whether a trellis GEMV can use legacy API today or needs esimd/CooperativeMatrix — ANGLE: grep dpct.hpp + compile a minimal legacy-API kernel with -I dpcpp-ct include
EOF; echo LOG_RC=$?

CORRECTION (lead, self-verified): recalled note claimed joint_matrix in ds4_gemv.dp.cpp/ds4_softmax.dp.cpp; grep -c = 0 both; real sites ds4_gguf.dp.cpp (16), ds4_iq.dp.cpp (13). Strata uses the UNIFIED sycl::ext::oneapi::experimental::matrix API, not legacy intel::experimental.
