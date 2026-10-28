# Intent diff
- User word 'trellis': in Strata this is ds4_tq.cu, the AMD RDNA4 (gfx12) WMMA path (F28). It does NOT exist in llama.cpp (Q1, one observer). So 'trellis GEMV' in the brief means: the decode GEMV for Strata's trellis/tq expert format, accelerated on Intel XMX - not a llama.cpp feature name.
- User asks for joint_matrix specifically; upstream's merged decode-side XMX path (PR #29864) is ESIMD DPAS, not joint_matrix (F3). joint_matrix upstream is MoE GEMM (PR #29245, F5). Both are XMX; only one is joint_matrix.
- Strata's decode GEMV is scalar today (F26); joint_matrix already serves expert SET tiles (F25). The gap the user wants closed is decode GEMV on XMX.
- Hardware reality changes the answer: at M=1 the joint_matrix f16 kernel is bandwidth-bound (F39/F40), so tensor-core compute is not the lever - weight bytes and the scale epilogue are.
