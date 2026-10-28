# E1 return (st_01a17c13)
Rows F25-F33. Key correction: 'trellis' is Strata's own ds4_tq.cu quant path (AMD gfx12 wmma), not a llama.cpp term. Decode GEMV is scalar on both CUDA and SYCL; joint_matrix already serves expert SET tiles (iq 16x16x16 ext_intel_packed, gguf 8x16x16).
