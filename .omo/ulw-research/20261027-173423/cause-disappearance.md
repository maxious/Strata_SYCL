# Cause disappearance
- Symptom: SYCL decode 1.65 tok/s on one card (Q7).
- Candidate cause A: scalar GEMV leaves XMX unused (F26). Disappears only if the kernel moves the same bytes faster - but at M=1 the XMX f16 kernel already runs at ~295 GB/s (F39/F40), so compute is not the bottleneck; A alone does not explain it.
- Candidate cause B: weight bytes dominate. int8 weights halve bytes (11.1 vs 22.2 us/expert, F39) - this is the lever that actually removes the symptom.
- Candidate cause C: scale epilogue cost. i8blk adds 6.8 us/expert over i8 (F40) - a fused dequant/scale path must be cheaper than that.
- Candidate cause D: expert offload to CPU (109/1680 experts, Q7) - a memory/paging issue, orthogonal to kernel choice.
- Disappearance test: run the decode-shaped kernel at M=1 for one expert (done, F39) and compare 22.2us/expert against the measured decode rate. 8 tokens x ~64 expert GEMVs at 22us = ~11 ms/token of expert compute, far below the measured 527 ms/token - so the measured decode rate is NOT kernel-bound; the symptom's cause lies outside the GEMV kernel (D and non-expert layers).
