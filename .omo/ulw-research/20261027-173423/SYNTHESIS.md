# Synthesis - joint_matrix trellis GEMV for SYCL decode on dsv41

STATUS: draft

## Verdict
Decode on dsv41 is memory-bound, and joint_matrix (XMX) does not win it: the M=1 XMX tile path measures {{F1:decode-f16-gateup-m1}} us/expert on gate+up, {{F4:decode-f16-down-m1}} us/expert on down - about the DRAM floor for these expert shapes. The port that matters is not a GEMV tile but the fused-gemm trellis for batched shapes: llama.cpp PR #29245 (merge d81d0d8f) runs XMX joint_matrix with FG_KSPLIT={{F15:fg-ksplit}} sub-groups per K range and GGML_SYCL_FG_MAX_N={{F26:fg-max-n}}, decoding quant blocks inside the A stage instead of repacking. On this machine the same shapes measure {{F20:batch-f16-gateup-m160-v2}} us (f16), {{F21:batch-i8-gateup-m160-v2}} us (i8), {{F22:batch-i8blk-gateup-m160-v2}} us (i8-block) for gate+up and {{F23:batch-f16-down-m160-v2}}, {{F24:batch-i8-down-m160-v2}}, {{F25:batch-i8blk-down-m160-v2}} us for down.

## Support fact (decides everything downstream)
{{Q4:libsycl-decider-q}: on Battlemage, joint_matrix availability is decided by the libsycl runtime, not the silicon - oneAPI 2025.3.2 (.so.8) reports unsupported, a 2026 build (.so.9) runs DPAS tiles with the driver and IGC byte-identical. Strata builds against 2026.2, so the tile path is available; the repo already gates it with STRATA_XMX_SG16.

## Sources, in brief order
- oneMKL: GEMV exists only as dense float/double/complex ({{F27:onemkl-gemv-decl-count}} declarations in usm_decls.hpp); no quantized or skinny path. Dead end for dsv41 decode.
- oneDNN: v3.13 headers installed locally, no dnnl_src link; upstream v3.14 adds an M=1 no-copy bypass, gemmstone has no skinny logic. Useful as a reference, not a drop-in.
- valarauca PR #1: legacy joint_matrix API (removed from current SYCL); pattern only.
- vllm-xpu-kernels PR #644: dedicated MoE int4 GEMV kernel, {{Q3:b70-bandwidth-q}} - the author's own framing is bandwidth, not tiles.
- dkhaldi kernels: joint_matrix reduce implementation, read through the librarian lane; raw path 404s.
- yiliu30 viz page: the libsycl root cause above.
- llama.cpp PR #29245: the trellis design worth porting.

## Implementation plan for Strata SYCL
1. Port the fused-gemm trellis (S4/S10) into sycl/src/dsv41 as a quant-block A stage for ds4_gguf, reusing the existing ext_intel_packed 16x16x16 sg16 tiles in ds4_iq.dp.cpp.
2. Keep ds4_gemv.dp.cpp gemv_f16/gemv_f32 for M=1; the measured floor says tiles cannot beat it.
3. Adopt the SoA weight layout (reorder_qw offsets) from PR #29245 so the A stage decodes whole blocks per lane.
4. Gate the new path behind STRATA_XMX_SG16 and validate with the xmx bench harness at M=160 and M=1 before touching dispatch in ds4_model.dp.cpp.

## Limits
Batch numbers are kernel-level on one B60, not end-to-end; end-to-end decode is {{Q1:strata-decode-one-card-q}} on one card and {{Q2:strata-decode-two-card-q}} on two (prompt {{F13:strata-prompt-one-card}} and {{F14:strata-prompt-two-card}} tok/s). B70 numbers from PR #644 do not transfer directly to B60.
