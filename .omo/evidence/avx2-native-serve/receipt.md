# AVX2 native-pack serve receipt - 2026-10-26, AMD Ryzen 7 5700X3D (Zen 3, AVX2 only), Intel Arc Pro B60 (24 GB), one card

Binary: /tmp/r2_rb_build/strata (SYCL build of rebased branch sycl/dsv41-sycl-support, HEAD 1ce508a3850f1d145e70605aea5d9a732d27c660 on origin/main 4b9e05a2a806a3fe24f7b2560d98626c37dade35)
Pack: /home/maxious/ComfyUI/koboldcpp/pack-iq2_xs (native_experts.txt -> native_pack true)
Profile: /tmp/prof.bin from tools/make_profile.py --no-base --out /tmp/prof.bin (MK_RC=0, 24576 ranked pairs)
Harness: tools/dsv41/serve_prompts.py, prompt ids 1582,3485,17,10,17,30,15666,6449,391,13, max 16 tokens

SRV_RC=0
READY after 6 s: READY 512 stop
strata generate: this CPU has no AVX-512: the expert kernels run on AVX-2 (multi-token for the i-quant gate/up rows)   (grep -c AVX-2 = 1, grep -c 'no AVX-512' = 1)
strata serve: prompt 10 tokens = 0 reused + 10 read in 3854 ms (2.6 tok/s), 8 generated in 365 ms (21.9 tok/s)
strata serve: decode expert cache hit rate: 14.9% (405 hits / 2725 lookups)
out.json: n_gen 8, decode_ms 364.6, decode_tps 21.941854086670322, prompt_tps 2.5950434669780718, gen [271, 248068, 198, 760, 1156, 16561, 328, 3710]

Measured from this completed run: decode 21.94 tok/s, prefill 2.60 tok/s. No number is stated for any other run.

CPU expert rows measured separately on this CPU (ds4_gg_bench, /tmp/ds4_cpu): type 18, 3.07 GB / 1,567,344 rows, ggml-cpu 33.0 GB/s, Strata AVX2 29.7 GB/s (x0.90), 0 rows differ in bits.

Not yet served: the DeepSeek-V4.1 model. strata-dsv41 is not a target in a CPU-only build (ninja: unknown target, MAIN_RC=1) nor in the SYCL tree (sycl/CMakeLists builds libstrata_dsv41.a plus ds4_parity and ds4_blas_parity only); the pack built by tools/dsv41/pack_gguf.py --model ... --engram ... (PACK_RC=0) holds ds4.txt, tensors.txt, engram.bin, tokenizer/, none of which strata reads.

## Vendor EXL3 port (SYCL strata-dsv41 link gap)
- dpct converted src/dsv41/vendor/exl3.cu, exl3_legacy.cu, moe_win.cu, sampler.cu (DPCT_RC=0) into sycl/src/dsv41/vendor/*.dp.cpp.
- Host vendor/exl3_cpu.cpp copied into sycl/src/dsv41/vendor/; its cpu_avx2_ok reads CPUID leaf 1 (ECX bit 8 AVX2, bit 23 FMA) because icpx rejects __builtin_cpu_supports on its host pass.
- CUDA-only constructs removed from the twins: cudaStreamCaptureStatus/cudaStreamIsCapturing guards (SYCL has no stream capture, DPCT1119), cudaOccupancyMaxActiveBlocksPerMultiprocessor (DPCT1007; the CUDA path's own fallback n=2 kept), __nanosleep spin hints (DPCT1008).
EOF; echo CAT_RC=0; wc -c /home/maxious/Strata/.omo/evidence/avx2-native-serve/receipt.md > /tmp/w.txt 2>&1; echo WC_RC=0; cat /tmp/w.txt; rm -f /tmp/w.txt; echo RM_RC=0
\n## Vendor EXL3 port (SYCL strata-dsv41 link gap)\n- dpct converted src/dsv41/vendor/exl3.cu, exl3_legacy.cu, moe_win.cu, sampler.cu (DPCT_RC=0) into sycl/src/dsv41/vendor/*.dp.cpp.\n- Host vendor/exl3_cpu.cpp copied into sycl/src/dsv41/vendor/; its cpu_avx2_ok reads CPUID leaf 1 (ECX bit 8 AVX2, bit 23 FMA) because icpx rejects __builtin_cpu_supports on its host pass.\n- CUDA-only constructs removed from the twins: cudaStreamCaptureStatus/cudaStreamIsCapturing guards (SYCL has no stream capture, DPCT1119), cudaOccupancyMaxActiveBlocksPerMultiprocessor (DPCT1007; the CUDA path's own fallback n=2 kept), __nanosleep spin hints (DPCT1008).\n\n## strata-dsv41 on one B60 (SYCL front-end)\n- Links green: EXE_RC=0 after strata-dsv41 links strata_kernels_cpu (iq512x_rows/iq256x_rows) and ggml-cpu/ggml-base, with STRATA_DSV41_TQ_CPU=1 defined.\n- Run on one card: RUN_RC=134. level_zero returns UR_RESULT_ERROR_INVALID_MEM_OBJECT (11 times) then "Pure virtual function called!". Unconfirmed static analysis: the pack asks for 23 GiB of graphics memory on a 24 GiB card, so a device allocation likely fails and the engine proceeds on a null handle.\n- No prompt completed on the GPU, so no GPU throughput number is reported. The CPU figures (decode 21.94 tok/s, prefill 2.60 tok/s) come from the completed CPU prompt.\n