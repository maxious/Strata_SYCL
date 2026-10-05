# Experiment 01 - XMX runtime capability probe (from llama.cpp)

## Question

Strata's XMX kernel paths are gated purely by environment flags (`STRATA_PROMPT_ATTN_XMX=1`, and the fused
XMX GEMM's caller). llama.cpp gates XMX behind a runtime device probe - `gpu_has_xmx()` in
`ggml/src/ggml-sycl/common.cpp` doing `dev.has(sycl::aspect::ext_intel_matrix)` - so a `joint_matrix` kernel is
only ever launched on a device that actually has matrix units. Can the same probe be adopted into the port, so
the capability (not just a flag) decides whether an XMX path is reachable?

Reference: llama.cpp `gpu_has_xmx()` = `dev.has(sycl::aspect::ext_intel_matrix)` (aspect 53), consumed for the
SYCL_USE_XMX mmq tiles and the XMX attention dispatch.

## Change

`sycl/include/strata/sycl_queue.hpp` - added `strata::gpu_has_xmx(device)` / `(queue)` / `(queue*)`, the SYCL
analog of llama.cpp's probe, via `sycl::aspect::ext_intel_matrix`. The port's matrix aspect (53) is present in
oneAPI 2026.1 (`device_aspect_macros.hpp`).

Two XMX entry points now require the capability in addition to their existing gates:
- `qsa_prompt_attn_xmx` (`sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp`): after the `STRATA_PROMPT_ATTN_XMX`
  env opt-in and shape checks, `if (!strata::gpu_has_xmx(st)) return false;` so the caller keeps its FP32
  fallback.
- `xmx_gemm_iq` (`sycl/src/kernels/cuda/iq_kernels.dp.cpp`): same gate before launching the fused dequant+XMX
  GEMM.

Neither changes when XMX is used - both were already opt-in and still are - the probe only makes the "can this
device run a matrix kernel" decision explicit and device-driven, matching llama.cpp. A device without matrix
units now cleanly falls back instead of launching (or failing on) a `joint_matrix` kernel.

## Benchmark: probe result on this box

`sycl/src/kernels/xmx_probe_test.cpp` (new: `xmx_probe_test`, registered in `sycl/CMakeLists.txt` and as a
ctest) enumerates every visible GPU and reports the probe:

    device 0: Intel(R) Arc(TM) Pro B60 Graphics xmx=1 (ext_intel_matrix)
    device 1: Intel(R) Arc(TM) Pro B60 Graphics xmx=1 (ext_intel_matrix)
    device 2: Intel(R) Arc(TM) Pro B60 Graphics xmx=1 (ext_intel_matrix)
    device 3: Intel(R) Arc(TM) Pro B60 Graphics xmx=1 (ext_intel_matrix)
    4 GPU device(s), 0 without XMX     -> exit 0

The 4 GPU entries are the two B60s enumerated twice (Level-Zero and OpenCL). Both cards advertise
`ext_intel_matrix`, so the probe returns true everywhere and both gated XMX paths stay reachable - the change
did not disable XMX on the hardware it runs on.

Since every device here has XMX, the "probe returns false -> fallback" branch cannot be exercised on this box;
it is covered by construction (the same `dev.has(ext_intel_matrix)` ll:ama.cpp uses on non-XMX iGPUs) and the
test binary's `--allow-non-xmx` path.

## Validity

The two parity tests that *force* `STRATA_PROMPT_ATTN_XMX=1` (exactly the gated dispatch) still pass on the B60:

- `kv_hybrid_parity`: **ALL PASS**, including `[5/5] qsa_prompt_attn mode 3 (tensor cores): PASS (vs dequant ref
  7.60e-06)`.
- `qsa_prompt_attn_parity`: **0 failures** across int8/fp16 contexts at ctx 32768/1500/2100. Sample:
  `PASS int8 ctx 32768, 2048 queries: ... 61.420 -> 171.040 ms per chunk (0.36x)` (per-chunk; the FP32 fallback
  is faster, as INTEL.md already records).

`xmx_gemm_bench` still runs (devices have XMX): fused quantized GEMM `0.34x` of dequant+oneMKL
(`dequant 0.078 + mkl 0.044 = 0.126 ms` vs `xmx 0.371 ms`), unchanged by the gate.

Full rebuild: `reconf` exit 0, `ninja strata xmx_gemm_bench xmx_probe_test` exit 0, 58/58 objects, 0 errors,
only the pre-existing 36 warnings (`argmax` unused, dpct helpers).

## Conclusion

llama.cpp's XMX capability probe ports cleanly into Strata (~5 lines in `sycl_queue.hpp`, two one-line gates).
On the dual-B60 box both cards advertise the matrix aspect, so XMX paths remain usable; the change's value is
that a future non-XMX Intel iGPU/dGPU now falls back through the documented FP32 path automatically instead of
relying on the operator to unset `STRATA_PROMPT_ATTN_XMX`. No parity regression, no engine behavior change on
this hardware.