# Plan: SYCL support for the dsv41 (DeepSeek) engine

Goal: research vllm-xpu-kernels + SYCLomatic, push a plan branch on the fork
(maxious/Strata_SYCL), then hillclimb a SYCL port of the core dsv41 kernels on
this owner PC (2× Intel Arc Pro B60).

## Facts from the discovery wave
- remote "fork" = https://github.com/maxious/Strata_SYCL.git, gh authed as maxious.
- Machine: 2× Intel Battlemage G21 [Arc Pro B60]; sudo passwordless; /opt/intel/oneapi exists but icpx not on PATH.
- sycl/ tree exists for the main engine: src/{kernels,core,program,prefill,ngram,platform}, CMakeLists.txt maps src/X.cu → sycl/src/X.dp.cpp (CMakeLists.txt:12-22 returns early — dsv41 option at :800 is CUDA/HIP-only).
- src/dsv41: ~15.4k lines .cu (ds4_kernels.cu 2183, ds4_gguf.cu 1469, ds4_iq.cu, ds4_tq.cu, ds4_model.cu, ds4_vision.cu) + vendor/exl3.cu; CPU reference implementations ds4_iq_cpu.cpp, ds4_tq_cpu.cpp, ds4_gguf_cpu.cpp, ds4_gg_cpu.cpp.
- HIP precedent for dsv41: hip_compat maps cublas→hipblaslt (CMakeLists.txt:71,159-163); gfx12 WMMA path ds4_iq_hip.inc; __ldcv at ds4_kernels.cu:2038-2050.
- fork already carries SYCL main-engine branches (sycl/* series); bench/results/2026-10-22-gyro-sycl-2xArcProB60/ exists.

## Ordered steps
1. Toolchain: probe /opt/intel/oneapi (setvars.sh? compiler present?), install dpcpp + SYCLomatic (Intel apt repo or GitHub release tarball). Verify: icpx --version, SYCLomatic --version. (C2)
2. Research: integrate librarian findings — vllm-xpu-kernels inventory; SYCLomatic capability matrix.
3. Explore: sycl/ wiring + exact CUDA-construct inventory of ds4_iq.cu/ds4_kernels.cu/ds4_gguf.cu/ds4_tq.cu for the port table.
4. Branch: create sycl/dsv41-sycl-support off main (8183c6b6), commit docs/SYCL_DSV41_PLAN.md, push to fork. Verify ls-remote. (C1)
5. Convert: SYCLomatic-run selected quant kernels into sycl/dsv41/*.dp.cpp following the sycl/ pattern; wire opt-in CMake target (STRATA_BUILD_DSV41_SYCL) — default path byte-identical.
6. Hillclimb: icpx -fsycl compile loop until green. (C3)
7. Device test: dequant equality vs CPU reference on Arc Pro B60 (SYCL_DEVICE_FILTER=accelerator), prints MATCH. (C4)
8. Regression: pytest tests/terminal green; git diff touches only opt-in SYCL files + docs. (C5)
9. Commits per increment on the branch; final push; cleanup RUN_TMP/monitors; self-review in notepad.

## Verification scenarios
- C1: `GIT_MASTER=1 git ls-remote fork sycl/dsv41-sycl-support` → sha; commit contains docs/SYCL_DSV41_PLAN.md (`git show --stat`).
- C2: `icpx --version` + `SYCLomatic --version` output captured.
- C3: `icpx -fsycl --target spir64-x64 <converted ds4 sources>` exit 0 log.
- C4: test binary on accelerator prints `MATCH`; device name line proves Arc Pro B60.
- C5: `python -m pytest tests/terminal` output green; `git diff --stat` scope excludes default build files.
