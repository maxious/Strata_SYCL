# ds4_model SYCL Wiring Map

**Task**: Wiring map for todo 5 of `.omo/plans/dsv41-sycl-round2.md` — "ds4_model SYCL wiring + first measured tok/s on one Arc Pro B60"
**Date**: 2026-10-25
**Worktree**: `/home/maxious/Strata-wt/dsv41-round2-w1` (branch `dsv41-sycl-w1`, HEAD `e8319a37f879874bdf62dc133344a058391387d1`)
**Method**: Read-only investigation by child `st_01a1727e` (librarian). No edits to product files. Every claim labeled MEASURED (ran a command, output shown) or STATIC (read code only). Section 10 adds facts the orchestrator measured in this session, because the librarian's tool set could not measure file sizes or GPU presence.

## 1. Files that reference ds4_model / dsv41

### 1.1 Source files that include or define ds4_model

**STATIC** (read from worktree):

- `src/dsv41/ds4_model.cu:3` — "DeepSeek-V4.1-Flash on Maya (see ds4_model.hpp)" — the 9,151-line CUDA implementation
- `src/dsv41/ds4_model.hpp:3` — header declaring the model interface
- `src/dsv41/ds4_main.cpp:21` — `#include "ds4_model.hpp"` — the CLI entry point that calls ds4_model

### 1.2 CMake targets that compile ds4_model.cu

**STATIC** (read from CMakeLists.txt):

- `CMakeLists.txt:809` — `add_executable(strata-dsv41 src/dsv41/ds4_main.cpp src/dsv41/ds4_model.cu src/dsv41/ds4_kernels.cu src/dsv41/ds4_vision.cu ...)` — the CUDA executable target
- This target is built only when `STRATA_ENABLE_CUDA=ON` or `STRATA_ENABLE_HIP=ON` (CUDA path via HIP compat layer)

### 1.3 SYCL build status

**STATIC** (read from sycl/CMakeLists.txt):

- `sycl/CMakeLists.txt:257` — `option(STRATA_SYCL_DSV41 "Build the ported dsv41 quant kernels (sycl/src/dsv41/*.dp.cpp)" OFF)`
- `sycl/CMakeLists.txt:258-262` — When `STRATA_SYCL_DSV41=ON`, builds `src/dsv41/ds4_tq.cu src/dsv41/ds4_gguf.cu src/dsv41/ds4_iq.cu` (resolved to `sycl/src/dsv41/*.dp.cpp` via `strata_resolve`)
- **ds4_model.cu is NOT in this list** — it is not built for SYCL yet
- The option is OFF by default, so the SYCL build is byte-identical to the release when off

### 1.4 Server and CLI integration

**STATIC** (read from serve/ and strata_cli/):

- `serve/server.py:8048` — `FAMILY_MODEL_NAMES = {..., "deepseek41": "deepseek-v4.1-flash"}`
- `strata_cli/launch/app.py:151` — `if (cfg or {}).get("arch") == "deepseek41"` — uses strata-dsv41, not the Qwen engine
- `strata_cli/launch/cmd.py:87` — `--dsv41-package` flag for manual engine package installation
- `strata_cli/launch/runner.py:1091-1128` — `deepseek_files()` detects DeepSeek shards, sets `s.family = "deepseek41"`
- `strata_cli/launch/runner.py:1265-1313` — `tune_deepseek()` runs `tools/dsv41/tune.py`
- `strata_cli/launch/runner.py:1853-1858` — `use_own_dsv41_engine()` points config at this install's engine
- `strata_control/dsv41_engine.py:1-134` — DeepSeek engine as its own package (feed member `dsv41`), downloaded separately
- `strata_control/dsv41_fit.py:1-317` — Memory layout: 508.4 GB download (Unsloth UD-Q8_K_XL), 305.6 GB in memory (288.8 GB experts + 16.8 GB dense)

## 2. How ds4_model.cu is compiled today

### 2.1 CUDA build (NVIDIA)

**STATIC** (read from CMakeLists.txt):

- **Target**: `strata-dsv41` executable
- **Sources**: `src/dsv41/ds4_main.cpp src/dsv41/ds4_model.cu src/dsv41/ds4_kernels.cu src/dsv41/ds4_vision.cu` (CMakeLists.txt:809)
- **Compiler**: nvcc (CUDA 13.x for RTX 50, CUDA 12.x for experimental sm_60/sm_70)
- **Flags**: `-std=c++20`, arch-specific (`-arch=sm_86` for RTX 30/40, `-arch=sm_120` for RTX 50)
- **Headers**: `include/strata/dsv41/exl3.hpp`, `include/strata/dsv41/exl3_cpu.hpp`, `include/strata/dsv41/moe_win.hpp`, `include/strata/dsv41/sampler.hpp` (ds4_model.cu:26-30); `<cublas_v2.h>`, `<cuda_runtime.h>` (ds4_model.cu:32-33); `third_party/ggml` (GGUF); `include/strata/artifact` (dequantization)
- **Links**: `strata_engine`, `strata_prefill`, `strata_spec`, `CUDA::cudart`, `CUDA::cublas`

### 2.2 HIP build (AMD)

**STATIC** (read from CMakeLists.txt and include/strata/platform/hip_compat/):

- Same sources as CUDA, compiled through HIP (hipcc)
- `include/strata/platform/hip_compat/cuda_runtime.h` maps `cuda*` -> `hip*` names
- `include/strata/platform/hip_compat/cublas_v2.h` maps `cublas*` -> `rocblas*` names
- Arch: `gfx906` (MI50, wave64) or `gfx1100`/`gfx1201` (RDNA3/RDNA4)

### 2.3 SYCL build (Intel Arc) — current state

**STATIC** (read from sycl/CMakeLists.txt):

- **Option**: `STRATA_SYCL_DSV41=OFF` by default (sycl/CMakeLists.txt:257)
- **When ON**: builds `strata_dsv41` static library from `sycl/src/dsv41/ds4_tq.dp.cpp`, `ds4_gguf.dp.cpp`, `ds4_iq.dp.cpp` (sycl/CMakeLists.txt:259)
- **Compiler**: `icpx -fsycl` (oneAPI DPC++ 2026.1 or newer)
- **Flags**: `-fsycl`, `-fsycl-default-sub-group-size=32`, `-fp-model=precise`, `-qmkl=sequential` (for oneMKL)
- **AOT**: `-DSTRATA_SYCL_AOT=bmg-g21` for Arc Pro B60 (Battlemage G21 chip)
- **Headers**: `sycl/include/compat/cuda_runtime.h` (stub for `cudaStream_t` etc.), `sycl/include/compat/cuda_fp16.h` (stub for `half`)
- **Links**: `strata_includes`, `strata_core`, `oneMKL` (via `-qmkl=sequential`)

### 2.4 What would have to be added to get a SYCL ds4_model

**STATIC** (analysis of sycl/CMakeLists.txt and ds4_model.cu):

1. Add `src/dsv41/ds4_model.cu` to the `STRATA_SYCL_DSV41` source list (sycl/CMakeLists.txt:259)
2. The `strata_resolve` function would look for `sycl/src/dsv41/ds4_model.dp.cpp` — this file does not exist yet
3. Port ds4_model.cu to SYCL: replace `cublasGemmEx` / `cublasSgemm` with `dpct::blas::gemm` (oneMKL; precedent `sycl/src/prefill/gemm.dp.cpp`), `cudaMemcpy*` with `queue.memcpy()`, `cudaStream_t` with `sycl::queue` (`sycl/include/compat/cuda_runtime.h`), `<<<...>>>` launches with `queue.parallel_for()`, `cudaEvent*` with `sycl::event`
4. Link `strata_dsv41` into `strata_engine` or a new `strata_dsv41_engine` library
5. Wire the SYCL engine into the serving path (currently only CUDA/HIP engines are started)

### 2.5 What blocks it today

**STATIC** (analysis):

- **No migrated copy of ds4_model.cu**: the 9,151-line file has not been converted by SYCLomatic or hand-ported
- **cuBLAS calls**: 3 call sites in ds4_model.cu (lines 2910, 2924, 2937) plus more in ds4_vision.cu — need `dpct::blas::gemm` wrappers
- **Custom kernels**: `pred_proj_k` (line 509) is a `__global__` kernel launched at line 7052 — needs SYCL port
- **gemv_f16, gemv_f32, sparse_attn**: custom kernels called from ds4_model.cu (lines 6967, 6995, 8650, 8725, 8755, 8765, 8852) — not in `sycl/src/dsv41/` yet
- **Engine wiring**: the SYCL engine does not know about dsv41 yet

## 3. CUDA-only constructs in ds4_model.cu that have no SYCL equivalent yet

### 3.1 cuBLAS references (11 total: 1 include, 1 handle field, 7 call expressions on 6 lines, 3 error strings)

**STATIC** (read from ds4_model.cu):

| Line | Construct | SYCL replacement |
|------|-----------|------------------|
| 32 | `#include <cublas_v2.h>` | `sycl/include/compat/cublas_v2.h` (needs to be created, mirroring the HIP shim) |
| 548 | `cublasHandle_t blas = nullptr;` | `sycl::queue` or `oneapi::mkl::blas::descriptor` (needs wrapper) |
| 70-71 | `cudaGetErrorString(e_)` in the CK macro | `sycl::exception::what()` or a custom error mapping |
| 2884 | `cublasCreate(&c.blas)`, `cublasSetStream(c.blas, d.st)` | queue creation; no separate BLAS handle (oneMKL manages workspace) |
| 2910 | `cublasGemmEx(...)` fp16 -> fp32 | `dpct::blas::gemm(...)` — precedent `sycl/src/prefill/gemm.dp.cpp` |
| 2924 | `cublasGemmEx(...)` fp16 -> fp32 | same as above |
| 2937 | `cublasSgemm(...)` fp32 | `dpct::blas::gemm(...)` fp32 variant |
| 1245, 2565 | `cublasDestroy(chunks[i]->blas)` | no-op (queue destruction handles cleanup) |
| 2914, 2928, 2939 | error strings | keep as-is or map to oneMKL exceptions |

**Precedent in repo**: `include/strata/hip_compat/cublas_v2.h` (macro shim for HIP), `sycl/src/prefill/gemm.dp.cpp` (`dpct::blas::gemm`, with a 4 GiB staging workaround).

### 3.2 `__global__` / `__device__` kernels

**STATIC** (read from ds4_model.cu):

| Line | Construct | SYCL replacement |
|------|-----------|------------------|
| 509 | `__global__ void pred_proj_k(const float* x, int D, float* out)` | `queue.parallel_for(sycl::range<2>(T, 256), ...)` |
| 7052 | `pred_proj_k<<<T, 256, 0, s>>>(c.x, (int) D, dp);` | `queue.submit([&](sycl::handler& h) { h.parallel_for(...); })` |

**Precedent**: every migrated kernel in `sycl/src/kernels/cuda/`.

### 3.3 `cuda*` API calls (~100 references, grouped)

**STATIC** (read from ds4_model.cu):

- Device management (~24): `cudaSetDevice` (22), `cudaDeviceGetAttribute`, `cudaGetLastError` (2), `cudaGetDeviceCount` (1) -> device selection at queue creation, `sycl::device::get_info<>()`, `sycl::device::get_devices().size()`
- Memory management (~11): `cudaMalloc` (2), `cudaFree` (4), `cudaHostAlloc` (2), `cudaFreeHost` (2), `cudaHostRegister` (1) -> `sycl::malloc_device()`, `sycl::malloc_host()`, `sycl::free()`, `queue.mem_advise()`
- Memory copies (~25): `cudaMemcpyAsync` (~7), `cudaMemcpy` (~9), `cudaMemcpy2DAsync` (~3) -> `queue.memcpy()`, `queue.memcpy().wait()`, manual loop or strided copy for 2D
- Streams and events (~30): `cudaStream_t` (2), `cudaEvent_t` (~9), `cudaEventCreate*` (~8), `cudaEventRecord` (2), `cudaEventElapsedTime` (1), `cudaStreamSynchronize` (4), `cudaDeviceSynchronize` (1), `cudaStreamCreateWithFlags` (2) -> `sycl::queue`, `sycl::event`, implicit submit, profiling info, `queue.wait()`, `queue.wait_and_throw()`
- Memory info (2): `cudaMemGetInfo` -> `device.get_info<sycl::info::device::global_mem_size>()`
- memset (1): `cudaMemsetAsync` -> `queue.fill()`

### 3.4 Custom kernels called from ds4_model.cu

**STATIC** (read from ds4_model.cu):

| Line | Kernel | SYCL port status |
|------|--------|------------------|
| 6967, 8725 | `sparse_attn(...)` | no precedent in `sycl/src/dsv41/` — needs port |
| 6995, 8755, 8765, 8852 | `gemv_f32(...)` | no precedent — needs port |
| 8650 | `gemv_f16_rows(...)` | no precedent — needs port |
| 6993 | `hc_pre_norm(...)` | no precedent — needs port |
| 2939 | `hc_mix_scale(...)` | no precedent — needs port |
| 2900-2918 | `gg::rows(...)` / `x3::reconstruct_rows(...)` | partially ported: `ds4_gguf.dp.cpp`, `ds4_iq.dp.cpp` |

## 4. Serving path: how a dsv41 model is started

### 4.1 User-facing command

**STATIC** (quoted from `docs/TERMINAL_LAUNCH.md`):

```sh
strata launch --model q2_0 --yes --no-tui                              # no questions (line 67)
strata launch --plan --model q2_0                                      # what would happen (line 66)
strata launch --list-presets | --delete-preset NAME | --status | --stop [--preset NAME] | --show-key  # (line 70)
```

For DeepSeek-V4.1-Flash:

```sh
strata launch --family deepseek41 --model-path /path/to/DeepSeek-V4.1-Flash-1017-GSQ-RCO --yes --no-tui
```

### 4.2 Internal flow

**STATIC** (read from strata_cli/ and strata_control/):

1. `--dsv41-package` or a model path (`strata_cli/launch/cmd.py:87`)
2. `deepseek_files()` detects DeepSeek shards, sets `s.family = "deepseek41"` (`strata_cli/launch/runner.py:1091-1128`)
3. `tune_deepseek()` runs `tools/dsv41/tune.py` (`strata_cli/launch/runner.py:1265-1313`)
4. `use_own_dsv41_engine()` points config at this install's engine (`strata_cli/launch/runner.py:1853-1858`)
5. `find(dest)` locates the newest installed `strata-dsv41` executable (`strata_control/dsv41_engine.py:23-35`)
6. `plan()` computes where the model goes (`strata_control/dsv41_fit.py:249-317`)
7. The server starts `strata-dsv41` with the computed config

### 4.3 Minimal end-to-end smoke test on one GPU

**STATIC** (from docs/SYCL_DSV41_PLAN.md and docs/INTEL_ARC.md; not run this session):

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S sycl -B build-sycl -G Ninja -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_AOT=bmg-g21
cmake --build build-sycl --target strata
./build-sycl/ds4_parity --selftest
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
SYCL_CACHE_PERSISTENT=0 \
ZES_ENABLE_SYSMAN=1 \
STRATA_VERIFY_DEVICE_PLAN=1 \
STRATA_VERIFY_NO_HOST=1 \
./build-sycl/strata --pack /path/to/DeepSeek-V4.1-Flash-1017-GSQ-RCO --context 4096 --gpu 0
```

From `docs/INTEL_ARC.md:151-152`: the Arc Pro B60 PCI id is `8086:e211`, `sycl-ls` prints `Intel(R) Arc(TM) Pro B60 Graphics 20.1.0`, so `-DSTRATA_SYCL_AOT=bmg-g21` is the right target.

## 5. Memory facts

### 5.1 System RAM

**MEASURED** by the librarian (read `/proc/meminfo`, 2026-10-25):

```
MemTotal:       127361560 kB  ~ 121.5 GB
MemFree:        118256940 kB  ~ 112.8 GB
MemAvailable:   123189300 kB  ~ 117.5 GB
SwapTotal:      138412024 kB  ~ 132.0 GB
SwapFree:       138412024 kB  ~ 132.0 GB
```

### 5.2 Model files on disk

**MEASURED** (listed `/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/`): `.cache/`, four `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000N-of-00004.gguf` shards, `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, `tensor-allocation/`. Exact sizes were not measured by the librarian (see section 10 for the orchestrator's measured sizes).

### 5.3 Model memory requirements (from code)

**STATIC** (read from `strata_control/dsv41_fit.py`):

- IST-DAS IQ2_XXS set on this PC: 4 shards + Engram (110.6 GB) + DSpark drafter (8.0 GB)
- `ISTA_SETS["IST-DAS-2.00bpw"]`: 138.2 GB of shards, 132.1 GB experts, 6.16 GB dense weights
- In RAM: experts (CPU lane) + dense weights (GPU) + ~9 GB overhead -> IQ2_XXS: 132.1 + 6.16 + 9 = 147.3 GB

### 5.4 Does the full set fit in RAM on this PC?

**STATIC** (analysis) + **MEASURED** (section 10): 117.5 GB available against 147.3 GB needed, so the full set does not fit. `dsv41_fit.py:219-231` classifies RAM as "fits", "tight" (>=40 GB) or "spills" (<40 GB); 117.5 GB reads as "tight" for IQ2_XXS.

## 6. Stop sequence: how the server is stopped

**STATIC** (read from code):

- SIGTERM triggers `request_stop("sigterm")` in `serve/lifeline.py:277-301`, which drains the server and closes the engine process cleanly
- Engine close sequence `serve/server.py:2460-2478`: send STOP and QUIT on the engine's stdin, wait `ENGINE_QUIT_WAIT_S` (30-60 s), then `proc.terminate()` (`ENGINE_TERM_WAIT_S`, ~15 s), then `proc.kill()` (`ENGINE_KILL_WAIT_S`, ~5 s)
- From `docs/SYCL_DSV41_PLAN2.md`: killing the engine mid-cleanup leaves the GPU in a `PAGEFAULT -> CAT error -> DEVICE_LOST` state that blocks later starts; stop with SIGTERM and wait
- Observable health after stop: `sycl-ls` still lists both Arc Pro B60 devices, the engine process has exited, and a second `strata launch` starts without error

## 7. Open questions for the orchestrator

**STATIC** (decisions the wiring map cannot settle from code):

1. Which SYCL BLAS wrapper to use — `dpct::blas::gemm` or a hand-written `oneapi::mkl::blas` shim mirroring `include/strata/hip_compat/cublas_v2.h`
2. Whether `ds4_model.dp.cpp` should be SYCLomatic-converted or hand-ported (9,151 lines)
3. Which custom kernel families to port first (6 families)
4. Single-GPU or multi-GPU for the first measurement, given 117.5 GB RAM against 147.3 GB needed
5. What the acceptance criterion for "first measured tok/s" is — the plan says measure prefill and decode, no target
6. How to handle the Engram (110.6 GB, IQ4_NL): read from disk or skip
7. Which context length for the first measurement (4K or 8K to reduce memory pressure)

## 8. Static vs measured (librarian)

- **MEASURED**: system RAM from `/proc/meminfo`; the model directory listing
- **STATIC**: all file:line citations in sections 1-4 and 6; CUDA constructs in section 3; SYCL build status in section 2; memory requirements in 5.3; serving path in 4; stop sequence in 6; doc command lines in 4.1 and 4.3
- **Not verified by the librarian**: exact file sizes, GPU presence, build verification, runtime verification (its tool set had no shell)

## 9. Summary of blockers (STATIC)

1. `ds4_model.cu` (9,151 lines) is not ported to SYCL
2. cuBLAS calls need oneMKL wrappers (3 gemm call sites; 11 references total counting include, handle field, create/set-stream, destroy, error strings)
3. 6 custom kernel families need SYCL ports: `sparse_attn`, `gemv_f32`, `gemv_f16_rows`, `hc_pre_norm`, `hc_mix_scale`, `gg::rows`/`x3::reconstruct_rows`
4. Engine wiring missing: the SYCL build does not know about dsv41
5. Memory pressure: 117.5 GB available against 147.3 GB needed for the IQ2_XXS set
6. No Arc Pro B60 baseline exists for dsv41 tok/s

Recommendation from the librarian: finish Phases A-C of `docs/SYCL_DSV41_PLAN2.md` before attempting the full `ds4_model` port; the first smoke test should use a minimal config (4K context, one GPU, a subset of experts if needed).

## 10. Orchestrator-measured facts added in this session (this PC, 2026-10-25)

**MEASURED** by the orchestrator (`/home/maxious/Strata-wt/dsv41-round2-w1`, integration HEAD `e8319a37`):

- `free -h`: Mem 121Gi total, avail 117Gi; Swap 131Gi. Matches the librarian's `/proc/meminfo` reading.
- `du -sh /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO`: 232G total. The four IQ2_XXS shards are ~118 GB together; `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` is 110.59 GB.
- `sycl-ls` (inside `bash -c 'source /opt/intel/oneapi/setvars.sh'`): 2x `Intel(R) Arc(TM) Pro B60 Graphics` plus an OpenCL CPU device. GPU presence is therefore MEASURED, not assumed.
- `icpx --version` prints 2026.2.0 only inside `bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; icpx --version'`; a bare probe returns empty and must not be read as "not installed".
- Two CMake facts, MEASURED in this session, that the librarian's static reading did not capture:
  - `-fp-model=precise` at `sycl/CMakeLists.txt:102-103` fails when the compiler is GNU cc; the flag must sit inside the `COMPILE_LANGUAGE:CXX` generator expression. This is measured evidence and beats the librarian's static claim that `:102-103` is correct.
  - `iq_avx512.cpp` and `nvfp4_avx512.cpp` need `-mavx512vpopcntdq;-mbmi2` in their per-file `COMPILE_OPTIONS` (`sycl/CMakeLists.txt:210-211`, matching root `:1727-1728`).
- A parity harness exists today only as a hand-built command (4 translation units: `ds4_parity.cpp ds4_iq.dp.cpp ds4_tq.dp.cpp ds4_gguf.dp.cpp`, includes `-I sycl/include -I include -I third_party/ggml -I sycl/include/compat -I include/strata/artifact -I sycl/src/dsv41`; the tile form adds `-DSTRATA_XMX_SG16=1 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21 -options -cl-fp32-correctly-rounded-divide-sqrt"`). There is no CMake parity target until todo 4 lands; a CMake configure must carry `-DCMAKE_C_COMPILER=icx`.
- `ds4_gemv.cuh` and `ds4_softmax.cuh` do not exist in the repo; the gemv and softmax originals live in `src/dsv41/ds4_kernels.cu:372/398/438` and `src/dsv41/ds4_vision.cu:126`. Any todo 4 shim must cite those, not invented headers.
- oneMKL is present: `oneapi/mkl/blas.hpp` and `libmkl_sycl_blas.so` resolve after `setvars.sh`.

**STATIC, unconfirmed**: the llama.cpp line citations and Phase A-C ordering quoted from `docs/SYCL_DSV41_PLAN.md` / `docs/SYCL_DSV41_PLAN2.md` are readings of docs, not measurements.

**No tok/s number appears in this map.** No dsv41 run has been measured on this PC in this session, so todo 5 must produce the number itself or report `[~]` naming the memory blocker.
