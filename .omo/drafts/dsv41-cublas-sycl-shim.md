# dsv41 SYCL port, todo 4: the cublas -> oneMKL shim (read-only mapping)

Status: static reading of this checkout plus one toolchain probe on this PC (2026-10-25). No product file was
edited. Everything below is either a file-and-line fact read here or marked as a recommendation.

Scope of todo 4 (`.omo/plans/dsv41-sycl-round2.md`, "Add the cublas->oneMKL shim for SYCL and device parity for gemv
and softmax"): write `sycl/include/sycl_compat/cublas_v2.h` mirroring `include/strata/hip_compat/cublas_v2.h`, port
the gemv/softmax helpers to a SYCL translation unit, register `ds4_parity` and a gemv parity binary under
`STRATA_SYCL_DSV41` so `ctest -R ds4` runs them. Must NOT: touch upstream CUDA files or the default build, invent a
speed claim.

## 1. The 11 `cublas` references in `src/dsv41/ds4_model.cu`

11 lines in the file name `cublas` (case-sensitive grep). 7 are call expressions on 6 lines; the other 4 are the
include, the handle field and three error strings. All 11 are tabled so the count reconciles.

| # | line | exact call / reference | what it computes or does | oneMKL or plain-SYCL equivalent | oneMKL here? |
|---|------|------------------------|--------------------------|--------------------------------|--------------|
| 1 | :32 | `#include <cublas_v2.h>` | pulls in the cuBLAS API the file uses | the shim header itself; resolved by putting `sycl/include/sycl_compat` on the include path | n/a (include) |
| 2 | :548 | `cublasHandle_t blas = nullptr;` (field of `Chunk`) | one BLAS handle per device chunk, bound to that device's compute stream | `cublasHandle_t` = shim handle wrapping `dpct::blas::descriptor` | n/a (type) |
| 3 | :1245 | `cublasDestroy(chunks[i]->blas);` (`Impl::~Impl`, after `cudaSetDevice`) | releases each chunk's handle at shutdown | `delete` of the shim handle; nothing else to free | n/a (lifecycle) |
| 4 | :2565 | `if (chunks[di]->blas) cublasDestroy(chunks[di]->blas);` (`alloc_chunks` rollback) | releases the handle when the chunked prompt buffers did not fit and the chunk size is halved | same as #3 | n/a (lifecycle) |
| 5 | :2884 | `cublasCreate(&c.blas) != CUBLAS_STATUS_SUCCESS \|\| cublasSetStream(c.blas, d.st) != CUBLAS_STATUS_SUCCESS` (`alloc_chunks`) | creates the handle and binds it to device `d`'s compute stream `d.st` | `cublasCreate`: allocate handle; `cublasSetStream`: `desc.set_queue(st)` (`dpct::blas::descriptor::set_queue`, `sycl/include/dpct/blas_utils.hpp:298`; `cudaStream_t` is `dpct::queue_ptr`, `sycl/include/compat/cuda_runtime.h:12`) | n/a (lifecycle) |
| 6 | :2910-2912 | `cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, m.k, &one, c.w16, CUDA_R_16F, m.k, x16, CUDA_R_16F, ldx, &beta, y + c0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)` (`Impl::gemm`, :2901) | `y[r][c] = x16[r] . W[:, c] (+ beta y)` for `r < rows`: the chunked prompt path's dense products, W reconstructed to FP16 in whole 128-column blocks into `c.w16` (`gg::rows` / `x3::reconstruct_rows`), FP32 out, FP32 accumulation. Column block loop `c0 += cols`, `cols = max(128, min(m.n, w16_elems/m.k)/128*128)`; `rows` = chunk tokens | `dpct::blas::gemm(desc, trans, nontrans, nc, rows, m.k, &one, c.w16, library_data_t::real_half, m.k, x16, real_half, ldx, &beta, y+c0, real_float, ldy, compute f32)` (`blas_utils.hpp:920`; half x half -> float is a dispatched combination, same type family as the bf16 case read at :1000-1010) | **yes** |
| 7 | :2914 | `err = "cublasGemmEx: status " + std::to_string((int) st);` | error string of #6 | unchanged (pure host string) | n/a (diagnostic) |
| 8 | :2924-2926 | `cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, W, CUDA_R_16F, k, x16, CUDA_R_16F, ldx, &zero, y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)` (`Impl::gemm_f16w`, :2919) | `y[r][c] = alpha x16[r] . W[c]` for F16 weights `W [n][k]` already in memory (the indexer's `weights_proj`); beta = 0 | same as #6 with `m = n`, `k = k`, `alpha` kept, `beta = &zero` | **yes** |
| 9 | :2928 | `err = "cublasGemmEx (F16 weights): status " + ...` | error string of #8 | unchanged | n/a (diagnostic) |
| 10 | :2937-2938 | `cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n_mix, T, hcd, &one, fn, hcd, h, hcd, &zero, mixes, 32)` (`Impl::hc_mix_chunk`, :2934) | FP32 GEMM `mixes[n_mix][T] = fn^T . h^T`, `n_mix = (2 + g.hc) * g.hc`, `hcd = g.hc * g.dim`, `ldc = 32`; the mHC mix dots over a chunk, then `hc_mix_scale` applies the rms scale (own kernel, :2943) | `oneapi::mkl::blas::column_major::gemm(q, trans, nontrans, n_mix, T, hcd, 1.0f, fn, hcd, h, hcd, 0.0f, mixes, 32)` (the shape `sycl/probe/gemm_launch.cpp:33` already runs) or `dpct::blas::gemm` with `real_float` throughout | **yes** |
| 11 | :2939 | `err = "cublasSgemm (mHC mixes)";` | error string of #10 | unchanged | n/a (diagnostic) |

Outside this file but inside the shim surface: `src/dsv41/ds4_vision.cu` has 10 more `cublas` references (one
`gemm()` wrapper :185/:188, `cublasCreate`/`cublasSetStream`/`cublasSetWorkspace` :370-379, `cublasDestroy` :221,
two BF16 `cublasGemmEx` :470/:475). `docs/SYCL_DSV41_PLAN2.md:15` counts 13 math calls across both files, all plain
gemms with transposes, no cublasLt and no fused epilogues. Vision is not in todo 4 or 5, but `cublasSetWorkspace`
and `CUDA_R_16BF` are part of the mirrored surface because of it.

### Recommendation: oneMKL for every call site above, hand-written SYCL gemv only where CUDA already uses one

- **Use oneMKL (through `dpct::blas::gemm` or `oneapi::mkl::blas::column_major::gemm`) for #6, #8, #10.** These are
  not skinny MoE gemv: they are the chunked prompt path's dense matrix products with `rows` = the chunk's tokens
  (hundreds), full-weight `k` (up to `g.dim` / `g.hc * g.dim`), and FP32 accumulation (FP16 operands in #6/#8,
  FP32 in #10). oneMKL covers the exact type and transpose combinations 1:1 with the cuBLAS calls, column-major
  like cuBLAS so `m/n/k/lda/ldb/ldc` pass through unchanged, and the repo already runs this pairing
  (`strata_prefill` links it, `sycl/src/prefill/gemm_bf16_parity.dp.cpp` checks a oneMKL-backed gemm against
  `cublasGemmEx`). The repo measures oneMKL's FP16 GEMMs on the XMX units at 30-60 TFLOP/s in `xmx_gemm_bench`
  (`docs/INTEL.md:306`; that number is the repo's, measured on its own cards, not produced in this session).
- **A hand-written SYCL gemv is the better choice only for the true skinny shapes, and those are not cublas call
  sites.** The MoE expert products never go through cublas: they are `ds4_iq`'s GEMV jobs (`src/dsv41/ds4_iq.cu:898`),
  `ds4_tq`'s GEMV jobs (`ds4_tq.cu:433`), and the exl3 `block_gemv` family (`src/dsv41/vendor/exl3.cu:254`). The
  window/decode gemvs are the three kernels in `ds4_kernels.cu` (below). Those keep their own kernels in the port,
  which is what todo 4's "kernel fallback" means: port the gemv kernels, do not re-point the cublas sites at them.
- **Why not replace #6/#8/#10 with a hand-written gemv:** (a) `rows > 1` in the prompt path, so a gemv would need a
  multi-row variant anyway; (b) a new kernel must reproduce cuBLAS's FP32 reduction order for parity, and nothing
  in the port would test it against the original until todo 5 runs the engine; (c) oneMKL keeps the change to a
  header plus CMake, and the plan's gate is "everything compiles, gemv parity against CPU on device" - no speed
  claim is permitted, so a hand-written path has no measured win to justify itself here.
- **Where a gemv does beat oneMKL:** `rows == 1..16` (decode/window calls). If a later change funnels those through
  `Impl::gemm`, route them to the ported `gemv_f16`/`gemv_f16_rows`/`gemv_f32` kernels, which already accumulate in
  the required order ("the same products in the same order", `ds4_kernels.cu:393-396`). That is a future decision,
  not part of the 11 sites as written.

### The CUDA gemv/softmax originals (the files named in the ask do not exist)

`src/dsv41/ds4_gemv.cuh` and `src/dsv41/ds4_softmax.cuh` are **not present** in this checkout (full `src/dsv41`
listing checked). The originals are:

| original | file | semantics | launcher |
|----------|------|-----------|----------|
| `gemv_f16_kernel` | `src/dsv41/ds4_kernels.cu:372` | `y[t][row] = alpha (W[row] . x[t])`, F16 weights, FP32 accumulate, warp a row | `gemv_f16` :1921, grid `(cdiv(n, kThreads/32), T)` |
| `gemv_f16_rows_kernel` | `ds4_kernels.cu:398` | same product for `T` activations (`kRowsT = 2` a block), rows `ldx` apart, outputs `ldy` apart, bit for bit `gemv_f16_kernel` | `gemv_f16_rows` :1924 |
| `gemv_f32_kernel` | `ds4_kernels.cu:438` | F32 router GEMV, `fma` accumulate, `y[t*ldy+row]` | `gemv_f32` :1931 |
| `softmax_kernel` | `src/dsv41/ds4_vision.cu:126` | per-row softmax over up to `kSmPer*kSmT = 10240` keys, max/sum in `block_reduce`, BF16 output `p`; used :473 (vision attention) | inline launch |
| attention slab softmax | `src/dsv41/ds4_attn_split.cuh:134` | "local softmax: warp g handles head g" over the slab's keys | inside the split-attention kernel |

The plan's "port the CUDA gemv/softmax helpers ... into a SYCL translation unit" therefore means the three gemv
kernels of `ds4_kernels.cu` plus the softmax kernel (plan wording: "the three gemv kernels and the vision softmax",
`docs/SYCL_DSV41_PLAN2.md:29`).

## 2. Shim header shape: `sycl/include/sycl_compat/cublas_v2.h`

Mirror contract (from `include/strata/hip_compat/cublas_v2.h`, 63 lines, read here): same call names, same argument
shapes; the mapping lives behind the names so upstream files compile unmodified. The HIP shim's surface is
`cublasCreate, cublasDestroy, cublasHandle_t, cublasSetStream, cublasSetWorkspace, cublasSetMathMode, cublasGemmEx,
cublasSgemm, cublasSgemmStridedBatched, cublasGemmStridedBatchedEx, cublasStatus_t` plus constants
`CUBLAS_COMPUTE_32F, CUBLAS_DEFAULT_MATH, CUBLAS_GEMM_DEFAULT, CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_STATUS_SUCCESS,
CUDA_R_16BF, CUDA_R_16F, CUDA_R_32F`. The dsv41 HIP build also force-includes `include/strata/dsv41/hip_extra.h`,
which adds the two names the shim must carry itself because SYCL has no force-include for them:
`CUBLAS_GEMM_DEFAULT_TENSOR_OP` (:45-46) and `cudaDataType` (:48-50).

Verified API facts the body builds on: `dpct::blas::descriptor` with `set_queue(dpct::cs::queue_ptr)` and
`get_math_mode()` (`sycl/include/dpct/blas_utils.hpp:296-301`); `dpct::blas::gemm(descriptor_ptr, trans, trans, m, n,
k, alpha, a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc, compute_type|library_data_t)` (:920) dispatching
type combinations over `oneapi::mkl::blas` (bf16 x bf16 -> float read at :1000-1010); `library_data_t` enumerators
`real_float, real_half, real_bfloat16` (`sycl/include/dpct/lib_common_utils.hpp:54-60`); `cudaStream_t` is
`dpct::queue_ptr` in the port (`sycl/include/compat/cuda_runtime.h:12`). Everything below is the proposed shape,
not a file that exists.

```cpp
// sycl/include/sycl_compat/cublas_v2.h - the cuBLAS surface src/dsv41 uses, over oneMKL (dpct::blas),
// mirroring include/strata/hip_compat/cublas_v2.h name for name and argument for argument.
#pragma once
#include <cstdint>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <sycl/sycl.hpp>

// --- types (the HIP shim maps these to hipblas/rocblas types; here they are the shim's own) ---
using cublasStatus_t = int;
constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
// every call site compares against CUBLAS_STATUS_SUCCESS only; any failure is any other value

struct cublasContext {
    dpct::blas::descriptor desc;   // set_queue(dpct::cs::queue_ptr) at cublasSetStream
};
using cublasHandle_t = cublasContext*;   // CUDA semantics: may be null-checked by callers (:2565)

enum cudaDataType { CUDA_R_16F, CUDA_R_16BF, CUDA_R_32F };        // hip_extra.h's cudaDataType role
enum cublasComputeType_t { CUBLAS_COMPUTE_32F };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT_TENSOR_OP };   // accepted, ignored
enum cublasOperation_t { CUBLAS_OP_N, CUBLAS_OP_T };              // -> oneapi::mkl::transpose::{nontrans,trans}
constexpr int CUBLAS_DEFAULT_MATH = 0;

// --- handle lifecycle ---
inline cublasStatus_t cublasCreate(cublasHandle_t* h);            // *h = new cublasContext
inline cublasStatus_t cublasDestroy(cublasHandle_t h);            // delete h
inline cublasStatus_t cublasSetStream(cublasHandle_t h, cudaStream_t s);   // h->desc.set_queue(s)
inline cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, std::size_t);  // no-op, SUCCESS (dpct owns scratch)
inline cublasStatus_t cublasSetMathMode(cublasHandle_t h, int mode);
    // mirror the HIP shim's rocblas branch: SUCCESS for CUBLAS_DEFAULT_MATH only, else CUBLAS_STATUS_* failure

// --- the math: column-major, identical m/n/k/lda/ldb/ldc to the cuBLAS calls ---
inline cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                                   int m, int n, int k, const void* alpha, const void* a, cudaDataType at, int lda,
                                   const void* b, cudaDataType bt, int ldb, const void* beta,
                                   void* c, cudaDataType ct, int ldc,
                                   cublasComputeType_t compute, cublasGemmAlgo_t algo);
    // -> dpct::blas::gemm(&h->desc, trans(ta), trans(tb), m, n, k, alpha, a, lib(at), lda,
    //                      b, lib(bt), ldb, beta, c, lib(ct), ldc, dpct::compute_type::f32)
    //    lib: CUDA_R_16F->real_half, CUDA_R_16BF->real_bfloat16, CUDA_R_32F->real_float
    //    (verify the compute_type enumerator spelling at compile time: the ct parameter is
    //     std::variant<compute_type, library_data_t>, blas_utils.hpp:920-930)
inline cublasStatus_t cublasSgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                                  int m, int n, int k, const float* alpha, const float* a, int lda,
                                  const float* b, int ldb, const float* beta, float* c, int ldc);
    // -> dpct::blas::gemm(... real_float ...) or oneapi::mkl::blas::column_major::gemm(h->desc.get_queue(), ...)
inline cublasStatus_t cublasSgemmStridedBatched(...);             // -> dpct::blas::gemm_batch (surface parity;
inline cublasStatus_t cublasGemmStridedBatchedEx(...);            //   Qwen3.6 calls these, ds4_model does not)
```

Notes the executor must hold:
- Failure mapping: catch `oneapi::mkl::exception` and return a non-success `cublasStatus_t`; never let an exception
  cross into upstream code (`sycl/include/dpct/detail/lapack_utils_detail.hpp:16` is the repo's catch-and-map
  precedent). The call sites only test `!= CUBLAS_STATUS_SUCCESS`.
- The algo argument is accepted and ignored (the HIP shim's rocblas branch does the same, `hip_compat/cublas_v2.h:39`
  passes `rocblas_gemm_algo_standard`); `CUBLAS_COMPUTE_32F` means FP32 scaling/accumulation, which is what the
  dpct compute-type parameter selects.
- `cublasSgemmStridedBatched` / `cublasGemmStridedBatchedEx` exist in the HIP shim's `>= 1` branch only
  (`hip_compat/cublas_v2.h:58-59`); mirroring "the same call names" means providing them, but todo 4's acceptance
  does not exercise them (no ds4 call site).
- Include resolution: `src/dsv41/ds4_model.cu:32` says `#include <cublas_v2.h>`, so the shim directory itself (not
  its parent) goes on the include path, ahead of any fetched CUDA header dir (the parity recipe passes one:
  `sycl/src/dsv41/ds4_parity.cpp:12-13`). The HIP build does exactly this with `include/strata/hip_compat` "first on
  the include path" (`CMakeLists.txt:159-163`).

## 3. What `sycl/CMakeLists.txt` adds under `STRATA_SYCL_DSV41`

Today (read here, guard block after the comment "the DeepSeek quant kernels (dsv41), opt-in"): the option
`STRATA_SYCL_DSV41` is **OFF by default**, `strata_resolve(_dsv41 src/dsv41/ds4_tq.cu src/dsv41/ds4_gguf.cu
src/dsv41/ds4_iq.cu)` maps each `.cu` to its `sycl/src/<same>.dp.cpp` (FATAL_ERROR if the copy is missing; `.cpp`
and port-only files are taken as-is or from `STRATA_ROOT`), `strata_dsv41` is a STATIC library with include dirs
`third_party/ggml` and `sycl/include/compat` (both PRIVATE) linking `strata_includes`. **No MKL link exists on this
target today.** oneMKL is linked in exactly one place: `strata_prefill` gets
`target_compile_options(... -qmkl=sequential)` and `target_link_options(... -qmkl=sequential)` above the comment
"cuBLAS -> oneMKL through dpct::blas (needs the MKL SYCL BLAS domain)".

Additions, all inside `if(STRATA_SYCL_DSV41)` / the parity section (parity loop is the `STRATA_SYCL_PARITY` block
whose `enable_testing()` runs first - the plan's reference points at it):

```cmake
if(STRATA_SYCL_DSV41)
  strata_resolve(_dsv41 src/dsv41/ds4_tq.cu src/dsv41/ds4_gguf.cu src/dsv41/ds4_iq.cu)
  # todo 4: the gemv/softmax SYCL kernels (new hand-written TU; upstream has no ds4_gemv.cu, so no strata_resolve)
  list(APPEND _dsv41 ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41/ds4_gemv.dp.cpp)
  add_library(strata_dsv41 STATIC ${_dsv41})
  target_include_directories(strata_dsv41 PRIVATE
    ${STRATA_ROOT}/third_party/ggml
    ${CMAKE_CURRENT_SOURCE_DIR}/include/compat
    ${CMAKE_CURRENT_SOURCE_DIR}/include/sycl_compat)   # #include <cublas_v2.h> resolves here (before any CUDA header dir)
  target_link_libraries(strata_dsv41 PUBLIC strata_includes)
  target_compile_options(strata_dsv41 PUBLIC -qmkl=sequential)   # the shim's oneMKL BLAS domain (strata_prefill precedent)
  target_link_options(strata_dsv41 PUBLIC -qmkl=sequential)
endif()
```

Parity registration (inside `if(STRATA_SYCL_PARITY)` after its `enable_testing()`, nested in
`if(STRATA_SYCL_DSV41)`, or in the block above with its own `enable_testing()` call - the tests must be registered
in a directory where testing is enabled):

```cmake
  if(STRATA_SYCL_DSV41)
    # ds4_parity: the device dequant parity (sycl/src/dsv41/ds4_parity.cpp), no CLI args, exit 0 on MATCH
    add_executable(ds4_parity ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41/ds4_parity.cpp)
    target_include_directories(ds4_parity PRIVATE ${STRATA_ROOT}/third_party/ggml
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/include/compat)
    target_link_libraries(ds4_parity PRIVATE strata_dsv41)
    add_test(NAME ds4_parity COMMAND ds4_parity)          # ctest -R ds4 matches the name
    # todo 4's gemv/softmax parity binary (wrong leading dimension -> outside tolerance -> exit non-zero)
    add_executable(ds4_gemv_parity ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41/ds4_gemv_parity.cpp)
    target_link_libraries(ds4_gemv_parity PRIVATE strata_dsv41)
    add_test(NAME ds4_gemv_parity COMMAND ds4_gemv_parity)
  endif()
```

Why the test targets repeat the include dirs: `strata_dsv41`'s `third_party/ggml` and `include/compat` are PRIVATE,
so they do not propagate to consumers. `ds4_parity.cpp` includes `ds4_iq.dp.hpp` and `ds4_numerics.dp.hpp` (its own
directory) and was hand-built with `-I third_party/ggml -I sycl/include/compat` (its header comment). Making those
two dirs PUBLIC on `strata_dsv41` instead is equivalent; pick one.

Facts behind the snippet: `ds4_parity` main takes no arguments (`sycl/src/dsv41/ds4_parity.cpp:181`), prints one
line per IQ type and ends `MATCH: ...`, exit 0 (:255 area); it only calls `iq::dequant_rows`, so linking
`strata_dsv41` covers it. Test names starting `ds4` satisfy the plan's `ctest -R ds4` gate, which is also todo 4's
acceptance command (`cmake -S sycl -B build-sycl-dsv41 -G Ninja -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON`,
build green, `ctest -R ds4 --output-on-failure` on the B60). `ds4_model.cu` is not migrated until todo 5, so the
shim's 11 call shapes need a compile exercise inside todo 4 (see the smoke TU below).

## 4. Files a todo-4 executor may touch

| file | action | note |
|------|--------|------|
| `sycl/include/sycl_compat/cublas_v2.h` | create | the shim (section 2) |
| `sycl/src/dsv41/ds4_gemv.dp.cpp` (and `ds4_gemv.dp.hpp` if it wants one) | create | the three gemv kernels (`ds4_kernels.cu:372/:398/:438`) and the vision softmax (`ds4_vision.cu:126`) as SYCL kernels, entry points mirroring `gemv_f16` / `gemv_f16_rows` / `gemv_f32` (`ds4_kernels.cuh:80-88`) |
| `sycl/src/dsv41/ds4_gemv_parity.cpp` | create | the gemv/softmax parity binary (CPU reference, device run, exit non-zero outside tolerance) |
| `sycl/src/dsv41/ds4_shim_smoke.dp.cpp` (recommended) | create | tiny TU that compiles the 11 call shapes through the shim (handle lifecycle + 2 `cublasGemmEx` + `cublasSgemm`) so the shim is exercised before `ds4_model.dp.cpp` exists in todo 5; link it into `ds4_gemv_parity` or its own target |
| `sycl/CMakeLists.txt` | edit | only inside `if(STRATA_SYCL_DSV41)` and the parity registration area (section 3) |
| `sycl/src/dsv41/ds4_parity.cpp` | edit (comment only) | replace the hand-build recipe at lines 12-13 (the line the todo names is :12) with a pointer to the CMake target: "Built by the strata_sycl build (`-DSTRATA_SYCL_DSV41=ON`), target `ds4_parity`; `ctest -R ds4` runs it." Do not touch its code |

Must NOT touch: `src/**` and `include/strata/**` (upstream CUDA and the HIP shim stay byte-identical),
`CMakeLists.txt` at the top level, the default build (`STRATA_SYCL_DSV41` stays OFF), and no speed claim may be
written anywhere. The plan also records that lane 1 owns `ds4_parity.cpp`; todo 4 is the exception for the header
comment above, nothing else in that file.

### Recorded instruction (not done here)

`sycl/src/dsv41/ds4_parity.cpp:12-13` carries the stale hand-build recipe ("Build (this PC, 2026-10-24): icpx
-fsycl -std=c++20 -O2 -I sycl/include -I include -I third_party/ggml -I sycl/include/compat <cuda-header dir>
ds4_parity.cpp ds4_iq.dp.cpp"). Todo 4 replaces those two comment lines with the CMake target reference once the
target exists. This session did not edit the file.

## 5. oneMKL on this machine (probe run 2026-10-25)

The exact command from the ask was run verbatim:

```
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ls /opt/intel/oneapi/mkl/latest/include/ 2>&1 | head -20; echo RC=$?'
```

It printed the MKL include listing (`fftw, i_malloc.h, intel64, mkl, mkl.fi, mkl.h, mkl_blas.h, ...`) and `RC=0`.
Caveat on "the real exit code": as written, `RC=$?` reports the exit of `head`, not of `ls`. A corrected probe
capturing `ls`'s own status gives `LS_RC=0` (and the lib directory `LIB_LS_RC=0`), so the real answer is the same:
**MKL headers are present and `ls` exits 0.**

- `/opt/intel/oneapi/mkl/latest/include/` exists with `mkl.h`, `mkl_blas.h`, `mkl_types.h` and the SYCL domain:
  `oneapi/mkl/blas.hpp`, `oneapi/mkl/blas/`, `oneapi/mkl/lapack.hpp`, `oneapi/mkl/rng.hpp`, `oneapi/mkl/vm.hpp`.
- `/opt/intel/oneapi/mkl/latest/lib/` carries `libmkl_sycl.so`, `libmkl_sycl_blas.so(.6)` and the rest of the
  oneMKL libraries.
- Compiler: `/opt/intel/oneapi/compiler/2026.2/bin/icpx`, Intel oneAPI DPC++/C++ 2026.2.0 (matches
  `docs/SYCL_DSV41_PLAN2.md:17`, which also notes the self-built llvm tree has no oneMKL and is not the default).
