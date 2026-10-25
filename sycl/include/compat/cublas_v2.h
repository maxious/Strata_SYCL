// Port-only stand-in for CUDA's cublas_v2.h, same scope as compat/cuda_runtime.h beside it: only the opt-in
// strata_dsv41 target (STRATA_SYCL_DSV41=ON) has include/compat on its path. It mirrors the call names and argument
// shapes of include/strata/hip_compat/cublas_v2.h over oneMKL (oneapi::mkl::blas::gemm, column-major, on a
// sycl::queue), covering exactly what src/dsv41/ds4_model.cu uses: the include, a handle type, cublasCreate and
// cublasSetStream (:2884), cublasGemmEx (the two FP16-in / FP32-out shapes at :2910 and :2924), cublasSgemm (:2937),
// cublasDestroy (:1245, :2565), and a status type the three error strings at :2914, :2928 and :2939 can format with
// (int) st. cublasSetWorkspace carries the engine's explicit workspace contract (the HIP compat header preserves it
// too) and is this shim's staging buffer's first choice. The strided-batched calls and cublasSetMathMode have no
// caller in src/dsv41 and are deliberately not stubbed: extend here when one appears.
#pragma once

#include <sycl/sycl.hpp>
// The supported umbrella: oneapi::mkl::blas.hpp declares column_major::gemm by including
// oneapi/mkl/blas/usm_decls.hpp inside oneapi::mkl::blas (ONEMKL_DECLARE_GEMM at usm_decls.hpp:22, the
// (sycl::half, sycl::half, float, float) instantiation at :43). A direct include of usm_decls.hpp lands that
// macro outside its namespaces and does not compile (measured 2026-10-25: 20 value_or_pointer errors), so the
// declaration is reached through the umbrella and MKL_GEMM_DECL_FILE is recorded in the lane's evidence log.
#include <oneapi/mkl/blas.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <new>

#include "strata/sycl_queue.hpp"   // q_of; device_offset_end / intel_gpu_gen for gemm.dp.cpp's 4 GiB staging

enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
};
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };
enum cudaDataType { CUDA_R_32F = 0, CUDA_R_16F = 2, CUDA_R_16BF = 14 };
enum cublasComputeType_t { CUBLAS_COMPUTE_16F = 0, CUBLAS_COMPUTE_32F = 1 };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT = -1, CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99 };

struct cublasContext {
    sycl::queue* q = nullptr;      // cublasSetStream's queue; unset: the default in-order queue (strata::q_of)
    void* workspace = nullptr;     // cublasSetWorkspace's buffer: the engine's explicit workspace contract
    size_t workspace_bytes = 0;
    uint8_t* stage = nullptr;      // a fresh allocation for operands past 4 GiB (gemm.dp.cpp's staging)
    size_t stage_bytes = 0;
    sycl::event last;              // the previous GEMM: staging is reused only after it has finished
};
typedef cublasContext* cublasHandle_t;

inline cublasStatus_t cublasCreate(cublasHandle_t* h) {
    if (h == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    *h = new (std::nothrow) cublasContext();
    return *h != nullptr ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_ALLOC_FAILED;
}

inline cublasStatus_t cublasDestroy(cublasHandle_t h) {
    if (h == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    h->last.wait();
    if (h->stage != nullptr) sycl::free(h->stage, *strata::q_of(h->q));
    delete h;
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasSetStream(cublasHandle_t h, sycl::queue* q) {
    if (h == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->q = q;
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasSetWorkspace(cublasHandle_t h, void* workspace, size_t bytes) {
    if (h == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->workspace = workspace;
    h->workspace_bytes = bytes;
    return CUBLAS_STATUS_SUCCESS;
}

namespace cublas_shim_detail {

inline oneapi::mkl::transpose mk_trans(cublasOperation_t t) {
    return t == CUBLAS_OP_N ? oneapi::mkl::transpose::nontrans
                            : t == CUBLAS_OP_T ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::conjtrans;
}

// gemm.dp.cpp's 4 GiB staging, kept here: oneMKL's GEMM reads an operand's last partial tile from the wrong place
// when the operand lies more than 4 GiB into its allocation (Arc A770, oneMKL 2026.1 - measured in gemm.dp.cpp;
// deliberately not applied on Battlemage, where 4K and 49K prompts with operands at 18 GiB were byte-identical with
// and without staging). STRATA_SYCL_GEMM_NO_STAGE forces the direct call; STRATA_SYCL_GEMM_STAGE=1 stages everywhere.
inline bool staging_off(sycl::queue& q) {
    const char* force = std::getenv("STRATA_SYCL_GEMM_STAGE");
    static const bool no_stage =
        std::getenv("STRATA_SYCL_GEMM_NO_STAGE") != nullptr ||
        (strata::intel_gpu_gen(q.get_device()) == strata::IntelGpuGen::Battlemage && !(force && force[0] == '1'));
    return no_stage;
}

// Any operand past 4 GiB is copied into one staging block ([A][B]) at offset 0 - the handle's explicit workspace
// when it fits, else a grown device allocation of the handle's own - and the GEMM runs on those pointers. Returns
// false when neither operand needs staging (the common, correct case) so the caller runs the direct GEMM.
template <class T>
inline bool stage_operands(cublasHandle_t h, sycl::queue& q, const T* a, size_t a_bytes, const T* b, size_t b_bytes,
                           const T** a_out, const T** b_out) {
    constexpr size_t k4G = 4ull << 30;
    const bool far_a = strata::device_offset_end(a, a_bytes) > k4G;
    const bool far_b = strata::device_offset_end(b, b_bytes) > k4G;
    *a_out = a;
    *b_out = b;
    if (staging_off(q) || (!far_a && !far_b)) return false;
    const size_t need = (far_a ? a_bytes : 0) + (far_b ? b_bytes : 0);
    h->last.wait();
    uint8_t* dst = (h->workspace != nullptr && h->workspace_bytes >= need) ? (uint8_t*) h->workspace : h->stage;
    if (dst != nullptr && dst == h->stage && h->stage_bytes < need) {
        sycl::free(h->stage, q);
        h->stage = nullptr;
        h->stage_bytes = 0;
        dst = nullptr;
    }
    if (dst == nullptr) {
        h->stage = (uint8_t*) sycl::malloc_device(need, q);
        h->stage_bytes = h->stage != nullptr ? need : 0;
        dst = h->stage;
    }
    if (dst == nullptr) {
        std::fprintf(stderr, "cublas shim: WARNING: a GEMM operand lies past 4 GiB and the %zu MiB staging buffer "
                             "could not be allocated; the GEMM runs directly and may be wrong on this card\n",
                     need >> 20);
        return false;
    }
    uint8_t* at = dst;
    if (far_a) {
        q.memcpy(at, a, a_bytes);
        *a_out = (const T*) at;
        at += a_bytes;
    }
    if (far_b) {
        q.memcpy(at, b, b_bytes);
        *b_out = (const T*) at;
    }
    return true;
}

template <class T, class TC>
inline cublasStatus_t gemm_run(cublasHandle_t h, cublasOperation_t trans_a, cublasOperation_t trans_b, int m, int n,
                               int k, float alpha, const T* a, int lda, const T* b, int ldb, float beta, TC* c,
                               int ldc) {
    if (h == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m <= 0 || n <= 0 || k <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (lda < (trans_a == CUBLAS_OP_N ? m : k) || ldb < (trans_b == CUBLAS_OP_N ? k : n) || ldc < m)
        return CUBLAS_STATUS_INVALID_VALUE;
    sycl::queue& q = *strata::q_of(h->q);
    const size_t a_bytes = (size_t) lda * (size_t) (trans_a == CUBLAS_OP_N ? k : m) * sizeof(T);
    const size_t b_bytes = (size_t) ldb * (size_t) (trans_b == CUBLAS_OP_N ? n : k) * sizeof(T);
    const T *A = a, *B = b;
    stage_operands<T>(h, q, a, a_bytes, b, b_bytes, &A, &B);
    h->last = oneapi::mkl::blas::column_major::gemm(q, mk_trans(trans_a), mk_trans(trans_b), m, n, k, alpha, A, lda,
                                                    B, ldb, beta, c, ldc, oneapi::mkl::blas::compute_mode::unset);
    return CUBLAS_STATUS_SUCCESS;
}

}  // namespace cublas_shim_detail

// ds4_model.cu's two FP16-in / FP32-out shapes (:2910, :2924) and nothing else: other type combinations return
// CUBLAS_STATUS_NOT_SUPPORTED, which the call sites' (int) st error strings report as-is.
inline cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t trans_a, cublasOperation_t trans_b, int m,
                                   int n, int k, const void* alpha, const void* a, cudaDataType a_type, int lda,
                                   const void* b, cudaDataType b_type, int ldb, const void* beta, void* c,
                                   cudaDataType c_type, int ldc, cublasComputeType_t compute_type,
                                   cublasGemmAlgo_t /*algo - oneMKL takes no algorithm choice*/) {
    if (compute_type != CUBLAS_COMPUTE_32F || a_type != CUDA_R_16F || b_type != CUDA_R_16F || c_type != CUDA_R_32F)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    if (alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    try {
        return cublas_shim_detail::gemm_run<sycl::half, float>(
            h, trans_a, trans_b, m, n, k, *static_cast<const float*>(alpha), static_cast<const sycl::half*>(a), lda,
            static_cast<const sycl::half*>(b), ldb, *static_cast<const float*>(beta), static_cast<float*>(c), ldc);
    } catch (const std::exception&) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    } catch (...) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}

// ds4_model.cu:2937 (the mHC mixes): CUBLAS_COMPUTE_32F, all FP32.
inline cublasStatus_t cublasSgemm(cublasHandle_t h, cublasOperation_t trans_a, cublasOperation_t trans_b, int m,
                                  int n, int k, const float* alpha, const float* a, int lda, const float* b, int ldb,
                                  const float* beta, float* c, int ldc) {
    if (alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    try {
        return cublas_shim_detail::gemm_run<float, float>(h, trans_a, trans_b, m, n, k, *alpha, a, lda, b, ldb,
                                                          *beta, c, ldc);
    } catch (const std::exception&) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    } catch (...) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}
