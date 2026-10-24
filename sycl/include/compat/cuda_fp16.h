// Port-only stand-in for CUDA's cuda_fp16.h, same scope as compat/cuda_runtime.h beside it: only the opt-in
// strata_dsv41 target (STRATA_SYCL_DSV41=ON) has include/compat on its path. It gives the fp16 names the
// converted dsv41 files and ggml-common.h's CUDA-decl branch still use (half, __half, __half2float), backed by
// sycl::half. The uint16_t overload carries the raw bits, which is how the engine's f16 activations travel.
#pragma once

#include <cstring>
#include <sycl/sycl.hpp>

using half = sycl::half;
using __half = sycl::half;
struct half2 { sycl::half x, y; };
struct __half2 { sycl::half x, y; };

inline float __half2float(__half h) { return (float)h; }
inline float __half2float(uint16_t bits) {
    sycl::half h;
    std::memcpy(&h, &bits, 2);
    return (float)h;
}
inline __half __float2half(float f) { return (__half)f; }
