// Port-only stand-in for CUDA's cuda_runtime.h, for compiling the converted dsv41 files (sycl/src/dsv41/)
// on a machine with no CUDA toolkit. It is reached only by the opt-in strata_dsv41 target
// (STRATA_SYCL_DSV41=ON, docs/SYCL_DSV41_PLAN.md): include/compat sits on that target's include path alone,
// so nothing else in the build sees it. The converted code carries dpct's queue_ptr everywhere it used a
// stream; an unconverted header (strata/dsv41/exl3.hpp) still names cudaStream_t in its declarations, and
// this is that name. Strata's CUDA sources never include this file - nvcc takes the real header.
#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

typedef dpct::queue_ptr cudaStream_t;
typedef int cudaError_t;
constexpr cudaError_t cudaErrorSuccess = 0;
constexpr cudaError_t cudaSuccess = 0;

// CUDA-only qualifiers nvcc understands and icpx does not; the unconverted ggml headers use them on
// declarations only, so empty is right for the SYCL build.
#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
