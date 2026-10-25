// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// SYCL copy of src/dsv41/ds4_kernels.cu's gemv launchers (gemv_f16_kernel :372, gemv_f16_rows_kernel :398,
// gemv_f32_kernel :438; converted by SYCLomatic, hand-fixed in ds4_gemv.dp.cpp). The shapes keep the CUDA kernels'
// contract: W holds n rows of k; x rows are exactly k apart for gemv_f16 and ldx apart for the others; y rows are n
// apart for gemv_f16 and ldy apart for the others. Every product is taken in the original's order, so a row's bits
// match the CUDA kernel's.
#pragma once

#include <cstddef>
#include <cstdint>

#include <dpct/dpct.hpp>

namespace strata::ds4 {

void gemv_f16(const uint16_t* W, const float* x, int n, int k, int T, float alpha, float* y, dpct::queue_ptr s);
void gemv_f16_rows(const uint16_t* W, const float* x, size_t ldx, int n, int k, int T, float alpha, float* y,
                   size_t ldy, dpct::queue_ptr s);
void gemv_f32(const float* W, const float* x, int n, int k, int T, float* y, int ldy, dpct::queue_ptr s);

}  // namespace strata::ds4
