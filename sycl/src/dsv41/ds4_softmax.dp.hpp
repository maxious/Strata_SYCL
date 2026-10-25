// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// SYCL copy of src/dsv41/ds4_vision.cu's softmax_kernel (:126, the vision tower's P = softmax(S); converted by
// SYCLomatic, hand-fixed in ds4_softmax.dp.cpp). rows of up to 10240 keys take the register-tile kernel; longer rows
// take the scalar fallback the CUDA kernel did not have (it read only the first 10240 keys). Both keep the original's
// arithmetic: max, exp(v - max), sum, BF16 out (ds4_numerics.dp.hpp's bf16_rn).
#pragma once

#include <cstdint>

#include <dpct/dpct.hpp>

namespace strata::ds4 {

void softmax(const float* s, int n, int rows, uint16_t* p, dpct::queue_ptr stream);

}  // namespace strata::ds4
