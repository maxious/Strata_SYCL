// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// SYCL copy of src/dsv41/ds4_numerics.cuh, converted by SYCLomatic and hand-fixed.
// The CUDA build uses the bf16/fp8 intrinsics; the HIP build and this SYCL copy use the plain arithmetic the
// original file documents as giving the same bits: BF16 round to nearest even; E4M3 (OCP "fn": no infinities,
// max 448) decoded exactly, rounded to nearest even and saturated.
#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>

namespace strata::ds4 {

/// F32 -> BF16 bits, round to nearest even (a NaN stays a NaN).
__dpct_inline__ uint16_t bf16_rn(float f) {
    const uint32_t u = sycl::bit_cast<uint32_t>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/// An E4M3 code's value (exact in F32).
__dpct_inline__ float e4m3_value(uint8_t q) {
    const uint32_t e = (q >> 3) & 15u, m = q & 7u;
    float v;
    if (e == 15u && m == 7u) v = sycl::bit_cast<float>(0x7fc00000u);                    // NaN
    else if (e == 0u) v = (float) m * 0.001953125f;                                      // m 2^-9
    else v = sycl::bit_cast<float>(((e + 120u) << 23) | (m << 20));                     // (1 + m / 8) 2^(e - 7)
    return (q & 0x80u) ? -v : v;
}

/// v rounded to the nearest E4M3 value (ties to even), saturated at +-448.
__dpct_inline__ float e4m3_round(float v) {
    const float a = sycl::fabs(v);
    if (!(a == a)) return v;
    float r;
    if (a >= 448.0f) r = 448.0f;
    else {
        const int e = (int) ((sycl::bit_cast<uint32_t>(a) >> 23) & 0xffu) - 127;        // a's binade (a > 0)
        const float step = e < -6 ? 0.001953125f : sycl::bit_cast<float>((uint32_t) (e - 3 + 127) << 23);
        r = a == 0.0f ? 0.0f : sycl::rint(a / step) * step;                             // (a power of two: exact)
        r = sycl::fmin(r, 448.0f);
    }
    return v < 0.0f ? -r : r;
}

__dpct_inline__ float bf16f(uint16_t b) {
  return sycl::bit_cast<float>((uint32_t)b << 16);
}

__dpct_inline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}
__dpct_inline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(
        v, dpct::permute_sub_group_by_xor(
               sycl::ext::oneapi::this_work_item::get_sub_group(), v, o));
    return v;
}

// block-wide sum for blockDim.x <= 1024 (a multiple of 32); every thread gets the result
__dpct_inline__ float block_sum(float v, float *red, float &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int lane = item_ct1.get_local_id(2) & 31,
              w = item_ct1.get_local_id(2) >> 5;
    v = warp_sum(v);
    if (lane == 0) red[w] = v;
    /*
    DPCT1065:76: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
    item_ct1.barrier();
    if (w == 0) {
        float t = lane < (int)(item_ct1.get_local_range(2) >> 5) ? red[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) total = t;
    }
    /*
    DPCT1065:77: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
    item_ct1.barrier();
    const float r = total;
    /*
    DPCT1065:78: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
    item_ct1.barrier();
    return r;
}

// an E2M1 code's value times 2^-14, by the conversion of an FP16 (the code's sign, exponent and mantissa bit are an
// FP16's at bits 15 and 11..9: exact, subnormals too) - the readers' decode: (value 2^-14) (scale 2^14) is value * scale
// exactly, both products exact
__dpct_inline__ float e2m1_val14(uint32_t c) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(
                   (unsigned short)(((c & 8u) << 12) | ((c & 7u) << 9))))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}

}  // namespace strata::ds4
