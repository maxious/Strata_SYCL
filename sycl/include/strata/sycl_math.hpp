// include/strata/sycl_math.hpp - the SYCL port's device math that dpct emulates but the hardware has.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>

namespace strata {
// CUDA's __dp4a(int, int, int): four signed byte products summed into c. One place to change it: the byte
// unpack + multiply-add form is what IGC pattern-matches into the DP4A instruction (sycl::ext::oneapi::dot_acc is
// the same emulation and its header defines non-inline functions, which breaks the link across TUs).
template <typename A, typename B, typename C>
inline int32_t dp4a(A a, B b, C c) {
    return dpct::dp4a((int32_t) a, (int32_t) b, (int32_t) c);
}

/// CUDA's __shfl_* onto the standard SYCL 2020 sub-group functions. dpct migrates each __shfl_*_sync to the
/// MASKED dpct::experimental::*_sub_group(0xFFFFFFFFu, sg, x, delta); on the Intel path that still runs the
/// ext::oneapi::experimental entangle/chunked_partition wrappers even with a full mask and full-width sub-group,
/// where the masked form is exact the standard function. Kernels replace the experimental call with these.
template <typename T>
inline T sub_group_shift_left(sycl::sub_group sg, T x, unsigned delta) {
    return sycl::shift_group_left(sg, x, delta);
}
template <typename T>
inline T sub_group_shift_right(sycl::sub_group sg, T x, unsigned delta) {
    return sycl::shift_group_right(sg, x, delta);
}
template <typename T>
inline T sub_group_permute_xor(sycl::sub_group sg, T x, unsigned delta) {
    return sycl::permute_group_by_xor(sg, x, delta);
}
template <typename T>
inline T sub_group_select(sycl::sub_group sg, T x, int lane) {
    return sycl::select_from_group(sg, x, lane);
}
}  // namespace strata
