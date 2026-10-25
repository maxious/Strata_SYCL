// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// SYCL copy of src/dsv41/ds4_vision.cu's softmax_kernel (:126) and its block_reduce (:57-80). The kernel bodies are
// SYCLomatic's conversion of the originals; the hand fixes are the launchers (dpct leaves the <<<grid, block>>>
// syntax and the hoisted __shared__ arrays behind), the [[sycl::reqd_sub_group_size(32)]] the butterfly needs, the
// exp call (dpct writes sycl::native::exp, which a 2026-10-24 probe on this PC measured up to 8 F32 ulps off host
// expf where sycl::exp matched bit for bit), and a scalar fallback for rows past the register tile: the original
// read only a row's first kSmT * kSmPer = 10240 keys, which is a wrong softmax rather than a fallback, so longer
// rows take a per-row pass.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <cfloat>
#include <cstdint>

#include "ds4_numerics.dp.hpp"
#include "ds4_softmax.dp.hpp"

namespace strata::ds4 {
namespace {

constexpr int kSmT = 512, kSmPer = 20;   // softmax: a row of up to 10240 keys in registers

__dpct_inline__ uint16_t bfr(float f) { return bf16_rn(f); }

float block_reduce(float v, bool is_max, float* red, float* tot) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    // the butterfly walks the width the launch actually bound (the wrappers pin 32; the width is read from the
    // sub-group so the reduce is right for whatever power-of-two width the device gives this kernel)
    const int sgw = (int) sycl::ext::oneapi::this_work_item::get_sub_group().get_local_range()[0];
    for (int o = sgw / 2; o > 0; o >>= 1) {
        const float u = dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
        v = is_max ? sycl::fmax(v, u) : v + u;
    }
    const int lane = (int) item_ct1.get_local_id(2) % sgw, w = (int) item_ct1.get_local_id(2) / sgw,
              nw = ((int) item_ct1.get_local_range(2) + sgw - 1) / sgw;
    item_ct1.barrier(sycl::access::fence_space::local_space);   // (an earlier call's readers of red / tot are done)
    if (lane == 0) red[w] = v;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (w == 0) {
        v = lane < nw ? red[lane] : (is_max ? -FLT_MAX : 0.0f);
        for (int o = sgw / 2; o > 0; o >>= 1) {
            const float u = dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
            v = is_max ? sycl::fmax(v, u) : v + u;
        }
        if (lane == 0) *tot = v;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    return *tot;
}

void softmax_kernel(const float* __restrict__ s, int n, uint16_t* __restrict__ p, float* red, float* tot) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float* sr = s + (size_t) item_ct1.get_group(2) * n;
    uint16_t* pr = p + (size_t) item_ct1.get_group(2) * n;
    float v[kSmPer];
    float m = -FLT_MAX;
#pragma unroll
    for (int k = 0; k < kSmPer; ++k) {
        const int e = item_ct1.get_local_id(2) + k * kSmT;
        v[k] = e < n ? sr[e] : -FLT_MAX;
        m = sycl::fmax(m, v[k]);
    }
    m = block_reduce(m, true, red, tot);
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kSmPer; ++k) {
        const int e = item_ct1.get_local_id(2) + k * kSmT;
        v[k] = e < n ? sycl::exp(v[k] - m) : 0.0f;
        sum += v[k];
    }
    const float inv = 1.0f / block_reduce(sum, false, red, tot);
#pragma unroll
    for (int k = 0; k < kSmPer; ++k) {
        const int e = item_ct1.get_local_id(2) + k * kSmT;
        if (e < n) pr[e] = bfr(v[k] * inv);
    }
}

void softmax_scalar_kernel(const float* __restrict__ s, int n, uint16_t* __restrict__ p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float* sr = s + (size_t) item_ct1.get_group(2) * n;
    uint16_t* pr = p + (size_t) item_ct1.get_group(2) * n;
    float m = -FLT_MAX;
    for (int i = 0; i < n; ++i) m = sycl::fmax(m, sr[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += sycl::exp(sr[i] - m);
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) pr[i] = bfr(sycl::exp(sr[i] - m) * inv);
}

}  // namespace

class softmax_wrapper {
    const float* s;
    int n;
    uint16_t* p;
    sycl::local_accessor<float, 1> red;
    sycl::local_accessor<float, 1> tot;

  public:
    softmax_wrapper(const float* s_, int n_, uint16_t* p_, sycl::local_accessor<float, 1> red_,
                    sycl::local_accessor<float, 1> tot_)
        : s(s_), n(n_), p(p_), red(red_), tot(tot_) {}
    [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<3>) const {
        softmax_kernel(s, n, p, red.get_multi_ptr<sycl::access::decorated::no>().get(),
                       tot.get_multi_ptr<sycl::access::decorated::no>().get());
    }
};

class softmax_scalar_wrapper {
    const float* s;
    int n;
    uint16_t* p;

  public:
    softmax_scalar_wrapper(const float* s_, int n_, uint16_t* p_) : s(s_), n(n_), p(p_) {}
    void operator()(sycl::nd_item<3>) const { softmax_scalar_kernel(s, n, p); }
};

void softmax(const float* s, int n, int rows, uint16_t* p, dpct::queue_ptr stream) {
    if (n <= 0 || rows <= 0) return;
    if (n <= kSmT * kSmPer) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> red(sycl::range<1>(32), cgh);
            sycl::local_accessor<float, 1> tot(sycl::range<1>(1), cgh);
            cgh.parallel_for<dpct_kernel_name<softmax_wrapper>>(
                sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) rows * kSmT), sycl::range<3>(1, 1, kSmT)),
                softmax_wrapper(s, n, p, red, tot));
        });
    } else {
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for<dpct_kernel_name<softmax_scalar_wrapper>>(
                sycl::nd_range<3>(sycl::range<3>(1, 1, rows), sycl::range<3>(1, 1, 1)),
                softmax_scalar_wrapper(s, n, p));
        });
    }
}

}  // namespace strata::ds4
