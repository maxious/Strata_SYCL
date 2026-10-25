// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// SYCL copy of src/dsv41/ds4_kernels.cu's three gemv kernels (gemv_f16_kernel :372, gemv_f16_rows_kernel :398,
// gemv_f32_kernel :438). The kernel bodies are SYCLomatic's conversion of the originals; the hand fixes are the
// launchers (dpct leaves the <<<grid, block>>> syntax behind), the [[sycl::reqd_sub_group_size(32)]] the warp-sum
// butterfly needs (the originals assume 32-lane warps), and a scalar fallback for the shapes the 16-byte tile loads
// cannot take (k not a multiple of 8 / 4, or unaligned rows): the fallback walks the same elements per lane in the
// same order, so both paths land within the parity rule of the CPU reference.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <cstdint>

#include "ds4_gemv.dp.hpp"

namespace strata::ds4 {
namespace {

constexpr int kThreads = 256;
constexpr int kRowsT = 2;

__dpct_inline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}

// ---------------------------------------------------------------- gemv over F16 weights, a warp a row
void gemv_f16_kernel(const uint16_t* __restrict__ W, const float* __restrict__ x, int n, int k, float alpha,
                     float* __restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * (kThreads / 32) + (item_ct1.get_local_id(2) >> 5);
    const int lane = item_ct1.get_local_id(2) & 31, t = item_ct1.get_group(1);
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    const float* xv = x + (size_t) t * k;
    float acc = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const sycl::uint4 q = *(const sycl::uint4*) (w + c);
        const sycl::half2* hp = (const sycl::half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const sycl::float2 f = hp[i].template convert<float, sycl::rounding_mode::automatic>();
            acc += f.x() * xv[c + 2 * i] + f.y() * xv[c + 2 * i + 1];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * n + row] = alpha * acc;
}

void gemv_f16_scalar_kernel(const uint16_t* __restrict__ W, const float* __restrict__ x, int n, int k, float alpha,
                            float* __restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * (kThreads / 32) + (item_ct1.get_local_id(2) >> 5);
    const int lane = item_ct1.get_local_id(2) & 31, t = item_ct1.get_group(1);
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    const float* xv = x + (size_t) t * k;
    float acc = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = c + 2 * i;
            if (j + 1 < k) {
                const float w0 = static_cast<float>(sycl::bit_cast<sycl::half>(w[j]));
                const float w1 = static_cast<float>(sycl::bit_cast<sycl::half>(w[j + 1]));
                acc += w0 * xv[j] + w1 * xv[j + 1];
            } else if (j < k) {
                acc += static_cast<float>(sycl::bit_cast<sycl::half>(w[j])) * xv[j];
            }
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * n + row] = alpha * acc;
}

// ... for T activations (rows ldx apart, 16-byte aligned; outputs ldy apart), kRowsT of them a block (grid y): each
// weight row read once for the block's activations, theirs in float4s - each activation's sum gemv_f16_kernel's (the
// same products in the same order), so its bits too.  (A block for all of them, 16 at most, was slower than a launch
// a token: its accumulators and guards kept the loop from unrolling - 66 against 29 us at 5 tokens on an RTX 5060 Ti,
// this 9)
void gemv_f16_rows_kernel(const uint16_t* __restrict__ W, const float* __restrict__ x, size_t ldx, int n, int k,
                          int T, float alpha, float* __restrict__ y, size_t ldy) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * (kThreads / 32) + (item_ct1.get_local_id(2) >> 5);
    const int lane = item_ct1.get_local_id(2) & 31;
    const int t0 = item_ct1.get_group(1) * kRowsT, nt = sycl::min(kRowsT, T - t0);
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    float acc[kRowsT];
#pragma unroll
    for (int t = 0; t < kRowsT; ++t) acc[t] = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const sycl::uint4 q = *(const sycl::uint4*) (w + c);
        const sycl::half2* hp = (const sycl::half2*) &q;
        float xv[kRowsT][8];
#pragma unroll
        for (int t = 0; t < kRowsT; ++t)
            if (t < nt) {
                const float* xp = x + (size_t) (t0 + t) * ldx + c;
                const sycl::float4 a = *(const sycl::float4*) xp, b = *(const sycl::float4*) (xp + 4);
                xv[t][0] = a.x(), xv[t][1] = a.y(), xv[t][2] = a.z(), xv[t][3] = a.w();
                xv[t][4] = b.x(), xv[t][5] = b.y(), xv[t][6] = b.z(), xv[t][7] = b.w();
            }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const sycl::float2 f = hp[i].template convert<float, sycl::rounding_mode::automatic>();
#pragma unroll
            for (int t = 0; t < kRowsT; ++t)
                if (t < nt) acc[t] += f.x() * xv[t][2 * i] + f.y() * xv[t][2 * i + 1];
        }
    }
#pragma unroll
    for (int t = 0; t < kRowsT; ++t)
        if (t < nt) {
            const float a = warp_sum(acc[t]);
            if (lane == 0) y[(size_t) (t0 + t) * ldy + row] = alpha * a;
        }
}

void gemv_f16_rows_scalar_kernel(const uint16_t* __restrict__ W, const float* __restrict__ x, size_t ldx, int n,
                                 int k, int T, float alpha, float* __restrict__ y, size_t ldy) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * (kThreads / 32) + (item_ct1.get_local_id(2) >> 5);
    const int lane = item_ct1.get_local_id(2) & 31;
    const int t0 = item_ct1.get_group(1) * kRowsT, nt = sycl::min(kRowsT, T - t0);
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    float acc[kRowsT];
#pragma unroll
    for (int t = 0; t < kRowsT; ++t) acc[t] = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = c + 2 * i;
            if (j + 1 >= k && j >= k) continue;
            const float w0 = static_cast<float>(sycl::bit_cast<sycl::half>(w[j]));
            if (j + 1 < k) {
                const float w1 = static_cast<float>(sycl::bit_cast<sycl::half>(w[j + 1]));
#pragma unroll
                for (int t = 0; t < kRowsT; ++t) {
                    const float* xp = x + (size_t) (t0 + t) * ldx;
                    if (t < nt) acc[t] += w0 * xp[j] + w1 * xp[j + 1];
                }
            } else {
#pragma unroll
                for (int t = 0; t < kRowsT; ++t) {
                    const float* xp = x + (size_t) (t0 + t) * ldx;
                    if (t < nt) acc[t] += w0 * xp[j];
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kRowsT; ++t)
        if (t < nt) {
            const float a = warp_sum(acc[t]);
            if (lane == 0) y[(size_t) (t0 + t) * ldy + row] = alpha * a;
        }
}

// ---------------------------------------------------------------- gemv over F32 weights, a warp a row (the router)
void gemv_f32_kernel(const float* __restrict__ W, const float* __restrict__ x, int n, int k, float* __restrict__ y,
                     int ldy) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * (kThreads / 32) + (item_ct1.get_local_id(2) >> 5);
    const int lane = item_ct1.get_local_id(2) & 31, t = item_ct1.get_group(1);
    if (row >= n) return;
    const sycl::float4* w = (const sycl::float4*) (W + (size_t) row * k);
    const sycl::float4* xv = (const sycl::float4*) (x + (size_t) t * k);
    float acc = 0.0f;
    for (int c = lane; c < k / 4; c += 32) {
        const sycl::float4 a = w[c], b = xv[c];
        acc = sycl::fma((float) (a.x()), (float) (b.x()), acc);
        acc = sycl::fma((float) (a.y()), (float) (b.y()), acc);
        acc = sycl::fma((float) (a.z()), (float) (b.z()), acc);
        acc = sycl::fma((float) (a.w()), (float) (b.w()), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * ldy + row] = acc;
}

void gemv_f32_scalar_kernel(const float* __restrict__ W, const float* __restrict__ x, int n, int k,
                            float* __restrict__ y, int ldy) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * (kThreads / 32) + (item_ct1.get_local_id(2) >> 5);
    const int lane = item_ct1.get_local_id(2) & 31, t = item_ct1.get_group(1);
    if (row >= n) return;
    const float* w = W + (size_t) row * k;
    const float* xv = x + (size_t) t * k;
    float acc = 0.0f;
    for (int c = lane; c < (k + 3) / 4; c += 32) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = 4 * c + i;
            if (j < k) acc = sycl::fma(w[j], xv[j], acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * ldy + row] = acc;
}

}  // namespace

static bool aligned16(const void* p) { return ((uintptr_t) p & 15) == 0; }

class gemv_f16_wrapper {
    const uint16_t* W;
    const float* x;
    int n, k;
    float alpha;
    float* y;
    bool tile;

  public:
    gemv_f16_wrapper(const uint16_t* W_, const float* x_, int n_, int k_, float alpha_, float* y_, bool tile_)
        : W(W_), x(x_), n(n_), k(k_), alpha(alpha_), y(y_), tile(tile_) {}
    [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<3>) const {
        if (tile) gemv_f16_kernel(W, x, n, k, alpha, y);
        else gemv_f16_scalar_kernel(W, x, n, k, alpha, y);
    }
};

class gemv_f16_rows_wrapper {
    const uint16_t* W;
    const float* x;
    size_t ldx, ldy;
    int n, k, T;
    float alpha;
    float* y;
    bool tile;

  public:
    gemv_f16_rows_wrapper(const uint16_t* W_, const float* x_, size_t ldx_, int n_, int k_, int T_, float alpha_,
                          float* y_, size_t ldy_, bool tile_)
        : W(W_), x(x_), ldx(ldx_), ldy(ldy_), n(n_), k(k_), T(T_), alpha(alpha_), y(y_), tile(tile_) {}
    [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<3>) const {
        if (tile) gemv_f16_rows_kernel(W, x, ldx, n, k, T, alpha, y, ldy);
        else gemv_f16_rows_scalar_kernel(W, x, ldx, n, k, T, alpha, y, ldy);
    }
};

class gemv_f32_wrapper {
    const float* W;
    const float* x;
    int n, k, ldy;
    float* y;
    bool tile;

  public:
    gemv_f32_wrapper(const float* W_, const float* x_, int n_, int k_, float* y_, int ldy_, bool tile_)
        : W(W_), x(x_), n(n_), k(k_), ldy(ldy_), y(y_), tile(tile_) {}
    [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<3>) const {
        if (tile) gemv_f32_kernel(W, x, n, k, y, ldy);
        else gemv_f32_scalar_kernel(W, x, n, k, y, ldy);
    }
};

void gemv_f16(const uint16_t* W, const float* x, int n, int k, int T, float alpha, float* y, dpct::queue_ptr s) {
    if (n <= 0 || k <= 0 || T <= 0) return;
    const int groups = (n + kThreads / 32 - 1) / (kThreads / 32);
    const bool tile = k % 8 == 0 && aligned16(W);
    s->submit([&](sycl::handler& cgh) {
        cgh.parallel_for<dpct_kernel_name<gemv_f16_wrapper>>(
            sycl::nd_range<3>(sycl::range<3>(1, T, (size_t) groups * kThreads), sycl::range<3>(1, 1, kThreads)),
            gemv_f16_wrapper(W, x, n, k, alpha, y, tile));
    });
}

void gemv_f16_rows(const uint16_t* W, const float* x, size_t ldx, int n, int k, int T, float alpha, float* y,
                   size_t ldy, dpct::queue_ptr s) {
    if (n <= 0 || k <= 0 || T <= 0) return;
    const int groups = (n + kThreads / 32 - 1) / (kThreads / 32), tgroups = (T + kRowsT - 1) / kRowsT;
    const bool tile = k % 8 == 0 && (ldx % 4) == 0 && aligned16(W) && aligned16(x);
    s->submit([&](sycl::handler& cgh) {
        cgh.parallel_for<dpct_kernel_name<gemv_f16_rows_wrapper>>(
            sycl::nd_range<3>(sycl::range<3>(1, tgroups, (size_t) groups * kThreads),
                              sycl::range<3>(1, 1, kThreads)),
            gemv_f16_rows_wrapper(W, x, ldx, n, k, T, alpha, y, ldy, tile));
    });
}

void gemv_f32(const float* W, const float* x, int n, int k, int T, float* y, int ldy, dpct::queue_ptr s) {
    if (n <= 0 || k <= 0 || T <= 0) return;
    const int groups = (n + kThreads / 32 - 1) / (kThreads / 32);
    const bool tile = k % 4 == 0 && aligned16(W) && aligned16(x);
    s->submit([&](sycl::handler& cgh) {
        cgh.parallel_for<dpct_kernel_name<gemv_f32_wrapper>>(
            sycl::nd_range<3>(sycl::range<3>(1, T, (size_t) groups * kThreads), sycl::range<3>(1, 1, kThreads)),
            gemv_f32_wrapper(W, x, n, k, y, ldy, tile));
    });
}

}  // namespace strata::ds4
