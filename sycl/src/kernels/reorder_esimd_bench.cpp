// sycl/src/kernels/reorder_esimd_bench.cpp - P0-L1 vertical slice (Q8_0).
//
// Proves the llama.cpp decode layout move on the B60 for Q8_0: a one-shot SoA
// weight reorder ([qs: nb*32 bytes][d: nb*sizeof(half)]) + an ESIMD
// dequantize_mul_mat_vec kernel that streams the contiguous qs bytes and MACs
// against the full-row activations. Compares byte-identical-everything against
// the port's existing AOS wide decode (native_q8_0_mmvq / Wide32Q8) and reports
// GB/s for both so the port can decide whether the reorder+ESIMD path is worth
// wiring into production dispatch.
//
// Reference (llama.cpp HEAD cb7934c52): dmmv.cpp q8_0_mac_stripe /
// dequantize_mul_mat_vec_q8_0_reorder_esimd (l2012-2098) + ggml-sycl.cpp
// reorder_qw_q8_0 (l4290). This file ports those two mechanisms verbatim in a
// self-contained harness; it does NOT touch the production decode dispatch.
//
// Env: STRATA_REORDER_ESIMD=0 turns the ESIMD variant off (compare AOS only).
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;

// AOS block layout of Strata's Q8_0 decode weights (native_mmvq.dp.cpp Q80Block).
struct AosQ80 {
    sycl::half d;
    int8_t qs[32];
};
static_assert(sizeof(AosQ80) == 34 && alignof(AosQ80) == 2);

// ---- the SoA reorder (llama.cpp reorder_qw_q8_0) ---------------------------
// AOS block (Q80Block = half d + int8 qs[32], 34 B) -> [qs: nb*32][d: nb*half].
static void reorder_q8_0_soa(const AosQ80* __restrict__ a, int nb,
                             int8_t* __restrict__ qs, sycl::half* __restrict__ d,
                             sycl::queue* s) {
    // one work-item per block (llama.cpp uses size/sizeof(block_q8_0) items)
    s->parallel_for(nb, [=](sycl::id<1> ib) [[sycl::reqd_sub_group_size(32)]] {
        for (int j = 0; j < 32; ++j) qs[ib * 32 + j] = a[ib].qs[j];
        d[ib] = a[ib].d;
    }).wait();
}

// ---- the ESIMD decode (llama.cpp dmmv.cpp q8_0_mac_stripe) -----------------
template <int NBLK>
ESIMD_INLINE void q8_0_mac_stripe(
        const int8_t* qs_a, const int8_t* qs_b,
        const sycl::half* d_a, const sycl::half* d_b, bool has_b,
        esimd::simd<float, 32 * NBLK>& y_vec,
        esimd::simd<float, 32>& acc_a, esimd::simd<float, 32>& acc_b) {
    esimd::simd<int8_t, 32 * NBLK> qa = esimd::block_load<int8_t, 32 * NBLK>(qs_a);
    esimd::simd<int8_t, 32 * NBLK> qb = 0;
    esimd::simd<sycl::half, NBLK> da = esimd::block_load<sycl::half, NBLK>(d_a, esimd::element_aligned_tag{});
    esimd::simd<sycl::half, NBLK> db = 0;
    if (has_b) {
        qb = esimd::block_load<int8_t, 32 * NBLK>(qs_b);
        db = esimd::block_load<sycl::half, NBLK>(d_b, esimd::element_aligned_tag{});
    }
    esimd::simd<float, NBLK> da_f = esimd::convert<float>(da);
    esimd::simd<float, NBLK> db_f = esimd::convert<float>(db);
#pragma unroll
    for (int s = 0; s < NBLK; ++s) {
        esimd::simd<float, 32> y_s = y_vec.template select<32, 1>(s * 32);
        esimd::simd<int8_t, 32> qa_s = qa.template select<32, 1>(s * 32);
        esimd::simd<int8_t, 32> qb_s = qb.template select<32, 1>(s * 32);
        const float sa = da_f[s];
        const float sb = db_f[s];
        acc_a += y_s * (esimd::convert<float>(qa_s) * sa);
        acc_b += y_s * (esimd::convert<float>(qb_s) * sb);
    }
}

template <int WG>
static void dequantize_mul_mat_vec_q8_0_reorder_esimd(
        const void* vx, const float* y, float* dst,
        const int ncols, const int nrows, sycl::queue* s) {
    constexpr int STRIPE = 8;
    const int nblk_row = ncols / 32;
    const size_t nb = (size_t) nrows * nblk_row;
    const int8_t* qs = (const int8_t*) vx;
    const sycl::half* d = (const sycl::half*) (qs + nb * 32);
    const int workgroups = (nrows + 1) / 2;
    s->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(WG * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                const int tid = it.get_local_id(0);
                const int row_pair = it.get_group(0);
                const int row0 = row_pair * 2;
                const bool has_row1 = row0 + 1 < nrows;
                const size_t base0 = (size_t) row0 * nblk_row;
                const size_t base1 = has_row1 ? (size_t) (row0 + 1) * nblk_row : base0;
                esimd::simd<float, 32> acc0 = 0.0f;
                esimd::simd<float, 32> acc1 = 0.0f;
                int ib = 0;
                for (; ib + WG * STRIPE <= nblk_row; ib += WG * STRIPE) {
                    const int b = ib + tid * STRIPE;
                    esimd::simd<float, 256> y_vec = esimd::block_load<float, 256>(y + (size_t) b * 32);
                    q8_0_mac_stripe<STRIPE>(qs + (base0 + b) * 32, qs + (base1 + b) * 32,
                                            d + base0 + b, d + base1 + b, has_row1, y_vec, acc0, acc1);
                }
                for (int b = ib + tid; b < nblk_row; b += WG) {
                    esimd::simd<float, 32> y_vec = esimd::block_load<float, 32>(y + (size_t) b * 32);
                    q8_0_mac_stripe<1>(qs + (base0 + b) * 32, qs + (base1 + b) * 32,
                                       d + base0 + b, d + base1 + b, has_row1, y_vec, acc0, acc1);
                }
                lmem[tid * 2 + 0] = esimd::reduce<float>(acc0, std::plus<>{});
                lmem[tid * 2 + 1] = esimd::reduce<float>(acc1, std::plus<>{});
                it.barrier(sycl::access::fence_space::local_space);
                if (tid == 0) {
                    float sum0 = 0.0f, sum1 = 0.0f;
                    for (int p = 0; p < WG; ++p) { sum0 += lmem[p * 2 + 0]; sum1 += lmem[p * 2 + 1]; }
                    dst[row0] = sum0;
                    if (has_row1) dst[row0 + 1] = sum1;
                }
            });
    });
}

static void esimd_launch(const void* soa, const float* y, float* dst,
                         int ncols, int nrows, sycl::queue* s) {
    const int nblk_row = ncols / 32;
    if (nblk_row >= 64) { dequantize_mul_mat_vec_q8_0_reorder_esimd<8>(soa, y, dst, ncols, nrows, s); }
    else if (nblk_row >= 32) { dequantize_mul_mat_vec_q8_0_reorder_esimd<4>(soa, y, dst, ncols, nrows, s); }
    else if (nblk_row >= 16) { dequantize_mul_mat_vec_q8_0_reorder_esimd<2>(soa, y, dst, ncols, nrows, s); }
    else { dequantize_mul_mat_vec_q8_0_reorder_esimd<1>(soa, y, dst, ncols, nrows, s); }
}

int main(int argc, char** argv) {
    sycl::queue* s = &dpct::get_in_order_queue();
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    const bool do_esimd = std::getenv("STRATA_REORDER_ESIMD") == nullptr ||
                          std::atoi(std::getenv("STRATA_REORDER_ESIMD")) != 0;
    const int shapes[][2] = {{2560, 10240}, {2560, 12288}, {6144, 2560}, {2560, 2560}, {2560, 640}};
    int worst_failures = 0;
    for (auto& sh : shapes) {
        const int n_in = sh[0], n_out = sh[1];
        const int blocks_per_row = n_in / 32;
        const size_t nb = (size_t) n_out * blocks_per_row;
        std::vector<AosQ80> hw(nb);
        for (size_t i = 0; i < nb; ++i) {
            for (int j = 0; j < 32; ++j) hw[i].qs[j] = (int8_t) ((i * 37 + j * 2654435761u) >> 13) & 0x7f;
            hw[i].d = sycl::half(0.5f * (0.5f + (float) (i % 7) / 7.0f));
        }
        void* w = sycl::malloc_device(nb * 34, *s);
        s->memcpy(w, hw.data(), nb * 34).wait();
        // SoA buffer: [qs: nb*32 bytes][d: nb*half]
        void* soa = sycl::malloc_device((size_t) nb * 32 + (size_t) nb * 2, *s);
        reorder_q8_0_soa((const AosQ80*) w, (int) nb, (int8_t*) soa, (sycl::half*) ((int8_t*) soa + nb * 32), s);

        for (int nc : {1, 2, 4, 6}) {
            std::vector<float> hx((size_t) nc * n_in);
            for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) ((i * 7919) % 1000) / 500.f - 1.f;
            float* xq_f = sycl::malloc_device<float>(hx.size(), *s);     // dequantized q8_1 (same act values for both engines)
            void* xq = sycl::malloc_device((size_t) nc * (n_in / 32) * 36, *s);
            float* y = sycl::malloc_device<float>((size_t) nc * n_out, *s);
            s->memcpy(xq_f, hx.data(), hx.size() * 4).wait();
            strata::kernels::native_quantize_q8_1(xq_f, xq, n_in, nc, s);
            auto dequant_q8_1 = [&] {
                s->parallel_for((size_t) nc * (n_in / 32), [=](sycl::id<1> i) {
                    const int8_t* qs = (const int8_t*) ((const uint8_t*) xq + i * 36 + 4);
                    const float d = (float) ((const sycl::half*) ((const uint8_t*) xq + i * 36))[0];
                    float* dst = xq_f + i * 32;
                    for (int j = 0; j < 32; ++j) dst[j] = (float) qs[j] * d;
                }).wait();
            };
            dequant_q8_1();
            auto time = [&](const std::function<void()>& run, std::vector<float>& out) {
                const auto w0 = std::chrono::steady_clock::now();
                while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 150) {
                    for (int i = 0; i < 10; ++i) run();
                    s->wait();
                }
                const int it = 200;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i) run();
                s->wait();
                out.resize((size_t) nc * n_out);
                s->memcpy(out.data(), y, out.size() * 4).wait();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
            };
            std::vector<float> y_aos, y_soa;
            const double a = time([&] { strata::kernels::native_q8_0_mmvq(w, xq, y, n_in, n_out, nc, s); }, y_aos);
            double b = a;
            if (do_esimd) {
                b = time([&] { esimd_launch((const void*) soa, xq_f, y, n_in, n_out, s); }, y_soa);
            } else {
                y_soa = y_aos;
            }
            double num = 0, den = 0;
            for (size_t i = 0; i < y_aos.size(); ++i) { num += std::fabs((double) y_aos[i] - y_soa[i]); den += std::fabs((double) y_aos[i]); }
            const double rel = num / (den > 0 ? den : 1);
            if (selftest && do_esimd && !(rel < 1e-5)) { ++worst_failures; std::fprintf(stderr, "esimd_kq_parity FAIL %dx%d cols%d rel %.2e\n", n_in, n_out, nc, rel); }
            const double eff_blocks = (double) nb;
            const double gb = eff_blocks * 34.0 / 1e3;
            if (do_esimd) {
                std::printf("q8_0 %5d x %6d cols %d: AOS(w32) %7.1f us %6.1f GB/s | reorder+ESIMD %7.1f us %6.1f GB/s (%.2fx) | rel diff %.1e\n",
                            n_in, n_out, nc, a, gb / a, b, gb / b, a / b, num / (den > 0 ? den : 1));
            } else {
                std::printf("q8_0 %5d x %6d cols %d: AOS(w32) %7.1f us %6.1f GB/s (ESIMD disabled)\n",
                            n_in, n_out, nc, a, gb / a);
            }
            sycl::free(xq_f, *s); sycl::free(xq, *s); sycl::free(y, *s);
        }
        sycl::free(w, *s); sycl::free(soa, *s);
    }
    if (selftest) { std::fprintf(stderr, "esimd_kq_parity: %d failures\n", worst_failures); return worst_failures == 0 ? 0 : 1; }
    return 0;
}