// decode_xmx_gemm_bench: the P0b #2/#3 A/B - is a oneMKL dense-FP16 GEMM on a *persistent* FP16 copy of the weights
// faster than the engine's native_mmvq decode matvec, at the decode batch (ncols = 1..8)?
//
// Background (README P0b #2/#3, exp 23): the dense Q6_K decode MMVQ is PIPE-bound (Pipe 16.8%, Send 0.0%), so it
// leaves the matrix units idle. P0b #3's untried XMX-for-decode version is "dequantize once into a persistent FP16
// copy and GEMM on XMX", cost = 2x dense weights resident in VRAM (FP16 13.1 MB vs 5.4 MB packed), paid once. This
// bench measures whether, GIVEN that persistent copy, oneMKL's dense FP16 GEMM (compute_type f32) beats the engine's
// native_mmvq at its real decode ncols. The FP16 materialization cost itself is NOT in the GEMM timing (that is what
// "persistent" means); dequant_bench bounds it separately (~12 us for 5.4 MB at Q2_0's 419-473 GB/s, amortized over
// the MTP window / many decode rounds).
//
// native_mmvq runs the exact `native_quantize_q8_1` + `native_q6_k_mmvq` calls mmvq_bench uses (Q6_K wide, random
// d=1.0 rows), so its numbers sit against mmvq_bench's published 19.1/21.5/26.1/35.9/55.4 us curve. The GEMM path is
// verified against an fp64 host reference on the persistent dw; the bench exits non-zero past 1e-3 rel error.
//
//   decode_xmx_gemm_bench [n_in=2560] [n_out=2560] [reps=200]
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static float   h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }
static uint16_t f2h(float f)   { return sycl::bit_cast<uint16_t>(sycl::half(f)); }

int main(int argc, char** argv) {
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 2560;
    const int n_out = argc > 2 ? std::atoi(argv[2]) : 2560;
    const int reps  = argc > 3 ? std::atoi(argv[3]) : 200;
    const int cols[] = { 1, 2, 4, 6, 8 };
    if (n_in % 256 || n_out % 256) { std::printf("n_in/n_out must be multiples of 256\n"); return 2; }

    sycl::queue* q = &dpct::get_in_order_queue();
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::normal_distribution<float> nw(0.f, 0.02f);

    const size_t wbytes = (size_t) n_out * (n_in / 256) * 210;
    std::vector<uint8_t> hw(wbytes);
    for (size_t i = 0; i < wbytes; ++i) hw[i] = (uint8_t) (i * 2654435761u >> 13);
    for (size_t b = 0; b < wbytes / 210; ++b) { hw[b * 210 + 208] = 0x00; hw[b * 210 + 209] = 0x3c; }
    void* w = sycl::malloc_device(wbytes, *q);
    q->memcpy(w, hw.data(), wbytes).wait();

    const size_t kN = (size_t) n_in * n_out;
    std::vector<uint16_t> hdw(kN);
    for (int64_t n = 0; n < n_out; ++n)
        for (int64_t k = 0; k < n_in; ++k) hdw[(size_t) k + n * n_in] = f2h(nw(rng));
    uint16_t* dw = sycl::malloc_device<uint16_t>(kN, *q);
    q->memcpy(dw, hdw.data(), kN * 2).wait();

    const int ncols_max = cols[4];
    std::vector<uint16_t> x_all((size_t) ncols_max * n_in);
    std::vector<float>    xf_all((size_t) ncols_max * n_in);
    for (int64_t t = 0; t < ncols_max; ++t)
        for (int64_t k = 0; k < n_in; ++k) {
            float v = nd(rng);
            xf_all[(size_t) k + t * n_in] = v;
            x_all[(size_t) k + t * n_in]  = f2h(v);
        }
    uint16_t* dxf = sycl::malloc_device<uint16_t>(x_all.size(), *q);
    q->memcpy(dxf, x_all.data(), x_all.size() * 2).wait();

    float* y_mmvq = sycl::malloc_device<float>((size_t) ncols_max * n_out, *q);
    float* y_gemm = sycl::malloc_device<float>((size_t) ncols_max * n_out, *q);
    dpct::blas::descriptor_ptr h = new dpct::blas::descriptor();
    h->set_queue(q);
    const float alpha = 1.0f, beta = 0.0f;
    auto gemm_run = [&](int ncols) {
        dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, n_out, ncols, n_in,
                         &alpha, dw, dpct::library_data_t::real_half, n_in, dxf, dpct::library_data_t::real_half,
                         n_in, &beta, y_gemm, dpct::library_data_t::real_float, n_out, dpct::compute_type::f32);
    };

    for (int ci = 0; ci < 5; ++ci) {
        const int ncols = cols[ci];
        void* xq = sycl::malloc_device((size_t) ncols * (n_in / 32) * 36, *q);
        auto mmvq_run = [&]() {
            strata::kernels::native_quantize_q8_1(xf_all.data(), xq, n_in, ncols, q);
            strata::kernels::native_q6_k_mmvq(w, xq, y_mmvq, n_in, n_out, ncols, q);
        };

        mmvq_run(); gemm_run(ncols); q->wait();
        {
            std::vector<float> g((size_t) ncols * n_out);
            const int St = std::min(ncols, 128);
            q->memcpy(g.data(), y_gemm, (size_t) ncols * n_out * 4).wait();
            const auto wk = [&](int64_t k, int64_t n) { return (double) h2f(hdw[(size_t) k + n * n_in]); };
            const auto xt = [&](int64_t k, int64_t t) { return (double) h2f(x_all[(size_t) k + t * n_in]); };
            double maxref = 0, maxerr = 0;
            for (int64_t n = 0; n < n_out; ++n)
                for (int64_t t = 0; t < St; ++t) {
                    double v = 0;
                    for (int64_t k = 0; k < n_in; ++k) v += wk(k, n) * xt(k, t);
                    const size_t i = (size_t) n + t * n_out;
                    maxref = std::max(maxref, std::fabs(v));
                    maxerr = std::max(maxerr, std::fabs((double) g[i] - v));
                }
            const double sc = maxref ? maxref : 1.0;
            std::printf("ncols %d: oneMKL-fp16-GEMM max-rel-err vs fp64 %.3g (%s)\n", ncols, maxerr / sc,
                        (maxerr / sc < 1e-3) ? "ok" : "FAIL");
            if (maxerr / sc >= 1e-3) return 1;
        }

        auto warm = [&](auto&& fn) {
            const auto w0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 300) {
                for (int i = 0; i < 20; ++i) fn(); q->wait();
            }
        };
        warm(mmvq_run); warm([&] { gemm_run(ncols); });
        auto time = [&](auto&& fn) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) fn();
            q->wait();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        const double t_m = time(mmvq_run), t_g = time([&] { gemm_run(ncols); });
        const double flop = 2.0 * ncols * n_out * n_in;
        std::printf("%2d cols:  native_mmvq %6.1f us (%5.0f GB/s weights) | oneMKL-fp16-GEMM %6.1f us (%5.0f GB/s, %7.1f GFLOP/s) %.2fx\n",
                    ncols, t_m, wbytes / t_m / 1e3, t_g, wbytes / t_g / 1e3, flop / t_g / 1e6, t_m / t_g);
        sycl::free(xq, *q);
    }
    return 0;
}