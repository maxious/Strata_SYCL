// int8_gemm_bench: FP16 vs INT8 oneMKL XMX GEMM at the prompt expert shapes (prefill/gemm.dp.cpp's f16 vs a
// hypothetical dequant-to-int8 + int8 gemm). Answers whether INT8 product work has headroom over FP16 on the
// B60 before wiring a quantize pipeline. Shapes follow the engine: down = W[N=2560,K=640] . X[K,T],
// gate/up = W[N=1280,K=2560] . X[K,T]; T is the prompt batch (rows of the chunk).
//   int8_gemm_bench [T=96] [mode=down|gu] [reps=50]
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
int main(int argc, char** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : 96;
    const std::string mode = argc > 2 ? argv[2] : "down";
    const int reps = argc > 3 ? std::atoi(argv[3]) : 50;
    const bool gu = mode == "gu";
    const int64_t K = gu ? 2560 : 640, N = gu ? 1280 : 2560;
    sycl::queue* q = &dpct::get_in_order_queue();
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    // Integer-valued operands for both paths: an EXACT int8 grid. fp16 (exact ints) and int8 both accumulate
    // K <= 2560 products of |v| <= 63, so every output is an exact integer sum for both paths.
    std::vector<uint16_t> hx((size_t) T * K), hw((size_t) N * K);
    std::vector<std::int8_t> hxq((size_t) T * K), hwq((size_t) N * K);
    std::uniform_int_distribution<int> idist(-63, 63);
    for (size_t i = 0; i < hx.size(); ++i) { const int v = idist(rng); hx[i] = f2h((float) v); hxq[i] = (std::int8_t) v; }
    for (size_t i = 0; i < hw.size(); ++i) { const int v = idist(rng); hw[i] = f2h((float) v); hwq[i] = (std::int8_t) v; }
    uint16_t* dx = sycl::malloc_device<uint16_t>(hx.size(), *q);
    uint16_t* dw = sycl::malloc_device<uint16_t>(hw.size(), *q);
    std::int8_t* dxq = sycl::malloc_device<std::int8_t>(hxq.size(), *q);
    std::int8_t* dwq = sycl::malloc_device<std::int8_t>(hwq.size(), *q);
    float* yf = sycl::malloc_device<float>((size_t) N * T, *q);
    float* yi = sycl::malloc_device<float>((size_t) N * T, *q);
    q->memcpy(dx, hx.data(), hx.size() * 2).wait();
    q->memcpy(dw, hw.data(), hw.size() * 2).wait();
    q->memcpy(dxq, hxq.data(), hxq.size()).wait();
    q->memcpy(dwq, hwq.data(), hwq.size()).wait();
    dpct::blas::descriptor_ptr h = new dpct::blas::descriptor();
    h->set_queue(q);
    const float alpha = 1.0f, beta = 0.0f;
    auto fp16 = [&]() {
        dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, (int) N, T, (int) K, &alpha, dw,
                         dpct::library_data_t::real_half, (int) K, dx, dpct::library_data_t::real_half, (int) K, &beta, yf,
                         dpct::library_data_t::real_float, (int) N, dpct::compute_type::f32);
    };
    auto int8 = [&]() {
        dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, (int) N, T, (int) K, &alpha, dwq,
                         dpct::library_data_t::real_int8, (int) K, dxq, dpct::library_data_t::real_int8, (int) K, &beta, yi,
                         dpct::library_data_t::real_float, (int) N, dpct::compute_type::f32);
    };
    fp16(); int8(); q->wait();
    auto time = [&](auto&& fn) {
        for (int i = 0; i < 3; ++i) fn(); q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) fn();
        q->wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    const double t16 = time(fp16), t8 = time(int8);
    const double flop = 2.0 * N * T * K;
    std::printf("%s N=%lld K=%lld T=%d: fp16 %.3f ms (%.0f GFLOP/s) | int8 %.3f ms (%.0f GFLOP/s, %.2fx)\n", mode.c_str(),
                (long long) N, (long long) K, T, t16, flop / t16 / 1e6, t8, flop / t8 / 1e6, t16 / t8);
    std::vector<float> r((size_t) N * T), x((size_t) N * T);
    q->memcpy(r.data(), yf, r.size() * 4).wait();
    q->memcpy(x.data(), yi, x.size() * 4).wait();
    double maxref = 0, maxerr = 0, sumerr = 0; size_t bad = 0, nonfinite = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        if (!std::isfinite(r[i]) || !std::isfinite(x[i])) { ++nonfinite; continue; }
        maxref = std::max(maxref, (double) std::fabs(r[i]));
        const double e = std::fabs((double) r[i] - x[i]);
        maxerr = std::max(maxerr, e); sumerr += e;
        if (e > 1e-2 * (1.0 + std::fabs(r[i]))) ++bad;
    }
    std::printf("  int8-vs-fp16 exactness (same int grid): max err %.3g mean err %.3g, %zu/%zu outside 1%% (nonfinite %zu)\n",
                maxerr, sumerr / r.size(), bad, r.size(), nonfinite);
    return bad > r.size() / 1000 ? 1 : 0;
}