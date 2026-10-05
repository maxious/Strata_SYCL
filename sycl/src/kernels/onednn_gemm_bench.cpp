// onednn_gemm_bench: dense FP16 oneDNN matmul vs the engine's oneMKL FP16 GEMM at the prompt expert shapes.
// The reframed P0b #1: the fused "Dequantize -> MatMul" premise is blocked (oneDNN cannot read ggml block-quantized
// weights, and llama.cpp dequantizes to F16 first anyway), so the honest library question is whether, on already-F16
// weights, oneDNN's dense FP16 matmul beats the oneMKL GEMM the prompt path uses today (Gemm::f16).
// Buffers are the engine's column-major FF layout (lda=ldb=K, ldc=N): W[k,n] at k + n*K, X[k,t] at k + t*K,
// output Y at n + t*N, Y[n,t] = sum_k W[k,n]*X[k,t].  Both paths read the same physical buffers.
// The 'int' data mode (|v|<=63, exact in FP16) checks exactness: sums reach ~1e7, past fp16's 65504, so it exposes
// whether a path accumulates in reduced precision.
//   onednn_gemm_bench [T=96] [mode=gu|down] [reps=50] [int|real]
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
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
    const std::string mode = argc > 2 ? argv[2] : "gu";
    const int reps = argc > 3 ? std::atoi(argv[3]) : 50;
    const std::string dkind = argc > 4 ? argv[4] : "real";
    const bool intgrid = dkind == "int";
    const bool gu = mode == "gu";
    const int64_t K = gu ? 2560 : 640, N = gu ? 1280 : 2560;
    sycl::queue* q = &dpct::get_in_order_queue();
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> idist(-63, 63);
    std::vector<uint16_t> hw((size_t) K * N), hx((size_t) K * T);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            hw[(size_t) k + n * K] = intgrid ? f2h((float) idist(rng)) : f2h(nd(rng) * 0.02f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k)
            hx[(size_t) k + t * K] = intgrid ? f2h((float) idist(rng)) : f2h(nd(rng));
    uint16_t* dw = sycl::malloc_device<uint16_t>(hw.size(), *q);
    uint16_t* dx = sycl::malloc_device<uint16_t>(hx.size(), *q);
    float* ym = sycl::malloc_device<float>((size_t) N * T, *q);
    float* yd = sycl::malloc_device<float>((size_t) N * T, *q);
    q->memcpy(dw, hw.data(), hw.size() * 2).wait();
    q->memcpy(dx, hx.data(), hx.size() * 2).wait();

    dpct::blas::descriptor_ptr h = new dpct::blas::descriptor();
    h->set_queue(q);
    const float alpha = 1.0f, beta = 0.0f;
    auto mkl = [&]() {
        dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, (int) N, T, (int) K,
                         &alpha, dw, dpct::library_data_t::real_half, (int) K, dx, dpct::library_data_t::real_half,
                         (int) K, &beta, ym, dpct::library_data_t::real_float, (int) N, dpct::compute_type::f32);
    };

    const dnnl::engine eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
    const dnnl::stream  st = dnnl::sycl_interop::make_stream(eng, *q);
    using dt = dnnl::memory::data_type;
    const dnnl::memory::desc a_md({N, K}, dt::f16, {K, 1});   // A[n,k] = W[k,n] at n*K + k
    const dnnl::memory::desc b_md({K, T}, dt::f16, {1, K});   // B[k,t] = X[k,t] at k + t*K
    const dnnl::memory::desc c_md({N, T}, dt::f32, {1, N});   // C[n,t] = Y at n + t*N
    dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md);
    dnnl::matmul prim(pd);
    const dnnl::memory a_mem(a_md, eng, dw);
    const dnnl::memory b_mem(b_md, eng, dx);
    const dnnl::memory c_mem(c_md, eng, yd);
    const std::unordered_map<int, dnnl::memory> args = {
        { DNNL_ARG_SRC, a_mem }, { DNNL_ARG_WEIGHTS, b_mem }, { DNNL_ARG_DST, c_mem }};
    auto dnn = [&]() { prim.execute(st, args); };

    mkl(); dnn(); q->wait();
    auto time = [&](auto&& fn) {
        for (int i = 0; i < 3; ++i) fn(); q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) fn();
        q->wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    const double tm = time(mkl), td = time(dnn);
    const double flop = 2.0 * N * T * K;
    std::printf("%s N=%lld K=%lld T=%d [%s]: oneMKL %.3f ms (%.0f GFLOP/s) | oneDNN %.3f ms (%.0f GFLOP/s, %.2fx) impl=%s\n",
                mode.c_str(), (long long) N, (long long) K, T, dkind.c_str(), tm, flop / tm / 1e6, td,
                flop / td / 1e6, tm / td, pd.impl_info_str());

    std::vector<float> r((size_t) N * T), x((size_t) N * T);
    q->memcpy(r.data(), ym, r.size() * 4).wait();
    q->memcpy(x.data(), yd, x.size() * 4).wait();
    const auto wn = [&](int64_t k, int64_t n) { return (double) sycl::bit_cast<sycl::half>(hw[(size_t) k + n * K]); };
    const auto xt = [&](int64_t k, int64_t t) { return (double) sycl::bit_cast<sycl::half>(hx[(size_t) k + t * K]); };
    const int64_t St = std::min<int64_t>(T, 128);
    double maxref = 0, maxm = 0, maxd = 0;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t t = 0; t < St; ++t) {
            double v = 0;
            for (int64_t k = 0; k < K; ++k) v += wn(k, n) * xt(k, t);
            const size_t i = (size_t) n + t * N;
            maxref = std::max(maxref, std::fabs(v));
            maxm   = std::max(maxm,   std::fabs((double) r[i] - v));
            maxd   = std::max(maxd,   std::fabs((double) x[i] - v));
        }
    const double sc = maxref ? maxref : 1.0;
    std::printf("  vs fp64 ref: oneMKL max-rel-err %.3g, oneDNN max-rel-err %.3g\n", maxm / sc, maxd / sc);
    return 0;
}