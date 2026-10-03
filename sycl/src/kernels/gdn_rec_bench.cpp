// sycl/src/kernels/gdn_rec_bench.cpp - the GDN recurrence alone, at the prompt chunk's shapes (README.sycl.md: the
// DPCT1053 lead - the key-head kernel was written around cp.async, which the SYCL port cannot use, and is an A/B).
//
//     gdn_rec_bench [T=640] [reps=50]
//
// Times strata::prefill::gdn_recurrence.  Its variant is chosen once per process from the environment, so one run
// measures one of: the default software-pipelined column kernel, the key-head kernel (STRATA_GDN_KEYHEAD=1), the
// plain column kernel (STRATA_GDN_PIPELINE=0) or the one-block-per-head kernel (STRATA_GDN_REC_HEADS=1).
#include "strata/prefill/kernels.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 640;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 50;
    const int64_t S = 128, HK = 16, HV = 48, C = 10240;
    sycl::queue* s = &dpct::get_in_order_queue();
    const char* tag = std::getenv("STRATA_GDN_REC_HEADS") ? "rec_heads" :
                      std::getenv("STRATA_GDN_KEYHEAD")    ? "keyhead" :
                      (std::getenv("STRATA_GDN_PIPELINE") && std::atoi(std::getenv("STRATA_GDN_PIPELINE")) == 0) ? "cols" : "cols_pipe";
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 0.3f);
    const size_t st_n = (size_t) S * HV * S, h_n = (size_t) T * C, g_n = (size_t) T * HV, z_n = (size_t) T * HV * S;
    std::vector<float> hst(st_n), hh(h_n), hg(g_n), hb(g_n), hz(z_n), hga(S);
    for (auto& v : hst) v = nd(rng);
    for (auto& v : hh) v = nd(rng);
    for (auto& v : hg) v = nd(rng);          // gate enters through exp(): keep it small
    for (auto& v : hb) v = (float) (0.5 + 0.5 * std::fabs(nd(rng)));
    for (auto& v : hz) v = nd(rng);
    for (auto& v : hga) v = 1.f + 0.1f * nd(rng);
    float *state = sycl::malloc_device<float>(st_n, *s), *h = sycl::malloc_device<float>(h_n, *s);
    float *gate = sycl::malloc_device<float>(g_n, *s), *beta = sycl::malloc_device<float>(g_n, *s);
    float *z = sycl::malloc_device<float>(z_n, *s), *gamma = sycl::malloc_device<float>(S, *s);
    float *y = sycl::malloc_device<float>(z_n, *s);
    uint16_t* y16 = sycl::malloc_device<uint16_t>(z_n, *s);
    s->memcpy(state, hst.data(), st_n * 4);
    s->memcpy(h, hh.data(), h_n * 4);
    s->memcpy(gate, hg.data(), g_n * 4);
    s->memcpy(beta, hb.data(), g_n * 4);
    s->memcpy(z, hz.data(), z_n * 4);
    s->memcpy(gamma, hga.data(), S * 4).wait();
    auto run = [&] { strata::prefill::gdn_recurrence(state, h, gate, beta, z, gamma, 1e-5f, y, y16, T, s); };
    for (int i = 0; i < 3; ++i) run();
    s->wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) run();
    s->wait();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    std::printf("gdn_recurrence  variant=%-10s T=%-6lld  %.4f ms  (%.2f us/token)\n", tag, (long long) T, ms,
                1000.0 * ms / (double) T);
    return 0;
}
