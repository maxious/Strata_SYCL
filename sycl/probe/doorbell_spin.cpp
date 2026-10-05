// doorbell_spin.cpp - the device's host-flag spin alone: the host writes a mapped flag, a one-thread kernel spins
// on it with strata::sys_load and reports the first-seen iteration (strata::sys_store). first-seen == kSpinMax is
// the "exhausted" case, anything less is "seen late".  Usage: doorbell_spin [trials] [delay_us]
#include <sycl/sycl.hpp>
#include "strata/sycl_doorbell.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
    const int trials = argc > 1 ? std::atoi(argv[1]) : 256;
    const int delay_us = argc > 2 ? std::atoi(argv[2]) : 1000;
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("doorbell_spin: %s  kSpinMax=%u\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), (unsigned) strata::kSpinMax);

    volatile uint32_t* flag = (volatile uint32_t*) sycl::malloc_host<uint32_t>(1, q);
    uint32_t* seen = sycl::malloc_host<uint32_t>((size_t) trials, q);
    *flag = 0;

    auto run = [&](int t, int us) {
        seen[t] = 0;
        strata::sys_store(flag, 0);
        q.single_task([=]() {
            uint32_t s = 0;
            while (s < strata::kSpinMax && strata::sys_load(flag) < 1u) ++s;
            strata::sys_store(seen + t, s);
        });
        if (us > 0) std::this_thread::sleep_for(std::chrono::microseconds(us));
        strata::sys_store(flag, 1);
        q.wait();
        return seen[t];
    };

    static const int delays[] = {0, 5, 50, 500, 2000, 10000, 50000};
    std::printf("sweep (first-seen iteration vs host-store delay):\n");
    for (int d : delays) {
        const uint32_t s = run(0, d);
        std::printf("  delay %6d us -> first-seen %u%s\n", d, s, s >= strata::kSpinMax ? "  (EXHAUSTED)" : "");
    }

    static const int NB = 7;
    int hist[NB] = {};
    uint32_t mn = ~0u, mx = 0;
    double sum = 0;
    int exhausted = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < trials; ++t) {
        const uint32_t s = run(t, delay_us);
        if (s >= strata::kSpinMax) { ++hist[NB - 1]; ++exhausted; }
        else if (s == 0) ++hist[0];
        else if (s < 10) ++hist[1];
        else if (s < 100) ++hist[2];
        else if (s < 1000) ++hist[3];
        else if (s < 10000) ++hist[4];
        else ++hist[5];
        if (s < mn) mn = s;
        if (s > mx) mx = s;
        sum += s;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("histogram: %d trials, host-store delay %d us (%.1f ms total):\n", trials, delay_us, ms);
    static const char* labels[NB] = {"=0", "1-9", "10-99", "100-999", "1k-9k", "10k-(kSpinMax-1)", "kSpinMax (exhausted)"};
    for (int i = 0; i < NB; ++i) std::printf("  %-20s %5d\n", labels[i], hist[i]);
    std::printf("  first-seen: min %u  mean %.1f  max %u  exhausted %d/%d\n",
                mn, sum / trials, mx, exhausted, trials);

    sycl::free(seen, q);
    sycl::free((void*) flag, q);
    return 0;
}
