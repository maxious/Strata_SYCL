// sycl/src/kernels/dequant_bench.cpp - the prompt's dequant phase (exp 07/11: the one open prompt lever).
//   dequant_bench [type=20] [rows=2560] [cols=1280] [reps=200]
// Times iq_dequant_f16 at a filled expert gate/up shape (rows*cols fp16 out), prints effective write GB/s and a
// checksum of the fp16 output. The checksum is the stability oracle for a dequant rewrite (see README P0 "GEMM-shaped
// INT8 prompt dequant path"): a change that keeps the checksum and raises GB/s is a real, bit-identical dequant win.
#include "strata/kernels/iq_kernels.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint64_t fnv(const uint16_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    const int ty = argc > 1 ? std::atoi(argv[1]) : 20;          // i-quant (default IQ4_NL)
    const int rows = argc > 2 ? std::atoi(argv[2]) : 2560;      // n_out of a gate/up expert
    const int cols = argc > 3 ? std::atoi(argv[3]) : 1280;      // n_embd
    const int reps = argc > 4 ? std::atoi(argv[4]) : 200;
    const int64_t n = (int64_t) rows * cols;                    // fp16 values out
    if (n % 256 != 0) { std::printf("n must be a multiple of 256\n"); return 2; }
    std::vector<uint8_t> hw((size_t) strata::kernels::iq_row_bytes(ty, cols) * rows);
    if (hw.size() == 0) { std::printf("type %d not supported\n", ty); return 2; }
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = (uint8_t) (i * 2654435761u >> 13) & 0x3b;   // finite fp16 fields
    sycl::queue* s = &dpct::get_in_order_queue();
    uint8_t* w = sycl::malloc_device<uint8_t>(hw.size(), *s);
    uint16_t* y = sycl::malloc_device<uint16_t>((size_t) n, *s);
    s->memcpy(w, hw.data(), hw.size()).wait();
    strata::kernels::iq_dequant_f16(ty, w, n, y, s);
    // warm + one reference checksum
    s->wait();
    std::vector<uint16_t> hr((size_t) n);
    s->memcpy(hr.data(), y, (size_t) n * 2).wait();
    const uint64_t cksum = fnv(hr.data(), hr.size());
    auto run = [&] { strata::kernels::iq_dequant_f16(ty, w, n, y, s); };
    for (int i = 0; i < 5; ++i) run();                         // warm clocks
    s->wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) run();
    s->wait();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    const double gb = (double) n * 2 / 1e9;
    std::printf("ty %d %d x %d: %.3f ms (%.0f GB/s), checksum %016llx\n", ty, cols, rows, ms,
                gb / (ms / 1e3), (unsigned long long) cksum);
    return 0;
}