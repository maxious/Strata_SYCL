// p2p_bench: device-to-device (P2P) handoff between two Intel GPUs, the transfer Strata's layer split uses to
// move a verify window from one card to the next (docs/MULTI_GPU.md: pinned-RAM handoff, ~a few hundred KB).
// Times sycl::memcpy D0->D1 and D1->D0 over several sizes and checks the bytes round-trip intact.
//   p2p_bench [bytes=262144] [reps=100]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
// The two discrete B60s are exposed as 4 GPU entries (2 Level-Zero + 2 OpenCL) with identical names, so the
// physical cards are the first two Level-Zero devices - one per discrete GPU, and the backend the engine uses.
static std::vector<sycl::device> gpu_devices() {
    std::vector<sycl::device> out;
    for (auto const& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) out.push_back(d);
        if (out.size() == 2) break;
    }
    return out;
}
int main(int argc, char** argv) {
    const size_t bytes = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1u << 18;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 100;
    auto cards = gpu_devices();
    if (cards.size() < 2) { std::printf("need 2 Level-Zero GPUs, found %zu\n", cards.size()); return 2; }
    std::printf("dev0=%s\ndev1=%s\n", cards[0].get_info<sycl::info::device::name>().c_str(),
                cards[1].get_info<sycl::info::device::name>().c_str());
    const size_t n = bytes / 4;
    sycl::queue q0(cards[0]), q1(cards[1]);
    std::vector<uint32_t> h0(n, 0x1234), h1(n, 0);
    uint32_t* d0 = sycl::malloc_device<uint32_t>(n, q0);
    uint32_t* d1 = sycl::malloc_device<uint32_t>(n, q1);
    q0.memcpy(d0, h0.data(), bytes).wait();
    q1.memcpy(d1, h1.data(), bytes).wait();
    auto time = [&](auto&& fn, int r) {
        for (int i = 0; i < 3; ++i) fn(); q0.wait(); q1.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < r; ++i) fn();
        q0.wait(); q1.wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / r;
    };
    const auto t01 = time([&] { q1.memcpy(d1, d0, bytes); }, reps);
    const auto t10 = time([&] { q0.memcpy(d0, d1, bytes); }, reps);
    const double gb01 = bytes / 1e9 / (t01 / 1e3);
    const double gb10 = bytes / 1e9 / (t10 / 1e3);
    std::printf("D0->D1 %.3f ms (%d bytes) %.1f GB/s | D1->D0 %.3f ms %.1f GB/s\n", t01, (int) bytes, gb01, t10, gb10);
    q1.memcpy(h1.data(), d1, bytes).wait();
    q0.memcpy(h0.data(), d0, bytes).wait();
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) if (h1[i] != 0x1234 || h0[i] != 0x1234) ++bad;
    std::printf("round-trip check: %zu mismatches (nonfinite n/a)\n", bad);
    return bad == 0 ? 0 : 1;
}