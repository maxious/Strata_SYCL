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
    // sycl_ext_oneapi_peer_access: what the port does NOT do today (zero can_access_peer / enable_peer_access calls),
    // so this prints both directions' support and, when supported, enables peer access and re-times the identical
    // copy.  The engine's --peer-device expert tier stages every expert batch through pinned host buffers instead
    // (remote_experts.cpp), so a win here is the tier's transfer, not the layer hand-off's 34 us.
    const bool can01 = cards[0].ext_oneapi_can_access_peer(cards[1]);
    const bool can10 = cards[1].ext_oneapi_can_access_peer(cards[0]);
    std::printf("peer access: dev0->dev1 %s, dev1->dev0 %s\n", can01 ? "supported" : "NOT supported",
                can10 ? "supported" : "NOT supported");
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
    if (can01 || can10) {
        try {
            if (can01) cards[0].ext_oneapi_enable_peer_access(cards[1]);
            if (can10) cards[1].ext_oneapi_enable_peer_access(cards[0]);
        } catch (const sycl::exception& e) {
            std::printf("enable_peer_access threw: %s\n", e.what());
        }
        const auto p01 = time([&] { q1.memcpy(d1, d0, bytes); }, reps);
        const auto p10 = time([&] { q0.memcpy(d0, d1, bytes); }, reps);
        std::printf("peer enabled: D0->D1 %.3f ms %.1f GB/s (%.2fx) | D1->D0 %.3f ms %.1f GB/s (%.2fx)\n", p01,
                    bytes / 1e9 / (p01 / 1e3), t01 / p01, p10, bytes / 1e9 / (p10 / 1e3), t10 / p10);
    } else {
        std::printf("peer enabled: skipped (unsupported)\n");
    }
    q1.memcpy(h1.data(), d1, bytes).wait();
    q0.memcpy(h0.data(), d0, bytes).wait();
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) if (h1[i] != 0x1234 || h0[i] != 0x1234) ++bad;
    std::printf("round-trip check: %zu mismatches (nonfinite n/a)\n", bad);
    return bad == 0 ? 0 : 1;
}