// q6k_dpas_parity: the engine's DPAS int8 Q6_K decode path must reproduce the shipped AOS dp4a kernel.
//
// exp 42's bench verified the kernel against an fp64 host reference at the bench's shapes; this test is the engine-
// side gate for the same kernel as it is wired: random Q6_K weights -> native_q6k_vnni_tiles ->
// native_mmvq_register_q6k_dpas -> native_q6_k_mmvq (which dispatches to DPAS) against the same call with the
// registration removed, at ncols 1-8. Tolerance is exp 27's: the two paths differ only in the summation order of
// the 32-element dot, so fp32 rounding is the whole difference.
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct Q6KBlock {                     // GGUF Q6_K block order, as the engine's AOS weights arrive
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
};
static_assert(sizeof(Q6KBlock) == 210);

int main(int argc, char** argv) {
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 2560;
    const int n_out = argc > 2 ? std::atoi(argv[2]) : 2560;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 64;
    if (n_in % 256 || n_out % 16) { std::printf("n_in %% 256 and n_out %% 16 required\n"); return 2; }
    const int bpr = n_in / 256;
    const size_t nb = (size_t) n_out * bpr;

    std::mt19937 rng(11);
    std::vector<Q6KBlock> hos(nb);
    for (size_t b = 0; b < nb; ++b) {
        for (int i = 0; i < 128; ++i) hos[b].ql[i] = (uint8_t) rng();
        for (int i = 0; i < 64; ++i) hos[b].qh[i] = (uint8_t) rng();
        for (int i = 0; i < 16; ++i) hos[b].scales[i] = (int8_t) (1 + (rng() % 120));
        hos[b].d = sycl::bit_cast<uint16_t>(sycl::half(0.005f + 0.01f * (float) (rng() % 100) / 100.f));
    }

    sycl::queue* q = &dpct::get_in_order_queue();
    void* w = sycl::malloc_device(nb * 210, *q);
    q->memcpy(w, hos.data(), nb * 210).wait();
    const size_t dbytes = strata::kernels::native_mmvq_q6k_dpas_bytes(n_in, n_out);
    if (dbytes == 0) { std::printf("q6k_dpas_parity: bad shape\n"); return 2; }
    void* tiles = sycl::malloc_device(dbytes, *q);
    strata::kernels::native_q6k_vnni_tiles(w, tiles, n_in, n_out, q);
    q->wait();

    std::vector<float> xf((size_t) 8 * n_in);
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = (float) ((int) (rng() % 2000) - 1000) / 500.f;
    void* xq = sycl::malloc_device((size_t) 8 * (n_in / 32) * 36, *q);
    strata::kernels::native_quantize_q8_1(xf.data(), xq, n_in, 8, q);
    float* yaos = sycl::malloc_device<float>((size_t) 8 * n_out, *q);
    float* ydpas = sycl::malloc_device<float>((size_t) 8 * n_out, *q);

    {   // the transform must actually have written the tile buffer: checksum its first bytes and the scales
        std::vector<uint8_t> head(64);
        q->memcpy(head.data(), tiles, 64).wait();
        unsigned long long sum = 0;
        for (unsigned char c : head) sum += c;
        std::printf("q6k_dpas_parity: tile k=0..3 for n=0..15:");
        for (int n = 0; n < 16; ++n) std::printf(" %02x", head[n * 4]);
        std::printf("  (checksum %llu)\n", sum);
        const std::size_t tb = dbytes - (std::size_t) n_out * (n_in / 256) * 18;
        std::vector<int8_t> sc((std::size_t) n_out * (n_in / 256) * 16);
        q->memcpy(sc.data(), (const uint8_t*) tiles + tb, sc.size()).wait();
        long scsum = 0;
        for (int8_t v : sc) scsum += v;
        std::printf("q6k_dpas_parity: scales sum %ld over %zu bytes, tiles %zu bytes\n", scsum, sc.size(), tb);
    }

    // host-built tiles, exactly as the bench's reorder_q6k_vnni lays them out: if the engine kernel is correct on
    // these and wrong on the device transform's output, the transform is the bug, not the kernel
    {
        const int bpr2 = n_in / 256, ntiles2 = n_out / 16;
        std::vector<uint8_t> ht((std::size_t) ntiles2 * bpr2 * 8 * 512);
        std::vector<int8_t> hs((std::size_t) n_out * bpr2 * 16);
        std::vector<uint16_t> hd((std::size_t) n_out * bpr2);
        for (int r = 0; r < n_out; ++r)
            for (int b = 0; b < bpr2; ++b) {
                const Q6KBlock& bk = hos[(size_t) r * bpr2 + b];
                const int tile = r / 16, n = r % 16;
                for (int j = 0; j < 8; ++j)
                    for (int kk = 0; kk < 32; ++kk) {
                        const int e = 32 * j + kk, e2 = e & 127;
                        const int qlb = (e >= 128) ? 64 : 0, qhb = (e >= 128) ? 32 : 0;
                        const int nib = (e2 >= 64) ? 4 : 0;
                        const int raw = ((bk.ql[qlb + (e2 & 63)] >> nib) & 0xF) |
                                        ((((bk.qh[qhb + (e2 & 31)] >> (2 * (e2 >> 5))) & 3)) << 4);
                        ht[(((std::size_t) tile * bpr2 + b) * 8 + j) * 512 + (kk / 4) * 64 + n * 4 + (kk % 4)] =
                            (uint8_t) (int8_t) (raw - 32);
                    }
                std::memcpy(hs.data() + ((std::size_t) r * bpr2 + b) * 16, bk.scales, 16);
                hd[(std::size_t) r * bpr2 + b] = bk.d;
            }
        uint8_t* dht = (uint8_t*) sycl::malloc_device(ht.size(), *q);
        int8_t* dhs = (int8_t*) sycl::malloc_device(hs.size(), *q);
        uint16_t* dhd = (uint16_t*) sycl::malloc_device(hd.size() * 2, *q);
        q->memcpy(dht, ht.data(), ht.size()).wait();
        q->memcpy(dhs, hs.data(), hs.size()).wait();
        q->memcpy(dhd, hd.data(), hd.size() * 2).wait();
        {   // where do the device-built tiles differ from the host-built ones?
            std::vector<uint8_t> dev(ht.size());
            q->memcpy(dev.data(), tiles, ht.size()).wait();
            long firstdiff = -1, ndiff = 0;
            for (size_t i = 0; i < ht.size(); ++i)
                if (dev[i] != ht[i]) { if (firstdiff < 0) firstdiff = (long) i; ++ndiff; }
            std::vector<int8_t> devs(hs.size());
            q->memcpy(devs.data(), (const uint8_t*) tiles + (dbytes - hs.size() - hd.size() * 2), hs.size()).wait();
            long sdiff = 0, sfirst = -1;
            for (size_t i = 0; i < hs.size(); ++i)
                if (devs[i] != hs[i]) { if (sfirst < 0) sfirst = (long) i; ++sdiff; }
            std::vector<uint16_t> devd(hd.size());
            q->memcpy(devd.data(), (const uint8_t*) tiles + (dbytes - hd.size() * 2), hd.size() * 2).wait();
            long ddiff = 0, dfirst = -1;
            for (size_t i = 0; i < hd.size(); ++i)
                if (devd[i] != hd[i]) { if (dfirst < 0) dfirst = (long) i; ++ddiff; }
            std::printf("tiles differ %ld/%zu (first %ld); scales %ld/%zu (first %ld); d %ld/%zu (first %ld)\n", ndiff,
                        ht.size(), firstdiff, sdiff, hs.size(), sfirst, ddiff, hd.size(), dfirst);
        }
        strata::kernels::native_mmvq_q6k_dpas(dht, dhs, dhd, xq, ydpas, n_in, n_out, 1, q);
        q->wait();
        std::vector<float> hr(8);
        q->memcpy(hr.data(), ydpas, 8 * 4).wait();
        std::vector<float> hr_ref(8);
        q->memcpy(hr_ref.data(), yaos, 8 * 4).wait();
        // same bytes, two pointer arrangements: one buffer with offsets (what the engine passes) vs three buffers
        const std::size_t tb3 = dbytes - (std::size_t) n_out * (n_in / 256) * 18;
        strata::kernels::native_mmvq_q6k_dpas(tiles, (const uint8_t*) tiles + tb3,
                                              (const uint8_t*) tiles + tb3 + (std::size_t) n_out * (n_in / 256) * 16,
                                              xq, ydpas, n_in, n_out, 1, q);
        q->wait();
        std::vector<float> one_buf(8);
        q->memcpy(one_buf.data(), ydpas, 8 * 4).wait();
        std::printf("single-buffer DPAS first 8: ");
        for (int i = 0; i < 8; ++i) std::printf(" %.3f", one_buf[i]);
        std::printf("\n");
        std::printf("host-tiled DPAS first 8: ");
        for (int i = 0; i < 8; ++i) std::printf(" %.3f", hr[i]);
        std::printf("   AOS first 8:");
        for (int i = 0; i < 8; ++i) std::printf(" %.3f", hr_ref[i]);
        std::printf("\n");
        sycl::free(dht, *q); sycl::free(dhs, *q); sycl::free(dhd, *q);
    }

    int fails = 0;
    for (int nc = 1; nc <= 8; ++nc) {
        strata::kernels::native_q6_k_mmvq(w, xq, yaos, n_in, n_out, nc, q);          // AOS dp4a
        strata::kernels::native_mmvq_register_q6k_dpas(w, tiles, n_in, n_out);
        strata::kernels::native_mmvq_set_q6k_dpas(true);
        strata::kernels::native_q6_k_mmvq(w, xq, ydpas, n_in, n_out, nc, q);         // DPAS
        strata::kernels::native_mmvq_set_q6k_dpas(false);
        strata::kernels::native_mmvq_unregister_q6k_dpas(w);
        q->wait();
        std::vector<float> a((size_t) nc * n_out), b((size_t) nc * n_out);
        q->memcpy(a.data(), yaos, a.size() * 4).wait();
        q->memcpy(b.data(), ydpas, b.size() * 4).wait();
        {   // direct call, bypassing the registry: tells a dead dispatch from a dead kernel
            const std::size_t tb2 = dbytes - (std::size_t) n_out * (n_in / 256) * 18;
            strata::kernels::native_mmvq_q6k_dpas(tiles, (const uint8_t*) tiles + tb2,
                                                  (const uint8_t*) tiles + tb2 + (std::size_t) n_out * (n_in / 256) * 16,
                                                  xq, ydpas, n_in, n_out, nc, q);
            q->wait();
            std::vector<float> direct(4);
            q->memcpy(direct.data(), ydpas, 4 * 4).wait();
            std::vector<float> probe(2);
            q->memcpy(probe.data(), ydpas + (std::size_t) nc * n_out, 2 * 4).wait();
            std::printf("  direct: out %.3f %.3f %.3f %.3f | debug probe (tile0 bytes, tile1 act[0]) %.3f %.3f\n", direct[0],
                        direct[1], direct[2], direct[3], probe[0], probe[1]);
        }

        double num = 0, den = 0, worst = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            num += std::fabs((double) a[i] - (double) b[i]);
            den += std::fabs((double) a[i]);
            worst = std::max(worst, std::fabs((double) a[i] - (double) b[i]) /
                                    std::max(1e-6, std::fabs((double) a[i])));
        }
        const double rel = num / (den > 0 ? den : 1.0);
        const bool ok = rel < 2e-3 && worst < 1e-2;
        if (!ok) ++fails;
        std::printf("q6k_dpas_parity %5d x %5d ncols %d: rel %.2e worst-element %.2e %s\n", n_out, n_in, nc, rel, worst,
                    ok ? "ok" : "FAIL");
        (void) reps;
    }
    std::printf("q6k_dpas_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}