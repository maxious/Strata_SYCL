// xmx_global_pack_probe: B60 microprobe for a GLOBAL ext_intel_packed B operand load (sg16 joint_matrix).
//
// The stageA kernel plan (todo 3) reads its B operand (packed F16 activations) straight from global memory
// instead of staging it through shared memory like set_tc_kernel does (ds4_iq.dp.cpp:1144-1174).  This probe
// proves the address bijection of that load before any real kernel is written: element (k, m) of the B plane
// must live at Bpack[(k / 2) * Mpad * 2 + m * 2 + (k & 1)], ld = Mpad * 2 halfs, Mpad = round_up(M, 16).
//
// Proof: the host fills the packed plane at those exact offsets with distinct positive half values (and checks
// the offsets are a bijection onto [0, K * Mpad)).  A minimal sg16 kernel then walks the plane 16 x 16 tiles at
// a time and reads each tile's B operand via sycl::address_space_cast on ext_intel_packed with global_space +
// decorated::no, the load's only variable being ld.  A joint_matrix_mad against an identity A tile copies the
// loaded tile element-by-element into a row-major accumulator store (0/1 half weights are exact, so the copy is
// bit-exact and the identity is transposition-invariant).  The host compares the copy against the expected
// packed values position by position and prints MATCH / NO MATCH with counts per (K, M) shape.
//
// Negative control: --wrong-ld passes ld = Mpad * 2 + 2 to the B load, which must print NO MATCH and exit 1.
//
// Build (AOT bmg/lnl convention: STRATA_XMX_SG16, sycl/CMakeLists.txt:40-41):
//   icpx -fsycl -O2 -DSTRATA_XMX_SG16=1 xmx_global_pack_probe.cpp -o probe_jit
//   icpx -fsycl -O2 -DSTRATA_XMX_SG16=1 -fsycl-targets=spir64_gen \
//        -Xsycl-target-backend=spir64_gen "-device bmg-g21 -options -cl-fp32-correctly-rounded-divide-sqrt" \
//        xmx_global_pack_probe.cpp -o probe_aot
#include <sycl/sycl.hpp>
#if !defined(STRATA_USE_HIP)
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#endif

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace jm = sycl::ext::oneapi::experimental::matrix;

#ifndef STRATA_XMX_SG16
// the sg16 tile family exists only in the STRATA_XMX_SG16 build (AOT bmg/lnl), as in ds4_iq.dp.cpp
int main() {
    std::fprintf(stderr, "xmx_global_pack_probe: build with -DSTRATA_XMX_SG16=1 (AOT bmg/lnl convention)\n");
    return 2;
}
#else

// one 16-wide sub-group walks every 16 x 16 tile of the packed B plane; reqd_sub_group_size(16) rides on the
// functor's operator() (set_tc_wrapper pattern, ds4_iq.dp.cpp:1241-1256)
struct probe_kernel {
    sycl::half *Bpack;   // the packed B plane: element (k, m) at [(k / 2) * Mpad * 2 + 2 * m + (k & 1)]
    sycl::half *ident;   // row-major identity A tile, ld 16: acc = I * B = B element by element (exact in F32)
    float *out;          // row-major (K x Mpad) copy of what the tiles read
    int K, Mpad, ld_arg; // ld_arg: Mpad * 2 for the real layout, + 2 for the negative control

    [[sycl::reqd_sub_group_size(16)]]
    void operator()(sycl::nd_item<1> item_ct1) const {
        const sycl::sub_group sg = item_ct1.get_sub_group();
        auto gptr = [](sycl::half *p) {
            return sycl::address_space_cast<sycl::access::address_space::global_space,
                                            sycl::access::decorated::no>(p);
        };
        const int ld = Mpad * 2;
        for (int kk = 0; kk < K; kk += 16) {
            for (int mm = 0; mm < Mpad; mm += 16) {
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 16, 16, jm::layout::row_major> fa;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> fb;
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 16, 16> acc;
                jm::joint_matrix_fill(sg, acc, 0.0f);
                jm::joint_matrix_load(sg, fa, gptr(ident), 16);
                // the tile base for (kk, mm) under the pack convention; only ld_arg may be wrong
                const size_t base = (size_t)(kk >> 1) * (size_t)ld + (size_t)2 * mm + (kk & 1);
                jm::joint_matrix_load(sg, fb, gptr(Bpack + base), ld_arg);
                jm::joint_matrix_mad(sg, acc, fa, fb, acc);
                jm::joint_matrix_store(sg, acc,
                                       sycl::address_space_cast<sycl::access::address_space::global_space,
                                                               sycl::access::decorated::no>(
                                           out + (size_t)kk * Mpad + mm),
                                       Mpad, jm::layout::row_major);
            }
        }
    }
};

// positive finite halfs, varied enough that an address error cannot hide (values are compared bit-exact)
static sycl::half value_at(int k, int m) {
    unsigned h = (unsigned)k * 2654435761u + (unsigned)m * 40503u + 0x9E37u;
    float v = 1.0f + (float)(h % 1024u) + (float)((h >> 10) & 15u) * 0.0625f;
    return sycl::vec<float, 1>(v).convert<sycl::half, sycl::rounding_mode::rte>()[0];
}

int main(int argc, char **argv) {
    bool wrong_ld = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--wrong-ld")) wrong_ld = true;
        else {
            std::fprintf(stderr, "usage: %s [--wrong-ld]\n", argv[0]);
            return 2;
        }
    }

    // the probe is a B60 gate: pick the card by name, list what exists if it is not there
    std::vector<sycl::device> all;
    for (const sycl::platform &p : sycl::platform::get_platforms())
        for (const sycl::device &d : p.get_devices()) all.push_back(d);
    const std::string want = "Arc(TM) Pro B60";
    const sycl::device *dev = nullptr;
    for (const sycl::device &d : all)
        if (d.get_info<sycl::info::device::name>().find(want) != std::string::npos) { dev = &d; break; }
    if (!dev) {
        std::printf("device not found: Intel(R) %s\n", want.c_str());
        for (const sycl::device &d : all)
            std::printf("available device: %s\n", d.get_info<sycl::info::device::name>().c_str());
        return 2;
    }
    sycl::queue q(*dev);
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("mode: %s\n", wrong_ld ? "wrong ld (Mpad*2+2) negative control" : "correct ld (Mpad*2)");

    // the identity A tile: row-major, ld 16 (symmetric, so the probe is insensitive to A's row/col convention)
    sycl::half hident[256];
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            hident[r * 16 + c] = sycl::vec<float, 1>(r == c ? 1.0f : 0.0f)
                                     .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    sycl::half *dident = sycl::malloc_device<sycl::half>(256, q);
    q.memcpy(dident, hident, sizeof hident).wait();

    const int Mlist[2] = {16, 32};
    const int Klist[4] = {32, 128, 640, 2560};   // the measured expert shapes: n_embd and n_ff of the model
    int cases_ok = 0, cases = 0;
    for (int mi = 0; mi < 2; ++mi) {
        for (int ki = 0; ki < 4; ++ki) {
            const int M = Mlist[mi], K = Klist[ki];
            const int Mpad = (M + 15) & ~15, ld = Mpad * 2;
            const int nB = K * Mpad;   // (K / 2) * (Mpad * 2)
            const int kPad = 4096;     // slack so the --wrong-ld overshoot (13 halfs worst case) stays in bounds

            // host: expected packed offsets; they must be a bijection onto [0, nB)
            std::vector<int> seen(nB, 0);
            std::vector<sycl::half> hb(nB + kPad);
            for (int k = 0; k < K; ++k) {
                for (int m = 0; m < Mpad; ++m) {
                    const int off = (k >> 1) * ld + 2 * m + (k & 1);
                    if (off < 0 || off >= nB) { std::printf("offset out of range: k %d m %d off %d\n", k, m, off); return 1; }
                    ++seen[off];
                    hb[off] = value_at(k, m);
                }
            }
            int uniq = 0;
            for (int c : seen) uniq += c == 1;
            std::printf("case K=%d M=%d Mpad=%d ld=%d bijection %d/%d unique offsets\n", K, M, Mpad, ld, uniq, nB);

            sycl::half *db = sycl::malloc_device<sycl::half>(nB + kPad, q);
            float *dout = sycl::malloc_device<float>(nB, q);
            q.memcpy(db, hb.data(), (nB + kPad) * sizeof(sycl::half)).wait();
            q.memset(dout, 0, nB * sizeof(float)).wait();
            q.submit([&](sycl::handler &h) {
                 h.parallel_for(sycl::nd_range<1>(sycl::range<1>(16), sycl::range<1>(16)),
                                probe_kernel{db, dident, dout, K, Mpad, wrong_ld ? ld + 2 : ld});
             }).wait();

            std::vector<float> ho(nB);
            q.memcpy(ho.data(), dout, nB * sizeof(float)).wait();
            int matched = 0;
            for (int k = 0; k < K; ++k) {
                for (int m = 0; m < Mpad; ++m) {
                    const int off = (k >> 1) * ld + 2 * m + (k & 1);
                    if (ho[k * Mpad + m] == static_cast<float>(hb[off])) ++matched;
                }
            }
            const bool ok = uniq == nB && matched == nB;
            std::printf("result K=%d M=%d %s %d/%d\n", K, M, ok ? "MATCH" : "NO MATCH", matched, nB);
            ++cases;
            cases_ok += ok;
            sycl::free(db, q);
            sycl::free(dout, q);
        }
    }
    sycl::free(dident, q);
    std::printf("summary %s %d/%d cases\n", cases_ok == cases ? "MATCH" : "NO MATCH", cases_ok, cases);
    return cases_ok == cases ? 0 : 1;
}
#endif
