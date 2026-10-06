// exp 42: the DPAS int8 Q6_K decode path - the one-time AOS -> DPAS-order transform, the tile registry, and the
// decode matvec itself.
//
// Why its own translation unit: identical code compiled inside native_mmvq.dp.cpp returns zeros on the B60 while the
// same kernel in a standalone TU is correct (the bench's xmx_mmvq_bench verifies at rel-RMS 3.4e-03), with identical
// compile flags. Isolating it keeps the DPAS codegen in one place and makes that difference testable.
//
// The dot moves off the ALU pipe onto the matrix engine. int8 DPAS on this device is fixed at K=32 with N in {8,16}
// and a repeat count of 1..8 (dpas.hpp asserts SystolicDepth == 8), so:
//   * the weights are transformed ONCE at load into the B order the hardware consumes - per (16-row tile, 256-block,
//     32-wide group) a 512-byte tile with element (k,n) at dword (k/4)*16+n and byte k%4 - so each group is one
//     contiguous 512-byte load and no in-register transpose is needed;
//   * Q6_K keeps one scale per 16 weights, so a 32-deep DPAS spans two scales and the kernel issues two DPAS per
//     group with the unused half of B zeroed (measured cost of that masked pair: ~1%);
//   * a repeat count above 1 does not reproduce in this toolchain (exp 42: only one row of the MxN result matches),
//     so the kernel issues one M=1 DPAS per activation column with the tile held in registers across columns - which
//     costs nothing, since M=1 wastes none of the matrix pipe.
// The buffer is 272 B per 256 weights, the Q6K->Q6U pre-unpack's own VRAM class; which shapes are worth it is decided
// at registration (STRATA_Q6K_DPAS_MIN_BYTES), not per call.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace strata::kernels {

namespace esimd_q6k = sycl::ext::intel::esimd;
namespace xmx_q6k   = sycl::ext::intel::esimd::xmx;

namespace {
struct Q6KBlockMirror {          // GGUF Q6_K block order (ql, qh, scales, d), 210 B
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
};
static_assert(sizeof(Q6KBlockMirror) == 210, "the transform reads the GGUF Q6_K block layout");

struct Q6kDpasBuf { const uint8_t* tiles; const int8_t* scales; const uint16_t* d; };
std::unordered_map<const void*, Q6kDpasBuf> g_q6k_dpas;
int g_q6k_dpas_enabled = 0;

std::size_t q6k_dpas_tiles_bytes(int n_in, int n_out) {
    return (std::size_t) (n_out / 16) * (n_in / 256) * 8 * 512;
}
}  // namespace

std::size_t native_mmvq_q6k_dpas_bytes(int n_in, int n_out) {
    if (n_in % 256 != 0 || n_out % 16 != 0 || n_in <= 0 || n_out <= 0) return 0;
    return q6k_dpas_tiles_bytes(n_in, n_out) + (std::size_t) n_out * (n_in / 256) * 18;
}

void native_mmvq_set_q6k_dpas(bool enabled) { g_q6k_dpas_enabled = enabled ? 1 : 0; }

void native_mmvq_register_q6k_dpas(const void* packed, const void* buf, int n_in, int n_out) {
    if (!buf || n_in % 256 != 0 || n_out % 16 != 0) return;
    const std::size_t tb = q6k_dpas_tiles_bytes(n_in, n_out), sb = (std::size_t) n_out * (n_in / 256) * 16;
    const uint8_t* p = (const uint8_t*) buf;
    g_q6k_dpas[packed] = Q6kDpasBuf{p, (const int8_t*) (p + tb), (const uint16_t*) (p + tb + sb)};
}
void native_mmvq_unregister_q6k_dpas(const void* packed) { g_q6k_dpas.erase(packed); }
void native_mmvq_clear_q6k_dpas() { g_q6k_dpas.clear(); }

bool native_mmvq_q6k_dpas_lookup(const void* packed, const void** tiles, const void** scl, const void** d) {
    if (!g_q6k_dpas_enabled) return false;
    auto it = g_q6k_dpas.find(packed);
    if (it == g_q6k_dpas.end()) return false;
    *tiles = it->second.tiles;
    *scl = it->second.scales;
    *d = it->second.d;
    return true;
}

// one work-item per (16-row tile, 256-block): it writes that tile's 4 KiB of DPAS B order plus row 0's scales and d
static void native_q6k_vnni_kernel(const Q6KBlockMirror* __restrict__ w, uint8_t* __restrict__ tiles,
                                   int8_t* __restrict__ scl, uint16_t* __restrict__ dd, int bpr, int ntiles) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int idx = int(item.get_global_id(0));
    if (idx >= ntiles * bpr) return;
    const int tile = idx / bpr, b = idx % bpr;
    uint8_t* tp = tiles + (std::size_t) idx * 8 * 512;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
        const Q6KBlockMirror& bk = w[(std::size_t) (tile * 16 + n) * bpr + b];
        for (int j = 0; j < 8; ++j) {
            uint8_t* g = tp + j * 512;
            for (int kk = 0; kk < 32; ++kk) {
                const int e = 32 * j + kk, e2 = e & 127;
                const int qlb = (e >= 128) ? 64 : 0, qhb = (e >= 128) ? 32 : 0;
                const int nib = (e2 >= 64) ? 4 : 0;
                const int raw = ((bk.ql[qlb + (e2 & 63)] >> nib) & 0xF) |
                                ((((bk.qh[qhb + (e2 & 31)] >> (2 * (e2 >> 5))) & 3)) << 4);
                g[(kk / 4) * 64 + n * 4 + (kk % 4)] = (uint8_t) (int8_t) (raw - 32);
            }
        }
        {   // every row's scales and d, not just row 0's: the decode kernel reads all 16 rows of the tile
            int8_t* rp = scl + ((std::size_t) (tile * 16 + n) * bpr + b) * 16;
            for (int i = 0; i < 16; ++i) rp[i] = bk.scales[i];
            dd[(std::size_t) (tile * 16 + n) * bpr + b] = bk.d;
        }
    }
}

void native_q6k_vnni_tiles(const void* weights, void* buf, int n_in, int n_out, void* stream) {
    const int bpr = n_in / 256, ntiles = n_out / 16;
    uint8_t* p = (uint8_t*) buf;
    const std::size_t tb = q6k_dpas_tiles_bytes(n_in, n_out);
    dpct::get_in_order_queue()
        .parallel_for((std::size_t) ntiles * bpr, [=](sycl::id<1>) {
            native_q6k_vnni_kernel((const Q6KBlockMirror*) weights, p, (int8_t*) (p + tb),
                                   (uint16_t*) (p + tb + (std::size_t) n_out * bpr * 16), bpr, ntiles);
        })
        .wait();
}

template <int NC, int KS>
static void launch_q6k_dpas(const uint8_t* __restrict__ tiles, const int8_t* __restrict__ sc,
                            const uint16_t* __restrict__ dh, const uint8_t* __restrict__ xq, float* __restrict__ y,
                            int n_in, int n_out, sycl::queue* q) {
    constexpr int NT = 16, K = 32;
    const int bpr = n_in / 256, ng = n_in / 32, ntiles = n_out / NT;

    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) KS * NC * NT), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ntiles * KS), sycl::range<1>(KS)),
                         [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            const int tile = (int) (it.get_group(0) / KS);
            const int lid = (int) it.get_local_id(0);
            const int row0 = tile * NT;
            esimd_q6k::simd<uint32_t, 128> half_mask = 0u;
#pragma unroll
            for (int i = 64; i < 128; ++i) half_mask[i] = 0xFFFFFFFFu;   // keep kd 4..7 (the upper 16 of K)
            esimd_q6k::simd<float, NC * NT> Cf = 0.0f;
            for (int sb = lid; sb < bpr; sb += KS) {
                esimd_q6k::simd<uint32_t, NT> rows;
#pragma unroll
                for (int n = 0; n < NT; ++n) rows[n] = (uint32_t) (row0 + n < n_out ? row0 + n : n_out - 1);
                int8_t scl[NT][16];
                float dsp[NT];
                {
#pragma unroll
                    for (int n = 0; n < NT; ++n) dsp[n] = sycl::bit_cast<sycl::half>(*(const uint16_t*) (dh + (size_t) rows[n] * bpr + sb));
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        const uint8_t* sp = (const uint8_t*) (sc + ((size_t) rows[n] * bpr + sb) * 16);
#pragma unroll
                        for (int i = 0; i < 16; ++i) scl[n][i] = (int8_t) sp[i];
                    }
                }
                for (int j = 0; j < 8; ++j) {
                    const uint8_t* tp = tiles + (((size_t) tile * bpr + sb) * 8 + j) * (NT * K);
                    esimd_q6k::simd<uint32_t, 128> Xt = esimd_q6k::block_load<uint32_t, 128>((const uint32_t*) tp);
                    esimd_q6k::simd<uint32_t, 128> Xl = Xt & ~half_mask, Xh = Xt & half_mask;
                    esimd_q6k::simd<int8_t, NT * K> Blo = Xl.template bit_cast_view<int8_t>();
                    esimd_q6k::simd<int8_t, NT * K> Bhi = Xh.template bit_cast_view<int8_t>();
                    esimd_q6k::simd<float, NT> rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        rs0[n] = (float) scl[n][2 * j] * dsp[n];
                        rs1[n] = (float) scl[n][2 * j + 1] * dsp[n];
                    }
                    const int g8 = sb * 8 + j;
                    for (int m = 0; m < NC; ++m) {
                        const uint8_t* xb = xq + ((size_t) m * ng + g8) * 36;
                        const esimd_q6k::simd<int8_t, K> A = esimd_q6k::block_load<int8_t, K>((const int8_t*) (xb + 4));
                        const float dx = (float) sycl::bit_cast<sycl::half>(*(const uint16_t*) xb);
                        esimd_q6k::simd<int, NT> C0 = xmx_q6k::dpas<8, 1, int>(Blo, A);
                        const esimd_q6k::simd<int, NT> C1 = xmx_q6k::dpas<8, 1, int>(Bhi, A);
                        Cf.template select<NT, 1>(m * NT) +=
                            (esimd_q6k::convert<float>(C0) * rs0 + esimd_q6k::convert<float>(C1) * rs1) * dx;
                    }
                }
            }
            float* sh = red.get_multi_ptr<sycl::access::decorated::no>().get();
#pragma unroll
            for (int i = 0; i < NC * NT; ++i) sh[lid * (NC * NT) + i] = Cf[i];
            it.barrier(sycl::access::fence_space::local_space);
            if (lid != 0) return;
#pragma unroll
            for (int m = 0; m < NC; ++m)
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    float v = 0.0f;
#pragma unroll
                    for (int k = 0; k < KS; ++k) v += sh[k * (NC * NT) + m * NT + n];
                    if (row0 + n < n_out) y[(size_t) m * n_out + row0 + n] = v;
                }
        });
    });
}



void native_mmvq_q6k_dpas(const void* tiles, const void* scl, const void* d, const void* x_q8_1, float* y,
                          int n_in, int n_out, int ncols, void* stream) {
    const int bpr = n_in / 256;
    sycl::queue* q = strata::q_of(stream);
#define STRATA_DPAS(NC, KS) \
    launch_q6k_dpas<NC, KS>((const uint8_t*) tiles, (const int8_t*) scl, (const uint16_t*) d, \
                            (const uint8_t*) x_q8_1, y, n_in, n_out, q)
// One work-item per 16-row tile (KS = 1) in the engine path. The K-split was tried three ways and the
// combine is wrong in this translation unit at KS > 1: local memory gives rel 0.87 against the shipped kernel where
// KS = 1 gives 1.7e-07, a scalar atomic_ref is not available under explicit SIMD, and a two-pass reduce over distinct
// global partial slots failed the same way. The identical kernel reduces correctly inside xmx_mmvq_bench, so this is a
// property of this build's code path, not of the algorithm - and until it is root-caused the engine path stays
// single-work-item, which is correct but leaves the GPU underfilled (160 tiles for 2560 rows).
#define STRATA_DPAS_KS(NC) STRATA_DPAS(NC, 1)
    switch (ncols) {
        case 1: STRATA_DPAS_KS(1); break;
        case 2: STRATA_DPAS_KS(2); break;
        case 3: STRATA_DPAS_KS(3); break;
        case 4: STRATA_DPAS_KS(4); break;
        case 5: STRATA_DPAS_KS(5); break;
        case 6: STRATA_DPAS_KS(6); break;
        case 7: STRATA_DPAS_KS(7); break;
        default: STRATA_DPAS_KS(8); break;
    }
#undef STRATA_DPAS_KS
#undef STRATA_DPAS
}

}  // namespace strata::kernels