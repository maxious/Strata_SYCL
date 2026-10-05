// xmx_gateup_bench (exp 43): DPAS int8 MMVQ, fused gate+up for the shared expert - does one activation pass plus a
// float32 GLU epilogue beat the unfused sequence (two DPAS projections + the port's `swiglu_kernel`)?
//
// This is exp 42's Q6_K DPAS mechanism with a second weight stream. Arms, all in one process so every ratio is
// same-run:
//   1  fused        this experiment: gate and up weight tiles, one activation read, silu(gate)*up in the epilogue
//   2  unfused      baseline: two xmx_dpas_q6k projections (gate, up) then the port's `swiglu_kernel` (double)
//   3  1 projection the raw exp 42 kernel on the gate stream - the self-check that this build reproduces exp 42's
//                   published plateau, so the fused/unfused ratio is anchored to a number already on the record
//
// Correctness checks all three values the design names: the gate projection, the up projection and their product,
// each against an fp64 host reference. The fused arm's raw gate/up accumulators are dumped by a second instantiation
// of the same kernel (STORE_GLU=false) that shares the exact DPAS accumulation of the timed arm, because the
// product alone hides an error in either input. The reference product uses float64 silu; the fused epilogue uses
// float32 silu, so the product is not bit-identical to the reference by construction - the bench reports the
// float32-vs-float64 epilogue gap separately.
//
//   xmx_gateup_bench [n_in=2560] [n_out=2560] [reps=400]
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

struct Q6KBlock {                     // the port's AOS Q6_K, GGUF block order
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
};
static_assert(sizeof(Q6KBlock) == 210);

static ESIMD_INLINE float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

// ---------------------------------------------------------------------------
// exp 42's single-projection kernel, verbatim. Used by the baseline's two projections and by the self-check arm.
// ---------------------------------------------------------------------------
template <int NC, int KS, bool ONE_DPAS>
static void xmx_dpas_q6k_kernel(const uint8_t* __restrict__ tiles, const uint8_t* __restrict__ xq,
                                float* __restrict__ y, int n_in, int n_out, int ntiles,
                                const int8_t* __restrict__ sc, const sycl::half* __restrict__ dh, sycl::queue* q) {
    constexpr int NT = 16, K = 32;
    const int bpr = n_in / 256, ng = n_in / 32;

    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) KS * NC * NT), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ntiles * KS), sycl::range<1>(KS)),
                         [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            const int tile = (int) (it.get_group(0) / KS);
            const int lid = (int) it.get_local_id(0);
            const int row0 = tile * NT;
            esimd::simd<uint32_t, 128> half_mask = 0u;
#pragma unroll
            for (int i = 64; i < 128; ++i) half_mask[i] = 0xFFFFFFFFu;   // keep kd 4..7 (the upper 16 of K)
            esimd::simd<float, NC * NT> Cf = 0.0f;
            for (int sb = lid; sb < bpr; sb += KS) {
                esimd::simd<uint32_t, NT> rows;
#pragma unroll
                for (int n = 0; n < NT; ++n) rows[n] = (uint32_t) (row0 + n < n_out ? row0 + n : n_out - 1);
                int8_t scl[NT][16];
                float dsp[NT];
                {
#pragma unroll
                    for (int n = 0; n < NT; ++n) dsp[n] = h2f(*(const uint16_t*) (dh + (size_t) rows[n] * bpr + sb));
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        const uint8_t* sp = (const uint8_t*) (sc + ((size_t) rows[n] * bpr + sb) * 16);
#pragma unroll
                        for (int i = 0; i < 16; ++i) scl[n][i] = (int8_t) sp[i];
                    }
                }
                for (int j = 0; j < 8; ++j) {
                    const uint8_t* tp = tiles + (((size_t) tile * bpr + sb) * 8 + j) * (NT * K);
                    esimd::simd<uint32_t, 128> Xt = esimd::block_load<uint32_t, 128>((const uint32_t*) tp);
                    esimd::simd<uint32_t, 128> Xl = Xt & ~half_mask, Xh = Xt & half_mask;
                    esimd::simd<int8_t, NT * K> Blo = Xl.template bit_cast_view<int8_t>();
                    esimd::simd<int8_t, NT * K> Bhi = Xh.template bit_cast_view<int8_t>();
                    esimd::simd<float, NT> rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        rs0[n] = (float) scl[n][2 * j] * dsp[n];
                        rs1[n] = (float) scl[n][2 * j + 1] * dsp[n];
                    }
                    const int g8 = sb * 8 + j;
                    for (int m = 0; m < NC; ++m) {
                        const uint8_t* xb = xq + ((size_t) m * ng + g8) * 36;
                        const esimd::simd<int8_t, K> A = esimd::block_load<int8_t, K>((const int8_t*) (xb + 4));
                        const float dx = (float) sycl::bit_cast<sycl::half>(*(const uint16_t*) xb);
                        esimd::simd<int, NT> C0 = xmx::dpas<8, 1, int>(Blo, A);
                        esimd::simd<int, NT> C1 = 0;
                        if constexpr (!ONE_DPAS) C1 = xmx::dpas<8, 1, int>(Bhi, A);
                        Cf.template select<NT, 1>(m * NT) +=
                            (esimd::convert<float>(C0) * rs0 + esimd::convert<float>(C1) * rs1) * dx;
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

template <int NC>
static void xmx_dpas_q6k(const void* tiles, const void* xq, float* y, int n_in, int n_out, const void* sc,
                         const void* dh, sycl::queue* q) {
    const int ntiles = (n_out + 15) / 16;
    const int bpr = n_in / 256;
    const uint8_t* t = (const uint8_t*) tiles;
    const uint8_t* x = (const uint8_t*) xq;
    const int8_t* scp = (const int8_t*) sc;
    const sycl::half* dhp = (const sycl::half*) dh;
#define XMX_RUN(KS)                                                                                          \
    do {                                                                                                       \
        if (std::getenv("XMX_ONE_DPAS"))                                                                      \
            xmx_dpas_q6k_kernel<NC, KS, true>(t, x, y, n_in, n_out, ntiles, scp, dhp, q);                      \
        else                                                                                                   \
            xmx_dpas_q6k_kernel<NC, KS, false>(t, x, y, n_in, n_out, ntiles, scp, dhp, q);                     \
    } while (0)
    if (bpr >= 8) XMX_RUN(8);
    else if (bpr >= 4) XMX_RUN(4);
    else if (bpr >= 2) XMX_RUN(2);
    else XMX_RUN(1);
#undef XMX_RUN
}

// ---------------------------------------------------------------------------
// The fused gate+up kernel. Two weight streams, two accumulators, one activation read, GLU in the epilogue.
// STORE_GLU=true  -> timed arm: out = silu(gate)*up in float32, written to `y`.
// STORE_GLU=false -> correctness dump: raw gate accumulator to `y`, raw up accumulator to `y2`. The DPAS
//                    accumulation is the same code path, so the dump is the timed arm's inputs.
// ---------------------------------------------------------------------------
template <int NC, int KS, bool ONE_DPAS, bool STORE_GLU>
static void xmx_dpas_q6k_fused_kernel(const uint8_t* __restrict__ tg, const uint8_t* __restrict__ tu,
                                      const uint8_t* __restrict__ xq, float* __restrict__ y, float* __restrict__ y2,
                                      int n_in, int n_out, int ntiles, const int8_t* __restrict__ scg,
                                      const sycl::half* __restrict__ dhg, const int8_t* __restrict__ scu,
                                      const sycl::half* __restrict__ dhu, sycl::queue* q) {
    constexpr int NT = 16, K = 32;
    const int bpr = n_in / 256, ng = n_in / 32;

    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) 2 * KS * NC * NT), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ntiles * KS), sycl::range<1>(KS)),
                         [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            const int tile = (int) (it.get_group(0) / KS);
            const int lid = (int) it.get_local_id(0);
            const int row0 = tile * NT;
            esimd::simd<uint32_t, 128> half_mask = 0u;
#pragma unroll
            for (int i = 64; i < 128; ++i) half_mask[i] = 0xFFFFFFFFu;
            esimd::simd<float, NC * NT> Cg = 0.0f, Cu = 0.0f;
            for (int sb = lid; sb < bpr; sb += KS) {
                esimd::simd<uint32_t, NT> rows;
#pragma unroll
                for (int n = 0; n < NT; ++n) rows[n] = (uint32_t) (row0 + n < n_out ? row0 + n : n_out - 1);
                int8_t sclg[NT][16], sclu[NT][16];
                float dspg[NT], dspu[NT];
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    dspg[n] = h2f(*(const uint16_t*) (dhg + (size_t) rows[n] * bpr + sb));
                    dspu[n] = h2f(*(const uint16_t*) (dhu + (size_t) rows[n] * bpr + sb));
                }
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const uint8_t* spg = (const uint8_t*) (scg + ((size_t) rows[n] * bpr + sb) * 16);
                    const uint8_t* spu = (const uint8_t*) (scu + ((size_t) rows[n] * bpr + sb) * 16);
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        sclg[n][i] = (int8_t) spg[i];
                        sclu[n][i] = (int8_t) spu[i];
                    }
                }
                for (int j = 0; j < 8; ++j) {
                    const uint8_t* tpg = tg + (((size_t) tile * bpr + sb) * 8 + j) * (NT * K);
                    const uint8_t* tpu = tu + (((size_t) tile * bpr + sb) * 8 + j) * (NT * K);
                    esimd::simd<uint32_t, 128> Xtg = esimd::block_load<uint32_t, 128>((const uint32_t*) tpg);
                    esimd::simd<uint32_t, 128> Xtu = esimd::block_load<uint32_t, 128>((const uint32_t*) tpu);
                    esimd::simd<uint32_t, 128> Xlg = Xtg & ~half_mask, Xhg = Xtg & half_mask;
                    esimd::simd<uint32_t, 128> Xlu = Xtu & ~half_mask, Xhu = Xtu & half_mask;
                    esimd::simd<int8_t, NT * K> Bglo = Xlg.template bit_cast_view<int8_t>();
                    esimd::simd<int8_t, NT * K> Bghi = Xhg.template bit_cast_view<int8_t>();
                    esimd::simd<int8_t, NT * K> Bulo = Xlu.template bit_cast_view<int8_t>();
                    esimd::simd<int8_t, NT * K> Buhi = Xhu.template bit_cast_view<int8_t>();
                    esimd::simd<float, NT> rsg0 = 0.0f, rsg1 = 0.0f, rsu0 = 0.0f, rsu1 = 0.0f;
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        rsg0[n] = (float) sclg[n][2 * j] * dspg[n];
                        rsg1[n] = (float) sclg[n][2 * j + 1] * dspg[n];
                        rsu0[n] = (float) sclu[n][2 * j] * dspu[n];
                        rsu1[n] = (float) sclu[n][2 * j + 1] * dspu[n];
                    }
                    const int g8 = sb * 8 + j;
                    for (int m = 0; m < NC; ++m) {
                        const uint8_t* xb = xq + ((size_t) m * ng + g8) * 36;
                        const esimd::simd<int8_t, K> A = esimd::block_load<int8_t, K>((const int8_t*) (xb + 4));
                        const float dx = (float) sycl::bit_cast<sycl::half>(*(const uint16_t*) xb);
                        esimd::simd<int, NT> Cg0 = xmx::dpas<8, 1, int>(Bglo, A);
                        esimd::simd<int, NT> Cg1 = 0;
                        if constexpr (!ONE_DPAS) Cg1 = xmx::dpas<8, 1, int>(Bghi, A);
                        esimd::simd<int, NT> Cu0 = xmx::dpas<8, 1, int>(Bulo, A);
                        esimd::simd<int, NT> Cu1 = 0;
                        if constexpr (!ONE_DPAS) Cu1 = xmx::dpas<8, 1, int>(Buhi, A);
                        Cg.template select<NT, 1>(m * NT) +=
                            (esimd::convert<float>(Cg0) * rsg0 + esimd::convert<float>(Cg1) * rsg1) * dx;
                        Cu.template select<NT, 1>(m * NT) +=
                            (esimd::convert<float>(Cu0) * rsu0 + esimd::convert<float>(Cu1) * rsu1) * dx;
                    }
                }
            }
            float* sh = red.get_multi_ptr<sycl::access::decorated::no>().get();
            const size_t half = (size_t) KS * NC * NT;
#pragma unroll
            for (int i = 0; i < NC * NT; ++i) {
                sh[lid * (NC * NT) + i] = Cg[i];
                sh[half + lid * (NC * NT) + i] = Cu[i];
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (lid != 0) return;
#pragma unroll
            for (int m = 0; m < NC; ++m)
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    float g = 0.0f, u = 0.0f;
#pragma unroll
                    for (int k = 0; k < KS; ++k) {
                        g += sh[(size_t) k * (NC * NT) + m * NT + n];
                        u += sh[half + (size_t) k * (NC * NT) + m * NT + n];
                    }
                    if (row0 + n < n_out) {
                        if constexpr (STORE_GLU) {
                            const float sg = g / (1.0f + sycl::exp(-g));   // float32 silu, then * up
                            y[(size_t) m * n_out + row0 + n] = sg * u;
                        } else {
                            y[(size_t) m * n_out + row0 + n] = g;
                            y2[(size_t) m * n_out + row0 + n] = u;
                        }
                    }
                }
        });
    });
}

template <int NC, bool STORE_GLU>
static void xmx_dpas_q6k_fused(const void* tg, const void* tu, const void* xq, float* y, float* y2, int n_in,
                               int n_out, const void* scg, const void* dhg, const void* scu, const void* dhu,
                               sycl::queue* q) {
    const int ntiles = (n_out + 15) / 16;
    const int bpr = n_in / 256;
    const uint8_t* tgg = (const uint8_t*) tg;
    const uint8_t* tuu = (const uint8_t*) tu;
    const uint8_t* x = (const uint8_t*) xq;
    const int8_t* scgp = (const int8_t*) scg;
    const sycl::half* dhgp = (const sycl::half*) dhg;
    const int8_t* scup = (const int8_t*) scu;
    const sycl::half* dhup = (const sycl::half*) dhu;
#define XMX_FUSED_RUN(KS)                                                                                       \
    do {                                                                                                          \
        if (std::getenv("XMX_ONE_DPAS"))                                                                         \
            xmx_dpas_q6k_fused_kernel<NC, KS, true, STORE_GLU>(tgg, tuu, x, y, y2, n_in, n_out, ntiles, scgp,     \
                                                               dhgp, scup, dhup, q);                              \
        else                                                                                                      \
            xmx_dpas_q6k_fused_kernel<NC, KS, false, STORE_GLU>(tgg, tuu, x, y, y2, n_in, n_out, ntiles, scgp,    \
                                                                dhgp, scup, dhup, q);                             \
    } while (0)
    if (bpr >= 8) XMX_FUSED_RUN(8);
    else if (bpr >= 4) XMX_FUSED_RUN(4);
    else if (bpr >= 2) XMX_FUSED_RUN(2);
    else XMX_FUSED_RUN(1);
#undef XMX_FUSED_RUN
}

// ---------------------------------------------------------------------------
// The port's `swiglu_kernel` (src/kernels/cuda/shared_expert.dp.cpp), reproduced verbatim: silu(gate) computed in
// double, multiplied by up, cast to float. This is the engine's own unfused epilogue, so the baseline is not a
// re-implementation with different rounding.
// ---------------------------------------------------------------------------
static void swiglu_f64(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int n,
                       sycl::queue* q) {
    constexpr int T = 128;
    q->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((n + T - 1) / T) * T), sycl::range<1>(T)),
                         [=](sycl::nd_item<1> it) {
                             const int i = (int) it.get_global_id(0);
                             if (i >= n) return;
                             const double x = (double) gate[i];
                             out[i] = (float)(x / (1.0 + sycl::exp(-x))) * up[i];
                         });
    });
}

// the Q6_K 6-bit code of element e, straight from the GGUF block (the value the DPAS sees, before scale and d)
static inline int q6k_code(const Q6KBlock& b, int e) {
    const int e2 = e & 127;
    const int qlb = (e >= 128) ? 64 : 0, qhb = (e >= 128) ? 32 : 0;
    const int nib = (e2 >= 64) ? 4 : 0;
    const int raw = ((b.ql[qlb + (e2 & 63)] >> nib) & 0xF) | ((((b.qh[qhb + (e2 & 31)] >> (2 * (e2 >> 5))) & 3)) << 4);
    return raw - 32;
}

// One-time reorder into DPAS B order, the exp 42 layout.
static void reorder_q6k_vnni(const std::vector<Q6KBlock>& aos, std::vector<uint8_t>& tiles, std::vector<int8_t>& scl,
                             std::vector<uint16_t>& dh, int n_out, int bpr) {
    const int ntiles = n_out / 16;
    tiles.assign((size_t) ntiles * bpr * 8 * 512, 0);
    scl.assign((size_t) n_out * bpr * 16, 0);
    dh.assign((size_t) n_out * bpr, 0);
    for (int r = 0; r < n_out; ++r)
        for (int b = 0; b < bpr; ++b) {
            const Q6KBlock& blk = aos[(size_t) r * bpr + b];
            std::memcpy(scl.data() + ((size_t) r * bpr + b) * 16, blk.scales, 16);
            dh[(size_t) r * bpr + b] = blk.d;
            const int tile = r / 16, n = r % 16;
            for (int j = 0; j < 8; ++j)
                for (int kk = 0; kk < 32; ++kk) {
                    const int code = q6k_code(blk, 32 * j + kk);
                    tiles[(((size_t) tile * bpr + b) * 8 + j) * 512 + (kk / 4) * 64 + n * 4 + (kk % 4)] = (uint8_t) code;
                }
        }
}

// llama.cpp's dequantize_row_q6_K, used only for the correctness reference
static void dequant_q6k_row(const Q6KBlock& x, float* y) {
    const float d = h2f(x.d);
    for (int n = 0; n < 256; n += 128) {
        const uint8_t* ql = x.ql + n / 2;
        const uint8_t* qh = x.qh + n / 4;
        const int8_t* sc = x.scales + n / 16;
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
    }
}

static std::vector<Q6KBlock> make_weights(int n_out, int bpr, uint32_t seed) {
    const size_t nb = (size_t) n_out * bpr;
    std::vector<Q6KBlock> hos(nb);
    for (size_t b = 0; b < nb; ++b) {
        for (int i = 0; i < 128; ++i) hos[b].ql[i] = (uint8_t) ((b * 2654435761u + i * 40503u + seed) >> 11);
        for (int i = 0; i < 64; ++i) hos[b].qh[i] = (uint8_t) ((b * 2246822519u + i * 3266489917u + seed) >> 13);
        for (int i = 0; i < 16; ++i) hos[b].scales[i] = (int8_t) (1 + ((b * 31 + i * 17 + seed) % 60));
        hos[b].d = sycl::bit_cast<uint16_t>(sycl::half(0.02f + 0.01f * (float) ((b + seed) % 7)));
    }
    return hos;
}

template <int NC>
static void run_fused_glu(const void* wvg, const void* wvu, const void* xq, float* y, int n_in, int n_out,
                          const void* wscg, const void* wdhg, const void* wscu, const void* wdhu, sycl::queue* q) {
    xmx_dpas_q6k_fused<NC, true>(wvg, wvu, xq, y, nullptr, n_in, n_out, wscg, wdhg, wscu, wdhu, q);
}

template <int NC>
static void run_fused_dump(const void* wvg, const void* wvu, const void* xq, float* yg, float* yu, int n_in,
                           int n_out, const void* wscg, const void* wdhg, const void* wscu, const void* wdhu,
                           sycl::queue* q) {
    xmx_dpas_q6k_fused<NC, false>(wvg, wvu, xq, yg, yu, n_in, n_out, wscg, wdhg, wscu, wdhu, q);
}

template <int NC>
static void run_proj(const void* wv, const void* xq, float* y, int n_in, int n_out, const void* wsc,
                     const void* wdh, sycl::queue* q) {
    xmx_dpas_q6k<NC>(wv, xq, y, n_in, n_out, wsc, wdh, q);
}

int main(int argc, char** argv) {
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 2560;
    const int n_out = argc > 2 ? std::atoi(argv[2]) : 2560;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 400;
    const int cols[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    const int nmax = 8;
    const int bpr = n_in / 256;
    const int ng = n_in / 32;
    if (n_in % 256 || n_out % 16) { std::printf("n_in must be a multiple of 256, n_out of 16\n"); return 2; }
    const size_t nb = (size_t) n_out * bpr;

    std::vector<Q6KBlock> hos_g = make_weights(n_out, bpr, 0u);
    std::vector<Q6KBlock> hos_u = make_weights(n_out, bpr, 0x9E3779B9u);
    std::vector<uint8_t> htiles_g, htiles_u;
    std::vector<int8_t> hsc_g, hsc_u;
    std::vector<uint16_t> hdh_g, hdh_u;
    reorder_q6k_vnni(hos_g, htiles_g, hsc_g, hdh_g, n_out, bpr);
    reorder_q6k_vnni(hos_u, htiles_u, hsc_u, hdh_u, n_out, bpr);

    sycl::queue* q = &dpct::get_in_order_queue();
    std::printf("device: %s   shape %d x %d  (%.2f MB packed Q6_K x2, %.2f MB VNNI tiles x2)\n",
                q->get_device().get_info<sycl::info::device::name>().c_str(), n_out, n_in, nb * 210 / 1e6,
                htiles_g.size() / 1e6);
    void* wvg = sycl::malloc_device(htiles_g.size(), *q);
    void* wvu = sycl::malloc_device(htiles_u.size(), *q);
    void* wscg = sycl::malloc_device(hsc_g.size(), *q);
    void* wscu = sycl::malloc_device(hsc_u.size(), *q);
    void* wdhg = sycl::malloc_device(hdh_g.size() * 2, *q);
    void* wdhu = sycl::malloc_device(hdh_u.size() * 2, *q);
    q->memcpy(wvg, htiles_g.data(), htiles_g.size()).wait();
    q->memcpy(wvu, htiles_u.data(), htiles_u.size()).wait();
    q->memcpy(wscg, hsc_g.data(), hsc_g.size()).wait();
    q->memcpy(wscu, hsc_u.data(), hsc_u.size()).wait();
    q->memcpy(wdhg, hdh_g.data(), hdh_g.size() * 2).wait();
    q->memcpy(wdhu, hdh_u.data(), hdh_u.size() * 2).wait();

    std::vector<float> xf((size_t) nmax * n_in);
    for (size_t t = 0; t < (size_t) nmax; ++t)
        for (int k = 0; k < n_in; ++k)
            xf[(size_t) k + t * n_in] = (float) ((int) ((k * 7919 + t * 104729) % 1000) - 500) / 250.f;
    void* xq = sycl::malloc_device((size_t) nmax * ng * 36, *q);
    strata::kernels::native_quantize_q8_1(xf.data(), xq, n_in, nmax, q);

    float* y_fus = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    float* y_fusg = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    float* y_fusu = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    float* y_g = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    float* y_u = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    float* y_base = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    q->wait();

    int fails = 0;
    for (int ci = 0; ci < 8; ++ci) {
        const int nc = cols[ci];
        auto run_fused = [&] {
            switch (nc) {
                case 1: run_fused_glu<1>(wvg, wvu, xq, y_fus, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 2: run_fused_glu<2>(wvg, wvu, xq, y_fus, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 3: run_fused_glu<3>(wvg, wvu, xq, y_fus, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 4: run_fused_glu<4>(wvg, wvu, xq, y_fus, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 5: run_fused_glu<5>(wvg, wvu, xq, y_fus, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 6: run_fused_glu<6>(wvg, wvu, xq, y_fus, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 7: run_fused_glu<7>(wvg, wvu, xq, y_fus, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                default: run_fused_glu<8>(wvg, wvu, xq, y_fus, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
            }
        };
        auto run_dump = [&] {
            switch (nc) {
                case 1: run_fused_dump<1>(wvg, wvu, xq, y_fusg, y_fusu, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 2: run_fused_dump<2>(wvg, wvu, xq, y_fusg, y_fusu, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 3: run_fused_dump<3>(wvg, wvu, xq, y_fusg, y_fusu, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 4: run_fused_dump<4>(wvg, wvu, xq, y_fusg, y_fusu, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 5: run_fused_dump<5>(wvg, wvu, xq, y_fusg, y_fusu, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 6: run_fused_dump<6>(wvg, wvu, xq, y_fusg, y_fusu, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                case 7: run_fused_dump<7>(wvg, wvu, xq, y_fusg, y_fusu, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
                default: run_fused_dump<8>(wvg, wvu, xq, y_fusg, y_fusu, n_in, n_out, wscg, wdhg, wscu, wdhu, q); break;
            }
        };
        auto run_unfused = [&] {
            switch (nc) {
                case 1:
                    run_proj<1>(wvg, xq, y_g, n_in, n_out, wscg, wdhg, q);
                    run_proj<1>(wvu, xq, y_u, n_in, n_out, wscu, wdhu, q);
                    break;
                case 2:
                    run_proj<2>(wvg, xq, y_g, n_in, n_out, wscg, wdhg, q);
                    run_proj<2>(wvu, xq, y_u, n_in, n_out, wscu, wdhu, q);
                    break;
                case 3:
                    run_proj<3>(wvg, xq, y_g, n_in, n_out, wscg, wdhg, q);
                    run_proj<3>(wvu, xq, y_u, n_in, n_out, wscu, wdhu, q);
                    break;
                case 4:
                    run_proj<4>(wvg, xq, y_g, n_in, n_out, wscg, wdhg, q);
                    run_proj<4>(wvu, xq, y_u, n_in, n_out, wscu, wdhu, q);
                    break;
                case 5:
                    run_proj<5>(wvg, xq, y_g, n_in, n_out, wscg, wdhg, q);
                    run_proj<5>(wvu, xq, y_u, n_in, n_out, wscu, wdhu, q);
                    break;
                case 6:
                    run_proj<6>(wvg, xq, y_g, n_in, n_out, wscg, wdhg, q);
                    run_proj<6>(wvu, xq, y_u, n_in, n_out, wscu, wdhu, q);
                    break;
                case 7:
                    run_proj<7>(wvg, xq, y_g, n_in, n_out, wscg, wdhg, q);
                    run_proj<7>(wvu, xq, y_u, n_in, n_out, wscu, wdhu, q);
                    break;
                default:
                    run_proj<8>(wvg, xq, y_g, n_in, n_out, wscg, wdhg, q);
                    run_proj<8>(wvu, xq, y_u, n_in, n_out, wscu, wdhu, q);
                    break;
            }
            swiglu_f64(y_g, y_u, y_base, nc * n_out, q);
        };
        auto run_proj1 = [&] { run_proj<1>(wvg, xq, y_g, n_in, n_out, wscg, wdhg, q); };

        // correctness: gate, up and product against fp64, using the fused arm's own accumulators (dump) so a
        // product error cannot hide behind an input error
        run_fused(); run_dump(); run_unfused(); q->wait();
        std::vector<float> rf((size_t) nc * n_out), rg((size_t) nc * n_out), ru((size_t) nc * n_out),
            rb((size_t) nc * n_out);
        q->memcpy(rf.data(), y_fus, rf.size() * 4).wait();
        q->memcpy(rg.data(), y_fusg, rg.size() * 4).wait();
        q->memcpy(ru.data(), y_fusu, ru.size() * 4).wait();
        q->memcpy(rb.data(), y_base, rb.size() * 4).wait();
        const int nref = 48;
        std::vector<float> wg(n_in), wu(n_in);
        double sg = 0, su = 0, sp = 0;
        double eg = 0, eu = 0, ep = 0, eb = 0, epil = 0;
        for (int row = 0; row < nref; ++row) {
            for (int b = 0; b < bpr; ++b) {
                dequant_q6k_row(hos_g[(size_t) row * bpr + b], &wg[(size_t) b * 256]);
                dequant_q6k_row(hos_u[(size_t) row * bpr + b], &wu[(size_t) b * 256]);
            }
            for (int t = 0; t < nc; ++t) {
                double ag = 0, au = 0;
                for (int k = 0; k < n_in; ++k) {
                    ag += (double) wg[k] * xf[(size_t) k + t * n_in];
                    au += (double) wu[k] * xf[(size_t) k + t * n_in];
                }
                const double silu = ag / (1.0 + std::exp(-ag));      // float64 silu
                const double prod = silu * au;                        // float64 product
                const size_t i = (size_t) row + t * n_out;
                sg += ag * ag; eg += (rg[i] - ag) * (rg[i] - ag);
                su += au * au; eu += (ru[i] - au) * (ru[i] - au);
                sp += prod * prod; ep += (rf[i] - prod) * (rf[i] - prod);
                eb += (rb[i] - prod) * (rb[i] - prod);
                // the float32 epilogue applied to the fp64 projections: isolates the epilogue precision gap
                const float g32 = (float) ag, u32 = (float) au;
                const float p32 = (g32 / (1.0f + std::exp(-g32))) * u32;
                epil += (p32 - prod) * (p32 - prod);
            }
        }
        const double rmsg = std::sqrt(eg / sg), rmsu = std::sqrt(eu / su), rmsp = std::sqrt(ep / sp),
                     rmsb = std::sqrt(eb / sp), rmse = std::sqrt(epil / sp);
        // stated tolerances: each projection < 1e-2 (exp 42's DPAS plateau is 3.4e-3, ~3x headroom);
        // the product < 2e-2 (both inputs plus the float32 epilogue)
        const bool ok = rmsg < 1e-2 && rmsu < 1e-2 && rmsp < 2e-2;
        if (!ok) ++fails;
        std::printf("ncols %d  rel-RMS vs fp64:  gate %.2e | up %.2e | fused product %.2e | baseline product %.2e |"
                    " fp32-vs-fp64 silu alone %.2e  %s\n",
                    nc, rmsg, rmsu, rmsp, rmsb, rmse, ok ? "ok" : "FAIL");

        auto warm = [&](auto&& fn) {
            const auto w0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 300) {
                for (int i = 0; i < 20; ++i) fn();
                q->wait();
            }
        };
        auto time = [&](auto&& fn) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) fn();
            q->wait();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        warm(run_fused); warm(run_unfused); warm(run_proj1);
        const double tf = time(run_fused), tu = time(run_unfused), t1 = time(run_proj1);
        std::printf("%2d cols:  fused gate+up %6.1f us | unfused 2-proj+swiglu %6.1f us (fused is %.2fx) |"
                    " 1 projection %6.1f us\n",
                    nc, tf, tu, tu / tf, t1);
    }
    std::printf(fails ? "xmx_gateup_bench: %d FAILURES\n" : "xmx_gateup_bench: all arms within tolerance\n", fails);
    return fails ? 1 : 0;
}
