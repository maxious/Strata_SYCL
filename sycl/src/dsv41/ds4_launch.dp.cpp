// Hand-ported SYCL copies of the dsv41 kernel-launcher families that src/dsv41/ds4_model.cu calls and that no other
// port covers: the window sparse attention (ds4_kernels.cu's sparse_attn_kernel :771, sparse_attn_packed_kernel :951
// and their launcher), the Hyper-Connections pre-norm (hc_pre_norm_kernel :338) and mix scaling
// (hc_mix_scale_kernel :1689), and the debug projection (pred_proj_k, ds4_model.cu:509).  The two F16 GEMVs and the
// F32 GEMV are ds4_gemv.dp.cpp's, the vision tower's softmax is ds4_softmax.dp.cpp's.  Hand-ported: dpct exits -46
// on ds4_model.cu and its draft does not compile.  A warp's butterfly sums become ordered work-group sums (a scalar
// stand-in; these families reduce over no joint_matrix tile), so the last bits can differ from CUDA while staying
// within one BF16 ulp - ds4_parity.cpp's rows check exactly that against CPU references.
#include "ds4_launch.dp.hpp"

#include "ds4_numerics.dp.hpp"

#include <atomic>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace strata::ds4 {
namespace {

constexpr int kThreads = 256;
constexpr int kAttTile = 64;
constexpr int kStage = 16;
constexpr int kAttnScratchT = 7;   // queries the scratch path serves at heads_per_block 0
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline float bf16f(uint16_t b) { return sycl::bit_cast<float>((uint32_t) b << 16); }

// an E2M1 code's value times 2^-14 (ds4_kernels.cu's e2m1_val14): (value 2^-14) (scale 2^14) is value * scale
inline float e2m1_val14(uint32_t c) {
    return (float) sycl::bit_cast<sycl::half>((uint16_t) (((c & 8u) << 12) | ((c & 7u) << 9)));
}

inline float grp_sum(float v, float* red, sycl::nd_item<3> it) {
    const int lid = (int) it.get_local_id(2), ls = (int) it.get_local_range(2);
    red[lid] = v;
    it.barrier();
    if (lid == 0) {
        float t = 0.0f;
        for (int i = 0; i < ls; ++i) t += red[i];
        red[0] = t;
    }
    it.barrier();
    return red[0];
}

// values part * 32 .. + 32 of a packed compressed row (kKvFp4, 512 values) as BF16 - exact, the values are E2M1
// times an E4M3 scale
inline void unpack32(const uint8_t* rr, int part, uint16_t* dst) {
    const float k0 = e4m3_value(rr[256 + part * 2]) * 16384.0f, k1 = e4m3_value(rr[256 + part * 2 + 1]) * 16384.0f;
    for (int i = 0; i < 16; ++i) {
        const uint32_t b = rr[part * 16 + i];
        const float ks = i < 8 ? k0 : k1;
        const float lo = e2m1_val14(b) * ks, hi = e2m1_val14(b >> 4) * ks;
        dst[2 * i] = (uint16_t) (sycl::bit_cast<uint32_t>(lo) >> 16);
        dst[2 * i + 1] = (uint16_t) (sycl::bit_cast<uint32_t>(hi) >> 16);
    }
}

inline void attn_rows_of(const AttnRows& ar, int t, int& n_win, int& n_sel, int& w0) {
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = sycl::max((long long) ar.win - ar.p0, ta + 1);   // window buffer row of max(0, pa - win + 1)
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) sycl::min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
}

struct AttnBf16Name {};
struct UnpackSelName {};
template <int G>
struct AttnPackedName {};
struct HcPreNormName {};
struct HcMixScaleName {};
struct PredProjKName {};

// one block a (head, query); the rows are BF16 (window rows, kKvBf16 rows, or a scratch's decoded rows)
void launch_attn_bf16(dpct::queue_ptr s, const float* q, const uint16_t* win, const uint16_t* ckv, const int* sel,
                      int sel_stride, AttnRows ar, const float* sink, int heads, int T, float scale, float* out,
                      bool contig) {
    s->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> qs_l(sycl::range<1>(512), cgh);
        sycl::local_accessor<float, 1> sc_l(sycl::range<1>(kAttTile), cgh);
        sycl::local_accessor<const uint16_t*, 1> rows_l(sycl::range<1>(kAttTile), cgh);
        sycl::local_accessor<float, 1> misc_l(sycl::range<1>(3), cgh);
        cgh.parallel_for<AttnBf16Name>(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) heads * T * kThreads), sycl::range<3>(1, 1, kThreads)),
            [=](sycl::nd_item<3> it) {
                const int lid = (int) it.get_local_id(2);
                const int h = (int) it.get_group(2) / T, t = (int) it.get_group(2) % T;
                float* qs = qs_l.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* sc = sc_l.template get_multi_ptr<sycl::access::decorated::no>().get();
                const uint16_t** rows = rows_l.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* misc = misc_l.template get_multi_ptr<sycl::access::decorated::no>().get();
                int n_win, n_sel, w0;
                attn_rows_of(ar, t, n_win, n_sel, w0);
                const uint16_t* wbase = win + (size_t) w0 * 512;
                const int* sl = sel + (size_t) t * sel_stride;
                for (int i = lid; i < 512; i += kThreads) qs[i] = q[((size_t) t * heads + h) * 512 + i];
                if (lid == 0) {
                    misc[0] = -1e30f;
                    misc[2] = 0.0f;
                }
                it.barrier();
                float o0 = 0.0f, o1 = 0.0f;
                const int total = n_win + n_sel;
                for (int t0 = 0; t0 < total; t0 += kAttTile) {
                    const int nt = sycl::min(kAttTile, total - t0);
                    for (int j = lid; j < nt; j += kThreads) {
                        const int ja = t0 + j;
                        rows[j] = ja < n_win ? wbase + (size_t) ja * 512
                                             : ckv + (contig ? (size_t) t * sel_stride + (ja - n_win)
                                                             : (size_t) sl[ja - n_win]) * 512;
                    }
                    it.barrier();
                    for (int j = lid; j < nt; j += kThreads) {   // one thread a row's dot, the values in order
                        const uint16_t* rr = rows[j];
                        float sv = 0.0f;
                        for (int c = 0; c < 512; ++c) sv += qs[c] * bf16f(rr[c]);
                        sc[j] = sv * scale;
                    }
                    it.barrier();
                    if (lid == 0) {
                        float m = kNegInf;
                        for (int j = 0; j < nt; ++j) m = sycl::fmax(m, sc[j]);
                        const float m_old = misc[0], m_new = sycl::fmax(m_old, m);
                        float sm = 0.0f;
                        for (int j = 0; j < nt; ++j) {
                            const float p = sycl::exp(sc[j] - m_new);
                            sc[j] = p;
                            sm += p;
                        }
                        misc[1] = sycl::exp(m_old - m_new);
                        misc[2] = misc[2] * misc[1] + sm;
                        misc[0] = m_new;
                    }
                    it.barrier();
                    const float cs = misc[1];
                    o0 *= cs;
                    o1 *= cs;
                    for (int j = 0; j < nt; ++j) {
                        const uint16_t* rr = rows[j];
                        o0 += sc[j] * bf16f(rr[2 * lid]);
                        o1 += sc[j] * bf16f(rr[2 * lid + 1]);
                    }
                    it.barrier();
                }
                const float den = misc[2] + sycl::exp(sink[h] - misc[0]);
                float* ov = out + ((size_t) t * heads + h) * 512;
                ov[2 * lid] = o0 / den;
                ov[2 * lid + 1] = o1 / den;
            });
    });
}

// a few queries' selected compressed rows decoded once to BF16 rows (out[t * cap + i] = ckv[sel[t][i]])
void launch_unpack_sel(dpct::queue_ptr s, const uint8_t* ckv, const int* sel, int sel_stride, AttnRows ar, int cap,
                       uint16_t* out, int T) {
    const int nb = (sel_stride + kStage - 1) / kStage;
    s->submit([&](sycl::handler& cgh) {
        cgh.parallel_for<UnpackSelName>(
            sycl::nd_range<3>(sycl::range<3>(1, T, (size_t) nb * kThreads), sycl::range<3>(1, 1, kThreads)),
            [=](sycl::nd_item<3> it) {
                const int lid = (int) it.get_local_id(2);
                const int t = (int) it.get_group(1), b = (int) it.get_group(2);
                const int i = b * kStage + (lid >> 4), part = lid & 15;
                int n_win, n_sel, w0;
                attn_rows_of(ar, t, n_win, n_sel, w0);
                if (i >= n_sel) return;
                unpack32(ckv + (size_t) sel[(size_t) t * sel_stride + i] * kv_row_bytes(kKvFp4, 512), part,
                         out + ((size_t) t * cap + i) * 512 + (size_t) part * 32);
            });
    });
}

// G heads of one query a block: each pass stages 16 rows of a tile as BF16 (window rows as they are, packed rows
// decoded) once for all G heads.  Each head's arithmetic is launch_attn_bf16's: its tiles of 64 rows, each row's
// dot in value order, the running softmax and each output value's weighted sum in row order - the same bits for
// any G.
template <int G>
void launch_attn_packed(dpct::queue_ptr s, const float* q, const uint16_t* win, const uint8_t* ckv, const int* sel,
                        int sel_stride, AttnRows ar, const float* sink, int heads, int T, float scale, float* out) {
    s->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint16_t, 1> tile_l(sycl::range<1>(kStage * 512), cgh);
        sycl::local_accessor<float, 1> sc_l(sycl::range<1>(G * kAttTile), cgh);
        sycl::local_accessor<const uint8_t*, 1> rows_l(sycl::range<1>(kAttTile), cgh);
        sycl::local_accessor<float, 1> misc_l(sycl::range<1>(3 * G), cgh);
        cgh.parallel_for<AttnPackedName<G>>(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) (heads / G) * T * kThreads),
                              sycl::range<3>(1, 1, kThreads)),
            [=](sycl::nd_item<3> it) {
                const int lid = (int) it.get_local_id(2);
                const int t = (int) it.get_group(2) % T, hb = ((int) it.get_group(2) / T) * G;
                uint16_t* tile = tile_l.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* sc = sc_l.template get_multi_ptr<sycl::access::decorated::no>().get();
                const uint8_t** rows = rows_l.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* misc = misc_l.template get_multi_ptr<sycl::access::decorated::no>().get();
                int n_win, n_sel, w0;
                attn_rows_of(ar, t, n_win, n_sel, w0);
                const uint16_t* wbase = win + (size_t) w0 * 512;
                const int* sl = sel + (size_t) t * sel_stride;
                const size_t crow = kv_row_bytes(kKvFp4, 512);
                if (lid < G) {
                    misc[lid * 3] = -1e30f;
                    misc[lid * 3 + 2] = 0.0f;
                }
                float o0[G], o1[G];
#pragma unroll
                for (int g = 0; g < G; ++g) o0[g] = o1[g] = 0.0f;
                const int sr = lid >> 4, part = lid & 15;
                auto stage = [&](int t0, int s0, int n) {
                    if (sr >= n) return;
                    const int j = s0 + sr;
                    uint16_t* dst = tile + (size_t) sr * 512 + (size_t) part * 32;
                    if (t0 + j < n_win) {
                        const uint16_t* src = (const uint16_t*) rows[j] + (size_t) part * 32;
#pragma unroll
                        for (int i = 0; i < 32; ++i) dst[i] = src[i];
                    } else {
                        unpack32(rows[j], part, dst);
                    }
                };
                const int total = n_win + n_sel;
                for (int t0 = 0; t0 < total; t0 += kAttTile) {
                    const int nt = sycl::min(kAttTile, total - t0);
                    for (int j = lid; j < nt; j += kThreads) {
                        const int ja = t0 + j;
                        rows[j] = ja < n_win ? (const uint8_t*) (wbase + (size_t) ja * 512)
                                             : ckv + (size_t) sl[ja - n_win] * crow;
                    }
                    it.barrier();
                    for (int s0 = 0; s0 < nt; s0 += kStage) {   // the dots
                        const int n = sycl::min(kStage, nt - s0);
                        stage(t0, s0, n);
                        it.barrier();
                        for (int idx = lid; idx < G * n; idx += kThreads) {
                            const int g = idx / n, rel = idx - g * n;
                            const uint16_t* vr = tile + (size_t) rel * 512;
                            const float* qr = q + ((size_t) t * heads + hb + g) * 512;
                            float sv = 0.0f;
                            for (int c = 0; c < 512; ++c) sv += qr[c] * bf16f(vr[c]);
                            sc[(size_t) g * kAttTile + s0 + rel] = sv * scale;
                        }
                        it.barrier();
                    }
                    if (lid < G) {   // head lid's running softmax over the tile
                        float* scg = sc + (size_t) lid * kAttTile;
                        float m = kNegInf;
                        for (int j = 0; j < nt; ++j) m = sycl::fmax(m, scg[j]);
                        const float m_old = misc[lid * 3], m_new = sycl::fmax(m_old, m);
                        float sm = 0.0f;
                        for (int j = 0; j < nt; ++j) {
                            const float p = sycl::exp(scg[j] - m_new);
                            scg[j] = p;
                            sm += p;
                        }
                        misc[lid * 3 + 1] = sycl::exp(m_old - m_new);
                        misc[lid * 3 + 2] = misc[lid * 3 + 2] * misc[lid * 3 + 1] + sm;
                        misc[lid * 3] = m_new;
                    }
                    it.barrier();
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        const float cs = misc[g * 3 + 1];
                        o0[g] *= cs;
                        o1[g] *= cs;
                    }
                    for (int s0 = 0; s0 < nt; s0 += kStage) {   // the weighted sums
                        const int n = sycl::min(kStage, nt - s0);
                        stage(t0, s0, n);
                        it.barrier();
                        for (int r = 0; r < n; ++r) {
                            const uint16_t* vr = tile + (size_t) r * 512;
                            const float v0 = bf16f(vr[2 * lid]), v1 = bf16f(vr[2 * lid + 1]);
#pragma unroll
                            for (int g = 0; g < G; ++g) {
                                const float p = sc[(size_t) g * kAttTile + s0 + r];
                                o0[g] += p * v0;
                                o1[g] += p * v1;
                            }
                        }
                        it.barrier();
                    }
                }
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float den = misc[g * 3 + 2] + sycl::exp(sink[hb + g] - misc[g * 3]);
                    float* ov = out + ((size_t) t * heads + hb + g) * 512;
                    ov[2 * lid] = o0[g] / den;
                    ov[2 * lid + 1] = o1[g] / den;
                }
            });
    });
}

}  // namespace

void hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x,
                 dpct::queue_ptr s) {
    s->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red_l(sycl::range<1>(1024), cgh);
        cgh.parallel_for<HcPreNormName>(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) T * 1024), sycl::range<3>(1, 1, 1024)),
            [=](sycl::nd_item<3> it) {
                const int lid = (int) it.get_local_id(2), t = (int) it.get_group(2);
                float* red = red_l.template get_multi_ptr<sycl::access::decorated::no>().get();
                const float* hh = h + (size_t) t * hc * dim;
                const float* mx = mix + (size_t) t * 8;
                float* xo = x + (size_t) t * dim;
                float r[8];
                float ss = 0.0f;
                int n = 0;
                for (int e = lid; e < dim; e += 1024, ++n) {
                    float a = 0.0f;
                    for (int si = 0; si < hc; ++si) a += mx[si] * hh[(size_t) si * dim + e];
                    r[n] = a;
                    ss += a * a;
                }
                const float inv = sycl::rsqrt(grp_sum(ss, red, it) / (float) dim + eps);
                n = 0;
                for (int e = lid; e < dim; e += 1024, ++n) xo[e] = w != nullptr ? r[n] * inv * w[e] : r[n];
            });
    });
}

void hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes, dpct::queue_ptr s) {
    s->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red_l(sycl::range<1>(kThreads), cgh);
        cgh.parallel_for<HcMixScaleName>(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) T * kThreads), sycl::range<3>(1, 1, kThreads)),
            [=](sycl::nd_item<3> it) {
                const int lid = (int) it.get_local_id(2), t = (int) it.get_group(2);
                float* red = red_l.template get_multi_ptr<sycl::access::decorated::no>().get();
                const float* x = h + (size_t) t * hcdim;
                float ss = 0.0f;
                for (int k = lid * 4; k < hcdim; k += kThreads * 4) {
                    const sycl::float4 hv = *(const sycl::float4*) (x + k);
                    ss += hv.x() * hv.x() + hv.y() * hv.y() + hv.z() * hv.z() + hv.w() * hv.w();
                }
                const float inv = sycl::rsqrt(grp_sum(ss, red, it) / (float) hcdim + eps);
                if (lid < n_mix) mixes[(size_t) t * 32 + lid] *= inv;
            });
    });
}

void pred_proj_k(const float* x, int D, int T, float* out, dpct::queue_ptr s) {
    s->submit([&](sycl::handler& cgh) {
        cgh.parallel_for<PredProjKName>(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) T * 256), sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> it) {
                const int j = (int) it.get_local_id(2), t = (int) it.get_group(2);
                const float* xr = x + (size_t) t * D;
                float a = 0.0f;
                for (int d = 0; d < D; ++d) {
                    unsigned hsh = (unsigned) d * 2654435761u ^ ((unsigned) j + 1u) * 40503u;
                    hsh ^= hsh >> 15;
                    hsh *= 2246822519u;
                    hsh ^= hsh >> 13;
                    hsh *= 3266489917u;
                    hsh ^= hsh >> 16;
                    a += (hsh & 1u) ? xr[d] : -xr[d];
                }
                out[(size_t) t * 256 + j] = a * sycl::rsqrt((float) D);
            });
    });
}

void sparse_attn(const float* q, const uint16_t* win, const void* ckv, const int* sel, int sel_stride,
                 const AttnRows& a, const float* sink, int heads, int d, int T, float scale, float* out,
                 dpct::queue_ptr s, int ckv_fmt, int heads_per_block, const AttnScratch* scr) {
    if (d != 512 || (ckv_fmt != kKvFp4 && ckv_fmt != kKvBf16)) {
        std::fprintf(stderr, "ds4 sparse_attn: head dim %d (rows %d) not covered\n", d, ckv_fmt);
        return;
    }
    if (ckv_fmt == kKvBf16) {
        launch_attn_bf16(s, q, win, (const uint16_t*) ckv, sel, sel_stride, a, sink, heads, T, scale, out, false);
        return;
    }
    const bool few = heads_per_block == -1 || (heads_per_block == 0 && T <= kAttnScratchT);
    if (few && scr != nullptr && scr->rows != nullptr && T <= scr->max_t && sel_stride <= scr->cap) {
        launch_unpack_sel(s, (const uint8_t*) ckv, sel, sel_stride, a, scr->cap, scr->rows, T);
        launch_attn_bf16(s, q, win, scr->rows, nullptr, scr->cap, a, sink, heads, T, scale, out, true);
        return;
    }
    int G = heads_per_block;
    if (G != 1 && G != 2 && G != 4 && G != 8) {
        static std::atomic<int> units{0};
        int u = units.load(std::memory_order_relaxed);
        if (u <= 0) {
            u = (int) s->get_device().get_info<sycl::info::device::max_compute_units>();
            if (u <= 0) u = 1;
            units.store(u, std::memory_order_relaxed);
        }
        G = 8;
        while (G > 1 && (heads % G != 0 || (heads / G) * T < 2 * u)) G /= 2;
    }
    if (heads % G != 0) G = 1;
    const uint8_t* c = (const uint8_t*) ckv;
    if (G == 8) launch_attn_packed<8>(s, q, win, c, sel, sel_stride, a, sink, heads, T, scale, out);
    else if (G == 4) launch_attn_packed<4>(s, q, win, c, sel, sel_stride, a, sink, heads, T, scale, out);
    else if (G == 2) launch_attn_packed<2>(s, q, win, c, sel, sel_stride, a, sink, heads, T, scale, out);
    else launch_attn_packed<1>(s, q, win, c, sel, sel_stride, a, sink, heads, T, scale, out);
}

}  // namespace strata::ds4
