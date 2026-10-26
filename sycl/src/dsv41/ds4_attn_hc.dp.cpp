// One home for the dsv41 window sparse attention and the Hyper-Connections pre-norm / mix-scaling kernels and their
// launchers: the SYCLomatic copies of src/dsv41/ds4_kernels.cu (sparse_attn_kernel :814, unpack_sel_kernel :977,
// sparse_attn_packed_kernel :994, hc_pre_norm_kernel :381, hc_mix_scale_kernel :1732; the launchers' dispatch is
// sparse_attn :2054 and hc_pre_norm / hc_mix_scale :1942 / :1954).  Kept operation for operation with those CUDA
// originals: the same butterflies (warp_sum / warp_max / block_sum, shared with ds4_kernels.dp.cpp through
// ds4_numerics.dp.hpp), the same dispatch, the split-KV variant behind DS4_ATTN_SPLIT as in CUDA.
// strata-dsv41 (the engine) and ds4_parity (the parity harness) both link this file, so the copies the harness checks
// are the copies the engine runs.  Before, these families lived twice - hand-ported in ds4_launch.dp.cpp (ordered
// work-group sums in place of the butterflies, no split-KV path) and SYCLomatic in ds4_kernels.dp.cpp - and the two
// sets could drift apart.  This file is the home now; ds4_launch.dp.cpp keeps only pred_proj_k.
#include "ds4_kernels.dp.hpp"

#include "ds4_attn_split.dp.hpp"
#include "ds4_numerics.dp.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>

namespace strata::ds4 {
namespace {

constexpr int kThreads = 256;

int cdiv(int a, int b) { return (a + b - 1) / b; }

void hc_pre_norm_kernel(const float* __restrict__ h, const float* __restrict__ mix,
                                                           const float* __restrict__ w, int dim, int hc, float eps,
                                                           float* __restrict__ x,
                                                           float *red,
                                                           float &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    h += (size_t) t * hc * dim;
    mix += (size_t) t * 8;
    x += (size_t) t * dim;
    float r[8];
    float ss = 0.0f;
    int n = 0;
    for (int e = item_ct1.get_local_id(2); e < dim;
         e += item_ct1.get_local_range(2), ++n) {
        float a = 0.0f;
        for (int s = 0; s < hc; ++s) a += mix[s] * h[(size_t) s * dim + e];
        r[n] = a;
        ss += a * a;
    }
    const float inv = sycl::rsqrt(block_sum(ss, red, total) / (float)dim + eps);
    n = 0;
    for (int e = item_ct1.get_local_id(2); e < dim;
         e += item_ct1.get_local_range(2), ++n)
        x[e] = w != nullptr ? r[n] * inv * w[e] : r[n];
}

constexpr int kAttTile = 64;

// kContig: query t's selected rows are ckv's rows t * sel_stride .. (the scratch of rows decoded once a query) - read
// without sel, a warp's rows of a tile loaded kPre at a time before their dots (0: as the BF16 kernel); the
// arithmetic the same
template <bool kContig, int kPre = 0>
/*
DPCT1110:19: The total declared local variable size in device function
sparse_attn_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
// audit note: DPCT1110 accepted — a register-pressure hint; the plan forbids a speculative rewrite, and a change needs a measurement first
void sparse_attn_kernel(const float *__restrict__ q,
                        const uint16_t *__restrict__ win,
                        const uint16_t *__restrict__ ckv,
                        const int *__restrict__ sel, int sel_stride,
                        AttnRows ar, const float *__restrict__ sink, int heads,
                        int d, float scale, float *__restrict__ out, float *qs,
                        float *sc, uint16_t **rows, float &m_run_s,
                        float &tile_scale_s, float &den_s) {
    // d == 512: a thread owns 2 output values, a lane 16 of a row's dot
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int h = item_ct1.get_group(2), t = item_ct1.get_group(1);
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first =
            sycl::max((long long)ar.win - ar.p0,
                      ta + 1); // the window buffer row of max(0, pa - win + 1)
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int)sycl::min((long long)ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;

    for (int i = item_ct1.get_local_id(2); i < d;
         i += item_ct1.get_local_range(2))
        qs[i] = q[((size_t)t * heads + h) * d + i];
    if (item_ct1.get_local_id(2) == 0) {
        m_run_s = -1e30f;
        den_s = 0.0f;
    }
    /*
    DPCT1065:94: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31,
              wp = item_ct1.get_local_id(2) >> 5;
    float o0 = 0.0f, o1 = 0.0f;
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = sycl::min(kAttTile, total - t0);
        if (item_ct1.get_local_id(2) < nt) {
            const int j = t0 + item_ct1.get_local_id(2);
            rows[item_ct1.get_local_id(2)] = (uint16_t *)(
                j < n_win
                    ? wbase + (size_t)j * d
                    : ckv + (kContig ? (size_t)t * sel_stride + (j - n_win)
                                     : (size_t)sl[j - n_win]) *
                                d);
        }
        /*
        DPCT1118:20: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        // audit note: DPCT1118 accepted — a convergence hint on group functions; the plan forbids a speculative restructure, and a change needs a failing case or a measurement first
        /*
        DPCT1065:95: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
        item_ct1.barrier();
        if (kContig && kPre > 0) {   // the warp's rows of the tile (wp, wp + 8, ..), kPre loads at a time, then dots
#pragma unroll
            for (int k0 = 0; k0 < kAttTile / 8; k0 += (kPre > 0 ? kPre : 1)) {
                sycl::uint4 ra[kPre > 0 ? kPre : 1], rb[kPre > 0 ? kPre : 1];
#pragma unroll
                for (int k = 0; k < kPre; ++k) {
                    const int jj = wp + 8 * (k0 + k);
                    if (jj < nt) {
                        ra[k] = *(const sycl::uint4 *)(rows[jj] + lane * 16);
                        rb[k] = *(const sycl::uint4 *)(rows[jj] + lane * 16 + 8);
                    }
                }
#pragma unroll
                for (int k = 0; k < kPre; ++k) {
                    const int jj = wp + 8 * (k0 + k);
                    if (jj < nt) {
                        const uint16_t* pa = (const uint16_t*) &ra[k];
                        const uint16_t* pb = (const uint16_t*) &rb[k];
                        float s = 0.0f;
#pragma unroll
                        for (int x = 0; x < 8; ++x) s += qs[lane * 16 + x] * bf16f(pa[x]);
#pragma unroll
                        for (int x = 0; x < 8; ++x) s += qs[lane * 16 + 8 + x] * bf16f(pb[x]);
                        s = warp_sum(s);
                        if (lane == 0) sc[jj] = s * scale;
                    }
                }
            }
        } else
            for (int jj = wp; jj < nt; jj += kThreads / 32) {
                const uint16_t* rr = rows[jj];
                const sycl::uint4 a = *(const sycl::uint4 *)(rr + lane * 16);
                const sycl::uint4 b = *(const sycl::uint4 *)(rr + lane * 16 + 8);
                const uint16_t* pa = (const uint16_t*) &a;
                const uint16_t* pb = (const uint16_t*) &b;
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qs[lane * 16 + x] * bf16f(pa[x]);
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qs[lane * 16 + 8 + x] * bf16f(pb[x]);
                s = warp_sum(s);
                if (lane == 0) sc[jj] = s * scale;
            }
        /*
        DPCT1118:21: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        // audit note: DPCT1118 accepted — a convergence hint on group functions; the plan forbids a speculative restructure, and a change needs a failing case or a measurement first
        /*
        DPCT1065:96: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
        item_ct1.barrier();
        if (wp == 0) {
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = sycl::fmax(m, sc[j]);
            m = warp_max(m);
            const float m_old = m_run_s;
            const float m_new = sycl::fmax(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float p = sycl::native::exp(sc[j] - m_new);
                sc[j] = p;
                s += p;
            }
            s = warp_sum(s);
            sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
            if (lane == 0) {
                tile_scale_s = sycl::native::exp(m_old - m_new);
                den_s = den_s * tile_scale_s + s;
                m_run_s = m_new;
            }
        }
        /*
        DPCT1118:22: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        // audit note: DPCT1118 accepted — a convergence hint on group functions; the plan forbids a speculative restructure, and a change needs a failing case or a measurement first
        /*
        DPCT1065:97: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
        item_ct1.barrier();
        const float cs = tile_scale_s;
        o0 *= cs;
        o1 *= cs;
        const int c = item_ct1.get_local_id(2) * 2;
        if (kContig && kPre > 0) {   // (eight rows' loads at once)
#pragma unroll 8
            for (int j = 0; j < nt; ++j) {
                const uint32_t pr = *(const uint32_t*) (rows[j] + c);
                o0 += sc[j] * bf16f((uint16_t) (pr & 0xffffu));
                o1 += sc[j] * bf16f((uint16_t) (pr >> 16));
            }
        } else
            for (int j = 0; j < nt; ++j) {
                const uint32_t pr = *(const uint32_t*) (rows[j] + c);
                o0 += sc[j] * bf16f((uint16_t) (pr & 0xffffu));
                o1 += sc[j] * bf16f((uint16_t) (pr >> 16));
            }
        /*
        DPCT1118:23: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        // audit note: DPCT1118 accepted — a convergence hint on group functions; the plan forbids a speculative restructure, and a change needs a failing case or a measurement first
        /*
        DPCT1065:98: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
        item_ct1.barrier();
    }
    const float den = den_s + sycl::native::exp(sink[h] - m_run_s);
    const int c = item_ct1.get_local_id(2) * 2;
    float* ov = out + ((size_t) t * heads + h) * d;
    ov[c] = o0 / den;
    ov[c + 1] = o1 / den;
}

// ---------------------------------------------------------------- sparse attention over the packed compressed rows
// G heads of one query a block (8 / G warps a head): each pass stages 16 of a tile's rows in shared memory as BF16
// (the window's rows as they are; the packed rows decoded - exact in BF16), once for all G heads, through one 16-byte
// load a thread.  Each head's arithmetic is sparse_attn_kernel's, operation for operation: its tiles of 64 rows, its
// dots by a warp's lanes (16 values each, in order) and the same butterfly, its softmax by one warp the same way, each
// output value's weighted sum by the same thread in row order - the same bits for any G.
constexpr int kStage = 16;   // rows a pass stages (a thread: 32 values of one)

// values part * 32 .. + 32 of a packed compressed row (kKvFp4, 512 values) as BF16 into dst[0 .. 4) (exact: the
// values are E2M1 times an E4M3 scale)
__dpct_inline__ void unpack32(const uint8_t *rr, int part, sycl::uint4 *dst) {
    const sycl::uint4 cw =
        *(const sycl::uint4 *)(rr + part * 16); // 32 codes, two scale groups
    const float k0 = e4m3_value(rr[256 + part * 2]) * 16384.0f, k1 = e4m3_value(rr[256 + part * 2 + 1]) * 16384.0f;
    const uint32_t wv[4] = {cw.x(), cw.y(), cw.z(), cw.w()};
    uint32_t o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {   // values 2i, 2i + 1 (the low and the high nibble of byte i)
        const uint32_t b = wv[i >> 2] >> (8 * (i & 3));
        const float ks = i < 8 ? k0 : k1;
        const float lo = e2m1_val14(b) * ks, hi = e2m1_val14(b >> 4) * ks;   // (exact: their BF16 bits)
        o[i] = (sycl::bit_cast<unsigned int>(lo) >> 16) |
               (sycl::bit_cast<unsigned int>(hi) & 0xffff0000u);
    }
    dst[0] = sycl::uint4(o[0], o[1], o[2], o[3]);
    dst[1] = sycl::uint4(o[4], o[5], o[6], o[7]);
    dst[2] = sycl::uint4(o[8], o[9], o[10], o[11]);
    dst[3] = sycl::uint4(o[12], o[13], o[14], o[15]);
}

// a few queries' attention (a decode step, a short window): the rows each query selects decoded once to BF16 rows -
// out[t * cap + i] = ckv[sel[t][i]] for i < the query's n_sel - for the BF16 kernel reading them in place (once a
// query, not once a head)
void unpack_sel_kernel(const uint8_t* __restrict__ ckv,
                                                              const int* __restrict__ sel, int sel_stride, AttnRows ar,
                                                              int cap, uint16_t* __restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              i = item_ct1.get_group(2) * kStage +
                  (int)(item_ct1.get_local_id(2) >> 4),
              part = item_ct1.get_local_id(2) & 15;
    int n_sel;
    if (ar.gen) {
        const long long pa = ar.p0 + ar.t0 + t;
        n_sel = ar.R > 0 ? (int)sycl::min((long long)ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_sel = ar.nsel[t];
    }
    if (i >= n_sel) return;
    const size_t r = (size_t) t * cap + i;
    unpack32(ckv + (size_t)sel[(size_t)t * sel_stride + i] *
                       kv_row_bytes(kKvFp4, 512),
             part, (sycl::uint4 *)(out + r * 512 + part * 32));
}

template <int G>
/*
DPCT1110:24: The total declared local variable size in device function
sparse_attn_packed_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
// audit note: DPCT1110 accepted — a register-pressure hint; the plan forbids a speculative rewrite, and a change needs a measurement first
void sparse_attn_packed_kernel(const float *__restrict__ q,
                               const uint16_t *__restrict__ win,
                               const uint8_t *__restrict__ ckv,
                               const int *__restrict__ sel, int sel_stride,
                               AttnRows ar, const float *__restrict__ sink,
                               int heads, float scale, float *__restrict__ out,
                               uint16_t *tile, float sc[G][64 /*kAttTile*/],
                               uint8_t **rows, float *m_run_s,
                               float *tile_scale_s, float *den_s) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int d = 512, W = 8 / G; // (kThreads / 32 warps)
    const int hb = item_ct1.get_group(2) * G, t = item_ct1.get_group(1);
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = sycl::max((long long)ar.win - ar.p0, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int)sycl::min((long long)ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;
    const size_t crow = kv_row_bytes(kKvFp4, d);

    const int lane = item_ct1.get_local_id(2) & 31,
              wp = item_ct1.get_local_id(2) >> 5, gw = wp / W, wr = wp % W;
    float qr[16];   // this lane's 16 values of its warp's head
#pragma unroll
    for (int x = 0; x < 16; ++x) qr[x] = q[((size_t) t * heads + hb + gw) * d + lane * 16 + x];
    if (item_ct1.get_local_id(2) < G) {
        m_run_s[item_ct1.get_local_id(2)] = -1e30f;
        den_s[item_ct1.get_local_id(2)] = 0.0f;
    }
    float o0[G], o1[G];
#pragma unroll
    for (int g = 0; g < G; ++g) o0[g] = o1[g] = 0.0f;
    const int c = item_ct1.get_local_id(2) * 2;
    const int sr = item_ct1.get_local_id(2) >> 4,
              part = item_ct1.get_local_id(2) &
                     15; // the staging: a thread's row and 32 values
    // rows [s0, s0 + n) of the tile at t0 into tile[] as BF16
    auto stage = [&](int t0, int s0, int n, uint16_t *tile,
                     uint8_t **rows) {
        if (sr >= n) return;
        const int j = s0 + sr;
        sycl::uint4 *dst = (sycl::uint4 *)(tile + sr * d + part * 32);
        if (t0 + j < n_win) {   // a window row: BF16 already
            const sycl::uint4 *src = (const sycl::uint4 *)(rows[j] + part * 64);
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            return;
        }
        unpack32(rows[j], part, dst);
    };
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = sycl::min(kAttTile, total - t0);
        if (item_ct1.get_local_id(2) < nt) {
            const int j = t0 + item_ct1.get_local_id(2);
            rows[item_ct1.get_local_id(2)] = (uint8_t *)(
                j < n_win ? (const uint8_t *)(wbase + (size_t)j * d)
                          : ckv + (size_t)sl[j - n_win] * crow);
        }
        /*
        DPCT1118:25: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        // audit note: DPCT1118 accepted — a convergence hint on group functions; the plan forbids a speculative restructure, and a change needs a failing case or a measurement first
        /*
        DPCT1065:99: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
        item_ct1.barrier();
        for (int s0 = 0; s0 < nt; s0 += kStage) {   // the dots
            const int n = sycl::min(kStage, nt - s0);
            stage(t0, s0, n, tile, rows);
            /*
            DPCT1118:27: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            // audit note: DPCT1118 accepted — a convergence hint on group functions; the plan forbids a speculative restructure, and a change needs a failing case or a measurement first
            /*
            DPCT1065:101: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
            item_ct1.barrier();
#if !defined(STRATA_USE_HIP)   // (HIP leaves a loop with shuffles and a remainder rolled anyway)
#pragma unroll 2
#endif
            for (int r = wr; r < n; r += W) {   // (two rows' chains at once)
                const uint16_t* vr = tile + r * d + lane * 16;
                const sycl::uint4 a = *(const sycl::uint4 *)vr;
                const sycl::uint4 b = *(const sycl::uint4 *)(vr + 8);
                const uint16_t* pa = (const uint16_t*) &a;
                const uint16_t* pb = (const uint16_t*) &b;
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qr[x] * bf16f(pa[x]);
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qr[8 + x] * bf16f(pb[x]);
                s = warp_sum(s);
                if (lane == 0) sc[gw][s0 + r] = s * scale;
            }
            /*
            DPCT1118:28: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            // audit note: DPCT1118 accepted — a convergence hint on group functions; the plan forbids a speculative restructure, and a change needs a failing case or a measurement first
            /*
            DPCT1065:102: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
            item_ct1.barrier();
        }
        if (wp < G) {   // head wp's softmax over the tile
            float* scg = sc[wp];
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = sycl::fmax(m, scg[j]);
            m = warp_max(m);
            const float m_old = m_run_s[wp];
            const float m_new = sycl::fmax(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float p = sycl::native::exp(scg[j] - m_new);
                scg[j] = p;
                s += p;
            }
            s = warp_sum(s);
            sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
            if (lane == 0) {
                tile_scale_s[wp] = sycl::native::exp(m_old - m_new);
                den_s[wp] = den_s[wp] * tile_scale_s[wp] + s;
                m_run_s[wp] = m_new;
            }
        }
        /*
        DPCT1118:26: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        // audit note: DPCT1118 accepted — a convergence hint on group functions; the plan forbids a speculative restructure, and a change needs a failing case or a measurement first
        /*
        DPCT1065:100: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
        item_ct1.barrier();
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float cs = tile_scale_s[g];
            o0[g] *= cs;
            o1[g] *= cs;
        }
        for (int s0 = 0; s0 < nt; s0 += kStage) {   // the weighted sums
            const int n = sycl::min(kStage, nt - s0);
            stage(t0, s0, n, tile, rows);
            /*
            DPCT1118:29: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            // audit note: DPCT1118 accepted — a convergence hint on group functions; the plan forbids a speculative restructure, and a change needs a failing case or a measurement first
            /*
            DPCT1065:103: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
            item_ct1.barrier();
            for (int r = 0; r < n; ++r) {
                const uint32_t pr = *(const uint32_t*) (tile + r * d + c);
                const float v0 = bf16f((uint16_t) (pr & 0xffffu)), v1 = bf16f((uint16_t) (pr >> 16));
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    o0[g] += sc[g][s0 + r] * v0;
                    o1[g] += sc[g][s0 + r] * v1;
                }
            }
            /*
            DPCT1118:30: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            // audit note: DPCT1118 accepted — a convergence hint on group functions; the plan forbids a speculative restructure, and a change needs a failing case or a measurement first
            /*
            DPCT1065:104: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            // audit note: DPCT1065 accepted — the note suggests a local-only barrier as a perf hint, and some of these barriers fence host-mapped flag traffic; a blanket narrowing is a regression hazard, so the full barriers stay
            item_ct1.barrier();
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float den = den_s[g] + sycl::native::exp(sink[hb + g] - m_run_s[g]);
        float* ov = out + ((size_t) t * heads + hb + g) * d;
        ov[c] = o0[g] / den;
        ov[c + 1] = o1[g] / den;
    }
}

void hc_mix_scale_kernel(const float* __restrict__ h, int hcdim, int n_mix,
                                                                float eps, float* mixes,
                                                                float *red,
                                                                float &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    const float* x = h + (size_t) t * hcdim;
    float ss = 0.0f;
    for (int k = item_ct1.get_local_id(2) * 4; k < hcdim; k += kThreads * 4) {
        const sycl::float4 hv = *(const sycl::float4 *)(x + k);
        ss += hv.x() * hv.x() + hv.y() * hv.y() + hv.z() * hv.z() + hv.w() * hv.w();
    }
    ss = block_sum(ss, red, total);
    const float inv = sycl::rsqrt(ss / (float)hcdim + eps);
    if (item_ct1.get_local_id(2) < n_mix)
        mixes[(size_t)t * 32 + item_ct1.get_local_id(2)] *= inv;
}

}  // namespace

namespace {

// ---- the split-KV attention's scratch: a device allocation a (device, stream), made on first use and kept
constexpr int kSplitG = 4, kSplitR = 40, kSplitCap = 24, kSplitMaxT = 5;
bool split_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("DS4_ATTN_SPLIT");
        return e != nullptr && std::atoi(e) != 0;   // opt-in: no gain measured on 4x R9700 (21.8 vs 21.9 tok/s)
    }();
    return on;
}
attnsplit::AttnSplitScratch split_scratch(dpct::queue_ptr s, int heads) try {
    static std::mutex mu;
    static std::map<std::pair<int, dpct::queue_ptr>,
                    attnsplit::AttnSplitScratch>
        m;
    int dev = 0;
    dev = dpct::get_current_device_id();
    std::lock_guard<std::mutex> lk(mu);
    auto it = m.find({dev, s});
    if (it != m.end()) return it->second;
    attnsplit::AttnSplitScratch sc;
    sc.nsplit = kSplitCap;
    sc.max_t = kSplitMaxT;
    const size_t np = (size_t) kSplitMaxT * heads * kSplitCap;
    if (DPCT_CHECK_ERROR(sc.part = sycl::malloc_device<float>(
                             np * 512, dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(sc.ml = sycl::malloc_device<float>(
                             np * 2, dpct::get_in_order_queue())) != 0) {
        /*
        DPCT1026:147: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        // audit note: DPCT1026 accepted — the removed calls are CUDA-only; cudaGetLastError is redundant in SYCL, and cudaFuncSetAttribute has no SYCL setting; local accessors size the shared memory instead
        if (sc.part != nullptr)
            dpct::dpct_free(sc.part, dpct::get_in_order_queue());
        sc = attnsplit::AttnSplitScratch{};
    }
    m[{dev, s}] = sc;
    return sc;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
}  // namespace

// the multiprocessors of the current device (cached a device)
static int sm_count() try {
    static std::atomic<int> cached[64];
    int dev = 0;
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64) dev = 0;
    int v = cached[dev].load(std::memory_order_relaxed);
    if (v <= 0) {
        if (DPCT_CHECK_ERROR(
                v = dpct::get_device(dev).get_max_compute_units()) != 0 ||
            v <= 0) v = 1;
        cached[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void hc_pre_norm(const float *h, const float *mix, const float *w, int dim,
                 int hc, int T, float eps, float *x, dpct::queue_ptr s) {
    /*
    DPCT1049:55: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    // audit note: DPCT1049 accepted — a work-group size hint on the migrated launches; the plan forbids a speculative resize, and a change needs a measurement first
    s->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> red_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 0> total_acc_ct1(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                  sycl::range<3>(1, 1, 1024),
                              sycl::range<3>(1, 1, 1024)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                hc_pre_norm_kernel(
                    h, mix, w, dim, hc, eps, x,
                    red_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    total_acc_ct1);
            });
    });
}

void hc_mix_scale(const float *h, int hcdim, int n_mix, int T, float eps,
                  float *mixes, dpct::queue_ptr s) {
    s->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> red_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 0> total_acc_ct1(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                  sycl::range<3>(1, 1, kThreads),
                              sycl::range<3>(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                hc_mix_scale_kernel(
                    h, hcdim, n_mix, eps, mixes,
                    red_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    total_acc_ct1);
            });
    });
}

void sparse_attn(const float *q, const uint16_t *win, const void *ckv,
                 const int *sel, int sel_stride, const AttnRows &a,
                 const float *sink, int heads, int d, int T, float scale,
                 float *out, dpct::queue_ptr s, int ckv_fmt,
                 int heads_per_block, const AttnScratch *scr) {
    if (d != 512 || (ckv_fmt != kKvFp4 && ckv_fmt != kKvBf16)) {
        std::fprintf(stderr, "ds4 sparse_attn: head dim %d (rows %d) not covered\n", d, ckv_fmt);
        return;
    }
    if (ckv_fmt == kKvBf16) {
        s->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> qs_acc_ct1(sycl::range<1>(512), cgh);
            /*
            DPCT1101:169: 'kAttTile' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<float, 1> sc_acc_ct1(
                sycl::range<1>(64 /*kAttTile*/), cgh);
            /*
            DPCT1101:170: 'kAttTile' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<uint16_t *, 1> rows_acc_ct1(
                sycl::range<1>(64 /*kAttTile*/), cgh);
            sycl::local_accessor<float, 0> m_run_s_acc_ct1(cgh);
            sycl::local_accessor<float, 0> tile_scale_s_acc_ct1(cgh);
            sycl::local_accessor<float, 0> den_s_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, T, heads) *
                                      sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                    32)]] {
                    sparse_attn_kernel<false>(
                        q, win, (const uint16_t *)ckv, sel, sel_stride, a, sink,
                        heads, d, scale, out,
                        qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        sc_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        rows_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        m_run_s_acc_ct1, tile_scale_s_acc_ct1, den_s_acc_ct1);
                });
        });
        return;
    }
    // a decode step / short verify window of packed rows: the split-KV kernel (Strata's CUDA lab, ds4_attn_split.cuh; 2.9x at
    // one query on an A4000).  The same inputs; the merge sums the chunks in another order, so the bits differ in the last
    // place (4e-7 against a double reference).  DS4_ATTN_SPLIT=1 turns it on (off by default: no measurable decode gain on the R9700 box, the decode is bound elsewhere).
    if (ckv_fmt == kKvFp4 && a.gen == 0 && T >= 1 && T <= kSplitMaxT && heads % kSplitG == 0 && split_enabled()) {
        int total_max = 0;
        for (int t = 0; t < T; ++t) total_max = std::max(total_max, a.nwin[t] + a.nsel[t]);
        if (total_max > 0 && total_max <= kSplitCap * kSplitR) {
            const attnsplit::AttnSplitScratch sc = split_scratch(s, heads);
            if (sc.part != nullptr) {
                attnsplit::sparse_attn_split<kSplitG, kSplitR>(q, win, (const uint8_t*) ckv, kv_row_bytes(kKvFp4, d), sel,
                                                               sel_stride, a, sink, heads, T, scale, out, sc, total_max, s);
                return;
            }
        }
    }
    // a few queries (a decode step, a short window): their selected rows decoded once a query into the scratch, then
    // the BF16 rows' kernel itself (heads_per_block -1: this way at any T)
    const bool few = heads_per_block == -1 || (heads_per_block == 0 && T <= kAttnScratchT);
    if (few && scr != nullptr && scr->rows != nullptr && T <= scr->max_t && sel_stride <= scr->cap) {
        {
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                auto scr_cap_ct4 = scr->cap;
                auto scr_rows_ct5 = scr->rows;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, T, cdiv(sel_stride, kStage)) *
                            sycl::range<3>(1, 1, kThreads),
                        sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        unpack_sel_kernel((const uint8_t *)ckv, sel, sel_stride,
                                          a, scr_cap_ct4, scr_rows_ct5);
                    });
            });
        }
#if !defined(STRATA_USE_HIP)
        if (T == 1)   // (V100: the loads ahead pay for the decode's kernel at one query, not at more)
            s->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<float, 1> qs_acc_ct1(sycl::range<1>(512),
                                                          cgh);
                /*
                DPCT1101:171: 'kAttTile' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
                sycl::local_accessor<float, 1> sc_acc_ct1(
                    sycl::range<1>(64 /*kAttTile*/), cgh);
                /*
                DPCT1101:172: 'kAttTile' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
                sycl::local_accessor<uint16_t *, 1> rows_acc_ct1(
                    sycl::range<1>(64 /*kAttTile*/), cgh);
                sycl::local_accessor<float, 0> m_run_s_acc_ct1(cgh);
                sycl::local_accessor<float, 0> tile_scale_s_acc_ct1(cgh);
                sycl::local_accessor<float, 0> den_s_acc_ct1(cgh);

                const uint16_t *scr_rows_ct2 = scr->rows;
                auto scr_cap_ct4 = scr->cap;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, T, heads) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        sparse_attn_kernel<true, 8>(
                            q, win, scr_rows_ct2, nullptr, scr_cap_ct4, a, sink,
                            heads, d, scale, out,
                            qs_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            sc_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            rows_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            m_run_s_acc_ct1, tile_scale_s_acc_ct1,
                            den_s_acc_ct1);
                    });
            });
        else
#endif
            s->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<float, 1> qs_acc_ct1(sycl::range<1>(512),
                                                          cgh);
                /*
                DPCT1101:173: 'kAttTile' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
                sycl::local_accessor<float, 1> sc_acc_ct1(
                    sycl::range<1>(64 /*kAttTile*/), cgh);
                /*
                DPCT1101:174: 'kAttTile' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
                sycl::local_accessor<uint16_t *, 1> rows_acc_ct1(
                    sycl::range<1>(64 /*kAttTile*/), cgh);
                sycl::local_accessor<float, 0> m_run_s_acc_ct1(cgh);
                sycl::local_accessor<float, 0> tile_scale_s_acc_ct1(cgh);
                sycl::local_accessor<float, 0> den_s_acc_ct1(cgh);

                const uint16_t *scr_rows_ct2 = scr->rows;
                auto scr_cap_ct4 = scr->cap;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, T, heads) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        sparse_attn_kernel<true, 0>(
                            q, win, scr_rows_ct2, nullptr, scr_cap_ct4, a, sink,
                            heads, d, scale, out,
                            qs_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            sc_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            rows_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            m_run_s_acc_ct1, tile_scale_s_acc_ct1,
                            den_s_acc_ct1);
                    });
            });
        return;
    }
    // packed rows: as many heads a block as still leaves two blocks a multiprocessor (a decoded row serves them all)
    int G = heads_per_block;
    if (G != 1 && G != 2 && G != 4 && G != 8) {
        G = 8;
        while (G > 1 && (heads % G != 0 || (heads / G) * T < 2 * sm_count())) G /= 2;
    }
    if (heads % G != 0) G = 1;
    const dpct::dim3 grid(heads / G, T);
    const uint8_t* c = (const uint8_t*) ckv;
    if (G == 8) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:175: 'kStage * d' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<uint16_t, 1> tile_acc_ct1(
                sycl::range<1>(8192 /*kStage * d*/), cgh);
            /*
            DPCT1101:176: 'kAttTile' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<float[8][64 /*kAttTile*/], 0> sc_acc_ct1(cgh);
            /*
            DPCT1101:177: 'kAttTile' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<uint8_t *, 1> rows_acc_ct1(
                sycl::range<1>(64 /*kAttTile*/), cgh);
            sycl::local_accessor<float, 1> m_run_s_acc_ct1(sycl::range<1>(8),
                                                           cgh);
            sycl::local_accessor<float, 1> tile_scale_s_acc_ct1(
                sycl::range<1>(8), cgh);
            sycl::local_accessor<float, 1> den_s_acc_ct1(sycl::range<1>(8),
                                                         cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        sparse_attn_packed_kernel<8>(
                            q, win, c, sel, sel_stride, a, sink, heads, scale,
                            out,
                            tile_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            sc_acc_ct1,
                            rows_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            m_run_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            tile_scale_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            den_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    } else if (G == 4) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:178: 'kStage * d' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<uint16_t, 1> tile_acc_ct1(
                sycl::range<1>(8192 /*kStage * d*/), cgh);
            /*
            DPCT1101:179: 'kAttTile' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<float[4][64 /*kAttTile*/], 0> sc_acc_ct1(cgh);
            /*
            DPCT1101:180: 'kAttTile' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<uint8_t *, 1> rows_acc_ct1(
                sycl::range<1>(64 /*kAttTile*/), cgh);
            sycl::local_accessor<float, 1> m_run_s_acc_ct1(sycl::range<1>(4),
                                                           cgh);
            sycl::local_accessor<float, 1> tile_scale_s_acc_ct1(
                sycl::range<1>(4), cgh);
            sycl::local_accessor<float, 1> den_s_acc_ct1(sycl::range<1>(4),
                                                         cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        sparse_attn_packed_kernel<4>(
                            q, win, c, sel, sel_stride, a, sink, heads, scale,
                            out,
                            tile_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            sc_acc_ct1,
                            rows_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            m_run_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            tile_scale_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            den_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    } else if (G == 2) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:181: 'kStage * d' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<uint16_t, 1> tile_acc_ct1(
                sycl::range<1>(8192 /*kStage * d*/), cgh);
            /*
            DPCT1101:182: 'kAttTile' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<float[2][64 /*kAttTile*/], 0> sc_acc_ct1(cgh);
            /*
            DPCT1101:183: 'kAttTile' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<uint8_t *, 1> rows_acc_ct1(
                sycl::range<1>(64 /*kAttTile*/), cgh);
            sycl::local_accessor<float, 1> m_run_s_acc_ct1(sycl::range<1>(2),
                                                           cgh);
            sycl::local_accessor<float, 1> tile_scale_s_acc_ct1(
                sycl::range<1>(2), cgh);
            sycl::local_accessor<float, 1> den_s_acc_ct1(sycl::range<1>(2),
                                                         cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        sparse_attn_packed_kernel<2>(
                            q, win, c, sel, sel_stride, a, sink, heads, scale,
                            out,
                            tile_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            sc_acc_ct1,
                            rows_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            m_run_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            tile_scale_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            den_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    } else {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:184: 'kStage * d' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<uint16_t, 1> tile_acc_ct1(
                sycl::range<1>(8192 /*kStage * d*/), cgh);
            /*
            DPCT1101:185: 'kAttTile' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<float[1][64 /*kAttTile*/], 0> sc_acc_ct1(cgh);
            /*
            DPCT1101:186: 'kAttTile' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            // audit note: DPCT1101 accepted — dpct folded compile-time expressions to values and kept each expression in the comment, and the ds4 launch parity rows run these launches
            sycl::local_accessor<uint8_t *, 1> rows_acc_ct1(
                sycl::range<1>(64 /*kAttTile*/), cgh);
            sycl::local_accessor<float, 1> m_run_s_acc_ct1(sycl::range<1>(1),
                                                           cgh);
            sycl::local_accessor<float, 1> tile_scale_s_acc_ct1(
                sycl::range<1>(1), cgh);
            sycl::local_accessor<float, 1> den_s_acc_ct1(sycl::range<1>(1),
                                                         cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        sparse_attn_packed_kernel<1>(
                            q, win, c, sel, sel_stride, a, sink, heads, scale,
                            out,
                            tile_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            sc_acc_ct1,
                            rows_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            m_run_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            tile_scale_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            den_s_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
}

}  // namespace strata::ds4
