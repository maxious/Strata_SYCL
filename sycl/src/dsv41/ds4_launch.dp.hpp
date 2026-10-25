// Hand-ported SYCL copies of the dsv41 kernel-launcher families that src/dsv41/ds4_model.cu calls and that no other
// port covers: the window sparse attention (ds4_kernels.cu's sparse_attn_kernel :771, sparse_attn_packed_kernel :951
// and their launcher), the Hyper-Connections pre-norm (hc_pre_norm_kernel :338) and mix scaling
// (hc_mix_scale_kernel :1689), and the debug projection (pred_proj_k, ds4_model.cu:509).  The two F16 GEMVs and the
// F32 GEMV are ds4_gemv.dp.hpp's and the vision tower's softmax is ds4_softmax.dp.hpp's.  Hand-ported: dpct exits
// -46 on ds4_model.cu and its draft does not compile.  Everything here is behind STRATA_SYCL_DSV41; the CUDA / HIP
// paths keep their own ds4_kernels.cu.
//
// The declarations and structs mirror the in-scope parts of src/dsv41/ds4_kernels.cuh (kWin, the KV row formats,
// AttnScratch / AttnRows, the launcher signatures with dpct::queue_ptr in place of cudaStream_t), so a later
// ds4_model port can call them the way the CUDA engine does.  What is NOT here: the split-KV attention variant
// (src/dsv41/ds4_attn_split.cuh, off by default behind DS4_ATTN_SPLIT) and ds4_kernels.cu's other families.
#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

namespace strata::ds4 {

constexpr int kWin = 16;      // tokens a window takes at most (ds4_kernels.cuh's value)

/// The caches' rows (ds4_kernels.cuh's formats and row size, verbatim).
constexpr int kKvBf16 = 0, kKvFp8 = 1, kKvFp4 = 2, kKvFp4p = 3;
constexpr size_t kv_row_bytes(int fmt, int d) {
    return fmt == kKvBf16  ? (size_t) d * 2
           : fmt == kKvFp8 ? ((size_t) d + (size_t) d / 32 + 15) / 16 * 16
           : fmt == kKvFp4 ? ((size_t) d / 2 + (size_t) d / 16 + 15) / 16 * 16
                           : ((size_t) d / 2 + (size_t) d / 32 + 15) / 16 * 16;
}

/// x[t][e] = sum_s mix[t][s] h[t][s][e], then x = rmsnorm(x) * w (w null: no norm).
void hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x,
                 dpct::queue_ptr s);

/// A chunk's hc_mix: the dots of each row with the n_mix weights already in mixes (rows 32 apart), times
/// rsqrt(mean(h_row^2) + eps).
void hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes, dpct::queue_ptr s);

/// The debug projection (DS4_PRED_DUMP): out[t][j] = (sum_d +-x[t][d]) / sqrt(D), the sign a fixed hash of (d, j).
void pred_proj_k(const float* x, int D, int T, float* out, dpct::queue_ptr s);

/// Sparse attention, query t: heads x d values of q[t] against win rows win0[t] .. win0[t] + nwin[t] - 1 of the
/// window buffer (d BF16 each) and the nsel[t] compressed rows ckv[sel[t][i]] (format ckv_fmt); K = V, a learned
/// sink logit per head.  As ds4_kernels.cuh: the packed rows' bits are the same either way - decoded once a query
/// into scr's BF16 rows when few queries come with a scratch (heads_per_block -1: any T; 0: T <= 7), else decoded
/// per staged row for heads_per_block heads of a query (1, 2, 4, 8).
struct AttnScratch {
    uint16_t* rows = nullptr;   // [max_t * cap][512] BF16
    int max_t = 0, cap = 0;     // queries, rows a query (at least the sel_stride)
};
struct AttnRows {
    int win0[kWin], nwin[kWin], nsel[kWin];
    int gen = 0, t0 = 0, win = 0, R = 0, kcap = 0;
    long long p0 = 0;
};
void sparse_attn(const float* q, const uint16_t* win, const void* ckv, const int* sel, int sel_stride,
                 const AttnRows& a, const float* sink, int heads, int d, int T, float scale, float* out,
                 dpct::queue_ptr s, int ckv_fmt = kKvFp4, int heads_per_block = 0, const AttnScratch* scr = nullptr);

}  // namespace strata::ds4
