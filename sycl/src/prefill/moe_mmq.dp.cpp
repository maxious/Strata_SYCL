// src/prefill/moe_mmq.dp.cpp - see include/strata/prefill/moe_mmq.hpp.  llama.cpp's SYCL matvec kernels
// (ggml-sycl, MIT) are compiled from the pinned llama.cpp checkout the build already takes ggml from:
// quantize.hpp's quantize_row_q8_1_sycl writes the q8_1 activations, the ported mmvq.cpp matvec kernels dot
// them against the GGUF blocks through vecdotq.hpp's vec_dot_*_q8_1 (ggml-common.h's block types, SYCL
// declaration mode - no ggml-cuda header is reachable from here).
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include "strata/prefill/moe_mmq.hpp"

#define GGML_SYCL_WARP_SIZE 32   // Strata's port-wide sub-group size (the build forces 32 everywhere)
#include "common.hpp"            // ggml-sycl: presets (WARP_SIZE, GGML_SYCL_MMV_Y) + ggml-common.h in SYCL declaration
                                 // and implementation mode (the block types, the codebook grids) + the
                                 // ggml_sycl_dp4a family - the include set mmvq.cpp itself runs on
#include "quantize.hpp"          // ggml-sycl: quantize_row_q8_1_sycl (the packed block_q8_1 activations)
#include "vecdotq.hpp"           // ggml-sycl: the vec_dot_*_q8_1 the matvec kernels call

// This TU's dpct is ggml-sycl's (above): Strata's vendored <dpct/dpct.hpp> - which "strata/sycl_queue.hpp" pulls
// in - is a second, different dpct (dpct::dp4a, err0, get_in_order_queue would each be defined twice), so the two
// pieces of Strata's the kept kernels use are declared here instead.
template <class... Args> class dpct_kernel_name;
namespace strata {
// strata/sycl_queue.hpp's q_of, with ggml-sycl's dpct::get_in_order_queue
inline sycl::queue* q_of(const void* stream) {
    return stream ? (sycl::queue*) stream : &dpct::get_in_order_queue();
}
}  // namespace strata

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {
namespace {

void ck(dpct::err0 e, const char *what) {
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

// ggml-cuda mmq.cuh's block_q8_1_mmq (the CUDA tensor-core MMQ's transposed q8_1 tile): q8_bytes() sizes the
// activation buffer in its terms, and its 36 bytes per 32 values is also exactly the packed block_q8_1 rows
// quantize() writes (mmq.cuh asserts sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1)), so the contract covers
// both.  The union's exact bytes are the MMQ layouts' business; the packed layout only needs the size.
struct block_q8_1_mmq {
    union {
        float d4[4];             // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        sycl::half2 ds4[4];      // 1 16 bit scale + 1 16 bit partial sum per 32 values
    };
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 4 * QK8_1 + 4 * sizeof(sycl::half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4 * sizeof(block_q8_1), "Unexpected block_q8_1_mmq size");

__dpct_inline__ void copy16_kernel(const sycl::uint4 *__restrict__ a,
                                   int64_t na,
                                   const sycl::uint4 *__restrict__ b,
                                   int64_t nb, sycl::uint4 *__restrict__ ab_dst,
                                   const sycl::uint4 *__restrict__ c,
                                   int64_t nc,
                                   sycl::uint4 *__restrict__ c_dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}
__dpct_inline__ void copy1_kernel(const uint8_t *__restrict__ a, int64_t na,
                                  const uint8_t *__restrict__ b, int64_t nb,
                                  uint8_t *__restrict__ ab_dst,
                                  const uint8_t *__restrict__ c, int64_t nc,
                                  uint8_t *__restrict__ c_dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales
// [2560][10] f16 (the layout of prefill/kernels.cu's blob_dequant_kernel).  A GGUF Q2_0 block is {f16 d; 16 code
// bytes} with the same 2-bit codes in the same order, so a block is a scale and a 16-byte run of codes.
/*
DPCT1110: The total declared local variable size in device function
strata_q2_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void strata_q2_kernel(const uint8_t *__restrict__ blob,
                                      uint16_t *__restrict__ gu,
                                      uint16_t *__restrict__ dn) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr size_t O_D_CODES = (size_t)1280 * 640,
                     O_GU_SC = O_D_CODES + (size_t)2560 * 160,
                     O_D_SC = O_GU_SC + (size_t)1280 * 40 * 2;
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2); // one GGUF block
    const int64_t n_gu = 1280LL * 40, n_d = 2560LL * 10;
    const uint8_t* codes;
    const uint16_t* scale;
    uint16_t* out;
    if (i < n_gu) {
        const int64_t row = i / 40, b = i % 40;
        codes = blob + row * 640 + b * 16;
        scale = (const uint16_t*) (blob + O_GU_SC) + row * 40 + b;
        out = gu + i * 9;
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 10, b = j % 10;
        codes = blob + O_D_CODES + row * 160 + b * 16;
        scale = (const uint16_t*) (blob + O_D_SC) + row * 10 + b;
        out = dn + j * 9;
    } else {
        return;
    }
    const sycl::uint4 q = *(const sycl::uint4 *)codes;
    const uint16_t* qh = (const uint16_t*) &q;
    out[0] = *scale;
#pragma unroll
    for (int k = 0; k < 8; ++k) out[1 + k] = qh[k];
}

__dpct_inline__ void swiglu_kernel(const float *__restrict__ gu,
                                   float *__restrict__ h, int64_t rows,
                                   int64_t n_ff, bool interleaved) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + sycl::native::exp(-g)) * u;
}

__dpct_inline__ void iota_kernel(int32_t *dst, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) dst[i] = (int32_t) i;
}

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

// ---------------------------------------------------- the q8_1 activation quantizer (ggml-sycl quantize.hpp)
// quantize.hpp's quantize_q8_1 (one WARP_SIZE sub-group per 32 values) with the row gather the native
// quantize_row_q8_1_sycl can't express: output row r takes x row ids[r] (x rows ld floats apart, the row
// itself when ids is null).  quantize_q8_1_impl's arithmetic and the packed block_q8_1 layout, unchanged.
template <int ElementsPerWI>
struct quantize_q8_1_gather {
    __dpct_inline__ void operator()(const float *__restrict__ x, const int32_t *__restrict__ ids, void *vy,
                                    const int kx, const int64_t ld, const sycl::nd_item<1> &it) const {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        const int num_blocks_per_row = kx / QK8_1;
        const int row                = (int) (subgroup_id / num_blocks_per_row);
        const int64_t src            = ids ? ids[row] : row;   // output row r takes x row ids[r]

        sycl::vec<float, ElementsPerWI> wi_f32_vals;
        wi_f32_vals = *reinterpret_cast<const sycl::vec<float, ElementsPerWI> *>(
            x + src * ld + (int64_t) (subgroup_id % num_blocks_per_row) * QK8_1 + ElementsPerWI * wi_id);

        sycl::vec<int8_t, ElementsPerWI> quantized_values;
        float d = 0.0f;
        float sum = 0.0f;
        float amax = 0.0f;

#pragma unroll(ElementsPerWI)
        for (int i = 0; i < ElementsPerWI; i++) {
            sum += wi_f32_vals[i];
            amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
            quantized_values[i] = 0;
        }
        sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
        amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());
        d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
        for (int i = 0; i < ElementsPerWI; i++) {
            quantized_values[i] = sycl::round(wi_f32_vals[i] / d);
        }

        d = amax == 0 ? 0 : d;

        block_q8_1 * b = (block_q8_1 *) vy + subgroup_id;   // packed: row * num_blocks_per_row + col
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(&b->qs[wi_id * ElementsPerWI]) = quantized_values;
        if (wi_id == 0) {
            b->ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

// quantize_row_q8_1_sycl's launch shape, for the gather above.
static void quantize_q8_1_gather_sycl(const float * x, const int32_t * ids, void * vy, const int kx,
                                      const int64_t ld, const int ky, dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto num_quant_blocks = (size_t) ky * (kx / QK8_1);
    auto global_range     = num_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_q8_1_gather<QK8_1 / WARP_SIZE>()(x, ids, vy, kx, ld, it);
                         });
}

// ---------------------------------------------------- the ported matvec kernels (ggml-sycl mmvq.cpp)
// mmvq.cpp's mul_mat_vec_q* with the batched head the prompt path needs: sub-group (row, token) of one expert's
// launch computes dst row `row` of token row `token` - token = bounds[0] + group(1) within the expert's
// [bounds[0], bounds[1]) (bounds on the device), the activation row xq + token*y_blocks (quantize()'s packed
// block_q8_1 rows), the result to dst[ids[token]*ld_dst + row] (an identity table works).  The dot loop and the
// sub-group reduction are mmvq.cpp's, unchanged.

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q(const void *__restrict__ vx, const int32_t *__restrict__ bounds,
                          const void *__restrict__ xq, const int32_t *__restrict__ ids,
                          float *__restrict__ dst, int64_t ld_dst, const int ncols, const int nrows,
                          const int y_blocks, const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }
    const int token = bounds[0] + item_ct1.get_group(1);
    if (token >= bounds[1]) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    assert(blocks_per_warp > 0);

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) xq + (int64_t) token * y_blocks;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[(int64_t) (ids ? ids[token] : token) * ld_dst + row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xxs_q8_1(const void *__restrict__ vx, const int32_t *__restrict__ bounds,
                                       const void *__restrict__ xq, const int32_t *__restrict__ ids,
                                       float *__restrict__ dst, int64_t ld_dst, const int ncols, const int nrows,
                                       const int y_blocks, const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }
    const int token = bounds[0] + item_ct1.get_group(1);
    if (token >= bounds[1]) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) xq + (int64_t) token * y_blocks;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[(int64_t) (ids ? ids[token] : token) * ld_dst + row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xs_q8_1(const void *__restrict__ vx, const int32_t *__restrict__ bounds,
                                      const void *__restrict__ xq, const int32_t *__restrict__ ids,
                                      float *__restrict__ dst, int64_t ld_dst, const int ncols, const int nrows,
                                      const int y_blocks, const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }
    const int token = bounds[0] + item_ct1.get_group(1);
    if (token >= bounds[1]) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) xq + (int64_t) token * y_blocks;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[(int64_t) (ids ? ids[token] : token) * ld_dst + row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_s_q8_1(const void *__restrict__ vx, const int32_t *__restrict__ bounds,
                                     const void *__restrict__ xq, const int32_t *__restrict__ ids,
                                     float *__restrict__ dst, int64_t ld_dst, const int ncols, const int nrows,
                                     const int y_blocks, const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }
    const int token = bounds[0] + item_ct1.get_group(1);
    if (token >= bounds[1]) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) xq + (int64_t) token * y_blocks;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[(int64_t) (ids ? ids[token] : token) * ld_dst + row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_xxs_q8_1(const void *__restrict__ vx, const int32_t *__restrict__ bounds,
                                       const void *__restrict__ xq, const int32_t *__restrict__ ids,
                                       float *__restrict__ dst, int64_t ld_dst, const int ncols, const int nrows,
                                       const int y_blocks, const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }
    const int token = bounds[0] + item_ct1.get_group(1);
    if (token >= bounds[1]) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) xq + (int64_t) token * y_blocks;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[(int64_t) (ids ? ids[token] : token) * ld_dst + row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_s_q8_1(const void *__restrict__ vx, const int32_t *__restrict__ bounds,
                                     const void *__restrict__ xq, const int32_t *__restrict__ ids,
                                     float *__restrict__ dst, int64_t ld_dst, const int ncols, const int nrows,
                                     const int y_blocks, const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }
    const int token = bounds[0] + item_ct1.get_group(1);
    if (token >= bounds[1]) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) xq + (int64_t) token * y_blocks;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[(int64_t) (ids ? ids[token] : token) * ld_dst + row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_nl_q8_1(const void *__restrict__ vx, const int32_t *__restrict__ bounds,
                                      const void *__restrict__ xq, const int32_t *__restrict__ ids,
                                      float *__restrict__ dst, int64_t ld_dst, const int ncols, const int nrows,
                                      const int y_blocks, const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }
    const int token = bounds[0] + item_ct1.get_group(1);
    if (token >= bounds[1]) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) xq + (int64_t) token * y_blocks;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[(int64_t) (ids ? ids[token] : token) * ld_dst + row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_xs_q8_1(const void *__restrict__ vx, const int32_t *__restrict__ bounds,
                                      const void *__restrict__ xq, const int32_t *__restrict__ ids,
                                      float *__restrict__ dst, int64_t ld_dst, const int ncols, const int nrows,
                                      const int y_blocks, const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }
    const int token = bounds[0] + item_ct1.get_group(1);
    if (token >= bounds[1]) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) xq + (int64_t) token * y_blocks;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[(int64_t) (ids ? ids[token] : token) * ld_dst + row] = tmp;
    }
}

// mmvq.cpp's mul_mat_vec_*_q8_1_sycl launch shape (GGML_SYCL_MMV_Y rows per sub-group), with group(1) = the
// token row within the expert (max_rows: the most rows one expert has, the launch grid).
template <typename Kernel>
static void mul_mat_vec_q_launch(const dpct::queue_ptr s, int64_t max_rows, const int nrows, Kernel body) {
    const sycl::range<3> block_nums(1, (size_t) max_rows,
                                    (size_t) ((nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y));
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    s->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             body(item_ct1);
                         });
    });
}

}  // namespace

bool built() {
    // opt-in: the i-quant matvec port is parity-clean but ~30x slower than dequant+oneMKL GEMM on batched
    // prompts (measured 2026-10-03, 91.5% of GPU time in the matvec gemm phases), so the FP16 path stays the
    // default; STRATA_PREFILL_MMQ=1 enables it
    static const bool v = [] { const char* e = std::getenv("STRATA_PREFILL_MMQ"); return e != nullptr && std::atoi(e) != 0; }();
    return v;
}

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

bool fits(int t, int64_t w_rows) {
    (void) w_rows;
    // the matvec kernels hold no shared-memory tile (the CUDA MMQ's J_best test had one), so every supported
    // type runs at any w_rows: the non-MMQ path (#420) is for the unsupported types only
    return supported(t);
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / ggml_blck_size((ggml_type) t)) * ggml_type_size((ggml_type) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    (void) t;   // the q8_1 activations do not depend on the weight type (run()'s kernels do)
    const dpct::queue_ptr s = strata::q_of(stream);
    if (ids == nullptr && ld == cols) {
        // the native quantizer: contiguous rows into the packed block_q8_1 layout the matvec kernels read
        quantize_row_q8_1_sycl<quantize_q8_1>(x, xq, (int) cols, (int) rows, (int) cols, s);
    } else {
        // the gather the native one can't express: output row r = x row ids[r], x rows ld floats apart
        quantize_q8_1_gather_sycl(x, ids, xq, (int) cols, ld, (int) rows, s);
    }
    ck(0, "quantize");
}

Context::Context() {
    // the matvec kernels keep no scratch pool (the CUDA MMQ's stream-k fixup did), ctx_ stays null
}
Context::~Context() {}

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const dpct::queue_ptr s = strata::q_of(stream);
    const int ncols = (int) p.w_cols, nrows = (int) p.w_rows;
    const int y_blocks = ncols / QK8_1;   // one token row of packed q8_1 blocks (quantize()'s layout)
    const void* xq = p.xq;
    const int32_t* ids = p.ids;
    float* dst = p.dst;
    const int64_t ld_dst = p.ld_dst;
    // one launch per expert: its token rows [bounds[e], bounds[e+1]) (bounds on the device) against its
    // [w_rows, w_cols] matrix; max_rows (the most rows one expert has) is the launch grid
    for (int e = 0; e < p.n; ++e) {
        const void* w = (const char*) p.w + (int64_t) e * (int64_t) p.expert_bytes;
        const int32_t* bnd = p.bounds + e;
        switch (t) {
            case GGML_TYPE_Q2_0:
                mul_mat_vec_q_launch(s, p.max_rows, nrows, [=](sycl::nd_item<3> it) {
                    mul_mat_vec_q<QK2_0, QI2_0, block_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                        w, bnd, xq, ids, dst, ld_dst, ncols, nrows, y_blocks, it);
                });
                break;
            case GGML_TYPE_IQ2_XXS:
                mul_mat_vec_q_launch(s, p.max_rows, nrows, [=](sycl::nd_item<3> it) {
                    mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, block_iq2_xxs, 1>(
                        w, bnd, xq, ids, dst, ld_dst, ncols, nrows, y_blocks, it);
                });
                break;
            case GGML_TYPE_IQ2_XS:
                mul_mat_vec_q_launch(s, p.max_rows, nrows, [=](sycl::nd_item<3> it) {
                    mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, block_iq2_xs, 1>(
                        w, bnd, xq, ids, dst, ld_dst, ncols, nrows, y_blocks, it);
                });
                break;
            case GGML_TYPE_IQ2_S:
                mul_mat_vec_q_launch(s, p.max_rows, nrows, [=](sycl::nd_item<3> it) {
                    mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, block_iq2_s, 1>(
                        w, bnd, xq, ids, dst, ld_dst, ncols, nrows, y_blocks, it);
                });
                break;
            case GGML_TYPE_IQ3_XXS:
                mul_mat_vec_q_launch(s, p.max_rows, nrows, [=](sycl::nd_item<3> it) {
                    mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, block_iq3_xxs, 1>(
                        w, bnd, xq, ids, dst, ld_dst, ncols, nrows, y_blocks, it);
                });
                break;
            case GGML_TYPE_IQ3_S:
                mul_mat_vec_q_launch(s, p.max_rows, nrows, [=](sycl::nd_item<3> it) {
                    mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, block_iq3_s, 1>(
                        w, bnd, xq, ids, dst, ld_dst, ncols, nrows, y_blocks, it);
                });
                break;
            case GGML_TYPE_IQ4_NL:
                mul_mat_vec_q_launch(s, p.max_rows, nrows, [=](sycl::nd_item<3> it) {
                    mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, block_iq4_nl, 2>(
                        w, bnd, xq, ids, dst, ld_dst, ncols, nrows, y_blocks, it);
                });
                break;
            case GGML_TYPE_IQ4_XS:
                mul_mat_vec_q_launch(s, p.max_rows, nrows, [=](sycl::nd_item<3> it) {
                    mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, block_iq4_xs, 1>(
                        w, bnd, xq, ids, dst, ld_dst, ncols, nrows, y_blocks, it);
                });
                break;
            default:
                std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
                std::exit(1);
        }
    }
    ck(0, "mul_mat_q");
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream) {
    const dpct::queue_ptr s = strata::q_of(stream);
    const bool a16 = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down | (uintptr_t) gu_dst | (uintptr_t) d_dst |
                      gu_half_bytes | d_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<dpct_kernel_name<class copy16_kernel_b12127>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks(2 * na + nc)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy16_kernel(
                        (const sycl::uint4 *)gate, na, (const sycl::uint4 *)up,
                        na, (sycl::uint4 *)gu_dst, (const sycl::uint4 *)down,
                        nc, (sycl::uint4 *)d_dst);
                });
        }
    } else {
        const int64_t na = (int64_t) gu_half_bytes, nc = (int64_t) d_bytes;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<dpct_kernel_name<class copy1_kernel_c557f3>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks(2 * na + nc)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy1_kernel((const uint8_t *)gate, na, (const uint8_t *)up,
                                 na, (uint8_t *)gu_dst, (const uint8_t *)down,
                                 nc, (uint8_t *)d_dst);
                });
        }
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "gather_native");
}

void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class strata_q2_kernel_793a65>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, blocks(1280LL * 40 + 2560LL * 10)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    strata_q2_kernel(blob, (uint16_t *)gu_dst,
                                     (uint16_t *)d_dst);
                });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "gather_strata_q2");
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class swiglu_kernel_3f7a6f>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks(rows * n_ff)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    swiglu_kernel(gu, h, rows, n_ff, interleaved);
                });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "swiglu");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class iota_kernel_9245a0>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks(n)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    iota_kernel(dst, n);
                });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "iota");
}

}  // namespace strata::prefill::mmq
