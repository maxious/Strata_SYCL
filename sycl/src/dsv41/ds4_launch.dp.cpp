// The debug projection's hand-ported SYCL copy (pred_proj_k, src/dsv41/ds4_model.cu's kernel :509):
// out[t][j] = (sum_d +-x[t][d]) / sqrt(D), the sign a fixed hash of (d, j).  Hand-ported because dpct exits -46 on
// ds4_model.cu and its draft does not compile.  The window sparse attention and the Hyper-Connections pre-norm /
// mix-scaling families used to be hand-ported here too; their one home is ds4_attn_hc.dp.cpp now - the SYCLomatic
// copies the engine (strata-dsv41) also links - so the parity harness checks the copies the engine runs.  Everything
// here is behind STRATA_SYCL_DSV41; the CUDA / HIP paths keep their own ds4_model.cu.
#include "ds4_launch.dp.hpp"

namespace strata::ds4 {
namespace {

struct PredProjKName {};

}  // namespace

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

}  // namespace strata::ds4
