// sycl/src/dsv41/ds4_blas_parity.cpp - device parity for the Wave 3 dsv41 additions: the cublas -> oneMKL shim's
// GEMMs (sycl/include/compat/cublas_v2.h - the two FP16-in / FP32-out shapes and the FP32 shape src/dsv41/ds4_model.cu
// calls at :2910, :2924 and :2937) and the ported gemv / softmax kernels (ds4_gemv.dp.cpp, ds4_softmax.dp.cpp),
// against CPU references computed through oneDNN's dnnl_sgemm (C API), not copied from the device code. Exit
// conventions follow src/kernels/dequant_s2_parity.cpp: vacuous 1, mismatch 1, success 0, usage 2; the print shape
// follows sycl/src/dsv41/ds4_parity.cpp: a device: line, one line per family, then MATCH / NO MATCH with the
// one-BF16-ulp rule (FMA vs separate multiply-and-add can cost one). Only the Arc Pro B60 is accepted: any other
// device is a refusal (exit 2), so a run on a CPU device cannot read as a pass. The negative controls are --corrupt
// (one device input element changed after the reference is computed), --corrupt-scale (the device GEMV scale
// changed) and --corrupt-ldx (the device GEMV leading dimension changed): every one must end NO MATCH, exit 1.
#include <sycl/sycl.hpp>

#include <cublas_v2.h>   // the oneMKL compat shim under test

#include <oneapi/dnnl/dnnl.h>   // the CPU reference's GEMM (oneDNN C API; this header set has no dnnl::gemm)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ds4_gemv.dp.hpp"
#include "ds4_numerics.dp.hpp"
#include "ds4_softmax.dp.hpp"

namespace {

constexpr const char* kDeviceName = "Intel(R) Arc(TM) Pro B60 Graphics";

// One BF16 ulp of a finite value: 2^(exponent - 7) (BF16 has 8 mantissa bits counting the implied one).
float bf16_ulp(float v) {
    if (!(v == v) || v == 0.0f) return 0.0f;
    const int e = (int) ((sycl::bit_cast<uint32_t>(std::fabs(v)) >> 23) & 0xFF) - 127;
    return std::ldexp(1.0f, e - 7);
}

// F16 bits of v rounded to nearest-even (sycl/src/dsv41/ds4_parity.cpp's helper: oneAPI's host convert truncates
// tiny values instead of rounding them, so the reference rounds explicitly).
uint16_t f16_rn(float v) {
    uint32_t b;
    std::memcpy(&b, &v, sizeof b);
    const uint16_t sign = (uint16_t) ((b >> 16) & 0x8000u);
    const uint32_t a = b & 0x7FFFFFFFu;
    if (a >= 0x7F800000u) return (uint16_t) (sign | 0x7C00u | (a > 0x7F800000u ? 0x0200u : 0u));
    if (a < 0x38800000u) {
        const long long n = (long long) std::nearbyint((double) v * 16777216.0);
        const unsigned long long m = (unsigned long long) (n < 0 ? -n : n);
        return (uint16_t) (sign | (uint16_t) (m <= 1024 ? m : 1024));
    }
    int e = (int) (a >> 23) - 112;
    if (e >= 31) return (uint16_t) (sign | 0x7C00u);
    uint32_t m10 = (a & 0x7FFFFFu) >> 13;
    const uint32_t rest = a & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (m10 & 1u))) {
        if (++m10 == 0x400u) {
            m10 = 0;
            if (++e >= 31) return (uint16_t) (sign | 0x7C00u);
        }
    }
    return (uint16_t) (sign | ((uint16_t) e << 10) | (uint16_t) m10);
}

// A seeded byte generator: the same bytes on every run, so the two runs this test compares never depend on luck.
struct Rng {
    unsigned long long s = 0x20261025ull;
    uint8_t next() {
        s = s * 6364136222842883251ull + 1442695040888963407ull;
        return (uint8_t) ((s >> 33) & 0xFF);
    }
};

// F16 input values drawn from a small rational grid: exactly representable, finite, away from zero and infinity, so
// the CPU reference and the device see the same bits.
uint16_t input_f16(Rng& rng) {
    const float v = ((int) (rng.next() & 63) - 32) / 32.0f;
    return f16_rn(v == 0.0f ? 1.0f / 32.0f : v);
}

float input_f32(Rng& rng) { return ((int) (rng.next() | ((int) rng.next() << 8)) - 32768) / 32768.0f; }

struct Counts {
    long long exact = 0, near = 0, bad = 0, total = 0;
    long long first_bad = -1;
    float first_got = 0.0f, first_ref = 0.0f;
    float max_got = -INFINITY, max_ref = -INFINITY;
    float max_diff = 0.0f;
};

void cmp_f32(const float* got, const float* ref, size_t n, Counts& c) {
    for (size_t i = 0; i < n; ++i) {
        ++c.total;
        if (got[i] > c.max_got) c.max_got = got[i];
        if (ref[i] > c.max_ref) c.max_ref = ref[i];
        if (got[i] == ref[i]) {
            ++c.exact;
            continue;
        }
        const float diff = std::fabs(got[i] - ref[i]);
        if (diff > c.max_diff) c.max_diff = diff;
        if (diff <= bf16_ulp(ref[i])) ++c.near;
        else {
            ++c.bad;
            if (c.first_bad < 0) {
                c.first_bad = (long long) i;
                c.first_got = got[i];
                c.first_ref = ref[i];
            }
        }
    }
}

void cmp_bf16(const uint16_t* got, const float* ref, size_t n, Counts& c) {
    for (size_t i = 0; i < n; ++i) {
        ++c.total;
        // the softmax outputs are BF16 bits (the kernels' bfr): widen them as BF16. Reading the bits as IEEE
        // FP16 (sycl::half) invents huge diffs and turns one-ulp rounding flips into false mismatches.
        const float a = sycl::bit_cast<float>((uint32_t) got[i] << 16);
        if (a > c.max_got) c.max_got = a;
        if (ref[i] > c.max_ref) c.max_ref = ref[i];
        if (got[i] == strata::ds4::bf16_rn(ref[i])) {
            ++c.exact;
            continue;
        }
        const float diff = std::fabs(a - ref[i]);
        if (diff > c.max_diff) c.max_diff = diff;
        if (diff <= bf16_ulp(ref[i])) ++c.near;
        else {
            ++c.bad;
            if (c.first_bad < 0) {
                c.first_bad = (long long) i;
                c.first_got = a;
                c.first_ref = ref[i];
            }
        }
    }
}

void report(const char* what, const Counts& c) {
    std::printf("%s: %lld/%lld values exact, %lld within one BF16 ulp, %lld outside\n", what, c.exact, c.total,
                c.near, c.bad);
    if (c.first_bad >= 0)
        std::printf("  first outside at %lld: got %.9g ref %.9g diff %.9g\n", c.first_bad, (double) c.first_got,
                    (double) c.first_ref, (double) std::fabs(c.first_got - c.first_ref));
    if (c.total) std::printf("  got max %.9g ref max %.9g\n", (double) c.max_got, (double) c.max_ref);
}

// Non-vacuity: the compared values must actually vary, or two buffers of a constant would agree forever. Eight
// distinct reference values per family is far below what the seeded inputs produce and far above what a broken
// (constant) reference would show.
long long distinct_values(const float* v, size_t n) {
    std::vector<float> a(v, v + n);
    std::sort(a.begin(), a.end());
    return (long long) (std::unique(a.begin(), a.end()) - a.begin());
}

struct Family {
    const char* name;
    std::vector<float> ref;
};

void ref_sgemm(char trans_a, char trans_b, int m, int n, int k, float alpha, const float* a, int lda, const float* b,
               int ldb, float beta, float* c, int ldc) {
    // dnnl_sgemm takes row-major matrices (dnnl.h:4102: "The matrices are assumed to be stored in row-major
    // order"), while these references use cuBLAS's column-major call shape. Transposing the product turns one
    // into the other at the same addresses: column-major C = alpha op(A) op(B) is row-major C^T = alpha op(B)^T
    // op(A)^T, so the operands swap, the trans flags travel with them, and the dims go n, m, k.
    const dnnl_status_t st = dnnl_sgemm(trans_b, trans_a, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
    if (st != dnnl_success) {
        std::fprintf(stderr, "dnnl_sgemm: status %d\n", (int) st);
        std::exit(1);
    }
}

void widen16(const uint16_t* p, size_t n, std::vector<float>& out) {
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(sycl::bit_cast<sycl::half>(p[i]));
}

void ref_gemm_f16(const uint16_t* a, int lda, const uint16_t* b, int ldb, float alpha, float* c, int ldc, int m,
                  int n, int k, float beta) {
    std::vector<float> fa, fb;
    widen16(a, (size_t) lda * m, fa);
    widen16(b, (size_t) ldb * n, fb);
    ref_sgemm('T', 'N', m, n, k, alpha, fa.data(), lda, fb.data(), ldb, beta, c, ldc);
}

void ref_gemm_f32(const float* a, int lda, const float* b, int ldb, float alpha, float* c, int ldc, int m, int n,
                  int k, float beta) {
    ref_sgemm('T', 'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ref_gemv_f16(const uint16_t* w, const float* x, size_t ldx, int n, int k, int t, float alpha, float* y,
                  size_t yld) {
    std::vector<float> fw;
    widen16(w, (size_t) n * k, fw);
    ref_sgemm('T', 'N', n, 1, k, alpha, fw.data(), k, x + (size_t) t * ldx, (int) ldx, 0.0f,
              y + (size_t) t * yld, (int) yld);
}

void ref_gemv_f32(const float* w, const float* x, size_t ldx, int n, int k, int t, float* y, size_t yld) {
    ref_sgemm('T', 'N', n, 1, k, 1.0f, w, k, x + (size_t) t * ldx, (int) ldx, 0.0f, y + (size_t) t * yld, (int) yld);
}

// The vision softmax reference: max, exp(v - max), the sum, then BF16 (the kernel's bfr output).
void ref_softmax(const float* s, int n, int rows, float* p) {
    for (int r = 0; r < rows; ++r) {
        const float* sr = s + (size_t) r * n;
        float* pr = p + (size_t) r * n;
        float m = -INFINITY;
        for (int i = 0; i < n; ++i) m = std::fmax(m, sr[i]);
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            pr[i] = std::exp(sr[i] - m);
            sum += pr[i];
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < n; ++i) pr[i] *= inv;
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, corrupt = false, corrupt_scale = false, corrupt_ldx = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--corrupt") == 0) corrupt = true;
        else if (std::strcmp(argv[i], "--corrupt-scale") == 0) corrupt_scale = true;
        else if (std::strcmp(argv[i], "--corrupt-ldx") == 0) corrupt_ldx = true;
        else {
            std::fprintf(stderr, "usage: ds4_blas_parity [--selftest] [--corrupt] [--corrupt-scale] [--corrupt-ldx]\n");
            return 2;
        }
    }

    sycl::device dev;
    bool have = false;
    try {
        for (const sycl::device& d : sycl::device::get_devices()) {
            if (!have) {
                dev = d;
                have = true;
            }
            if (d.get_info<sycl::info::device::name>() == kDeviceName) {
                dev = d;
                have = true;
                break;
            }
        }
    } catch (const sycl::exception&) {
        have = false;
    }
    if (!have) {
        std::printf("REFUSED: no SYCL device found (this parity run needs %s)\n", kDeviceName);
        return 2;
    }
    const std::string name = dev.get_info<sycl::info::device::name>();
    if (name != kDeviceName) {
        std::printf("REFUSED: device %s is not %s\n", name.c_str(), kDeviceName);
        return 2;
    }
    std::printf("device: %s\n", name.c_str());
    {
        std::string sgs;
        for (size_t sg : dev.get_info<sycl::info::device::sub_group_sizes>())
            sgs += (sgs.empty() ? "" : " ") + std::to_string(sg);
        std::printf("sub-group sizes: %s\n", sgs.c_str());
    }

    sycl::queue q(dev);
    dpct::queue_ptr s = &q;
    Rng rng;

    // ---- GEMM shape A: ds4_model.cu:2910 (W^T x, beta accumulating) -------------------------------------------
    const int ma = 96, na = 7, ka = 192, lda = ka, ldba = 224, ldca = 128;
    std::vector<uint16_t> wa((size_t) lda * ma), xa((size_t) ldba * na);
    std::vector<float> ya((size_t) ldca * na), ya_ref;
    for (uint16_t& v : wa) v = input_f16(rng);
    for (uint16_t& v : xa) v = input_f16(rng);
    for (float& v : ya) v = input_f32(rng);
    ya_ref = ya;
    ref_gemm_f16(wa.data(), lda, xa.data(), ldba, 1.0f, ya_ref.data(), ldca, ma, na, ka, 0.25f);

    // ---- GEMM shape B: ds4_model.cu:2924 (F16 weights, beta = 0) ----------------------------------------------
    const int mb = 128, nb = 5, kb = 64, ldb_ = kb, ldbb = kb, ldbc = 128;
    std::vector<uint16_t> wb((size_t) ldb_ * mb), xb((size_t) ldbb * nb);
    std::vector<float> yb((size_t) ldbc * nb, 1.0f), yb_ref((size_t) ldbc * nb, 1.0f);
    for (uint16_t& v : wb) v = input_f16(rng);
    for (uint16_t& v : xb) v = input_f16(rng);
    ref_gemm_f16(wb.data(), ldb_, xb.data(), ldbb, 0.75f, yb_ref.data(), ldbc, mb, nb, kb, 0.0f);

    // ---- cublasSgemm: ds4_model.cu:2937 (mHC mixes, ldc = 32 as the engine's) ---------------------------------
    const int mc = 20, nc = 5, kc = 48, sg_lda = kc, sg_ldb = kc, sg_ldc = 32;
    std::vector<float> ac((size_t) sg_lda * mc), bc((size_t) sg_ldb * nc), yc((size_t) sg_ldc * nc, -1.0f),
        yc_ref((size_t) sg_ldc * nc, -1.0f);
    for (float& v : ac) v = input_f32(rng);
    for (float& v : bc) v = input_f32(rng);
    ref_gemm_f32(ac.data(), sg_lda, bc.data(), sg_ldb, 1.0f, yc_ref.data(), sg_ldc, mc, nc, kc, 0.0f);

    // ---- gemv_f16 (tile shape and the scalar fallback the tile path cannot take) ------------------------------
    const int gn = 64, gtk = 256, gsk = 130, gtT = 3, gsT = 2;
    std::vector<uint16_t> wg((size_t) gtk * gn), wgs((size_t) gsk * gn);
    std::vector<float> xg((size_t) gtk * gtT), xgs((size_t) gsk * gsT);
    std::vector<float> yg((size_t) gn * gtT), ygs((size_t) gn * gsT);
    std::vector<float> yg_ref((size_t) gn * gtT), ygs_ref((size_t) gn * gsT);
    for (uint16_t& v : wg) v = input_f16(rng);
    for (uint16_t& v : wgs) v = input_f16(rng);
    for (float& v : xg) v = input_f32(rng);
    for (float& v : xgs) v = input_f32(rng);
    for (int t = 0; t < gtT; ++t) ref_gemv_f16(wg.data(), xg.data(), gtk, gn, gtk, t, 0.5f, yg_ref.data(), gn);
    for (int t = 0; t < gsT; ++t) ref_gemv_f16(wgs.data(), xgs.data(), gsk, gn, gsk, t, -0.25f, ygs_ref.data(), gn);

    // ---- gemv_f16_rows (kRowsT tiles, T odd; and the fallback with k not a multiple of 8 and unaligned rows) ----
    const int rn = 64, rtk = 256, rsk = 130, rtT = 5, rsT = 3, rtldx = 288, rsldx = 130, rldy = 128;
    std::vector<uint16_t> wr((size_t) rtk * rn), wrs((size_t) rsk * rn);
    std::vector<float> xr((size_t) rtldx * rtT), xrs((size_t) rsldx * rsT);
    std::vector<float> yr((size_t) rldy * rtT), yrs((size_t) rldy * rsT);
    std::vector<float> yr_ref((size_t) rldy * rtT), yrs_ref((size_t) rldy * rsT);
    for (uint16_t& v : wr) v = input_f16(rng);
    for (uint16_t& v : wrs) v = input_f16(rng);
    for (float& v : xr) v = input_f32(rng);
    for (float& v : xrs) v = input_f32(rng);
    for (int t = 0; t < rtT; ++t) ref_gemv_f16(wr.data(), xr.data(), rtldx, rn, rtk, t, 0.75f, yr_ref.data(), rldy);
    for (int t = 0; t < rsT; ++t) ref_gemv_f16(wrs.data(), xrs.data(), rsldx, rn, rsk, t, 2.0f, yrs_ref.data(), rldy);

    // ---- gemv_f32 (the router's; tile shape and fallback) ----------------------------------------------------
    const int fn = 64, ftk = 192, fsk = 130, ftT = 3, fsT = 2, ftldx = 192, fsldx = 130, fldy = 130;
    std::vector<float> wf((size_t) ftk * fn), wfs((size_t) fsk * fn), xf((size_t) ftldx * ftT),
        xfs((size_t) fsldx * fsT), yf((size_t) fldy * ftT), yfs((size_t) fldy * fsT), yf_ref((size_t) fldy * ftT),
        yfs_ref((size_t) fldy * fsT);
    for (float& v : wf) v = input_f32(rng);
    for (float& v : wfs) v = input_f32(rng);
    for (float& v : xf) v = input_f32(rng);
    for (float& v : xfs) v = input_f32(rng);
    for (int t = 0; t < ftT; ++t) ref_gemv_f32(wf.data(), xf.data(), ftldx, fn, ftk, t, yf_ref.data(), fldy);
    for (int t = 0; t < fsT; ++t) ref_gemv_f32(wfs.data(), xfs.data(), fsldx, fn, fsk, t, yfs_ref.data(), fldy);

    // ---- softmax (register tiles to 10240 keys, the boundary, and the scalar fallback past it) ----------------
    const int sn[3] = {1024, 10240, 12000}, sr_rows[3] = {3, 2, 2};
    std::vector<std::vector<float>> ss(3), sp_ref(3);
    for (int f = 0; f < 3; ++f) {
        ss[f].resize((size_t) sn[f] * sr_rows[f]);
        sp_ref[f].resize((size_t) sn[f] * sr_rows[f]);
        for (float& v : ss[f]) v = input_f32(rng) * 8.0f;
        ref_softmax(ss[f].data(), sn[f], sr_rows[f], sp_ref[f].data());
    }

    // ---- device buffers
    uint16_t *d_wa, *d_xa, *d_wb, *d_xb, *d_wg, *d_wgs, *d_wr, *d_wrs;
    float *d_ya, *d_yb, *d_ac, *d_bc, *d_yc, *d_xg, *d_xgs, *d_yg, *d_ygs, *d_xr, *d_xrs, *d_yr, *d_yrs, *d_wf,
        *d_wfs, *d_xf, *d_xfs, *d_yf, *d_yfs;
    std::vector<float*> d_ss(3), d_sp(3);
    d_wa = (uint16_t*) sycl::malloc_device(wa.size() * 2, q);
    d_xa = (uint16_t*) sycl::malloc_device(xa.size() * 2, q);
    d_ya = (float*) sycl::malloc_device(ya.size() * 4, q);
    d_wb = (uint16_t*) sycl::malloc_device(wb.size() * 2, q);
    d_xb = (uint16_t*) sycl::malloc_device(xb.size() * 2, q);
    d_yb = (float*) sycl::malloc_device(yb.size() * 4, q);
    d_ac = (float*) sycl::malloc_device(ac.size() * 4, q);
    d_bc = (float*) sycl::malloc_device(bc.size() * 4, q);
    d_yc = (float*) sycl::malloc_device(yc.size() * 4, q);
    d_wg = (uint16_t*) sycl::malloc_device(wg.size() * 2, q);
    d_wgs = (uint16_t*) sycl::malloc_device(wgs.size() * 2, q);
    d_xg = (float*) sycl::malloc_device(xg.size() * 4, q);
    d_xgs = (float*) sycl::malloc_device(xgs.size() * 4, q);
    d_yg = (float*) sycl::malloc_device(yg.size() * 4, q);
    d_ygs = (float*) sycl::malloc_device(ygs.size() * 4, q);
    d_wr = (uint16_t*) sycl::malloc_device(wr.size() * 2, q);
    d_wrs = (uint16_t*) sycl::malloc_device(wrs.size() * 2, q);
    d_xr = (float*) sycl::malloc_device(xr.size() * 4, q);
    d_xrs = (float*) sycl::malloc_device(xrs.size() * 4, q);
    d_yr = (float*) sycl::malloc_device(yr.size() * 4, q);
    d_yrs = (float*) sycl::malloc_device(yrs.size() * 4, q);
    d_wf = (float*) sycl::malloc_device(wf.size() * 4, q);
    d_wfs = (float*) sycl::malloc_device(wfs.size() * 4, q);
    d_xf = (float*) sycl::malloc_device(xf.size() * 4, q);
    d_xfs = (float*) sycl::malloc_device(xfs.size() * 4, q);
    d_yf = (float*) sycl::malloc_device(yf.size() * 4, q);
    d_yfs = (float*) sycl::malloc_device(yfs.size() * 4, q);
    for (int f = 0; f < 3; ++f) {
        d_ss[f] = (float*) sycl::malloc_device(ss[f].size() * 4, q);
        d_sp[f] = (float*) sycl::malloc_device(ss[f].size() * 2, q);
    }
    void* d_ws = sycl::malloc_device((size_t) 4 << 20, q);

    q.memcpy(d_wa, wa.data(), wa.size() * 2);
    q.memcpy(d_xa, xa.data(), xa.size() * 2);
    q.memcpy(d_ya, ya.data(), ya.size() * 4);
    q.memcpy(d_wb, wb.data(), wb.size() * 2);
    q.memcpy(d_xb, xb.data(), xb.size() * 2);
    q.memcpy(d_yb, yb.data(), yb.size() * 4);
    q.memcpy(d_ac, ac.data(), ac.size() * 4);
    q.memcpy(d_bc, bc.data(), bc.size() * 4);
    q.memcpy(d_yc, yc.data(), yc.size() * 4);
    q.memcpy(d_wg, wg.data(), wg.size() * 2);
    q.memcpy(d_wgs, wgs.data(), wgs.size() * 2);
    q.memcpy(d_xg, xg.data(), xg.size() * 4);
    q.memcpy(d_xgs, xgs.data(), xgs.size() * 4);
    q.memcpy(d_wr, wr.data(), wr.size() * 2);
    q.memcpy(d_wrs, wrs.data(), wrs.size() * 2);
    q.memcpy(d_xr, xr.data(), xr.size() * 4);
    q.memcpy(d_xrs, xrs.data(), xrs.size() * 4);
    q.memcpy(d_wf, wf.data(), wf.size() * 4);
    q.memcpy(d_wfs, wfs.data(), wfs.size() * 4);
    q.memcpy(d_xf, xf.data(), xf.size() * 4);
    q.memcpy(d_xfs, xfs.data(), xfs.size() * 4);
    for (int f = 0; f < 3; ++f) q.memcpy(d_ss[f], ss[f].data(), ss[f].size() * 4);
    q.wait();

    if (corrupt) {
        const float bad = xg[3] + 1.0f;
        q.memcpy(d_xg + 3, &bad, 4);
        q.wait();
        std::printf("corrupt: one gemv_f16 device input element differs from the CPU reference input\n");
    }
    const float ga = corrupt_scale ? 0.875f : 0.5f;
    if (corrupt_scale)
        std::printf("corrupt-scale: the gemv_f16 device scale differs from the CPU reference scale\n");
    if (corrupt_ldx)
        std::printf("corrupt-ldx: the gemv_f16_rows device leading dimension differs from the CPU reference's\n");

    // the shim's GEMMs
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS || cublasSetStream(h, s) != CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(h, d_ws, (size_t) 4 << 20) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS on CUDA0\n");
        return 1;
    }
    const float one = 1.0f, zero = 0.0f, alpha_b = 0.75f, beta_a = 0.25f;
    cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, ma, na, ka, &one, d_wa, CUDA_R_16F, lda, d_xa,
                                     CUDA_R_16F, ldba, &beta_a, d_ya, CUDA_R_32F, ldca, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::string err = "cublasGemmEx: status " + std::to_string((int) st);
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, mb, nb, kb, &alpha_b, d_wb, CUDA_R_16F, ldb_, d_xb, CUDA_R_16F,
                      ldbb, &zero, d_yb, CUDA_R_32F, ldbc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::string err = "cublasGemmEx (F16 weights): status " + std::to_string((int) st);
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, kc, &one, d_ac, sg_lda, d_bc, sg_ldb, &zero, d_yc,
                    sg_ldc) != CUBLAS_STATUS_SUCCESS) {
        std::string err = "cublasSgemm (mHC mixes)";
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const cublasStatus_t probe = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 4, 4, 4, &one, d_wa, CUDA_R_16F, 4,
                                              d_xa, CUDA_R_16F, 4, &zero, d_ya, CUDA_R_32F, 4, CUBLAS_COMPUTE_16F,
                                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    std::printf("shim status path: cublasGemmEx (F16 weights): status %d (want %d = NOT_SUPPORTED)\n",
                (int) probe, (int) CUBLAS_STATUS_NOT_SUPPORTED);

    // the ported kernels
    strata::ds4::gemv_f16(d_wg, d_xg, gn, gtk, gtT, ga, d_yg, s);
    strata::ds4::gemv_f16(d_wgs, d_xgs, gn, gsk, gsT, -0.25f, d_ygs, s);
    strata::ds4::gemv_f16_rows(d_wr, d_xr, corrupt_ldx ? (size_t) rtldx + 1 : (size_t) rtldx, rn, rtk, rtT, 0.75f,
                               d_yr, rldy, s);
    strata::ds4::gemv_f16_rows(d_wrs, d_xrs, rsldx, rn, rsk, rsT, 2.0f, d_yrs, rldy, s);
    strata::ds4::gemv_f32(d_wf, d_xf, fn, ftk, ftT, d_yf, fldy, s);
    strata::ds4::gemv_f32(d_wfs, d_xfs, fn, fsk, fsT, d_yfs, fldy, s);
    for (int f = 0; f < 3; ++f) strata::ds4::softmax(d_ss[f], sn[f], sr_rows[f], (uint16_t*) d_sp[f], s);
    q.wait();

    std::vector<float> got_a(ya.size()), got_b(yb.size()), got_c(yc.size());
    q.memcpy(got_a.data(), d_ya, got_a.size() * 4);
    q.memcpy(got_b.data(), d_yb, got_b.size() * 4);
    q.memcpy(got_c.data(), d_yc, got_c.size() * 4);
    std::vector<float> got_yg(yg.size()), got_ygs(ygs.size()), got_yr(yr.size()), got_yrs(yrs.size()),
        got_yf(yf.size()), got_yfs(yfs.size());
    q.memcpy(got_yg.data(), d_yg, got_yg.size() * 4);
    q.memcpy(got_ygs.data(), d_ygs, got_ygs.size() * 4);
    q.memcpy(got_yr.data(), d_yr, got_yr.size() * 4);
    q.memcpy(got_yrs.data(), d_yrs, got_yrs.size() * 4);
    q.memcpy(got_yf.data(), d_yf, got_yf.size() * 4);
    q.memcpy(got_yfs.data(), d_yfs, got_yfs.size() * 4);
    std::vector<std::vector<uint16_t>> got_sp(3);
    for (int f = 0; f < 3; ++f) {
        got_sp[f].resize((size_t) sn[f] * sr_rows[f]);
        q.memcpy(got_sp[f].data(), d_sp[f], got_sp[f].size() * 2);
    }
    q.wait();

    Counts c;
    Family fam[13];
    int nf = 0;
    fam[nf++] = {"gemm f16->f32 shape A (m=96 n=7 k=192 beta=0.25)", ya_ref};
    fam[nf++] = {"gemm f16->f32 shape B (m=128 n=5 k=64 alpha=0.75)", yb_ref};
    fam[nf++] = {"gemm f32 (m=20 n=5 k=48 ldc=32)", yc_ref};
    fam[nf++] = {"gemv_f16 tile (n=64 k=256 T=3 alpha=0.5)", yg_ref};
    fam[nf++] = {"gemv_f16 scalar fallback (n=64 k=130 T=2)", ygs_ref};
    fam[nf++] = {"gemv_f16_rows tile (n=64 k=256 T=5 ldx=288)", yr_ref};
    fam[nf++] = {"gemv_f16_rows scalar fallback (n=64 k=130 T=3 ldx=130)", yrs_ref};
    fam[nf++] = {"gemv_f32 tile (n=64 k=192 T=3 ldy=130)", yf_ref};
    fam[nf++] = {"gemv_f32 scalar fallback (n=64 k=130 T=2 ldy=130)", yfs_ref};
    fam[nf++] = {"softmax register tile (rows=3 n=1024)", sp_ref[0]};
    fam[nf++] = {"softmax register tile boundary (rows=2 n=10240)", sp_ref[1]};
    fam[nf++] = {"softmax scalar fallback (rows=2 n=12000)", sp_ref[2]};

    {
        Counts t;
        cmp_f32(got_a.data(), ya_ref.data(), ya_ref.size(), t);
        report(fam[0].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }
    {
        Counts t;
        cmp_f32(got_b.data(), yb_ref.data(), yb_ref.size(), t);
        report(fam[1].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }
    {
        Counts t;
        cmp_f32(got_c.data(), yc_ref.data(), yc_ref.size(), t);
        report(fam[2].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }
    {
        Counts t;
        cmp_f32(got_yg.data(), yg_ref.data(), yg_ref.size(), t);
        report(fam[3].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }
    {
        Counts t;
        cmp_f32(got_ygs.data(), ygs_ref.data(), ygs_ref.size(), t);
        report(fam[4].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }
    {
        Counts t;
        cmp_f32(got_yr.data(), yr_ref.data(), yr_ref.size(), t);
        report(fam[5].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }
    {
        Counts t;
        cmp_f32(got_yrs.data(), yrs_ref.data(), yrs_ref.size(), t);
        report(fam[6].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }
    {
        Counts t;
        cmp_f32(got_yf.data(), yf_ref.data(), yf_ref.size(), t);
        report(fam[7].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }
    {
        Counts t;
        cmp_f32(got_yfs.data(), yfs_ref.data(), yfs_ref.size(), t);
        report(fam[8].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }
    {
        Counts t;
        cmp_bf16(got_sp[0].data(), sp_ref[0].data(), sp_ref[0].size(), t);
        report(fam[9].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }
    {
        Counts t;
        cmp_bf16(got_sp[1].data(), sp_ref[1].data(), sp_ref[1].size(), t);
        report(fam[10].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }
    {
        Counts t;
        cmp_bf16(got_sp[2].data(), sp_ref[2].data(), sp_ref[2].size(), t);
        report(fam[11].name, t);
        c.exact += t.exact; c.near += t.near; c.bad += t.bad; c.total += t.total;
        if (t.max_diff > c.max_diff) c.max_diff = t.max_diff;
    }

    long long distinct = 0;
    for (int f = 0; f < nf; ++f) distinct += distinct_values(fam[f].ref.data(), fam[f].ref.size());
    std::printf("  non-vacuity: %lld distinct reference values across %d families, %lld values compared\n",
                distinct, nf, c.total);
    if (c.total == 0 || distinct < 8 * nf) {
        std::fprintf(stderr, "VACUOUS: the comparison could not have distinguished a wrong result\n");
        return 1;
    }

    if (c.bad == 0) {
        std::printf("MATCH: %lld/%lld values equal to the CPU reference, %lld within one BF16 ulp, max float diff %g\n",
                    c.exact, c.total, c.near, (double) c.max_diff);
        if (selftest) std::printf("ds4_blas_parity OK\n");
        cublasDestroy(h);
        return 0;
    }
    std::printf("NO MATCH: %lld values outside one BF16 ulp of the reference\n", c.bad);
    cublasDestroy(h);
    return 1;
}
