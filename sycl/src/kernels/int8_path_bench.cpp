// sycl/src/kernels/int8_path_bench.cpp - the INT8 prompt expert GEMM on real Q2_0 rows (README.sycl.md P0
// "GEMM-shaped INT8 prompt path"; docs/INTEL.md planned item 6).
//
//     int8_path_bench <shard1.gguf> [layer=0] [T=96] [reps=50]
//
// One real Q2_0 expert, two ways: (a) the shipped FP16 path (iq_dequant_gu_f16 -> oneMKL FP16 GEMM -> SwiGLU ->
// iq_dequant_f16 -> oneMKL FP16 GEMM), (b) the INT8 path (iq_quant_gu_i8/iq_quant_i8 -> quantize_act_i8 ->
// oneMKL INT8 GEMM -> scale_rows_i8 -> SwiGLU -> ...).  Prints the output difference (the INT8 path is NOT
// lossless: one scale per row cannot carry the Q2_0 per-64 block structure) and the time of each phase, so the
// speed win and its fidelity cost are measured together rather than assumed.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace prefill = strata::prefill;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: int8_path_bench <shard1.gguf> [layer=0] [T=96] [reps=50]\n"); return 2; }
    const int layer = argc > 2 ? std::atoi(argv[2]) : 0;
    const int T = argc > 3 ? std::atoi(argv[3]) : 96;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 50;
    const int E = 7;
    const int64_t H = 2560, FF = 640;
    sycl::queue* s = &dpct::get_in_order_queue();

    strata::GgufFile gguf(argv[1]);
    const strata::TensorInfo* t[3] = {};
    const char* roles[3] = {"gate", "up", "down"};
    for (const auto& ti : gguf.tensors())
        for (int r = 0; r < 3; ++r)
            if (ti.name == "blk." + std::to_string(layer) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
    if (!t[0] || !t[1] || !t[2]) { std::fprintf(stderr, "layer %d: no expert tensors\n", layer); return 2; }
    const int gu_ty = (int) t[0]->type, d_ty = (int) t[2]->type;
    std::printf("layer %d: gate/up type %d, down type %d, T=%d\n", layer, gu_ty, d_ty, T);
    if (!strata::kernels::iq_int8_supported(gu_ty, d_ty, H, FF)) {
        std::fprintf(stderr, "  the INT8 path handles Q2_0 gate/up and down only\n");
        return 2;
    }

    // the expert's blob on the host, then on the device (the layout native_expert_layout defines)
    const size_t gu_row = strata::kernels::iq_row_bytes(gu_ty, H);
    const size_t d_row = strata::kernels::iq_row_bytes(d_ty, FF);
    const size_t up_off = (size_t) FF * gu_row;
    const size_t down_off = 2 * up_off;
    const size_t d_bytes = (size_t) H * d_row;
    const size_t blob_bytes = down_off + d_bytes;
    std::vector<uint8_t> blob(blob_bytes);
    std::memcpy(blob.data(), gguf.tensor_data(*t[0]) + (size_t) E * up_off, up_off);
    std::memcpy(blob.data() + up_off, gguf.tensor_data(*t[1]) + (size_t) E * up_off, up_off);
    std::memcpy(blob.data() + down_off, gguf.tensor_data(*t[2]) + (size_t) E * d_bytes, d_bytes);
    uint8_t* dblob = sycl::malloc_device<uint8_t>(blob_bytes, *s);
    s->memcpy(dblob, blob.data(), blob_bytes).wait();

    // fp16 activations (what the prompt path feeds the expert GEMMs)
    std::mt19937 rng(1234 + layer);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> hx((size_t) T * H);
    for (auto& v : hx) v = sycl::bit_cast<uint16_t>(sycl::half(nd(rng)));
    uint16_t* X = sycl::malloc_device<uint16_t>((size_t) T * H, *s);
    s->memcpy(X, hx.data(), hx.size() * 2).wait();

    // ---- buffers
    uint16_t *dq_gu = sycl::malloc_device<uint16_t>((size_t) 1280 * H, *s);
    uint16_t *dq_d = sycl::malloc_device<uint16_t>((size_t) H * FF, *s);
    int8_t *wq_gu = sycl::malloc_device<int8_t>((size_t) 1280 * H, *s);
    int8_t *wq_d = sycl::malloc_device<int8_t>((size_t) H * FF, *s);
    float *s_gu = sycl::malloc_device<float>(1280, *s), *s_d = sycl::malloc_device<float>(H, *s);
    int8_t *xq = sycl::malloc_device<int8_t>((size_t) T * H, *s);
    int8_t *hq = sycl::malloc_device<int8_t>((size_t) T * FF, *s);
    float *sx = sycl::malloc_device<float>(T, *s), *sh = sycl::malloc_device<float>(T, *s);
    float *GU16 = sycl::malloc_device<float>((size_t) T * 1280, *s);
    float *D16 = sycl::malloc_device<float>((size_t) T * H, *s);
    float *GU8 = sycl::malloc_device<float>((size_t) T * 1280, *s);
    float *D8 = sycl::malloc_device<float>((size_t) T * H, *s);
    uint16_t *Hh16 = sycl::malloc_device<uint16_t>((size_t) T * FF, *s);
    uint16_t *Hh8 = sycl::malloc_device<uint16_t>((size_t) T * FF, *s);

    prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(s, 0, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    auto fp16_path = [&] {
        strata::kernels::iq_dequant_gu_f16(gu_ty, dblob, dblob + up_off, FF, H, dq_gu, s);
        strata::kernels::iq_dequant_f16(d_ty, dblob + down_off, H * FF, dq_d, s);
        gemm.f16(X, dq_gu, GU16, T, 1280, H);
        prefill::swiglu_interleaved(GU16, Hh16, T, s);
        gemm.f16(Hh16, dq_d, D16, T, H, FF);
    };
    auto int8_path = [&] {
        strata::kernels::iq_quant_gu_i8(gu_ty, dblob, dblob + up_off, FF, H, wq_gu, s_gu, s);
        strata::kernels::iq_quant_i8(d_ty, dblob + down_off, H, FF, wq_d, s_d, s);
        prefill::quantize_act_i8(X, xq, sx, T, H, s);
        gemm.int8(xq, wq_gu, GU8, T, 1280, H);
        prefill::scale_rows_i8(GU8, sx, s_gu, T, 1280, 1280, s);
        prefill::swiglu_interleaved(GU8, Hh8, T, s);
        prefill::quantize_act_i8(Hh8, hq, sh, T, FF, s);
        gemm.int8(hq, wq_d, D8, T, H, FF);
        prefill::scale_rows_i8(D8, sh, s_d, T, H, H, s);
    };

    fp16_path(); int8_path(); s->wait();


    // ---- fidelity: the weight quantizer's own error, then both expert outputs
    std::vector<float> hy16((size_t) T * 1280), hy8((size_t) T * 1280);
    std::vector<float> hd16((size_t) T * H), hd8((size_t) T * H);
    s->memcpy(hy16.data(), GU16, hy16.size() * 4).wait();
    s->memcpy(hy16.data(), GU16, hy16.size() * 4).wait();
    s->memcpy(hy8.data(), GU8, hy8.size() * 4).wait();
    s->memcpy(hd16.data(), D16, hd16.size() * 4).wait();
    s->memcpy(hd8.data(), D8, hd8.size() * 4).wait();
    auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
        double n = 0, d = 0;
        for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
        return n / (d + 1e-30);
    };
    auto errs = [](const std::vector<float>& a, const std::vector<float>& b, double& med, double& p90) {
        std::vector<double> e;
        for (size_t i = 0; i < a.size(); ++i) e.push_back(std::fabs((double) a[i] - b[i]) / (std::fabs((double) b[i]) + 1e-6));
        std::sort(e.begin(), e.end());
        med = e[e.size() / 2];
        p90 = e[(size_t) (e.size() * 0.9)];
    };
    double gm, gp, dm, dp;
    errs(hy8, hy16, gm, gp);
    errs(hd8, hd16, dm, dp);
    std::printf("gate/up: rel-mean %.4f  |rel| median %.4f p90 %.4f\n", rel(hy8, hy16), gm, gp);
    std::printf("down   : rel-mean %.4f  |rel| median %.4f p90 %.4f\n", rel(hd8, hd16), dm, dp);

    // ---- timing
    auto time = [&](auto&& fn) {
        for (int i = 0; i < 3; ++i) fn();
        s->wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) fn();
        s->wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    const double t16 = time(fp16_path), t8 = time(int8_path);
    std::printf("expert fp16 %.3f ms | int8 %.3f ms | %.2fx\n", t16, t8, t16 / t8);

    // where the INT8 path's time goes, against the two phases the FP16 path has
    const double t_dq16 = time([&] {
        strata::kernels::iq_dequant_gu_f16(gu_ty, dblob, dblob + up_off, FF, H, dq_gu, s);
        strata::kernels::iq_dequant_f16(d_ty, dblob + down_off, H * FF, dq_d, s);
    });
    const double t_wq = time([&] {
        strata::kernels::iq_quant_gu_i8(gu_ty, dblob, dblob + up_off, FF, H, wq_gu, s_gu, s);
        strata::kernels::iq_quant_i8(d_ty, dblob + down_off, H, FF, wq_d, s_d, s);
    });
    const double t_gu16 = time([&] { gemm.f16(X, dq_gu, GU16, T, 1280, H); });
    const double t_gu8 = time([&] { gemm.int8(xq, wq_gu, GU8, T, 1280, H); });
    const double t_d16 = time([&] { gemm.f16(Hh16, dq_d, D16, T, H, FF); });
    const double t_d8 = time([&] { gemm.int8(hq, wq_d, D8, T, H, FF); });
    const double t_aq = time([&] {
        prefill::quantize_act_i8(X, xq, sx, T, H, s);
        prefill::quantize_act_i8(Hh8, hq, sh, T, FF, s);
    });
    const double t_ep = time([&] {
        prefill::scale_rows_i8(GU8, sx, s_gu, T, 1280, 1280, s);
        prefill::scale_rows_i8(D8, sh, s_d, T, H, H, s);
    });
    std::printf("  fp16: dequant-w %.3f  gemm gu %.3f  gemm down %.3f\n", t_dq16, t_gu16, t_d16);
    std::printf("  int8: quant-w %.3f  act-quant %.3f  gemm gu %.3f  gemm down %.3f  epilogue %.3f\n", t_wq, t_aq,
                t_gu8, t_d8, t_ep);
    std::printf("  gemm rate: gate/up %.2fx  down %.2fx\n", t_gu16 / t_gu8, t_d16 / t_d8);
    return 0;
}
