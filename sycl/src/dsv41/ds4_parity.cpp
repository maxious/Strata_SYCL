// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// The device dequant parity test for the ported dsv41 kernels: iq::dequant_rows (ds4_iq.dp.cpp) on the card against a
// scalar CPU reference written from the same layouts the shipped CPU path uses (include/strata/artifact/dequant.hpp
// and ggml's dequantize_row_q4_K/q5_K/q6_K) - not copied from the device kernel, so it checks the layout and the
// arithmetic, not just the translation.
//
// Runs on the card: prints the device name, then one line per type, then MATCH when every decoded BF16 bit equals the
// reference (or differs by one BF16 ulp, which FMA vs separate multiply-and-subtract can cause). Exit 0 on MATCH.
//
// Build (this PC, 2026-10-24): icpx -fsycl -std=c++20 -O2 -I sycl/include -I include -I third_party/ggml
//   -I sycl/include/compat <cuda-header dir> ds4_parity.cpp ds4_iq.dp.cpp
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ds4_iq.dp.hpp"
#include "ds4_numerics.dp.hpp"

namespace {

constexpr int kQK = 256;      // values a block of the dense K-quants holds

// ggml's GGML_FP16_TO_FP32: the exact widening of a stored fp16.
float fp16_to_f32(const uint8_t* p) {
    const uint16_t bits = (uint16_t) (p[0] | (p[1] << 8));
    return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

// ggml's get_scale_min_k4: the 6-bit scale/min pairs packed into the block's 12 scale bytes.
void get_scale_min_k4(int j, const uint8_t* s, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = s[j] & 63;
        *m = s[j + 4] & 63;
    } else {
        *d = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        *m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// The three scalar references, transcribed from ggml-quants.c's dequantize_row_q4_K/q5_K/q6_K (b4050-era file; the
// layouts are ggml-common.h's block_q4_K/q5_K/q6_K: {dm; scales[12]; qs[128]}, {dm; scales[12]; qh[32]; qs[128]},
// {ql[128]; qh[64]; scales[16]; d}).
void ref_q4_K(const uint8_t* row, float* y) {
    const float d = fp16_to_f32(row + 0), min = fp16_to_f32(row + 2);
    const uint8_t* sc = row + 4, *q = row + 16;
    int is = 0;
    for (int j = 0; j < kQK; j += 64) {
        uint8_t s1, m1v, s2, m2v;
        get_scale_min_k4(is + 0, sc, &s1, &m1v);
        get_scale_min_k4(is + 1, sc, &s2, &m2v);
        const float d1 = d * s1, mm1 = min * m1v;
        const float d2 = d * s2, mm2 = min * m2v;
        for (int l = 0; l < 32; ++l) y[j + l] = d1 * (q[l] & 0xF) - mm1;
        for (int l = 0; l < 32; ++l) y[j + 32 + l] = d2 * (q[l] >> 4) - mm2;
        q += 32; is += 2;
    }
}

void ref_q5_K(const uint8_t* row, float* y) {
    const float d = fp16_to_f32(row + 0), min = fp16_to_f32(row + 2);
    const uint8_t* sc = row + 4, *qh = row + 16, *q = row + 48;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < kQK; j += 64) {
        uint8_t s1, m1v, s2, m2v;
        get_scale_min_k4(is + 0, sc, &s1, &m1v);
        get_scale_min_k4(is + 1, sc, &s2, &m2v);
        const float d1 = d * s1, mm1 = min * m1v;
        const float d2 = d * s2, mm2 = min * m2v;
        for (int l = 0; l < 32; ++l) y[j + l] = d1 * ((q[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - mm1;
        for (int l = 0; l < 32; ++l) y[j + 32 + l] = d2 * ((q[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - mm2;
        q += 32; is += 2;
        u1 <<= 2; u2 <<= 2;
    }
}

void ref_q6_K(const uint8_t* row, float* y) {
    const float d = fp16_to_f32(row + 208);
    const uint8_t* ql = row;
    const uint8_t* qh = row + 128;
    const int8_t* sc = (const int8_t*) (row + 192);
    for (int n = 0; n < kQK; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[n + l + 0] = d * sc[is + 0] * q1;
            y[n + l + 32] = d * sc[is + 2] * q2;
            y[n + l + 64] = d * sc[is + 4] * q3;
            y[n + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// ggml's kvalues_iq4nl codebook, verbatim (include/strata/artifact/dequant.hpp, ggml-common.h at 3cf03257).
const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                  1,    13,   25,  38,  53,  69,  89,  113};

// The IQ4_NL rows (8 blocks of 18 bytes) and the IQ4_XS superblock, transcribed from the same header.
void ref_iq4_nl(const uint8_t* row, float* y) {
    for (int b = 0; b < kQK / 32; ++b) {
        const uint8_t* block = row + 18 * b;
        const float d = fp16_to_f32(block);
        const uint8_t* qs = block + 2;
        for (int j = 0; j < 16; ++j) {
            y[b * 32 + j] = d * (float) kvalues_iq4nl[qs[j] & 0x0F];
            y[b * 32 + j + 16] = d * (float) kvalues_iq4nl[qs[j] >> 4];
        }
    }
}

void ref_iq4_xs(const uint8_t* row, float* y) {
    const float d = fp16_to_f32(row);
    const uint16_t scales_h = (uint16_t) (row[2] | (row[3] << 8));
    const uint8_t* scales_l = row + 4;
    const uint8_t* qs = row + 8;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[ib * 32 + j] = dl * (float) kvalues_iq4nl[qs[j] & 0x0F];
            y[ib * 32 + j + 16] = dl * (float) kvalues_iq4nl[qs[j] >> 4];
        }
        qs += 16;
    }
}

// A seeded byte generator: the same bytes on every run, so a pass never depends on luck.
struct Rng {
    unsigned long long s = 0x20261024ull;
    uint8_t next() {
        s = s * 6364136222842883251ull + 1442695040888963407ull;
        return (uint8_t) ((s >> 33) & 0xFF);
    }
};

// The block's scale fields get small finite fp16 values (random bits could land on NaN or huge scales); the quants and
// 6-bit scale pairs stay seeded-random.
void fill_block(int type, uint8_t* row, Rng& rng) {
    for (size_t i = 0; i < (size_t) strata::ds4::gguf::iq::block_bytes(type); ++i) row[i] = rng.next();
    if (type == strata::ds4::gguf::iq::kQ4_K || type == strata::ds4::gguf::iq::kQ5_K) {
        for (int i = 0; i < 2; ++i) {
            const float v = 0.01f + 0.25f * (float) (rng.next() & 31);
            const uint16_t u = sycl::bit_cast<uint16_t>(sycl::half(v));
            row[2 * i] = (uint8_t) (u & 0xFF);
            row[2 * i + 1] = (uint8_t) (u >> 8);
        }
    } else if (type == strata::ds4::gguf::iq::kQ6_K) {
        const float v = 0.01f + 0.25f * (float) (rng.next() & 31);
        const uint16_t u = sycl::bit_cast<uint16_t>(sycl::half(v));
        row[208] = (uint8_t) (u & 0xFF);
        row[209] = (uint8_t) (u >> 8);
    } else if (type == strata::ds4::gguf::iq::kIQ4_NL || type == strata::ds4::gguf::iq::kIQ4_XS) {
        const float v = 0.01f + 0.25f * (float) (rng.next() & 31);
        const uint16_t u = sycl::bit_cast<uint16_t>(sycl::half(v));
        row[0] = (uint8_t) (u & 0xFF);
        row[1] = (uint8_t) (u >> 8);
    }
}

// One BF16 ulp of a finite value: 2^(exponent - 7) (BF16 has 8 mantissa bits counting the implied one).
float bf16_ulp(float v) {
    if (!(v == v) || v == 0.0f) return 0.0f;
    const int e = (int) ((sycl::bit_cast<uint32_t>(std::fabs(v)) >> 23) & 0xFF) - 127;
    return std::ldexp(1.0f, e - 7);
}

}  // namespace

int main() {
    try {
        sycl::device dev(sycl::gpu_selector_v);
        sycl::queue q(dev);
        printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        dpct::queue_ptr s = &q;

        const int types[] = {strata::ds4::gguf::iq::kQ4_K, strata::ds4::gguf::iq::kQ5_K, strata::ds4::gguf::iq::kQ6_K,
                             strata::ds4::gguf::iq::kIQ4_NL, strata::ds4::gguf::iq::kIQ4_XS};
        const int n_rows = 4;
        Rng rng;
        int exact = 0, near = 0, bad = 0;
        float max_diff = 0.0f;

        for (const int type : types) {
            const size_t ld = strata::ds4::gguf::iq::row_bytes(type, kQK);
            std::vector<uint8_t> raw(ld * n_rows);
            for (int r = 0; r < n_rows; ++r) fill_block(type, raw.data() + (size_t) r * ld, rng);

            std::vector<float> ref((size_t) kQK * n_rows);
            for (int r = 0; r < n_rows; ++r) {
                const uint8_t* row = raw.data() + (size_t) r * ld;
                float* y = ref.data() + (size_t) r * kQK;
                if (type == strata::ds4::gguf::iq::kQ4_K) ref_q4_K(row, y);
                else if (type == strata::ds4::gguf::iq::kQ5_K) ref_q5_K(row, y);
                else if (type == strata::ds4::gguf::iq::kQ6_K) ref_q6_K(row, y);
                else if (type == strata::ds4::gguf::iq::kIQ4_NL) ref_iq4_nl(row, y);
                else ref_iq4_xs(row, y);
            }

            uint8_t* w = (uint8_t*) sycl::malloc_device(raw.size(), q);
            uint16_t* d = (uint16_t*) sycl::malloc_device((size_t) kQK * n_rows * sizeof(uint16_t), q);
            q.memcpy(w, raw.data(), raw.size());
            q.wait();
            const bool launched = strata::ds4::gguf::iq::dequant_rows(type, w, ld, kQK, n_rows, d, true, s);
            q.wait();

            std::vector<uint16_t> got((size_t) kQK * n_rows);
            q.memcpy(got.data(), d, got.size() * sizeof(uint16_t));
            q.wait();
            sycl::free(w, q);
            sycl::free(d, q);

            if (!launched) {
                if (type == strata::ds4::gguf::iq::kIQ4_NL) {
                    // (IQ4_NL's block is 32 values, not on the dq_rows path: CUDA's dequant_rows
                    // refuses it too, so refusing is the parity - not a failure.)
                    printf("IQ4_NL: correctly refused (32-value blocks are not a dq_rows path)\n");
                    continue;
                }
                printf("%s: FAIL (no kernel launched)\n", strata::ds4::gguf::iq::name(type));
                ++bad;
                continue;
            }
            int t_exact = 0, t_near = 0, t_bad = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                const float a = static_cast<float>(sycl::bit_cast<sycl::half>(got[i]));   // the device's BF16 value widened
                const uint16_t want = strata::ds4::bf16_rn(ref[i]);
                if (got[i] == want) ++t_exact;
                else {
                    const float b = ref[i];
                    const float diff = std::fabs(a - b);
                    if (diff > max_diff) max_diff = diff;
                    if (diff <= bf16_ulp(b)) ++t_near;
                    else ++t_bad;
                }
            }
            printf("%s: %d/%d bits exact, %d within one BF16 ulp, %d outside\n",
                   strata::ds4::gguf::iq::name(type), t_exact, (int) got.size(), t_near, t_bad);
            exact += t_exact; near += t_near; bad += t_bad;
        }

        if (bad == 0) {
            printf("MATCH: %d/%d BF16 bits equal to the CPU reference, %d within one ulp, max float diff %g\n",
                   exact, exact + near + bad, near, (double) max_diff);
            return 0;
        }
        printf("NO MATCH: %d values outside one BF16 ulp of the reference\n", bad);
        return 1;
    } catch (const sycl::exception& e) {
        printf("SYCL error: %s\n", e.what());
        return 2;
    }
}
