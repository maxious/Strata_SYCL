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
#include "ds4_gguf.dp.hpp"
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

// F16 bits of v rounded to nearest-even (the kernels' __float2half_rn / the device's convert<rte>).  Not
// sycl::vec::convert on the host: oneAPI 2026.2's host path truncates tiny values instead of rounding them
// (probe 2026-10-25: +0.525 steps -> 0x0000 where round-to-nearest gives 0x0001), which put the reference one
// F16 step below the kernels at 35 slots of the gate/up row.
uint16_t f16_rn(float v) {
    uint32_t b;
    std::memcpy(&b, &v, sizeof b);
    const uint16_t sign = (uint16_t) ((b >> 16) & 0x8000u);
    const uint32_t a = b & 0x7FFFFFFFu;
    if (a >= 0x7F800000u) return (uint16_t) (sign | 0x7C00u | (a > 0x7F800000u ? 0x0200u : 0u));
    if (a < 0x38800000u) {   // |v| < 2^-14: the 2^-24 subnormal grid
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

// The MXFP4 set-tile reference, written from the block_mxfp4 layout documented in ds4_gguf.dp.hpp: a 17-byte block is
// an E8M0 scale e then 16 bytes of E2M1 codes, value j in byte j's low nibble (j < 16) or byte j - 16's high nibble,
// worth kvalues[code] * 2^(e - 128) with kvalues {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}.  The tiles
// hold F16 inputs and weights, so the reference rounds the inputs to F16 (rte) and keeps F32 sums in row order; the
// output is the F16 (rte) clamped swiglu for gate/up, the raw F32 sums for down.
void ref_mx_set_gate_up(const uint8_t* blob, size_t tr0, size_t tr1, const float* x, int ldx, const int* idx,
                        int n_embd, int n_ff, float limit, int r0, int r1, uint16_t* h16, int ldh) {
    const float lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const size_t rb = strata::ds4::gguf::mx_row_bytes(n_embd);
    const int nb = n_embd / 32;
    for (int r = r0; r < r1; ++r) {
        const int src = idx[r];
        for (int c = 0; c < n_ff; ++c) {
            const uint8_t* g = blob + tr0 + (size_t) c * rb;
            const uint8_t* u = blob + tr1 + (size_t) c * rb;
            float gv = 0.0f, uv = 0.0f;
            for (int kb = 0; kb < nb; ++kb) {
                const uint32_t eg = g[kb * 17], eu = u[kb * 17];
                const float dg = sycl::bit_cast<float>(eg < 2 ? 0x00200000u << eg : (eg - 1) << 23);
                const float du = sycl::bit_cast<float>(eu < 2 ? 0x00200000u << eu : (eu - 1) << 23);
                for (int j = 0; j < 32; ++j) {
                    const int k = kb * 32 + j;
                    const float xv = static_cast<float>(sycl::bit_cast<sycl::half>(f16_rn(x[(size_t) src * ldx + k])));
                    const uint8_t cg = j < 16 ? g[kb * 17 + 1 + j] : g[kb * 17 + 1 + j - 16] >> 4;
                    const uint8_t cu = j < 16 ? u[kb * 17 + 1 + j] : u[kb * 17 + 1 + j - 16] >> 4;
                    gv += (float) xv * (lut[cg & 15u] * dg);
                    uv += (float) xv * (lut[cu & 15u] * du);
                }
            }
            float gg = gv, uu = uv;
            if (limit > 0.0f) {
                gg = std::fmin(gg, limit);
                uu = std::fmin(std::fmax(uu, -limit), limit);
            }
            const float o = gg / (1.0f + std::exp(-gg)) * uu;
            h16[(size_t) r * ldh + c] = f16_rn(o);
        }
    }
}

void ref_mx_set_down(const uint8_t* blob, size_t tr2, const uint16_t* h16, int ldh, int n_embd, int n_ff,
                     int r0, int r1, float* y, int ldy) {
    const float lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const size_t rb = strata::ds4::gguf::mx_row_bytes(n_ff);
    const int nb = n_ff / 32;
    for (int r = r0; r < r1; ++r) {
        for (int c = 0; c < n_embd; ++c) {
            const uint8_t* d = blob + tr2 + (size_t) c * rb;
            float v = 0.0f;
            for (int kb = 0; kb < nb; ++kb) {
                const uint32_t ed = d[kb * 17];
                const float dd = sycl::bit_cast<float>(ed < 2 ? 0x00200000u << ed : (ed - 1) << 23);
                for (int j = 0; j < 32; ++j) {
                    const int k = kb * 32 + j;
                    const float xv = static_cast<float>(sycl::bit_cast<sycl::half>(h16[(size_t) r * ldh + k]));
                    const uint8_t cd = j < 16 ? d[kb * 17 + 1 + j] : d[kb * 17 + 1 + j - 16] >> 4;
                    v += xv * (lut[cd & 15u] * dd);
                }
            }
            y[(size_t) r * ldy + c] = v;
        }
    }
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

        {   // The MXFP4 expert-set tiles (ds4_gguf's mx_set_tc_kernel; joint_matrix on Xe2): one expert of 32 rows,
            // gate/up and down, rows [16, 32) - they come from the tile kernel whatever the light kernels' switch
            // state (the light ones only take experts of at most 16 rows).
            const int n_embd = 256, n_ff = 256, max_rows = 32, ldx = n_embd, ldh = n_ff, ldy = n_embd;
            const size_t rb = strata::ds4::gguf::mx_row_bytes(n_embd);
            const size_t tr0 = 0, tr1 = (size_t) n_ff * rb, tr2 = tr1 * 2;
            const size_t stride = tr2 + (size_t) n_embd * strata::ds4::gguf::mx_row_bytes(n_ff);
            std::vector<uint8_t> blob(stride);
            for (size_t i = 0; i < blob.size(); ++i) blob[i] = rng.next();
            for (int part = 0; part < 3; ++part) {   // the E8M0 scales: small powers of two keep the weights exact in F16
                const size_t base = part == 0 ? tr0 : part == 1 ? tr1 : tr2;
                const int rows = part < 2 ? n_ff : n_embd, k = part < 2 ? n_embd : n_ff;
                const size_t prb = strata::ds4::gguf::mx_row_bytes(k);
                for (int r = 0; r < rows; ++r)
                    for (int b = 0; b < k / 32; ++b)
                        blob[base + (size_t) r * prb + (size_t) b * 17] = (uint8_t) (128 + (rng.next() & 3));
            }
            std::vector<float> xf((size_t) max_rows * ldx);
            for (size_t i = 0; i < xf.size(); ++i)
                xf[i] = ((int) (rng.next() | (rng.next() << 8)) - 32768) / 32768.0f;
            std::vector<int> idx(max_rows);
            for (int r = 0; r < max_rows; ++r) idx[r] = r;
            std::vector<uint16_t> xh((size_t) max_rows * ldh);
            for (size_t i = 0; i < xh.size(); ++i)   // F16 rows in +-[0.5, 1): finite, away from zero
                xh[i] = (uint16_t) ((rng.next() | (rng.next() << 8)) & 0xBBFFu);
            const int bounds_h[2] = {0, max_rows};
            uint8_t* w = (uint8_t*) sycl::malloc_device(blob.size(), q);
            int* bounds = (int*) sycl::malloc_device(sizeof(bounds_h), q);
            int* didx = (int*) sycl::malloc_device(idx.size() * sizeof(int), q);
            float* dxf = (float*) sycl::malloc_device(xf.size() * sizeof(float), q);
            uint16_t* dxh = (uint16_t*) sycl::malloc_device(xh.size() * sizeof(uint16_t), q);
            uint16_t* dh = (uint16_t*) sycl::malloc_device(xh.size() * sizeof(uint16_t), q);
            float* dy = (float*) sycl::malloc_device((size_t) max_rows * ldy * sizeof(float), q);
            q.memcpy(w, blob.data(), blob.size());
            q.memcpy(bounds, bounds_h, sizeof(bounds_h));
            q.memcpy(didx, idx.data(), idx.size() * sizeof(int));
            q.memcpy(dxf, xf.data(), xf.size() * sizeof(float));
            q.memcpy(dxh, xh.data(), xh.size() * sizeof(uint16_t));
            q.wait();
            if (!strata::ds4::gguf::prepare(s)) {
                printf("MXFP4 set tile: FAIL (no scratch)\n");
                ++bad;
            } else {
                strata::dsv41k::exl3::SetArgs sa{};
                sa.wbase = w;
                sa.stride = stride;
                sa.lay.fmt = strata::ds4::gguf::kFmtMXFP4;
                sa.lay.trellis[0] = tr0; sa.lay.trellis[1] = tr1; sa.lay.trellis[2] = tr2;
                sa.n_exp = 1;
                sa.bounds = bounds;
                sa.row0 = 0;
                sa.max_rows = max_rows;
                sa.n_embd = n_embd;
                sa.n_ff = n_ff;
                sa.limit = 10.0f;
                strata::ds4::gguf::set_gate_up(sa, dxf, ldx, didx, dh, ldh, s);
                strata::ds4::gguf::set_down(sa, dxh, ldh, dy, ldy, s);
                q.wait();
                if (strata::ds4::gguf::launch_errors() != 0) {
                    printf("MXFP4 set tile: FAIL (%d launch errors)\n", strata::ds4::gguf::launch_errors());
                    ++bad;
                }
                std::vector<uint16_t> hg((size_t) max_rows * ldh);
                std::vector<float> yg((size_t) max_rows * ldy);
                q.memcpy(hg.data(), dh, hg.size() * sizeof(uint16_t));
                q.memcpy(yg.data(), dy, yg.size() * sizeof(float));
                q.wait();
                std::vector<uint16_t> href((size_t) max_rows * ldh);
                std::vector<float> yref((size_t) max_rows * ldy);
                ref_mx_set_gate_up(blob.data(), tr0, tr1, xf.data(), ldx, idx.data(), n_embd, n_ff, 10.0f,
                                   16, max_rows, href.data(), ldh);
                ref_mx_set_down(blob.data(), tr2, xh.data(), ldh, n_embd, n_ff, 16, max_rows, yref.data(), ldy);
                int g_exact = 0, g_near = 0, g_bad = 0, d_exact = 0, d_near = 0, d_bad = 0;
                for (int r = 16; r < max_rows; ++r) {
                    for (int c = 0; c < n_ff; ++c) {
                        const float a = static_cast<float>(sycl::bit_cast<sycl::half>(hg[(size_t) r * ldh + c]));
                        const float b = static_cast<float>(sycl::bit_cast<sycl::half>(href[(size_t) r * ldh + c]));
                        if (hg[(size_t) r * ldh + c] == href[(size_t) r * ldh + c]) ++g_exact;
                        else {
                            const float diff = std::fabs(a - b);
                            if (diff > max_diff) max_diff = diff;
                            if (diff <= bf16_ulp(b)) ++g_near; else ++g_bad;
                        }
                    }
                    for (int c = 0; c < n_embd; ++c) {
                        const float a = yg[(size_t) r * ldy + c], b = yref[(size_t) r * ldy + c];
                        if (a == b) ++d_exact;
                        else {
                            const float diff = std::fabs(a - b);
                            if (diff > max_diff) max_diff = diff;
                            if (diff <= bf16_ulp(b)) ++d_near; else ++d_bad;
                        }
                    }
                }
                printf("MXFP4 set gate/up tile: %d/%d fp16 bits exact, %d within one BF16 ulp, %d outside\n",
                       g_exact, g_exact + g_near + g_bad, g_near, g_bad);
                printf("MXFP4 set down tile: %d/%d fp32 values exact, %d within one BF16 ulp, %d outside\n",
                       d_exact, d_exact + d_near + d_bad, d_near, d_bad);
                exact += g_exact + d_exact; near += g_near + d_near; bad += g_bad + d_bad;
            }
            sycl::free(w, q); sycl::free(bounds, q); sycl::free(didx, q); sycl::free(dxf, q); sycl::free(dxh, q);
            sycl::free(dh, q); sycl::free(dy, q);
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
