// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// The device dequant parity test for the ported dsv41 kernels: iq::dequant_rows (ds4_iq.dp.cpp) on the card against a
// scalar CPU reference written from the same layouts the shipped CPU path uses (include/strata/artifact/dequant.hpp
// and ggml's dequantize_row_q4_K/q5_K/q6_K) - not copied from the device kernel, so it checks the layout and the
// arithmetic, not just the translation.
//
// Runs on the card: prints the device name, then one line per type, per set-tile row and per K-quant GEMV type,
// then MATCH when every decoded
// BF16 bit equals the reference (or differs by one BF16 ulp, which FMA vs separate multiply-and-subtract can cause).
// Exit 0 on MATCH.  --bad-scale / --bad-scale-gu each flip one IQ4_XS scale byte of the set tiles' input after the
// reference is computed (down / gate row 0's scales_l[0]): the affected row must then report values outside one BF16
// ulp and the process must exit non-zero - a row that cannot fail is not evidence.
//
// The decode_q8_0 / decode_iq4_nl / decode_mxfp4 rows are the canonical block-decode oracle (plan todo 2): 32 F16
// (rte) values per canonical row-major block (bid = n * bpr + kb) bit-exact against the shipped CPU scalar
// dequantizer (include/strata/artifact/dequant.hpp; MXFP4 from ggml-common.h's tables) - no ulp tolerance there.
// --neg <decode row> flips one payload or scale byte of its input after the reference is computed and must exit 1;
// --weird-scale feeds NaN / denormal scale blocks (deterministic output, run twice to see it).
//
// The iq4_nl stageA set rows (plan todo 5) drive the sg16 A-stage tile through the dispatch arm at M = 2 .. 160 with
// the K and N tails, against a scalar reference written from the tile's F16-input contract (the four sub-groups'
// whole-block ranges, the fixed partial order); --neg-stageA corrupts one scale byte after the reference, so the row
// must report NO MATCH and exit 1 - a row that cannot fail is not evidence.
//
// Build (this PC, 2026-10-25, verified green): icpx -fsycl -std=c++20 -O2 -fsycl-default-sub-group-size=32
//   -fsycl-device-code-split=per_kernel -fp-model=precise -I sycl/include -I include -I third_party/ggml
//   -I sycl/include/compat -I include/strata/artifact -I sycl/src/dsv41 ds4_parity.cpp ds4_iq.dp.cpp
//   ds4_tq.dp.cpp ds4_gguf.dp.cpp -Xsycl-target-backend=spir64
//   "-cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required" -o ds4_parity
//   (all four TUs: the harness drives the iq, tq and gguf entry points; the set-tile rows need ds4_gguf.dp.cpp
//   and ds4_tq.dp.cpp for iq::tq::*). The ds4_parity CMake target (STRATA_SYCL_DSV41=ON) builds the same thing;
//   the XMX tile form adds -DSTRATA_XMX_SG16=1 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
//   "-device bmg-g21 -options -cl-fp32-correctly-rounded-divide-sqrt".
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>   // compat shim (sycl/include/compat): __device__ for ggml-common.h
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"   // the shipped codebook tables (kvalues_iq4nl / kvalues_mxfp4), never re-transcribed
#include "strata/artifact/dequant.hpp"   // the shipped CPU scalar dequantizer (dequantize_q8_0 / dequantize_iq4_nl)

#include "ds4_iq.dp.hpp"
#include "ds4_gguf.dp.hpp"
#include "ds4_numerics.dp.hpp"
#include "ds4_launch.dp.hpp"
#include "ds4_gemv.dp.hpp"
#include "ds4_softmax.dp.hpp"

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

// ggml's kvalues_iq4nl codebook now comes from third_party/ggml/ggml-common.h (included above with
// GGML_COMMON_IMPL_CUDA, the way ds4_iq.dp.cpp takes it): the same table include/strata/artifact/dequant.hpp and
// the device kernels carry.  The rows below use that shipped table - never a hand-transcribed copy; its values are
// checked against dequant.hpp's own copy by the decode rows themselves.

// The IQ4_NL rows (blocks of 18 bytes) and the IQ4_XS superblock, transcribed from the same header.
void ref_iq4_nl_blocks(const uint8_t* row, int nb, float* y) {
    for (int b = 0; b < nb; ++b) {
        const uint8_t* block = row + 18 * b;
        const float d = fp16_to_f32(block);
        const uint8_t* qs = block + 2;
        for (int j = 0; j < 16; ++j) {
            y[b * 32 + j] = d * (float) kvalues_iq4nl[qs[j] & 0x0F];
            y[b * 32 + j + 16] = d * (float) kvalues_iq4nl[qs[j] >> 4];
        }
    }
}

void ref_iq4_nl(const uint8_t* row, float* y) { ref_iq4_nl_blocks(row, kQK / 32, y); }

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
    unsigned long long s;
    explicit Rng(unsigned long long seed = 0x20261024ull) : s(seed) {}
    uint8_t next() {
        s = s * 6364136222842883251ull + 1442695040888963407ull;
        return (uint8_t) ((s >> 33) & 0xFF);
    }
};

// Deterministic per-index byte for the iq_set_tc rows below: the same value on every run, so the staged and
// default legs of that family can be diffed body-for-body.
uint8_t rng_next(size_t i) {
    unsigned long long s = i * 6364136222842883251ull + 0x20261029ull;
    s = s * 6364136222842883251ull + 1442695040888963407ull;
    return (uint8_t) ((s >> 33) & 0xFF);
}

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
    } else if (type == strata::ds4::gguf::iq::kQ2_K) {
        // block_q2_K's dm pair sits after the scales and quants (bytes 80/82): d in the common small range, dmin
        // smaller, so the min term stays a real share of each weight (a min-term regression moves the row) while no
        // output lands near zero, where one BF16 ulp would not cover a normal float rounding.
        const float v = 0.01f + 0.25f * (float) (rng.next() & 31);
        const uint16_t u = sycl::bit_cast<uint16_t>(sycl::half(v));
        row[80] = (uint8_t) (u & 0xFF);
        row[81] = (uint8_t) (u >> 8);
        const float vm = 0.01f + 0.05f * (float) (rng.next() & 31);
        const uint16_t um = sycl::bit_cast<uint16_t>(sycl::half(vm));
        row[82] = (uint8_t) (um & 0xFF);
        row[83] = (uint8_t) (um >> 8);
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

// The F32 down sums can cancel to near zero, where a relative ulp tolerance is meaningless: two FP32
// accumulation orders (the XMX 16x16 hardware order vs this sequential reference) differ by ~1e-5 there,
// so the down comparison takes the larger of the relative ulp and this small absolute floor.  A wrong
// weight or a stale strip moves a sum by ~10 or more, far above the floor, so the negative control still fails.
constexpr float kDownAbsTol = 1e-3f;

// One F32 ulp of a finite value, with the +-0 rule the reviewed stageA gate suite fixed: at a zero reference the
// subnormal floor 2^-24, not ulp(0) = 0 (a sub-ulp residual at +-0 would otherwise read as outside).
float ulp_f32(float v) {
    if (!(v == v)) return 0.0f;
    if (v == 0.0f) return 1.0f / 16777216.0f;   // 2^-24
    const uint32_t bits = sycl::bit_cast<uint32_t>(std::fabs(v));
    if ((bits >> 23) == 0) return 1.0f / 8388608.0f * 1.40129846e-45f;   // a denormal grid step: 2^-149
    const int e = (int) ((bits >> 23) & 0xFF) - 127;
    return std::ldexp(1.0f, e - 23);
}

// The element bound: one BF16 ulp floored at 2^-24 (the same +-0 rule).
float elem_bound(float v) {
    const float u = bf16_ulp(v);
    return u > 0.0f ? u : 1.0f / 16777216.0f;
}

// The tile-sum bound of the plan: abs_err <= 32*2^-24*sum_abs + 4*ulp_f32(ref).
double tile_bound(double sum_abs, float ref) {
    return 32.0 * 5.960464477539063e-08 * sum_abs + 4.0 * (double) ulp_f32(ref);
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

// ---- the IQ4_XS expert-set tiles (iq::set_tc_kernel) ----
//
// The rows below drive the set path for one expert of 32 rows - more than kLightRows, so the tiles take every row of it
// whatever the light kernels' switch state (ds4_iq.dp.cpp's light_item / light_kernel_rows return early for an expert
// of more rows).  The weights are IQ4_XS blocks decoded with ref_iq4_xs above: the layout the shipped CPU path uses
// (include/strata/artifact/dequant.hpp), not the device kernel's deq8.  Each decoded value is rounded to F16 with
// round-to-nearest-even, matching CUDA's __float2half_rn (that rounding, not a truncation, was e8319a37's reference
// fix); the lines being matched sit beside the f16_rn calls below:
//     src/dsv41/ds4_gguf.cu:758  dst[j] = __float2half_rn(lut[hn ? (c >> 4) : (c & 15u)] * d);
//     src/dsv41/ds4_gguf.cu:800  if (GATE_UP) a.h[lr * a.ldo + c0 + c] = __float2half_rn(swiglu(O.g[r][c], O.u[r][c], a.limit));
// (the work plan cites those two statements as :757 and :793).
//
// The rounded weights are packed as the B strips in the convention the XMX GEMM takes, the one
// sycl/src/kernels/cuda/iq_kernels.dp.cpp's gemm_kernel uses and set_tc_kernel repeats per 32-value step:
//     sm.packed[(k >> 1) * (NT * 2) + n * 2 + (k & 1)] = sm.stage[i];         (gemm_kernel, NT = 16)
//     Tl.pw[n >> 4][((k >> 1) * 32) + 2 * (n & 15) + (k & 1)] = Tl.wg[n][k];  (set_tc_kernel, same pack)
// 32 elements a k-pair, 2 of them per column.  The sums below read the weights back out of those strips, so the
// reference multiplies exactly the operands the tiles hold: F16 weights, F16 inputs, F32 sums.
struct BStrips {
    std::vector<uint16_t> p;
    int K = 0;
    void init(int k, int n) {
        K = k;
        p.assign((size_t) (n / 16) * (size_t) (k / 2) * 32, 0);
    }
    size_t slot(int n, int k) const {
        return (size_t) (n >> 4) * (size_t) (K / 2) * 32 + (size_t) (k >> 1) * 32 + 2 * (n & 15) + (k & 1);
    }
};

// One weight row (K values of IQ4_XS blocks) as output column n's strip: decoded, rounded to F16 (rte), packed.
void iq_pack_row(BStrips& s, int n, const uint8_t* row, int K) {
    const int bb = strata::ds4::gguf::iq::block_bytes(strata::ds4::gguf::iq::kIQ4_XS);
    std::vector<float> w(256);
    for (int b = 0; b < K / 256; ++b) {
        ref_iq4_xs(row + (size_t) b * bb, w.data());
        for (int j = 0; j < 256; ++j) s.p[s.slot(n, b * 256 + j)] = f16_rn(w[j]);
    }
}

float iq_dot(const BStrips& s, int n, const float* x, int K) {
    float v = 0.0f;
    for (int k = 0; k < K; ++k) v += x[k] * static_cast<float>(sycl::bit_cast<sycl::half>(s.p[s.slot(n, k)]));
    return v;
}

// gate/up: h[r][c] = F16 (rte) clamped swiglu of the two dots, as the tile kernel's store:
// ds4_gguf.cu:800 above (and ds4_iq.dp.cpp's set_tc_kernel store, sycl::vec<>::convert rte).
// x_f16 says whether the inputs are rounded to F16 first.  The MXFP4 tiles take F16 inputs, so their rows pass true.
// The IQ4_XS rows pass false: set_kernel, the light path, keeps a.xf as fp32 (ds4_iq.dp.cpp:959-961), and the tile
// kernel rounds it to F16 (rte) at :1162-1168 - a per-term loss of at most half an F16 ulp, which over 512 terms
// stays inside this family's tolerance (rel 2.0e-4, floor 1.0e-3), so one reference serves both the default and the
// staged leg, where the staged kernel hands the tile the fp32 value again (:1174-1188, gate/up rows only).  Measured
// on this PC's Intel Arc Pro B60 with this branch's bmg-g21 AOT artifact: the two legs do NOT report the same counts.
// The staged leg makes far more bits exact - gate/up 1792/8192 to 7936/8192 at n_embd=256, 2048/8192 to 8192/8192 at
// n_embd=512 - while both legs report 0 values outside, and the down rows keep their default counts (1024/8192,
// 2048/16384).  --bad-scale-gu still lands outside on the staged leg: 32 values on the 256-wide row, 31 on the
// 512-wide one.  Logs: w2-parity-aot-default.log, w2-parity-aot-staged.log, w2-parity-aot-bad-scale-gu-staged.log.
void ref_iq_set_gate_up(const uint8_t* blob, size_t tr0, size_t tr1, const float* x, int ldx, const int* idx,
                        int n_embd, int n_ff, float limit, int r0, int r1, uint16_t* h, int ldh, bool x_f16 = true) {
    // A gate/up weight row covers the whole K = n_embd, so it holds n_embd / 256 IQ4_XS blocks, and the kernel
// reads a row at row_bytes_t<T>(K) = (K / kQK<T>) * Ty<T>::B (ds4_iq.dp.cpp:69, kQK 256 for IQ4_XS at :59).
    // block_bytes() is one block, so it is the row size only when K is one block wide.
    const size_t rb = strata::ds4::gguf::iq::row_bytes(strata::ds4::gguf::iq::kIQ4_XS, n_embd);
    BStrips sg, su;
    sg.init(n_embd, n_ff);
    su.init(n_embd, n_ff);
    for (int c = 0; c < n_ff; ++c) {
        iq_pack_row(sg, c, blob + tr0 + (size_t) c * rb, n_embd);
        iq_pack_row(su, c, blob + tr1 + (size_t) c * rb, n_embd);
    }
    std::vector<float> xk(n_embd);
    for (int r = r0; r < r1; ++r) {
        const int src = idx[r];
        for (int k = 0; k < n_embd; ++k)
            xk[k] = x_f16 ? static_cast<float>(sycl::bit_cast<sycl::half>(f16_rn(x[(size_t) src * ldx + k])))
                          : x[(size_t) src * ldx + k];
        for (int c = 0; c < n_ff; ++c) {
            float gg = iq_dot(sg, c, xk.data(), n_embd), uu = iq_dot(su, c, xk.data(), n_embd);
            if (limit > 0.0f) {
                gg = std::fmin(gg, limit);
                uu = std::fmin(std::fmax(uu, -limit), limit);
            }
            const float o = gg / (1.0f + std::exp(-gg)) * uu;
            h[(size_t) r * ldh + c] = f16_rn(o);
        }
    }
}

// down: y[r][c] = the raw F32 sums (the kernel's y store rounds nothing).
void ref_iq_set_down(const uint8_t* blob, size_t tr2, const uint16_t* xh, int ldx, int n_embd, int n_ff,
                     int r0, int r1, float* y, int ldy) {
    // A down weight row covers the whole K = n_ff, so it holds n_ff / 256 IQ4_XS blocks, and the kernel reads a
    // row at row_bytes_t<T>(K) (ds4_iq.dp.cpp:68).  block_bytes() alone is one block.
    const size_t rb = strata::ds4::gguf::iq::row_bytes(strata::ds4::gguf::iq::kIQ4_XS, n_ff);
    BStrips sd;
    sd.init(n_ff, n_embd);
    for (int c = 0; c < n_embd; ++c) iq_pack_row(sd, c, blob + tr2 + (size_t) c * rb, n_ff);
    std::vector<float> xk(n_ff);
    for (int r = r0; r < r1; ++r) {
        for (int k = 0; k < n_ff; ++k)
            xk[k] = static_cast<float>(sycl::bit_cast<sycl::half>(xh[(size_t) r * ldx + k]));
        for (int c = 0; c < n_embd; ++c) y[(size_t) r * ldy + c] = iq_dot(sd, c, xk.data(), n_ff);
    }
}

// ---------------------------------------------------------------- the kernel-launcher families (ds4_launch.dp.cpp)
// One row a family, each against a CPU reference written from the family's contract (ds4_kernels.cuh's comments):
// double-precision sums in row order, or - pred_proj_k alone - the kernel's own float sum order, since its +-1
// projection cancels to near zero often enough that a double reference would measure rounding, not the port.
// The inputs are strictly positive (and the packed rows' E2M1 codes nonzero) so no output lands near zero, where one
// BF16 ulp would not cover a normal float rounding.

struct Tally {
    int exact = 0, near = 0, bad = 0;
    float max_diff = 0.0f;
};

void tally_f32(float got, double ref, Tally& t) {
    if ((double) got == ref) {
        ++t.exact;
        return;
    }
    const double diff = std::fabs((double) got - ref);
    if (diff > t.max_diff) t.max_diff = (float) diff;
    if (diff <= (double) bf16_ulp((float) ref)) ++t.near;
    else ++t.bad;
}

void tally_bf16(uint16_t got, double ref, Tally& t) {
    if (got == strata::ds4::bf16_rn((float) ref)) {
        ++t.exact;
        return;
    }
    tally_f32(static_cast<float>(sycl::bit_cast<sycl::half>(got)), ref, t);
}

void tally_add(Tally& to, const Tally& from) {
    to.exact += from.exact;
    to.near += from.near;
    to.bad += from.bad;
    if (from.max_diff > to.max_diff) to.max_diff = from.max_diff;
}

void row_print(const char* name, const Tally& t) {
    printf("%s: %d/%d exact, %d within one BF16 ulp, %d outside\n", name, t.exact, t.exact + t.near + t.bad, t.near,
           t.bad);
}

float rand_pos(Rng& rng, float lo) { return lo + (float) (rng.next() & 255) / 256.0f; }

void run_gemv_f32_row(sycl::queue& q, bool neg, Tally& t) {
    const int n = 17, k = 128, T = 3, ldy = n + 5;
    Rng rng(0x473332320001ull);
    std::vector<float> W((size_t) n * k), x((size_t) T * k), y((size_t) T * ldy);
    for (float& v : W) v = rand_pos(rng, 0.5f);
    for (float& v : x) v = rand_pos(rng, 0.5f);
    if (neg) ((uint8_t*) W.data())[15] = 0x7E;   // one wrong input byte: row 0's third weight to +2^125
    dpct::queue_ptr s = &q;
    float* dW = (float*) sycl::malloc_device(W.size() * sizeof(float), q);
    float* dx = (float*) sycl::malloc_device(x.size() * sizeof(float), q);
    float* dy = (float*) sycl::malloc_device(y.size() * sizeof(float), q);
    q.memcpy(dW, W.data(), W.size() * sizeof(float));
    q.memcpy(dx, x.data(), x.size() * sizeof(float));
    q.wait();
    strata::ds4::gemv_f32(dW, dx, n, k, T, dy, ldy, s);
    q.wait();
    q.memcpy(y.data(), dy, y.size() * sizeof(float));
    q.wait();
    sycl::free(dW, q);
    sycl::free(dx, q);
    sycl::free(dy, q);
    for (int ti = 0; ti < T; ++ti)
        for (int r = 0; r < n; ++r) {
            double acc = 0.0;
            for (int c = 0; c < k; ++c) acc += (double) W[(size_t) r * k + c] * (double) x[(size_t) ti * k + c];
            tally_f32(y[(size_t) ti * ldy + r], acc, t);
        }
    row_print("gemv_f32", t);
}

void run_gemv_f16_row(sycl::queue& q, bool neg, Tally& t) {
    const int n = 13, k = 512, T = 2;
    const float alpha = 0.75f;
    Rng rng(0x473136320002ull);
    std::vector<uint16_t> W((size_t) n * k);
    std::vector<float> x((size_t) T * k), y((size_t) T * n);
    for (uint16_t& v : W) v = f16_rn(rand_pos(rng, 0.5f));
    for (float& v : x) v = rand_pos(rng, 0.5f);
    if (neg) ((uint8_t*) W.data())[1] = 0x7A;   // one wrong input byte: row 0's first weight to the top of F16
    dpct::queue_ptr s = &q;
    uint16_t* dW = (uint16_t*) sycl::malloc_device(W.size() * sizeof(uint16_t), q);
    float* dx = (float*) sycl::malloc_device(x.size() * sizeof(float), q);
    float* dy = (float*) sycl::malloc_device(y.size() * sizeof(float), q);
    q.memcpy(dW, W.data(), W.size() * sizeof(uint16_t));
    q.memcpy(dx, x.data(), x.size() * sizeof(float));
    q.wait();
    strata::ds4::gemv_f16(dW, dx, n, k, T, alpha, dy, s);
    q.wait();
    q.memcpy(y.data(), dy, y.size() * sizeof(float));
    q.wait();
    sycl::free(dW, q);
    sycl::free(dx, q);
    sycl::free(dy, q);
    for (int ti = 0; ti < T; ++ti)
        for (int r = 0; r < n; ++r) {
            double acc = 0.0;
            for (int c = 0; c < k; ++c) {
                const double wv = (double) static_cast<float>(sycl::bit_cast<sycl::half>(W[(size_t) r * k + c]));
                acc += wv * (double) x[(size_t) ti * k + c];
            }
            tally_f32(y[(size_t) ti * n + r], (double) alpha * acc, t);
        }
    row_print("gemv_f16", t);
}

void run_gemv_f16_rows_row(sycl::queue& q, bool neg, Tally& t) {
    const int n = 11, k = 512, T = 5, ldx = k + 32, ldy = n + 16;
    const float alpha = 0.5f;
    Rng rng(0x473136520003ull);
    std::vector<uint16_t> W((size_t) n * k);
    std::vector<float> x((size_t) T * ldx, 0.0f), y((size_t) T * ldy);
    for (uint16_t& v : W) v = f16_rn(rand_pos(rng, 0.5f));
    for (int ti = 0; ti < T; ++ti)
        for (int c = 0; c < k; ++c) x[(size_t) ti * ldx + c] = rand_pos(rng, 0.5f);
    if (neg) ((uint8_t*) x.data())[15] = 0x7E;   // one wrong input byte: activation 0's third value to +2^125
    dpct::queue_ptr s = &q;
    uint16_t* dW = (uint16_t*) sycl::malloc_device(W.size() * sizeof(uint16_t), q);
    float* dx = (float*) sycl::malloc_device(x.size() * sizeof(float), q);
    float* dy = (float*) sycl::malloc_device(y.size() * sizeof(float), q);
    q.memcpy(dW, W.data(), W.size() * sizeof(uint16_t));
    q.memcpy(dx, x.data(), x.size() * sizeof(float));
    q.wait();
    strata::ds4::gemv_f16_rows(dW, dx, ldx, n, k, T, alpha, dy, ldy, s);
    q.wait();
    q.memcpy(y.data(), dy, y.size() * sizeof(float));
    q.wait();
    sycl::free(dW, q);
    sycl::free(dx, q);
    sycl::free(dy, q);
    for (int ti = 0; ti < T; ++ti)
        for (int r = 0; r < n; ++r) {
            double acc = 0.0;
            for (int c = 0; c < k; ++c) {
                const double wv = (double) static_cast<float>(sycl::bit_cast<sycl::half>(W[(size_t) r * k + c]));
                acc += wv * (double) x[(size_t) ti * ldx + c];
            }
            tally_f32(y[(size_t) ti * ldy + r], (double) alpha * acc, t);
        }
    row_print("gemv_f16_rows", t);
}

void run_hc_pre_norm_row(sycl::queue& q, bool neg, Tally& t) {
    const int dim = 1024, hc = 8, T = 2;
    const float eps = 1e-5f;
    Rng rng(0x48504e310004ull);
    std::vector<float> h((size_t) T * hc * dim), mix((size_t) T * 8), w(dim), x((size_t) T * dim);
    for (float& v : h) v = rand_pos(rng, 0.5f);
    for (float& v : mix) v = rand_pos(rng, 0.25f);
    for (float& v : w) v = rand_pos(rng, 0.5f);
    if (neg) ((uint8_t*) h.data())[15] = 0x7E;   // one wrong input byte: stream 0's third value to +2^125
    dpct::queue_ptr s = &q;
    float* dh = (float*) sycl::malloc_device(h.size() * sizeof(float), q);
    float* dmix = (float*) sycl::malloc_device(mix.size() * sizeof(float), q);
    float* dw = (float*) sycl::malloc_device(w.size() * sizeof(float), q);
    float* dx = (float*) sycl::malloc_device(x.size() * sizeof(float), q);
    q.memcpy(dh, h.data(), h.size() * sizeof(float));
    q.memcpy(dmix, mix.data(), mix.size() * sizeof(float));
    q.memcpy(dw, w.data(), w.size() * sizeof(float));
    q.wait();
    for (int cfg = 0; cfg < 2; ++cfg) {
        const bool use_w = cfg == 0;
        strata::ds4::hc_pre_norm(dh, dmix, use_w ? dw : nullptr, dim, hc, T, eps, dx, s);
        q.wait();
        q.memcpy(x.data(), dx, x.size() * sizeof(float));
        q.wait();
        Tally ct;
        for (int ti = 0; ti < T; ++ti) {
            std::vector<double> a(dim);
            double ss = 0.0;
            for (int e = 0; e < dim; ++e) {
                double av = 0.0;
                for (int si = 0; si < hc; ++si) av += (double) mix[(size_t) ti * 8 + si] * (double) h[((size_t) ti * hc + si) * dim + e];
                a[e] = av;
                ss += av * av;
            }
            const double inv = 1.0 / std::sqrt(ss / (double) dim + (double) eps);
            for (int e = 0; e < dim; ++e) {
                const double ref = use_w ? a[e] * inv * (double) w[e] : a[e];
                tally_f32(x[(size_t) ti * dim + e], ref, ct);
            }
        }
        row_print(use_w ? "hc_pre_norm/norm+weight" : "hc_pre_norm/no weight", ct);
        tally_add(t, ct);
    }
    sycl::free(dh, q);
    sycl::free(dmix, q);
    sycl::free(dw, q);
    sycl::free(dx, q);
    row_print("hc_pre_norm", t);
}

void run_hc_mix_scale_row(sycl::queue& q, bool neg, Tally& t) {
    const int hcdim = 512, n_mix = 8, T = 3;
    const float eps = 1e-5f;
    Rng rng(0x484d53310005ull);
    std::vector<float> h((size_t) T * hcdim), mixes((size_t) T * 32, 0.0f);
    for (float& v : h) v = rand_pos(rng, 0.5f);
    for (int ti = 0; ti < T; ++ti)
        for (int m = 0; m < n_mix; ++m) mixes[(size_t) ti * 32 + m] = rand_pos(rng, 0.25f);
    if (neg) ((uint8_t*) h.data())[15] = 0x7E;   // one wrong input byte: token 0's third value to +2^125
    dpct::queue_ptr s = &q;
    float* dh = (float*) sycl::malloc_device(h.size() * sizeof(float), q);
    float* dmix = (float*) sycl::malloc_device(mixes.size() * sizeof(float), q);
    q.memcpy(dh, h.data(), h.size() * sizeof(float));
    q.memcpy(dmix, mixes.data(), mixes.size() * sizeof(float));
    q.wait();
    strata::ds4::hc_mix_scale(dh, hcdim, n_mix, T, eps, dmix, s);
    q.wait();
    std::vector<float> got(mixes.size());
    q.memcpy(got.data(), dmix, got.size() * sizeof(float));
    q.wait();
    sycl::free(dh, q);
    sycl::free(dmix, q);
    for (int ti = 0; ti < T; ++ti) {
        double ss = 0.0;
        for (int e = 0; e < hcdim; ++e) ss += (double) h[(size_t) ti * hcdim + e] * (double) h[(size_t) ti * hcdim + e];
        const double inv = 1.0 / std::sqrt(ss / (double) hcdim + (double) eps);
        for (int m = 0; m < n_mix; ++m)
            tally_f32(got[(size_t) ti * 32 + m], (double) mixes[(size_t) ti * 32 + m] * inv, t);
    }
    row_print("hc_mix_scale", t);
}

void run_pred_proj_k_row(sycl::queue& q, bool neg, Tally& t) {
    const int D = 512, T = 2;
    Rng rng(0x50524f4a0006ull);
    std::vector<float> xv((size_t) T * D), out((size_t) T * 256);
    for (float& v : xv) v = rand_pos(rng, 0.5f);
    if (neg) ((uint8_t*) xv.data())[15] = 0x7E;   // one wrong input byte: token 0's third value to +2^125
    dpct::queue_ptr s = &q;
    float* dx = (float*) sycl::malloc_device(xv.size() * sizeof(float), q);
    float* dout = (float*) sycl::malloc_device(out.size() * sizeof(float), q);
    q.memcpy(dx, xv.data(), xv.size() * sizeof(float));
    q.wait();
    strata::ds4::pred_proj_k(dx, D, T, dout, s);
    q.wait();
    q.memcpy(out.data(), dout, out.size() * sizeof(float));
    q.wait();
    sycl::free(dx, q);
    sycl::free(dout, q);
    const double rs = 1.0 / std::sqrt((double) D);
    for (int ti = 0; ti < T; ++ti)
        for (int j = 0; j < 256; ++j) {
            float a = 0.0f;   // the kernel's own sum order, so the projection's cancellation is measured away
            for (int d = 0; d < D; ++d) {
                unsigned hsh = (unsigned) d * 2654435761u ^ ((unsigned) j + 1u) * 40503u;
                hsh ^= hsh >> 15;
                hsh *= 2246822519u;
                hsh ^= hsh >> 13;
                hsh *= 3266489917u;
                hsh ^= hsh >> 16;
                a += (hsh & 1u) ? xv[(size_t) ti * D + d] : -xv[(size_t) ti * D + d];
            }
            tally_f32(out[(size_t) ti * 256 + j], (double) a * rs, t);
        }
    row_print("pred_proj_k", t);
}

void run_softmax_row(sycl::queue& q, bool neg, Tally& t) {
    const int n = 300, T = 2;
    Rng rng(0x534f46540007ull);
    std::vector<float> sv((size_t) T * n);
    std::vector<uint16_t> got((size_t) T * n);
    for (float& v : sv) v = -4.0f + (float) (rng.next() & 255) / 32.0f;
    if (neg) ((uint8_t*) sv.data())[15] = 0x7E;   // one wrong input byte: row 0's third logit to +2^125
    dpct::queue_ptr s = &q;
    float* ds = (float*) sycl::malloc_device(sv.size() * sizeof(float), q);
    uint16_t* dp = (uint16_t*) sycl::malloc_device(got.size() * sizeof(uint16_t), q);
    q.memcpy(ds, sv.data(), sv.size() * sizeof(float));
    q.wait();
    strata::ds4::softmax(ds, n, T, dp, s);
    q.wait();
    q.memcpy(got.data(), dp, got.size() * sizeof(uint16_t));
    q.wait();
    sycl::free(ds, q);
    sycl::free(dp, q);
    for (int ti = 0; ti < T; ++ti) {
        double m = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) m = std::fmax(m, (double) sv[(size_t) ti * n + i]);
        double den = 0.0;
        std::vector<double> e(n);
        for (int i = 0; i < n; ++i) {
            e[i] = std::exp((double) sv[(size_t) ti * n + i] - m);
            den += e[i];
        }
        for (int i = 0; i < n; ++i) tally_bf16(got[(size_t) ti * n + i], e[i] / den, t);
    }
    row_print("softmax", t);
}

void run_sparse_attn_row(sycl::queue& q, bool neg, Tally& t) {
    const int heads = 4, T = 3, d = 512, nwin_max = 40, sel_stride = 24, nckv = 32;
    const double scale = 1.0 / std::sqrt(512.0);
    Rng rng(0x535041540008ull);
    std::vector<float> qv((size_t) T * heads * d);
    for (float& v : qv) v = rand_pos(rng, 0.5f);
    std::vector<uint16_t> win((size_t) nwin_max * d);
    for (uint16_t& v : win) v = strata::ds4::bf16_rn(rand_pos(rng, 0.5f));
    const size_t crow = strata::ds4::kv_row_bytes(strata::ds4::kKvFp4, d);
    std::vector<uint8_t> ckv((size_t) nckv * crow);
    for (int r = 0; r < nckv; ++r) {
        uint8_t* row = ckv.data() + (size_t) r * crow;
        for (int i = 0; i < d / 2; ++i) {
            const uint8_t lo = (uint8_t) (1 + rng.next() % 7), hi = (uint8_t) (1 + rng.next() % 7);
            row[i] = (uint8_t) (lo | (hi << 4));
        }
        for (int i = 0; i < d / 16; ++i) row[d / 2 + i] = (uint8_t) (0x20 + (rng.next() & 7));
    }
    std::vector<double> ckv_val((size_t) nckv * d);
    std::vector<uint16_t> ckv_bf((size_t) nckv * d);
    static const double lut[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    for (int r = 0; r < nckv; ++r) {
        const uint8_t* row = ckv.data() + (size_t) r * crow;
        for (int i = 0; i < d; ++i) {
            const uint8_t code = (uint8_t) ((row[i / 2] >> (4 * (i & 1))) & 15u);
            const double v = lut[code & 7u] * (double) strata::ds4::e4m3_value(row[d / 2 + i / 16]);
            ckv_val[(size_t) r * d + i] = v;
            ckv_bf[(size_t) r * d + i] = strata::ds4::bf16_rn((float) v);   // exact: E2M1 x E4M3 fits BF16
        }
    }
    std::vector<int> sel((size_t) T * sel_stride);
    std::vector<float> sink(heads);
    strata::ds4::AttnRows ar{};
    for (int ti = 0; ti < T; ++ti) {
        ar.win0[ti] = 0;
        ar.nwin[ti] = 20 + 5 * ti;
        ar.nsel[ti] = 8 + 4 * ti;
        for (int i = 0; i < sel_stride; ++i)
            sel[(size_t) ti * sel_stride + i] = i < ar.nsel[ti] ? (ti * 7 + i * 3) % nckv : 0;
    }
    for (int hh = 0; hh < heads; ++hh) sink[hh] = rand_pos(rng, 0.25f);
    if (neg) ((uint8_t*) qv.data())[15] = 0x7E;   // one wrong input byte: query 0's first head value to +2^125
    std::vector<double> ref((size_t) T * heads * d);
    for (int ti = 0; ti < T; ++ti) {
        const int nt = ar.nwin[ti] + ar.nsel[ti];
        std::vector<double> rv((size_t) nt * d);
        for (int j = 0; j < ar.nwin[ti]; ++j) {
            const uint16_t* wr = win.data() + (size_t) (ar.win0[ti] + j) * d;
            for (int c = 0; c < d; ++c)   // the window rows are BF16 bits: shift into the F32 exponent
                rv[(size_t) j * d + c] = (double) sycl::bit_cast<float>((uint32_t) wr[c] << 16);
        }
        for (int i = 0; i < ar.nsel[ti]; ++i) {
            const int row = sel[(size_t) ti * sel_stride + i];
            for (int c = 0; c < d; ++c) rv[(size_t) (ar.nwin[ti] + i) * d + c] = ckv_val[(size_t) row * d + c];
        }
        for (int hh = 0; hh < heads; ++hh) {
            const float* qr = qv.data() + ((size_t) ti * heads + hh) * d;
            std::vector<double> wexp(nt);
            double den = std::exp((double) sink[hh]);
            for (int j = 0; j < nt; ++j) {
                double dot = 0.0;
                for (int c = 0; c < d; ++c) dot += (double) qr[c] * rv[(size_t) j * d + c];
                wexp[j] = std::exp(dot * scale);
                den += wexp[j];
            }
            for (int c = 0; c < d; ++c) {
                double acc = 0.0;
                for (int j = 0; j < nt; ++j) acc += wexp[j] * rv[(size_t) j * d + c];
                ref[((size_t) ti * heads + hh) * d + c] = acc / den;
            }
        }
    }
    dpct::queue_ptr s = &q;
    float* dq = (float*) sycl::malloc_device(qv.size() * sizeof(float), q);
    uint16_t* dwin = (uint16_t*) sycl::malloc_device(win.size() * sizeof(uint16_t), q);
    uint8_t* dckv = (uint8_t*) sycl::malloc_device(ckv.size(), q);
    uint16_t* dckvbf = (uint16_t*) sycl::malloc_device(ckv_bf.size() * sizeof(uint16_t), q);
    int* dsel = (int*) sycl::malloc_device(sel.size() * sizeof(int), q);
    float* dsink = (float*) sycl::malloc_device(sink.size() * sizeof(float), q);
    float* dout = (float*) sycl::malloc_device(ref.size() * sizeof(float), q);
    uint16_t* dscr = (uint16_t*) sycl::malloc_device((size_t) T * sel_stride * d * sizeof(uint16_t), q);
    q.memcpy(dq, qv.data(), qv.size() * sizeof(float));
    q.memcpy(dwin, win.data(), win.size() * sizeof(uint16_t));
    q.memcpy(dckv, ckv.data(), ckv.size());
    q.memcpy(dckvbf, ckv_bf.data(), ckv_bf.size() * sizeof(uint16_t));
    q.memcpy(dsel, sel.data(), sel.size() * sizeof(int));
    q.memcpy(dsink, sink.data(), sink.size() * sizeof(float));
    q.wait();
    strata::ds4::AttnScratch scr;
    scr.rows = dscr;
    scr.max_t = T;
    scr.cap = sel_stride;
    const int ncfg = 4;
    const char* cfg_name[ncfg] = {"sparse_attn/bf16 rows", "sparse_attn/packed G=2", "sparse_attn/packed G=1",
                                  "sparse_attn/scratch rows"};
    for (int cfg = 0; cfg < ncfg; ++cfg) {
        if (cfg == 0)
            strata::ds4::sparse_attn(dq, dwin, dckvbf, dsel, sel_stride, ar, dsink, heads, d, T, (float) scale, dout,
                                     s, strata::ds4::kKvBf16);
        else if (cfg == 1)
            strata::ds4::sparse_attn(dq, dwin, dckv, dsel, sel_stride, ar, dsink, heads, d, T, (float) scale, dout, s,
                                     strata::ds4::kKvFp4, 2);
        else if (cfg == 2)
            strata::ds4::sparse_attn(dq, dwin, dckv, dsel, sel_stride, ar, dsink, heads, d, T, (float) scale, dout, s,
                                     strata::ds4::kKvFp4, 1);
        else
            strata::ds4::sparse_attn(dq, dwin, dckv, dsel, sel_stride, ar, dsink, heads, d, T, (float) scale, dout, s,
                                     strata::ds4::kKvFp4, -1, &scr);
        q.wait();
        std::vector<float> got(ref.size());
        q.memcpy(got.data(), dout, got.size() * sizeof(float));
        q.wait();
        Tally ct;
        for (size_t i = 0; i < got.size(); ++i) tally_f32(got[i], ref[i], ct);
        row_print(cfg_name[cfg], ct);
        tally_add(t, ct);
    }
    sycl::free(dq, q);
    sycl::free(dwin, q);
    sycl::free(dckv, q);
    sycl::free(dckvbf, q);
    sycl::free(dsel, q);
    sycl::free(dsink, q);
    sycl::free(dout, q);
    sycl::free(dscr, q);
    row_print("sparse_attn", t);
}

// ---- the K-quant GEMVs (iq::mv -> mv_kernel -> sub_dot: the dense matrices' Q2_K/Q4_K/Q5_K take this path alone) ----
//
// One row a quant type.  The reference is the block dot written from the same layouts the kernels decode
// (third_party/ggml/ggml-common.h's block_q2_K / block_q4_K / block_q5_K through decode_b in
// src/dsv41/ds4_iq.cu:152 and sycl/src/dsv41/ds4_iq.dp.cpp:179) and sub_dot's documented form
// (src/dsv41/ds4_iq.cu:353-373, sycl/src/dsv41/ds4_iq.dp.cpp:391-425): a weight is d s q - dmin m of its 16-value
// half, so a 32-value sub-block's dot is the exact integer dot of the codes and the int8 activations, its scales in
// F32, minus the min term's both halves - the `v - (m0 sl + m1 sh)` form the 28bd2aaa fix restored, which no other
// row exercises.  Computed in double; the device's F32 sums round to it within one BF16 ulp.
double ref_k_sub_dot(int type, const uint8_t* b, int ib, const int8_t* x) {
    int q[32], s0, s1;
    double d, m0, m1;
    if (type == strata::ds4::gguf::iq::kQ2_K) {   // scales[16] then qs[64], the dm pair after them (block_q2_K)
        const int n = ib >> 2, j = ib & 3;
        const uint8_t* qs = b + 16 + 32 * n;
        for (int t = 0; t < 32; ++t) q[t] = (qs[t] >> (2 * j)) & 3;
        const int c0 = b[8 * n + 2 * j], c1 = b[8 * n + 2 * j + 1];   // the two halves' scale/min pairs
        const double dmin = fp16_to_f32(b + 82);
        s0 = c0 & 15;
        s1 = c1 & 15;
        d = fp16_to_f32(b + 80);
        m0 = dmin * (double) (c0 >> 4);
        m1 = dmin * (double) (c1 >> 4);
    } else {   // kQ4_K / kQ5_K: dm then scales[12] (get_scale_min_k4's 6-bit pairs); q5_K's qh[32] before its qs[128]
        uint8_t sc, m;
        get_scale_min_k4(ib, b + 4, &sc, &m);
        const bool q5 = type == strata::ds4::gguf::iq::kQ5_K;
        const uint8_t* qs = b + (q5 ? 48 : 16) + 32 * (ib >> 1);
        const uint8_t* qh = q5 ? b + 16 : nullptr;
        const int shl = 4 * (ib & 1);
        for (int t = 0; t < 32; ++t) {
            q[t] = (qs[t] >> shl) & 15;
            if (q5) q[t] |= ((qh[t] >> ib) & 1) << 4;
        }
        s0 = s1 = sc;
        d = fp16_to_f32(b);
        m0 = m1 = fp16_to_f32(b + 2) * (double) m;
    }
    double lo = 0.0, hi = 0.0, sl = 0.0, sh = 0.0;
    for (int t = 0; t < 16; ++t) { lo += (double) q[t] * x[t]; sl += x[t]; }
    for (int t = 16; t < 32; ++t) { hi += (double) q[t] * x[t]; sh += x[t]; }
    return d * (s0 * lo + s1 * hi) - (m0 * sl + m1 * sh);
}

// One row: n rows of k values of one type against ncol int8 activations (MvJob: y[c][r] = alpha (row r . x_c) +
// bias[r], the q8_quant input form: int8 values, one F32 scale a 32).  The fill stays fill_block's seeded bytes and
// the activations are strictly positive, so no output lands near zero and a pass never depends on luck.
void run_iq_mv_row(sycl::queue& q, int type, Tally& t) {
    const int n = 17, k = 512, ncol = 3, nsb = k / 32;
    const float alpha = 0.75f;
    Rng rng(0x49514D560009ull + (unsigned) type);
    const size_t bb = (size_t) strata::ds4::gguf::iq::block_bytes(type);
    const size_t ld = strata::ds4::gguf::iq::row_bytes(type, k);
    std::vector<uint8_t> w((size_t) n * ld);
    for (int r = 0; r < n; ++r) fill_block(type, w.data() + (size_t) r * ld, rng);
    std::vector<int8_t> xq((size_t) ncol * k);
    for (int8_t& v : xq) v = (int8_t) (1 + rng.next() % 127);
    std::vector<float> xd((size_t) ncol * nsb);
    for (float& v : xd) v = rand_pos(rng, 0.5f);
    std::vector<float> bias(n), y((size_t) ncol * n);
    for (float& v : bias) v = rand_pos(rng, 0.25f);
    dpct::queue_ptr s = &q;
    uint8_t* dw = (uint8_t*) sycl::malloc_device(w.size(), q);
    int8_t* dxq = (int8_t*) sycl::malloc_device(xq.size(), q);
    float* dxd = (float*) sycl::malloc_device(xd.size() * sizeof(float), q);
    float* dbias = (float*) sycl::malloc_device(bias.size() * sizeof(float), q);
    float* dy = (float*) sycl::malloc_device(y.size() * sizeof(float), q);
    q.memcpy(dw, w.data(), w.size());
    q.memcpy(dxq, xq.data(), xq.size());
    q.memcpy(dxd, xd.data(), xd.size() * sizeof(float));
    q.memcpy(dbias, bias.data(), bias.size() * sizeof(float));
    q.wait();
    strata::ds4::gguf::iq::MvJob job{};
    job.w = dw;
    job.ld = ld;
    job.type = type;
    job.k = k;
    job.n = n;
    job.ncol = ncol;
    job.xq = dxq;
    job.xd = dxd;
    job.y = dy;
    job.ycs = n;
    job.bias = dbias;
    job.alpha = alpha;
    const bool launched = strata::ds4::gguf::iq::mv(&job, 1, s);
    q.wait();
    q.memcpy(y.data(), dy, y.size() * sizeof(float));
    q.wait();
    sycl::free(dw, q);
    sycl::free(dxq, q);
    sycl::free(dxd, q);
    sycl::free(dbias, q);
    sycl::free(dy, q);
    char nm[32];
    std::snprintf(nm, sizeof nm, "%s gemv", strata::ds4::gguf::iq::name(type));
    if (!launched) {
        printf("%s: FAIL (no kernel launched)\n", nm);
        ++t.bad;
        return;
    }
    for (int c = 0; c < ncol; ++c)
        for (int r = 0; r < n; ++r) {
            double acc = 0.0;
            for (int kb = 0; kb < nsb; ++kb) {
                const uint8_t* b = w.data() + (size_t) r * ld + (size_t) (kb >> 3) * bb;
                acc += (double) xd[(size_t) c * nsb + kb] *
                       ref_k_sub_dot(type, b, kb & 7, xq.data() + (size_t) c * k + (size_t) kb * 32);
            }
            double yref = (double) alpha * acc + (double) bias[r];
            tally_f32(y[(size_t) c * n + r], yref, t);
        }
    row_print(nm, t);
}

// ---- the canonical block-decode oracle (plan todo 2: the tile A-stage's decode contract) ----
//
// One row per canonical block format (Q8_0, IQ4_NL, MXFP4).  A matrix of kDecRows rows of kDecBpr blocks of 32
// values, the canonical row-major block index bid = n * bpr + kb (ds4_iq.dp.cpp:67-70, 442-445: block kb of row n
// starts at bid * block_bytes).  The device kernel decodes each block's 32 values to F16 with round-to-nearest-even
// exactly as the tile A-stage stages its F16 fragments, from the shipped decode sites' layouts (q8_block
// ds4_gguf.dp.cpp:283-295, IQ4_NL ds4_iq.dp.cpp:314-323, MXFP4 ds4_gguf.dp.cpp:227-235) with ggml-common.h's
// shipped codebook tables.  The reference is the shipped CPU scalar dequantizer,
// include/strata/artifact/dequant.hpp's dequantize_q8_0 / dequantize_iq4_nl; dequant.hpp ships no MXFP4 function
// (the documented gap), so that reference is assembled from ggml-common.h's block_mxfp4 layout and kvalues_mxfp4
// table, its E8M0 scale 2^(e - 128).
// The comparison is bit-exact F16 only - no BF16-ulp rule, no absolute floor: decoded operands are exact contract
// values, not accumulated sums (IS-2: decoded operands are bit-exact).  A row that cannot fail is not evidence:
// --neg flips one byte of the device input after the reference is computed (per format: one payload or one scale
// byte) and the row must report NO MATCH; the non-vacuity guard rejects a row whose reference values are not
// diverse, so an all-zero input cannot pass.

constexpr int kDecRows = 4, kDecBpr = 4;   // 16 canonical blocks of 32 values: 512 F16 bits a row
constexpr int kDecFmtQ8_0 = 0, kDecFmtIQ4_NL = 1, kDecFmtMXFP4 = 2;
bool g_weird_scale = false;   // --weird-scale: NaN / denormal scale blocks (adversarial class malformed_input)
bool g_bad_scale_gu = false;  // --bad-scale-gu: one wrong gate scale byte (the set-tile rows read these)
bool g_bad_scale_dn = false;  // --bad-scale: one wrong down scale byte

int dec_block_bytes(int fmt) { return fmt == kDecFmtQ8_0 ? 34 : fmt == kDecFmtIQ4_NL ? 18 : 17; }

// the CPU reference, one canonical block's 32 F32 values, from the shipped scalar dequantizers
void dec_ref_block(int fmt, const uint8_t* b, float* out) {
    if (fmt == kDecFmtQ8_0) {
        strata::dequantize_q8_0(b, out);
    } else if (fmt == kDecFmtIQ4_NL) {
        strata::dequantize_iq4_nl(b, out);
    } else {   // MXFP4: block_mxfp4 {uint8 e; uint8 qs[16]} (ggml-common.h): value j in byte j's low nibble
        // (j < 16) or byte j - 16's high one, worth kvalues_mxfp4 * 2^(e - 128)
        const float d = std::ldexp(1.0f, (int) b[0] - 128);
        for (int j = 0; j < 32; ++j) {
            const uint8_t c = j < 16 ? b[1 + j] & 15u : b[1 + j - 16] >> 4;
            out[j] = (float) kvalues_mxfp4[c] * d;
        }
    }
}

// the device side: one work-item a canonical block (bid = n * bpr + kb), 32 F16 (rte) values a block - the decode
// the tile A-stage reuses, per the shipped decode sites
void dec_rows_kernel(sycl::queue& q, const uint8_t* raw, int fmt, int nb, uint16_t* dst) {
    q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> it) {
        const int bid = (int) it[0];
        const uint8_t* b = raw + (size_t) bid * dec_block_bytes(fmt);
        float v[32];
        if (fmt == kDecFmtQ8_0) {   // {fp16 d; int8 qs[32]}: v = d q (q8_block, ds4_gguf.dp.cpp:283-295)
            const float d = fp16_to_f32(b);
            for (int j = 0; j < 32; ++j) v[j] = d * (float) (int8_t) b[2 + j];
        } else if (fmt == kDecFmtIQ4_NL) {   // {fp16 d; u8 qs[16]}: value j in byte j's low nibble (j < 16) or
            // byte j - 16's high one (ds4_iq.dp.cpp:314-323)
            const float d = fp16_to_f32(b);
            for (int j = 0; j < 32; ++j) {
                const uint8_t s = b[2 + (j & 15)];
                v[j] = d * (float) kvalues_iq4nl[j < 16 ? s & 15u : s >> 4];
            }
        } else {   // MXFP4: {u8 e; u8 qs[16]}, E8M0 2^(e - 128) (mx_block, ds4_gguf.dp.cpp:227-235)
            const uint32_t e = b[0];
            const float d = sycl::bit_cast<float>(e < 2 ? 0x00200000u << e : (e - 1) << 23);
            for (int j = 0; j < 32; ++j) {
                const uint8_t s = b[1 + (j & 15)];
                v[j] = (float) kvalues_mxfp4[j < 16 ? s & 15u : s >> 4] * d;
            }
        }
        for (int j = 0; j < 32; ++j)
            dst[(size_t) bid * 32 + j] = sycl::bit_cast<unsigned short, sycl::half>(
                sycl::vec<float, 1>(v[j]).convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    });
}

// seeded fill: random payload bytes, small finite scales (a random fp16 could land on NaN or a huge scale, where
// one F16 step is not a normal rounding) unless --weird-scale asks for the malformed blocks
void dec_fill(int fmt, uint8_t* raw, Rng& rng, bool weird) {
    const int B = dec_block_bytes(fmt);
    for (int bid = 0; bid < kDecRows * kDecBpr; ++bid) {
        uint8_t* b = raw + (size_t) bid * B;
        for (int i = 0; i < B; ++i) b[i] = rng.next();
        if (fmt == kDecFmtMXFP4)
            b[0] = (uint8_t) (128 + (rng.next() & 3));   // E8M0: 2^0..2^3, exact in F16
        else {
            const float v = 0.01f + 0.25f * (float) (rng.next() & 31);
            const uint16_t u = sycl::bit_cast<uint16_t>(sycl::half(v));
            b[0] = (uint8_t) (u & 0xFF);
            b[1] = (uint8_t) (u >> 8);
        }
    }
    if (!weird) return;
    // malformed scales (adversarial class malformed_input): block 1's scale a quiet NaN, block 3's a denormal;
    // MXFP4's E8M0 byte has no NaN, so 0xFF (2^127: the values overflow F16 to infinity) and 0x00 (2^-128, an F32
    // subnormal product) take those roles.  The row must stay deterministic under them.
    uint8_t* b1 = raw + (size_t) 1 * B, *b3 = raw + (size_t) 3 * B;
    if (fmt == kDecFmtMXFP4) {
        b1[0] = 0xFF;
        b3[0] = 0x00;
    } else {
        b1[0] = 0xE0;
        b1[1] = 0x7E;   // fp16 +qNaN (0x7E00)
        b3[0] = 0x01;
        b3[1] = 0x00;   // fp16 +2^-24 (denormal)
    }
}

// One row: nb canonical blocks of 32 values decoded on the device to F16 (rte) against the shipped CPU scalar
// dequantizer's values rounded to F16 (rte) - bit-exact only.  neg: one corrupted input byte after the reference.
void run_decode_row(sycl::queue& q, int fmt, const char* name, bool neg, Tally& t) {
    const int B = dec_block_bytes(fmt), nb = kDecRows * kDecBpr;
    Rng rng(0x444551310010ull + (unsigned) fmt);   // fixed seeds: a pass never depends on luck
    std::vector<uint8_t> raw((size_t) nb * B);
    dec_fill(fmt, raw.data(), rng, g_weird_scale);
    std::vector<uint16_t> want((size_t) nb * 32);
    for (int bid = 0; bid < nb; ++bid) {
        float f[32];
        dec_ref_block(fmt, raw.data() + (size_t) bid * B, f);
        for (int j = 0; j < 32; ++j) want[(size_t) bid * 32 + j] = f16_rn(f[j]);
    }
    if (neg) {
        // the negative control, one byte per format: Q8_0 a payload (code) byte, IQ4_NL its scale byte, MXFP4 its
        // E8M0 scale byte - each changes at least one decoded value, so the row must report NO MATCH
        const size_t at = fmt == kDecFmtQ8_0 ? 2 + 5 : 0;
        const uint8_t by = fmt == kDecFmtQ8_0 ? 0x11 : fmt == kDecFmtIQ4_NL ? 0x40 : 0x04;
        raw[at] ^= by;
        printf("%s negative control: one wrong %s byte, raw[%zu] ^= 0x%02X; the reference is from the clean blocks\n",
               name, fmt == kDecFmtQ8_0 ? "payload" : "scale", at, by);
    }
    uint8_t* dw = (uint8_t*) sycl::malloc_device(raw.size(), q);
    uint16_t* dd = (uint16_t*) sycl::malloc_device(want.size() * sizeof(uint16_t), q);
    q.memcpy(dw, raw.data(), raw.size());
    q.wait();
    dec_rows_kernel(q, dw, fmt, nb, dd);
    q.wait();
    std::vector<uint16_t> got(want.size());
    q.memcpy(got.data(), dd, got.size() * sizeof(uint16_t));
    q.wait();
    sycl::free(dw, q);
    sycl::free(dd, q);
    int same = 0, diff = 0;
    for (size_t i = 0; i < got.size(); ++i) (got[i] == want[i] ? ++same : ++diff);
    std::vector<uint16_t> u = want;
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    if (u.size() < 8) {   // non-vacuity: an all-zero input decodes to one value on both sides and cannot pass
        printf("%s: VACUOUS (%zu distinct reference values): the comparison could not distinguish a wrong decode\n",
               name, u.size());
        t.bad += diff + 1;
        return;
    }
    if (diff == 0)
        printf("%s: MATCH %d/%d F16 bits exact (bit-exact decode of %d canonical blocks x 32 values, "
               "bid = n*bpr + kb)\n",
               name, same, (int) got.size(), nb);
    else
        printf("%s: NO MATCH: %d/%d F16 bits differ from the CPU scalar dequantizer (bit-exact required)\n",
               name, diff, (int) got.size());
    t.exact += same;
    t.bad += diff;
}

// ---------------------------------------------------------------- the i8 A-stage set tiles (plan todo 5)
//
// The tiles this section drives are iq::set_gate_up / iq::set_down with IQ4_NL - the one type the dispatch arm (plan
// todo 4) routes to set_tc_stageA_kernel: the arm packs the set's activation rows into the ext_intel_packed F16 plane
// (stageA_pack_kernel) and the tile reads them from global memory.  The reference below is written from that
// contract, not from the kernel: the weights are the shipped IQ4_NL codebook values (ref_iq4_nl_blocks) rounded to
// F16 (rte, the A fragments' contract - deq8 decoded the same values bit-exactly in the plan's todo-3 gate suite),
// the activations are the F16 bits the pack kernel stores (gate/up: the row idx[r] of xf rounded rte; down: the xh
// row's bits), and the sums have the tile's own shape - the K range in the four sub-groups' whole-block ranges
// (kb0 = (s*nblk)/4, kb1 = ((s+1)*nblk)/4), each partial summed in double over the F16 products in block order,
// rounded to F32 and combined ((p0 + p1) + p2) + p3.  The gate/up output is the F16 (rte) clamped swiglu, accepted
// within one BF16 ulp of the reference; the down output is the raw F32 sum, accepted within the plan's
// 32*2^-24*sum_abs + 4*ulp_f32(ref) bound (or one BF16 ulp, which the reviewed gate suite's corrected +-0 rule also
// allows where a sum cancels to near zero).
//
// The data are scale-sensitive as the legacy XS set row's: d = 0.002 a 32-value block and activations in +-[0, 0.05),
// so the sums land where the clamped swiglu is not saturated and a wrong weight or plane slot moves an output.  The
// shapes are the gate suite's M list (2, 15, 16, 17, 31, 32, 33, 160 at the model's 2560 x 640) plus the K and N
// tails: 544 x 352 (17 and 11 blocks - the K split uneven, N not a multiple of the legacy BN = 64) and 128 x 128
// (one block a sub-group: the split edge).  n_exp = 2 splits M in half, so M = 2 is one row an expert.

struct StageAShape {
    int M, n_embd, n_ff, n_exp;
};

// One slab's weight rows as the A fragments hold them: the shipped IQ4_NL decode rounded to F16 (rte).
void stageA_w_plane(const uint8_t* slab, int rows, int K, std::vector<uint16_t>& w) {
    const size_t bb = (size_t) strata::ds4::gguf::iq::block_bytes(strata::ds4::gguf::iq::kIQ4_NL);
    w.assign((size_t) rows * (size_t) K, 0);
    std::vector<float> v((size_t) K);
    for (int n = 0; n < rows; ++n) {
        ref_iq4_nl_blocks(slab + (size_t) n * (size_t) (K / 32) * bb, K / 32, v.data());
        for (int k = 0; k < K; ++k) w[(size_t) n * (size_t) K + (size_t) k] = f16_rn(v[(size_t) k]);
    }
}

// The tile's reduction of one output element: the four whole-block ranges of the K split, the partials the double
// sums of the F16 products in block order, rounded to F32 and combined in the fixed ((p0 + p1) + p2) + p3 order;
// *sum_abs: the double sum of |w*x| for the tile bound.
float stageA_reduce(const uint16_t* w, const float* x, int K, double* sum_abs) {
    const int nblk = K / 32;
    float p[4];
    double sa = 0.0;
    for (int s = 0; s < 4; ++s) {
        const int kb0 = (s * nblk) / 4, kb1 = ((s + 1) * nblk) / 4;
        double acc = 0.0;
        for (int k = kb0 * 32; k < kb1 * 32; ++k) {
            const double pr = (double) static_cast<float>(sycl::bit_cast<sycl::half>(w[k])) * (double) x[k];
            acc += pr;
            if (sum_abs != nullptr) sa += std::fabs(pr);
        }
        p[s] = (float) acc;
    }
    if (sum_abs != nullptr) *sum_abs = sa;
    return ((p[0] + p[1]) + p[2]) + p[3];
}

// One shape: the IQ4_NL blob, the reference from the clean blob, then the arm's set calls (gate/up, down) and the
// comparison.  neg: one down scale byte flipped after the reference - the row must then leave the bounds.
void run_stageA_row(sycl::queue& q, const StageAShape& sh, bool neg, Tally& t) {
    const int M = sh.M, n_embd = sh.n_embd, n_ff = sh.n_ff, n_exp = sh.n_exp;
    const int tt = strata::ds4::gguf::iq::kIQ4_NL;
    const size_t bb = (size_t) strata::ds4::gguf::iq::block_bytes(tt);
    const size_t rb_gu = (size_t) (n_embd / 32) * bb, rb_dn = (size_t) (n_ff / 32) * bb;
    const size_t tr0 = 0, tr1 = (size_t) n_ff * rb_gu, tr2 = tr1 + (size_t) n_ff * rb_gu;
    const size_t stride = tr2 + (size_t) n_embd * rb_dn;
    Rng rng(0x5354414741ull + (unsigned long long) M * 7919ull + (unsigned long long) n_embd * 131ull);
    std::vector<uint8_t> blob((size_t) n_exp * stride);
    const uint16_t dbits = sycl::bit_cast<uint16_t>(sycl::half(0.002f));
    for (int e = 0; e < n_exp; ++e) {
        uint8_t* eb = blob.data() + (size_t) e * stride;
        for (int part = 0; part < 3; ++part) {
            uint8_t* base = eb + (part == 0 ? tr0 : part == 1 ? tr1 : tr2);
            const int rows = part < 2 ? n_ff : n_embd;
            const int K = part < 2 ? n_embd : n_ff;
            const size_t rb = part < 2 ? rb_gu : rb_dn;
            for (int r = 0; r < rows; ++r) {
                uint8_t* row = base + (size_t) r * rb;
                for (size_t i = 0; i < rb; ++i) row[i] = rng.next();
                for (int kb = 0; kb < K / 32; ++kb) {   // a small scale a 32-value block keeps the sums sensitive
                    row[(size_t) kb * bb + 0] = (uint8_t) (dbits & 0xFF);
                    row[(size_t) kb * bb + 1] = (uint8_t) (dbits >> 8);
                }
            }
        }
    }
    std::vector<float> xf((size_t) M * (size_t) n_embd);
    for (size_t i = 0; i < xf.size(); ++i)   // +-[0, 1): d = 0.002 a block keeps the sums scale-sensitive, and a row
        xf[i] = ((int) (rng.next() | (rng.next() << 8)) - 32768) / 32768.0f;   // wide enough that the F16 outputs stay distinct
    std::vector<int> idx((size_t) M);   // a rotation: the pack reads xf[idx[local row]], the reference the same row
    for (int r = 0; r < M; ++r) idx[(size_t) r] = (r + M / 2) % M;
    std::vector<uint16_t> xh((size_t) M * (size_t) n_ff);
    for (size_t i = 0; i < xh.size(); ++i)   // F16 rows in +-[0.5, 1): finite, away from zero (the XS set row's form)
        xh[i] = (uint16_t) ((rng.next() | (rng.next() << 8)) & 0xBBFFu);
    std::vector<int> bounds((size_t) n_exp + 1);
    for (int e = 0; e <= n_exp; ++e) bounds[(size_t) e] = e * M / n_exp;
    const float limit = 10.0f;
    std::vector<uint16_t> h_ref((size_t) M * (size_t) n_ff);
    std::vector<float> g_of(h_ref.size()), u_of(h_ref.size());   // the clamped gate / up sums, for the role check below
    std::vector<float> y_ref((size_t) M * (size_t) n_embd);
    std::vector<double> h_elem(h_ref.size()), y_elem(y_ref.size()), y_tile(y_ref.size());
    std::vector<float> xrow((size_t) (n_embd > n_ff ? n_embd : n_ff));
    int clamped = 0;
    for (int e = 0; e < n_exp; ++e) {
        const uint8_t* eb = blob.data() + (size_t) e * stride;
        std::vector<uint16_t> wg, wu, wd;
        stageA_w_plane(eb + tr0, n_ff, n_embd, wg);
        stageA_w_plane(eb + tr1, n_ff, n_embd, wu);
        stageA_w_plane(eb + tr2, n_embd, n_ff, wd);
        for (int r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r) {
            for (int k = 0; k < n_embd; ++k)
                xrow[(size_t) k] =
                    static_cast<float>(sycl::bit_cast<sycl::half>(f16_rn(xf[(size_t) idx[(size_t) r] * (size_t) n_embd +
                                                                       (size_t) k])));
            for (int n = 0; n < n_ff; ++n) {
                const float g = stageA_reduce(&wg[(size_t) n * (size_t) n_embd], xrow.data(), n_embd, nullptr);
                const float u = stageA_reduce(&wu[(size_t) n * (size_t) n_embd], xrow.data(), n_embd, nullptr);
                float gv = g, uv = u;
                if (limit > 0.0f) {
                    gv = std::fmin(gv, limit);
                    uv = std::fmin(std::fmax(uv, -limit), limit);
                    if (g > limit || std::fabs(u) > limit) ++clamped;
                }
                const float o = gv / (1.0f + std::exp(-gv)) * uv;
                const size_t i = (size_t) r * (size_t) n_ff + (size_t) n;
                g_of[i] = gv;
                u_of[i] = uv;
                h_ref[i] = f16_rn(o);
                h_elem[i] = (double) elem_bound(o);
            }
            for (int k = 0; k < n_ff; ++k)
                xrow[(size_t) k] = static_cast<float>(
                    sycl::bit_cast<sycl::half>(xh[(size_t) r * (size_t) n_ff + (size_t) k]));
            for (int n = 0; n < n_embd; ++n) {
                double sa = 0.0;
                const float v = stageA_reduce(&wd[(size_t) n * (size_t) n_ff], xrow.data(), n_ff, &sa);
                const size_t i = (size_t) r * (size_t) n_embd + (size_t) n;
                y_ref[i] = v;
                y_elem[i] = (double) elem_bound(v);
                y_tile[i] = tile_bound(sa, v);
            }
        }
    }
    {   // the non-vacuity guard: a row whose reference values collapse cannot tell a wrong tile from a right one
        std::vector<uint16_t> hu = h_ref;
        std::sort(hu.begin(), hu.end());
        hu.erase(std::unique(hu.begin(), hu.end()), hu.end());
        std::vector<float> yu = y_ref;
        std::sort(yu.begin(), yu.end());
        yu.erase(std::unique(yu.begin(), yu.end()), yu.end());
        if ((double) hu.size() < 0.75 * (double) h_ref.size() ||
            (double) yu.size() < 0.75 * (double) y_ref.size()) {
            // a vacuous row is no verdict, not a failure: it prints and adds nothing to the tallies (the wide rows'
            // F16 outputs cannot be 75% distinct however the tile is built, so counting them would make the row
            // unable to pass - and the negative control below would "catch" the vacuity instead of a corruption,
            // which the repair of task 5 found it doing: M=32 exited here before the scale byte was ever flipped)
            printf("stageA set M=%d n_embd=%d n_ff=%d: VACUOUS (%zu/%zu distinct gate/up values, %zu/%zu down): the "
                   "comparison could not distinguish a wrong tile\n", M, n_embd, n_ff, hu.size(), h_ref.size(),
                   yu.size(), y_ref.size());
            return;
        }
        if ((long) clamped * 10 > (long) h_ref.size()) {
            printf("stageA set M=%d n_embd=%d n_ff=%d: VACUOUS (%d of %zu gate/up sums at the swiglu clamp): the data "
                   "are not scale-sensitive\n", M, n_embd, n_ff, clamped, h_ref.size());
            return;
        }
    }
    if (neg) {
        blob[tr2 + 0] ^= 0x40;   // expert 0, down weight row 0, block 0's d (the low byte); the reference is clean
        printf("stageA set negative control: one wrong scale byte, blob[%zu] ^= 0x40 (down weight row 0, block 0); "
               "the reference is from the clean blob\n", tr2);
    }
    uint8_t* w = (uint8_t*) sycl::malloc_device(blob.size(), q);
    int* dbounds = (int*) sycl::malloc_device(bounds.size() * sizeof(int), q);
    int* didx = (int*) sycl::malloc_device(idx.size() * sizeof(int), q);
    float* dxf = (float*) sycl::malloc_device(xf.size() * sizeof(float), q);
    uint16_t* dxh = (uint16_t*) sycl::malloc_device(xh.size() * sizeof(uint16_t), q);
    uint16_t* dh = (uint16_t*) sycl::malloc_device(h_ref.size() * sizeof(uint16_t), q);
    float* dy = (float*) sycl::malloc_device(y_ref.size() * sizeof(float), q);
    q.memcpy(w, blob.data(), blob.size());
    q.memcpy(dbounds, bounds.data(), bounds.size() * sizeof(int));
    q.memcpy(didx, idx.data(), idx.size() * sizeof(int));
    q.memcpy(dxf, xf.data(), xf.size() * sizeof(float));
    q.memcpy(dxh, xh.data(), xh.size() * sizeof(uint16_t));
    q.wait();
    strata::ds4::gguf::iq::SetArgs sa{};
    sa.wbase = w;
    sa.stride = stride;
    sa.tr[0] = tr0; sa.tr[1] = tr1; sa.tr[2] = tr2;
    sa.bounds = dbounds;
    sa.row0 = 0;
    sa.max_rows = M;
    sa.n_exp = n_exp;
    sa.n_embd = n_embd;
    sa.n_ff = n_ff;
    sa.limit = limit;
    sa.xf = dxf;
    sa.idx = didx;
    sa.ldx = n_embd;
    sa.h = dh;
    sa.ldo = n_ff;
    sa.tg = tt;
    sa.td = tt;
    const bool ok_gu = strata::ds4::gguf::iq::set_gate_up(sa, &q);
    strata::ds4::gguf::iq::SetArgs sd = sa;
    sd.xf = nullptr;
    sd.idx = nullptr;
    sd.xh = dxh;
    sd.ldx = n_ff;
    sd.h = nullptr;
    sd.y = dy;
    sd.ldo = n_embd;
    const bool ok_dn = strata::ds4::gguf::iq::set_down(sd, &q);
    q.wait();
    std::vector<uint16_t> hg(h_ref.size());
    std::vector<float> yg(y_ref.size());
    if (ok_gu) q.memcpy(hg.data(), dh, hg.size() * sizeof(uint16_t));
    if (ok_dn) q.memcpy(yg.data(), dy, yg.size() * sizeof(float));
    q.wait();
    sycl::free(w, q); sycl::free(dbounds, q); sycl::free(didx, q); sycl::free(dxf, q); sycl::free(dxh, q);
    sycl::free(dh, q); sycl::free(dy, q);
    if (!ok_gu || !ok_dn) {
        printf("stageA set M=%d n_embd=%d n_ff=%d: FAIL (the arm launched nothing: gate/up %s, down %s)\n", M, n_embd,
               n_ff, ok_gu ? "ok" : "refused", ok_dn ? "ok" : "refused");
        t.bad += 1;
        return;
    }
    int g_exact = 0, g_near = 0, g_bad = 0, d_exact = 0, d_near = 0, d_bad = 0, g_swap = 0;
    for (size_t i = 0; i < hg.size(); ++i) {
        if (hg[i] == h_ref[i]) { ++g_exact; continue; }
        const float a = static_cast<float>(sycl::bit_cast<sycl::half>(hg[i]));
        const float b = static_cast<float>(sycl::bit_cast<sycl::half>(h_ref[i]));
        const double diff = std::fabs((double) a - (double) b);
        if (diff > (double) t.max_diff) t.max_diff = (float) diff;
        if (diff <= h_elem[i]) ++g_near; else {
            ++g_bad;
            // does the device value instead follow the gate / up roles swapped (tr[1] as the gate)?
            const float gs = u_of[i], us = g_of[i];
            const float os = gs / (1.0f + std::exp(-gs)) * us;
            if (std::fabs((double) a - (double) (float) sycl::bit_cast<sycl::half>(f16_rn(os))) <= h_elem[i]) ++g_swap;
        }
    }
    for (size_t i = 0; i < yg.size(); ++i) {
        if (yg[i] == y_ref[i]) { ++d_exact; continue; }
        const double diff = std::fabs((double) yg[i] - (double) y_ref[i]);
        if (diff > (double) t.max_diff) t.max_diff = (float) diff;
        if (diff <= y_elem[i] || diff <= y_tile[i]) ++d_near; else ++d_bad;
    }
    printf("stageA set M=%d n_embd=%d n_ff=%d exp=%d: gate/up %d/%d fp16 bits exact, %d within one BF16 ulp, %d "
           "outside; down %d/%d fp32 exact, %d within elem/tile bounds, %d outside\n", M, n_embd, n_ff, n_exp, g_exact,
           (int) hg.size(), g_near, g_bad, d_exact, (int) yg.size(), d_near, d_bad);
    if (g_bad) printf("  gate/up outside that follow the roles swapped (tr[1] as the gate): %d of %d\n", g_swap, g_bad);
    for (size_t i = 0, shown = 0; i < hg.size() && shown < 4; ++i) {   // the first outside values, for the log
        const double diff = std::fabs((double) (float) sycl::bit_cast<sycl::half>(hg[i]) - (double) (float) sycl::bit_cast<sycl::half>(h_ref[i]));
        if (diff > h_elem[i]) {
            printf("  gate/up outside: row %d col %d ref %04x (%.6f) device %04x (%.6f) bound %.3e\n",
                   (int) (i / (size_t) n_ff), (int) (i % (size_t) n_ff), h_ref[i], (float) sycl::bit_cast<sycl::half>(h_ref[i]),
                   hg[i], (float) sycl::bit_cast<sycl::half>(hg[i]), h_elem[i]);
            ++shown;
        }
    }
    for (size_t i = 0, shown = 0; i < yg.size() && shown < 4; ++i) {
        const double diff = std::fabs((double) yg[i] - (double) y_ref[i]);
        if (diff > y_elem[i] && diff > y_tile[i]) {
            printf("  down outside: row %d col %d ref %.6f device %.6f bound %.3e\n", (int) (i / (size_t) n_embd),
                   (int) (i % (size_t) n_embd), y_ref[i], yg[i], std::fmin(y_elem[i], y_tile[i]));
            ++shown;
        }
    }
    t.exact += g_exact + d_exact;
    t.near += g_near + d_near;
    t.bad += g_bad + d_bad;
}

const char* kFamilies[] = {"sparse_attn", "gemv_f32", "gemv_f16", "gemv_f16_rows",
                           "hc_pre_norm", "hc_mix_scale", "pred_proj_k", "softmax",
                           "decode_q8_0", "decode_iq4_nl", "decode_mxfp4"};

// ---- the staged-path rows for the expert-set tiles (plan todo 2.3) ----
// The same two dispatches the default run drives inline: the IQ4_XS set through gguf::iq::set_gate_up /
// set_down (ds4_iq.dp.cpp's launch_set, which reads STRATA_XMX_F16_STAGE and picks the staged instantiation
// where the joint_matrix tiles exist) and the MXFP4 set through gguf::set_gate_up / set_down.
// Behind --only iq_set_tc.  Tolerance: a value is inside when |dev-ref| <= 2.0e-4 * |ref| OR
// |dev-ref| <= kDownAbsTol, the absolute floor the down rows already use (ds4_parity.cpp:241).
// The default run's inline blocks above are untouched, so its output stays byte-identical.
// Every row here is a tile row: the harness gives one expert all 32 rows (idx[r] = r, n_exp = 1), so its span is
// hi - lo = 32.  A tile skips an expert only when that span is <= skip_le (ds4_iq.dp.cpp:1110 for IQ4_XS,
// ds4_gguf.dp.cpp:1135 for MXFP4), and skip_le is kLightRows = 16 whenever the set has more rows than that
// (ds4_gguf.dp.cpp:2202-2211, ds4_iq.dp.cpp:2819, :2844).  32 > 16, so the tiles take rows 0..31:
// the IQ4_XS light kernel early-returns on a span over kLightRows (ds4_iq.dp.cpp:1622) and the MXFP4 tile
// guard drops a tile only when the span is <= skip_le (ds4_gguf.dp.cpp:1356; :1398 is inside the wmma kk
// loop).  Both families below compare rows 0..max_rows, so every row the tiles wrote is checked.
constexpr float kSetRelTol = 2.0e-4f;

bool set_inside(float dev, float ref, float& max_diff) {
    const float diff = std::fabs(dev - ref);
    if (diff > max_diff) max_diff = diff;
    return diff <= std::fabs(ref) * kSetRelTol || diff <= kDownAbsTol;
}

void run_iq_set_tc_rows(sycl::queue& q, Tally& t) {
    const int t_q = strata::ds4::gguf::iq::kIQ4_XS;
    // One weight row is K / block_values blocks wide, and the kernel reads a row at row_bytes_t<T>(K)
    // (ds4_iq.dp.cpp:68) - the same bytes iq::row_bytes() reports (ds4_iq.dp.hpp:46).  A 512-wide IQ4_XS row is
    // 272 bytes, not the 136 of one block: laying a slab out at one block per row makes the kernel read row 2c
    // for column c, and columns past n_ff / 2 reach into the next slab.
    // The shapes the IQ4_XS set dispatch accepts: gate/up needs n_embd a multiple of 256 and n_ff a multiple
    // of BN, down needs n_ff a multiple of 256 and n_embd a multiple of BN (ds4_iq.dp.cpp:2814, :2839, BN = 64
    // at :899).  The ne gates at :2665 and :2674 use kWarps*kGuRW and kWarps*kDnRW, and the light kernels need
    // both multiples of 256 (:2683, :2688).  544x352 fails n_ff % BN (352 % 64 = 32), so it launches nothing.
    const int shapes[2][2] = {{256, 256}, {512, 256}};
    for (int si = 0; si < 2; ++si) {
        const int n_embd = shapes[si][0], n_ff = shapes[si][1];
        const int max_rows = 32, ldx = n_embd, ldh = n_ff, ldy = n_embd;
        const size_t rb_gu = strata::ds4::gguf::iq::row_bytes(t_q, n_embd);   // gate/up rows: K = n_embd
        const size_t rb_dn = strata::ds4::gguf::iq::row_bytes(t_q, n_ff);     // down rows: K = n_ff
        const size_t tr0 = 0, tr1 = (size_t) n_ff * rb_gu, tr2 = tr1 + (size_t) n_ff * rb_gu;
        const size_t stride = tr2 + (size_t) n_embd * rb_dn;
        std::vector<uint8_t> blob(stride);
        for (int part = 0; part < 3; ++part) {
            const size_t base = part == 0 ? tr0 : part == 1 ? tr1 : tr2;
            const size_t rb = part < 2 ? rb_gu : rb_dn;
            const int rows = part < 2 ? n_ff : n_embd;
            for (int r = 0; r < rows; ++r) {
                uint8_t* row = blob.data() + base + (size_t) r * rb;
                for (size_t i = 0; i < rb; ++i) row[i] = rng_next(i * 251 + (size_t) part * 977);
                // The scale fields stay small and finite, the way fill_block does for the types it fills
                // (ds4_parity.cpp:196): IQ4_XS's block is d at bytes 0/1, scales_h at 2/3, scales_l at 4..11,
                // the quants at 8..135 (Ty<kIQ4_XS>::B = 136, ds4_iq.dp.cpp:43, decode at :268,:269 and :509,:510).
                // A random scales_h/scales_l makes ls - 32 up to 31, so the sub-block scale reaches d * 31 and
                // the sums run past the +-10 clamp; one fp16 ulp then exceeds this family's tolerance.  scales_h
                // = 0xAAAA pins every sub-block's high bits to 2 (ls >= 32), and each scales_l nibble to 0..3,
                // so ls - 32 is 0..3 and the sub-block scale stays d * 3 at most.
                const uint16_t u = sycl::bit_cast<uint16_t>(sycl::half(0.01f));
                const size_t bb = (size_t) strata::ds4::gguf::iq::block_bytes(t_q);
                for (size_t blk = 0; blk < rb / bb; ++blk) {   // a row of width K holds rb / bb blocks
                    uint8_t* b = row + blk * bb;
                    b[0] = (uint8_t) (u & 0xFF);
                    b[1] = (uint8_t) (u >> 8);
                    b[2] = 0xAA;
                    b[3] = 0xAA;
                    for (int i = 0; i < 8; ++i) {
                        const uint8_t lo = (uint8_t) (rng_next(i * 61 + (size_t) part * 131 + blk * 7919) & 3u);
                        b[4 + i] = (uint8_t) (lo | (lo << 4));
                    }
                }
            }
        }
        std::vector<float> xf((size_t) max_rows * ldx);
        for (size_t i = 0; i < xf.size(); ++i)
            xf[i] = ((int) (rng_next(i * 131 + 7) | (rng_next(i * 131 + 7) << 8)) - 32768) / 655360.0f;
        std::vector<int> idx(max_rows);
        for (int r = 0; r < max_rows; ++r) idx[r] = r;
        std::vector<uint16_t> xh((size_t) max_rows * ldh);
        for (size_t i = 0; i < xh.size(); ++i)   // F16 rows in +-[0.5, 1): finite, away from zero
            xh[i] = (uint16_t) ((rng_next(i * 197 + 13) | (rng_next(i * 197 + 13) << 8)) & 0xBBFFu);
        std::vector<uint16_t> href((size_t) max_rows * ldh), hg((size_t) max_rows * ldh);
        std::vector<float> yref((size_t) max_rows * ldy), yg((size_t) max_rows * ldy);
        ref_iq_set_gate_up(blob.data(), tr0, tr1, xf.data(), ldx, idx.data(), n_embd, n_ff, 10.0f,
                           0, max_rows, href.data(), ldh, false);
        ref_iq_set_down(blob.data(), tr2, xh.data(), ldh, n_embd, n_ff, 0, max_rows, yref.data(), ldy);
        if (g_bad_scale_gu) {
            const size_t at = tr0 + 2;   // scales_h's low byte: bit 3 is sub-block 1's ls high bit, so ls goes up by 32
            // (the pair at sh bits 2*ib..2*ib+1 is shifted << 4: ds4_iq.dp.cpp:268,:269, and :509,:510).
            blob[at] ^= 0x08;
            printf("iq_set_tc negative control: one wrong gate scale byte, blob[%zu] ^= 0x08 (gate row 0 "
                   "scales_h); the reference is from the clean blob\n", at);
        }
        if (g_bad_scale_dn) {
            const size_t at = tr2 + 2;
            blob[at] ^= 0x08;
            printf("iq_set_tc negative control: one wrong down scale byte, blob[%zu] ^= 0x08 (down row 0 "
                   "scales_h); the reference is from the clean blob\n", at);
        }
        const int bounds_h[2] = {0, max_rows};
        uint8_t* w = (uint8_t*) sycl::malloc_device(blob.size(), q);
        int* bounds = (int*) sycl::malloc_device(sizeof(bounds_h), q);
        int* didx = (int*) sycl::malloc_device(idx.size() * sizeof(int), q);
        float* dxf = (float*) sycl::malloc_device(xf.size() * sizeof(float), q);
        uint16_t* dxh = (uint16_t*) sycl::malloc_device(xh.size() * sizeof(uint16_t), q);
        uint16_t* dh = (uint16_t*) sycl::malloc_device(hg.size() * sizeof(uint16_t), q);
        float* dy = (float*) sycl::malloc_device(yg.size() * sizeof(float), q);
        q.memcpy(w, blob.data(), blob.size());
        q.memcpy(bounds, bounds_h, sizeof(bounds_h));
        q.memcpy(didx, idx.data(), idx.size() * sizeof(int));
        q.memcpy(dxf, xf.data(), xf.size() * sizeof(float));
        q.memcpy(dxh, xh.data(), xh.size() * sizeof(uint16_t));
        q.wait();
        strata::ds4::gguf::iq::SetArgs sa{};
        sa.wbase = w;
        sa.stride = stride;
        sa.tr[0] = tr0; sa.tr[1] = tr1; sa.tr[2] = tr2;
        sa.bounds = bounds;
        sa.row0 = 0;
        sa.max_rows = max_rows;
        sa.n_exp = 1;
        sa.n_embd = n_embd;
        sa.n_ff = n_ff;
        sa.limit = 10.0f;
        sa.xf = dxf;
        sa.idx = didx;
        sa.ldx = ldx;
        sa.h = dh;
        sa.ldo = ldh;
        sa.tg = t_q;
        sa.td = t_q;
        const bool ok_gu = strata::ds4::gguf::iq::set_gate_up(sa, &q);
        strata::ds4::gguf::iq::SetArgs sd = sa;
        sd.xf = nullptr;
        sd.idx = nullptr;
        sd.xh = dxh;
        sd.ldx = ldh;
        sd.h = nullptr;
        sd.y = dy;
        sd.ldo = ldy;
        const bool ok_dn = strata::ds4::gguf::iq::set_down(sd, &q);
        q.wait();
        if (ok_gu) q.memcpy(hg.data(), dh, hg.size() * sizeof(uint16_t));
        if (ok_dn) q.memcpy(yg.data(), dy, yg.size() * sizeof(float));
        q.wait();
        int g_exact = 0, g_near = 0, g_bad = 0, d_exact = 0, d_near = 0, d_bad = 0;
        if (!ok_gu) {
            printf("iq_set_tc IQ4_XS set gate/up tile n_embd=%d n_ff=%d: FAIL (set_gate_up launched nothing)\n",
                   n_embd, n_ff);
            ++g_bad;
        } else {
            int shown = 0;
            for (size_t i = 0; i < hg.size(); ++i) {
                const float a = static_cast<float>(sycl::bit_cast<sycl::half>(hg[i]));
                const float b = static_cast<float>(sycl::bit_cast<sycl::half>(href[i]));
                if (hg[i] == href[i]) ++g_exact;
                else if (set_inside(a, b, t.max_diff)) ++g_near;
                else {
                    ++g_bad;
                    if (shown < 3) {
                        printf("  gate/up outside: row %d col %d ref %04x (%g) device %04x (%g)\n",
                               (int) (i / (size_t) ldh), (int) (i % (size_t) ldh), href[i], (double) b, hg[i], (double) a);
                        ++shown;
                    }
                }
            }
        }
        printf("iq_set_tc IQ4_XS set gate/up tile n_embd=%d n_ff=%d: %d/%d fp16 bits exact, %d within tol "
               "(rel %.1e, floor %.1e), %d outside\n",
               n_embd, n_ff, g_exact, g_exact + g_near + g_bad, g_near, kSetRelTol, kDownAbsTol, g_bad);
        if (!ok_dn) {
            printf("iq_set_tc IQ4_XS set down tile n_embd=%d n_ff=%d: FAIL (set_down launched nothing)\n",
                   n_embd, n_ff);
            ++d_bad;
        } else {
            int shown = 0;
            for (size_t i = 0; i < yg.size(); ++i) {
                const float a = yg[i], b = yref[i];
                if (a == b) ++d_exact;
                else if (set_inside(a, b, t.max_diff)) ++d_near;
                else {
                    ++d_bad;
                    if (shown < 3) {
                        printf("  down outside: row %d col %d ref %g device %g\n",
                               (int) (i / (size_t) ldy), (int) (i % (size_t) ldy), (double) b, (double) a);
                        ++shown;
                    }
                }
            }
        }
        printf("iq_set_tc IQ4_XS set down tile n_embd=%d n_ff=%d: %d/%d fp32 values exact, %d within tol "
               "(rel %.1e, floor %.1e), %d outside\n",
               n_embd, n_ff, d_exact, d_exact + d_near + d_bad, d_near, kSetRelTol, kDownAbsTol, d_bad);
        t.exact += g_exact + d_exact; t.near += g_near + d_near; t.bad += g_bad + d_bad;
        sycl::free(w, q); sycl::free(bounds, q); sycl::free(didx, q); sycl::free(dxf, q); sycl::free(dxh, q);
        sycl::free(dh, q); sycl::free(dy, q);
    }
}

void run_mx_set_tc_rows(sycl::queue& q, Tally& t) {
    const int shapes[2][2] = {{2560, 640}, {128, 128}};
    for (int si = 0; si < 2; ++si) {
        const int n_embd = shapes[si][0], n_ff = shapes[si][1];
        const int max_rows = 32, ldx = n_embd, ldh = n_ff, ldy = n_embd;
        // The layout the default MXFP4 set rows use: a row is mx_row_bytes(k) bytes, one E8M0 byte + 16 payload
        // bytes per 32-element block, and gate/up and down rows sit at tr0, tr1, tr2.
        const size_t rb = strata::ds4::gguf::mx_row_bytes(n_embd);
        const size_t prb_dn = strata::ds4::gguf::mx_row_bytes(n_ff);
        const size_t tr0 = 0, tr1 = (size_t) n_ff * rb, tr2 = tr1 * 2;
        const size_t stride = tr2 + (size_t) n_embd * prb_dn;
        std::vector<uint8_t> blob(stride);
        for (size_t i = 0; i < blob.size(); ++i) blob[i] = rng_next(i);
        for (int part = 0; part < 3; ++part) {   // the E8M0 scales: small powers of two keep the weights exact in F16
            const size_t base = part == 0 ? tr0 : part == 1 ? tr1 : tr2;
            const int rows = part < 2 ? n_ff : n_embd, k = part < 2 ? n_embd : n_ff;
            const size_t prb = strata::ds4::gguf::mx_row_bytes(k);
            for (int r = 0; r < rows; ++r)
                for (int b = 0; b < k / 32; ++b)
                    blob[base + (size_t) r * prb + (size_t) b * 17] =
                        (uint8_t) (128 + (rng_next(base + (size_t) r * prb + (size_t) b * 17) & 3));
        }
        std::vector<float> xf((size_t) max_rows * ldx);
        for (size_t i = 0; i < xf.size(); ++i)
            xf[i] = ((int) (rng_next(i * 131 + 7) | (rng_next(i * 131 + 7) << 8)) - 32768) / 32768.0f;
        std::vector<int> idx(max_rows);
        for (int r = 0; r < max_rows; ++r) idx[r] = r;
        std::vector<uint16_t> xh((size_t) max_rows * ldh);
        for (size_t i = 0; i < xh.size(); ++i)   // F16 rows in +-[0.5, 1): finite, away from zero
            xh[i] = (uint16_t) ((rng_next(i * 197 + 13) | (rng_next(i * 197 + 13) << 8)) & 0xBBFFu);
        std::vector<uint16_t> href((size_t) max_rows * ldh), hg((size_t) max_rows * ldh);
        std::vector<float> yref((size_t) max_rows * ldy), yg((size_t) max_rows * ldy);
        ref_mx_set_gate_up(blob.data(), tr0, tr1, xf.data(), ldx, idx.data(), n_embd, n_ff, 10.0f,
                           0, max_rows, href.data(), ldh);
        ref_mx_set_down(blob.data(), tr2, xh.data(), ldh, n_embd, n_ff, 0, max_rows, yref.data(), ldy);
        if (g_bad_scale_gu) {
            const size_t at = tr0 + (size_t) 16 * rb;   // gate row 16's first E8M0 byte, inside the compared rows
            blob[at] ^= 0x10;
            printf("iq_set_tc negative control: one wrong MXFP4 gate scale byte, blob[%zu] ^= 0x10 (gate row 16 "
                   "E8M0); the reference is from the clean blob\n", at);
        }
        if (g_bad_scale_dn) {
            const size_t at = tr2 + (size_t) 16 * prb_dn;   // down row 16's first E8M0 byte
            blob[at] ^= 0x10;
            printf("iq_set_tc negative control: one wrong MXFP4 down scale byte, blob[%zu] ^= 0x10 (down row 16 "
                   "E8M0); the reference is from the clean blob\n", at);
        }
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
        if (!strata::ds4::gguf::prepare(&q)) {
            printf("iq_set_tc MXFP4 set tile n_embd=%d n_ff=%d: FAIL (no scratch)\n", n_embd, n_ff);
            t.bad += 1;
            sycl::free(w, q); sycl::free(bounds, q); sycl::free(didx, q); sycl::free(dxf, q); sycl::free(dxh, q);
            sycl::free(dh, q); sycl::free(dy, q);
            continue;
        }
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
        strata::ds4::gguf::set_gate_up(sa, dxf, ldx, didx, dh, ldh, &q);
        strata::ds4::gguf::set_down(sa, dxh, ldh, dy, ldy, &q);
        q.wait();
        if (strata::ds4::gguf::launch_errors() != 0) {
            printf("iq_set_tc MXFP4 set tile n_embd=%d n_ff=%d: FAIL (%d launch errors)\n", n_embd, n_ff,
                   strata::ds4::gguf::launch_errors());
            t.bad += 1;
        }
        q.memcpy(hg.data(), dh, hg.size() * sizeof(uint16_t));
        q.memcpy(yg.data(), dy, yg.size() * sizeof(float));
        q.wait();
        int g_exact = 0, g_near = 0, g_bad = 0, d_exact = 0, d_near = 0, d_bad = 0;
        for (int r = 0; r < max_rows; ++r) {   // rows 0..15 are tile-written here too: see the header at :1441
            for (int c = 0; c < n_ff; ++c) {
                const size_t i = (size_t) r * ldh + c;
                const float a = static_cast<float>(sycl::bit_cast<sycl::half>(hg[i]));
                const float b = static_cast<float>(sycl::bit_cast<sycl::half>(href[i]));
                if (hg[i] == href[i]) ++g_exact;
                else if (set_inside(a, b, t.max_diff)) ++g_near;
                else ++g_bad;
            }
            for (int c = 0; c < n_embd; ++c) {
                const float a = yg[(size_t) r * ldy + c], b = yref[(size_t) r * ldy + c];
                if (a == b) ++d_exact;
                else if (set_inside(a, b, t.max_diff)) ++d_near;
                else ++d_bad;
            }
        }
        printf("iq_set_tc MXFP4 set gate/up tile n_embd=%d n_ff=%d: %d/%d fp16 bits exact, %d within tol "
               "(rel %.1e, floor %.1e), %d outside\n",
               n_embd, n_ff, g_exact, g_exact + g_near + g_bad, g_near, kSetRelTol, kDownAbsTol, g_bad);
        printf("iq_set_tc MXFP4 set down tile n_embd=%d n_ff=%d: %d/%d fp32 values exact, %d within tol "
               "(rel %.1e, floor %.1e), %d outside\n",
               n_embd, n_ff, d_exact, d_exact + d_near + d_bad, d_near, kSetRelTol, kDownAbsTol, d_bad);
        t.exact += g_exact + d_exact; t.near += g_near + d_near; t.bad += g_bad + d_bad;
        sycl::free(w, q); sycl::free(bounds, q); sycl::free(didx, q); sycl::free(dxf, q); sycl::free(dxh, q);
        sycl::free(dh, q); sycl::free(dy, q);
    }
}

bool run_family(const char* name, sycl::queue& q, bool neg, Tally& t) {
    if (!std::strcmp(name, "sparse_attn")) run_sparse_attn_row(q, neg, t);
    else if (!std::strcmp(name, "gemv_f32")) run_gemv_f32_row(q, neg, t);
    else if (!std::strcmp(name, "gemv_f16")) run_gemv_f16_row(q, neg, t);
    else if (!std::strcmp(name, "gemv_f16_rows")) run_gemv_f16_rows_row(q, neg, t);
    else if (!std::strcmp(name, "hc_pre_norm")) run_hc_pre_norm_row(q, neg, t);
    else if (!std::strcmp(name, "hc_mix_scale")) run_hc_mix_scale_row(q, neg, t);
    else if (!std::strcmp(name, "pred_proj_k")) run_pred_proj_k_row(q, neg, t);
    else if (!std::strcmp(name, "softmax")) run_softmax_row(q, neg, t);
    else if (!std::strcmp(name, "decode_q8_0")) run_decode_row(q, kDecFmtQ8_0, "decode_q8_0", neg, t);
    else if (!std::strcmp(name, "decode_iq4_nl")) run_decode_row(q, kDecFmtIQ4_NL, "decode_iq4_nl", neg, t);
    else if (!std::strcmp(name, "decode_mxfp4")) run_decode_row(q, kDecFmtMXFP4, "decode_mxfp4", neg, t);
    else if (!std::strcmp(name, "iq_set_tc")) {
        // The expert-set tiles through the set dispatch, with the staged-path tolerance.  In the JIT build
        // STRATA_XMX_SG16 is undefined, so launch_set has no staged instantiation and these rows measure the
        // scalar stand-in; only the AOT XMX build (bmg|lnl) compiles the joint_matrix tiles.
        run_iq_set_tc_rows(q, t);
        run_mx_set_tc_rows(q, t);
    }
    else return false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* dump = nullptr;
    bool bad_scale_gu = false, bad_scale_dn = false;
    const char* only = nullptr;
    bool neg = false, neg_stageA = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bad-scale-gu")) bad_scale_gu = true, g_bad_scale_gu = true;
        else if (!std::strcmp(argv[i], "--bad-scale")) bad_scale_dn = true, g_bad_scale_dn = true;
        else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!std::strcmp(argv[i], "--neg") && i + 1 < argc) only = argv[++i], neg = true;
        else if (!std::strcmp(argv[i], "--weird-scale")) g_weird_scale = true;
        else if (!std::strcmp(argv[i], "--neg-stageA")) neg_stageA = true;
    }
    try {
        sycl::device dev(sycl::gpu_selector_v);
        sycl::queue q(dev);
        printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        dpct::queue_ptr s = &q;

        if (neg_stageA) {   // the negative control alone: the reference from the clean blob, one scale byte flipped.
            // M=17, a non-vacuous shape: the wide M=32 rows fall under the non-vacuity guard, which exits before the
            // byte is flipped or anything launches - the control would report its own vacuity as a catch.
            Tally tt;
            run_stageA_row(q, StageAShape{17, 2560, 640, 2}, true, tt);
            if (tt.bad == 0) {
                printf("stageA negative control: NOT CAUGHT - the row failed to fail\n");
                return 2;
            }
            printf("NO MATCH: %d stageA set values outside the bounds after one scale byte was corrupted\n", tt.bad);
            return 1;
        }

        if (only != nullptr) {
            Tally tt;
            if (!run_family(only, q, neg, tt)) {
                std::fprintf(stderr, "ds4_parity: no family '%s'\n", only);
                return 2;
            }
            if (tt.bad == 0) {
                printf("MATCH: %d/%d values equal to the CPU reference, %d within one ulp, max float diff %g\n",
                       tt.exact, tt.exact + tt.near + tt.bad, tt.near, (double) tt.max_diff);
                return 0;
            }
            printf("NO MATCH: %d values outside one BF16 ulp of the reference\n", tt.bad);
            return 1;
        }

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
                            if (diff <= bf16_ulp(b) || diff <= kDownAbsTol) ++d_near; else ++d_bad;
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

        {   // The IQ4_XS expert-set tiles (iq::set_tc_kernel; joint_matrix on Xe2 under STRATA_XMX_SG16, its scalar
            // stand-in in the JIT form): one expert of 32 rows - more than kLightRows, so the tiles take every row of
            // it - gate/up and down.  The data scale is chosen so the sums land where the clamped swiglu is sensitive
            // (d 0.01, inputs in +-[0, 0.05): with the fill_block scales a sum would clip at +-10 or underflow and
            // every slot would match whatever the kernel computed - a row that cannot fail is not evidence).
            const int t = strata::ds4::gguf::iq::kIQ4_XS;
            const int n_embd = 256, n_ff = 256, max_rows = 32, ldx = n_embd, ldh = n_ff, ldy = n_embd;
            const size_t rb = (size_t) strata::ds4::gguf::iq::block_bytes(t);
            const size_t tr0 = 0, tr1 = (size_t) n_ff * rb, tr2 = tr1 + (size_t) n_ff * rb;
            const size_t stride = tr2 + (size_t) n_embd * rb;
            std::vector<uint8_t> blob(stride);
            for (int part = 0; part < 3; ++part) {
                const size_t base = part == 0 ? tr0 : part == 1 ? tr1 : tr2;
                const int rows = part < 2 ? n_ff : n_embd;
                for (int r = 0; r < rows; ++r) {
                    uint8_t* row = blob.data() + base + (size_t) r * rb;
                    for (size_t i = 0; i < rb; ++i) row[i] = rng.next();
                    const uint16_t u = sycl::bit_cast<uint16_t>(sycl::half(0.01f));
                    row[0] = (uint8_t) (u & 0xFF);
                    row[1] = (uint8_t) (u >> 8);
                }
            }
            std::vector<float> xf((size_t) max_rows * ldx);
            for (size_t i = 0; i < xf.size(); ++i)
                xf[i] = ((int) (rng.next() | (rng.next() << 8)) - 32768) / 655360.0f;
            std::vector<int> idx(max_rows);
            for (int r = 0; r < max_rows; ++r) idx[r] = r;
            std::vector<uint16_t> xh((size_t) max_rows * ldh);
            for (size_t i = 0; i < xh.size(); ++i)   // F16 rows in +-[0.5, 1): finite, away from zero
                xh[i] = (uint16_t) ((rng.next() | (rng.next() << 8)) & 0xBBFFu);
            std::vector<uint16_t> href((size_t) max_rows * ldh), hg((size_t) max_rows * ldh);
            std::vector<float> yref((size_t) max_rows * ldy), yg((size_t) max_rows * ldy);
            ref_iq_set_gate_up(blob.data(), tr0, tr1, xf.data(), ldx, idx.data(), n_embd, n_ff, 10.0f,
                               0, max_rows, href.data(), ldh);
            ref_iq_set_down(blob.data(), tr2, xh.data(), ldh, n_embd, n_ff, 0, max_rows, yref.data(), ldy);
            if (bad_scale_gu || bad_scale_dn) {
                const size_t at = (bad_scale_gu ? tr0 : tr2) + 4;   // scales_l[0]: IQ4_XS sub-blocks 0 and 1's ls
                blob[at] ^= 0x11;
                printf("negative control: one wrong scale byte, blob[%zu] ^= 0x11 (%s weight row 0 scales_l[0]); "
                       "the reference is from the clean blob\n", at, bad_scale_gu ? "gate" : "down");
            }
            const int bounds_h[2] = {0, max_rows};
            uint8_t* w = (uint8_t*) sycl::malloc_device(blob.size(), q);
            int* bounds = (int*) sycl::malloc_device(sizeof(bounds_h), q);
            int* didx = (int*) sycl::malloc_device(idx.size() * sizeof(int), q);
            float* dxf = (float*) sycl::malloc_device(xf.size() * sizeof(float), q);
            uint16_t* dxh = (uint16_t*) sycl::malloc_device(xh.size() * sizeof(uint16_t), q);
            uint16_t* dh = (uint16_t*) sycl::malloc_device(hg.size() * sizeof(uint16_t), q);
            float* dy = (float*) sycl::malloc_device(yg.size() * sizeof(float), q);
            q.memcpy(w, blob.data(), blob.size());
            q.memcpy(bounds, bounds_h, sizeof(bounds_h));
            q.memcpy(didx, idx.data(), idx.size() * sizeof(int));
            q.memcpy(dxf, xf.data(), xf.size() * sizeof(float));
            q.memcpy(dxh, xh.data(), xh.size() * sizeof(uint16_t));
            q.wait();
            strata::ds4::gguf::iq::SetArgs sa{};
            sa.wbase = w;
            sa.stride = stride;
            sa.tr[0] = tr0; sa.tr[1] = tr1; sa.tr[2] = tr2;
            sa.bounds = bounds;
            sa.row0 = 0;
            sa.max_rows = max_rows;
            sa.n_exp = 1;
            sa.n_embd = n_embd;
            sa.n_ff = n_ff;
            sa.limit = 10.0f;
            sa.xf = dxf;
            sa.idx = didx;
            sa.ldx = ldx;
            sa.h = dh;
            sa.ldo = ldh;
            sa.tg = t;
            sa.td = t;
            const bool ok_gu = strata::ds4::gguf::iq::set_gate_up(sa, s);
            strata::ds4::gguf::iq::SetArgs sd = sa;
            sd.xf = nullptr;
            sd.idx = nullptr;
            sd.xh = dxh;
            sd.ldx = ldh;
            sd.h = nullptr;
            sd.y = dy;
            sd.ldo = ldy;
            const bool ok_dn = strata::ds4::gguf::iq::set_down(sd, s);
            q.wait();
            if (ok_gu) q.memcpy(hg.data(), dh, hg.size() * sizeof(uint16_t));
            if (ok_dn) q.memcpy(yg.data(), dy, yg.size() * sizeof(float));
            q.wait();
            if (dump != nullptr) {
                const struct { const char* tag; const void* p; size_t n; } out[4] = {
                    {"gu_dev", hg.data(), hg.size() * sizeof(uint16_t)}, {"gu_ref", href.data(), href.size() * sizeof(uint16_t)},
                    {"dn_dev", yg.data(), yg.size() * sizeof(float)}, {"dn_ref", yref.data(), yref.size() * sizeof(float)}};
                for (const auto& f : out) {
                    char path[512];
                    std::snprintf(path, sizeof path, "%s_%s.bin", dump, f.tag);
                    FILE* fp = std::fopen(path, "wb");
                    if (fp == nullptr) { printf("dump: cannot write %s\n", path); ++bad; }
                    else { std::fwrite(f.p, 1, f.n, fp); std::fclose(fp); printf("dump: wrote %s (%zu bytes)\n", path, f.n); }
                }
            }
            if (!ok_gu) {
                printf("%s set gate/up tile: FAIL (set_gate_up launched nothing)\n", strata::ds4::gguf::iq::name(t));
                ++bad;
            } else {
                int g_exact = 0, g_near = 0, g_bad = 0;
                for (size_t i = 0; i < hg.size(); ++i) {
                    const float a = static_cast<float>(sycl::bit_cast<sycl::half>(hg[i]));
                    const float b = static_cast<float>(sycl::bit_cast<sycl::half>(href[i]));
                    if (hg[i] == href[i]) ++g_exact;
                    else {
                        const float diff = std::fabs(a - b);
                        if (diff > max_diff) max_diff = diff;
                        if (diff <= bf16_ulp(b)) ++g_near; else ++g_bad;
                    }
                }
                printf("%s set gate/up tile: %d/%d fp16 bits exact, %d within one BF16 ulp, %d outside\n",
                       strata::ds4::gguf::iq::name(t), g_exact, (int) hg.size(), g_near, g_bad);
                exact += g_exact; near += g_near; bad += g_bad;
            }
            if (!ok_dn) {
                printf("%s set down tile: FAIL (set_down launched nothing)\n", strata::ds4::gguf::iq::name(t));
                ++bad;
            } else {
                int d_exact = 0, d_near = 0, d_bad = 0;
                for (size_t i = 0; i < yg.size(); ++i) {
                    const float a = yg[i], b = yref[i];
                    if (a == b) ++d_exact;
                    else {
                        const float diff = std::fabs(a - b);
                        if (diff > max_diff) max_diff = diff;
                        if (diff <= bf16_ulp(b) || diff <= kDownAbsTol) ++d_near; else ++d_bad;
                    }
                }
                printf("%s set down tile: %d/%d fp32 values exact, %d within one BF16 ulp, %d outside\n",
                       strata::ds4::gguf::iq::name(t), d_exact, (int) yg.size(), d_near, d_bad);
                exact += d_exact; near += d_near; bad += d_bad;
            }
            sycl::free(w, q); sycl::free(bounds, q); sycl::free(didx, q); sycl::free(dxf, q); sycl::free(dxh, q);
            sycl::free(dh, q); sycl::free(dy, q);
        }

        {   // The K-quant GEMVs (iq::mv -> mv_kernel -> sub_dot): one row a type.  These rows alone cover sub_dot's
            // min term of both halves (`v - (m0 sl + m1 sh)`, the 28bd2aaa fix).
            for (const int type : {strata::ds4::gguf::iq::kQ2_K, strata::ds4::gguf::iq::kQ4_K,
                                   strata::ds4::gguf::iq::kQ5_K}) {
                Tally tt;
                run_iq_mv_row(q, type, tt);
                exact += tt.exact;
                near += tt.near;
                bad += tt.bad;
                if (tt.max_diff > max_diff) max_diff = tt.max_diff;
            }
        }

        for (const char* fam : kFamilies) {
            Tally tt;
            run_family(fam, q, false, tt);
            exact += tt.exact;
            near += tt.near;
            bad += tt.bad;
            if (tt.max_diff > max_diff) max_diff = tt.max_diff;
        }

        {   // The i8 A-stage set tiles (plan todo 5): the sg16 stage-A tile through the dispatch arm, IQ4_NL, at the gate
            // suite's M list and the K / N tails.  Last in the run: the gate suite records that the legacy set kernels
            // fault when they launch after the stageA tiles (level_zero error 20), so no legacy set launch follows.
            const StageAShape shapes[] = {
                {2, 2560, 640, 2}, {15, 2560, 640, 2}, {16, 2560, 640, 2}, {17, 2560, 640, 2},
                {31, 2560, 640, 2}, {32, 2560, 640, 2}, {33, 2560, 640, 2}, {160, 2560, 640, 2},
                {17, 544, 352, 2}, {160, 544, 352, 2}, {17, 128, 128, 2},
            };
            for (const StageAShape& sh : shapes) {
                Tally tt;
                run_stageA_row(q, sh, false, tt);
                exact += tt.exact;
                near += tt.near;
                bad += tt.bad;
                if (tt.max_diff > max_diff) max_diff = tt.max_diff;
            }
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
