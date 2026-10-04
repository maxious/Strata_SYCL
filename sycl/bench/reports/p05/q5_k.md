# Q5_K pre-unpack feasibility + codegen reference (ggml_type 13)

Source transcript (no code written in this report; every claim cites lines from the two in-scope files):

- `sycl/src/kernels/cuda/native_mmvq.dp.cpp` (native ncols=1 oracle)
- `include/strata/artifact/dequant.hpp` (scalar reference dequantizer)

---

## 1. Exact `Q5KBlock` byte layout native reads

From `native_mmvq.dp.cpp:58-63`:

```cpp
struct Q5KBlock {
    sycl::half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
```

Offsets are pinned by the `static_assert` at `native_mmvq.dp.cpp:151-153`:

- `scales` at offset 4, `qh` at offset 16, `qs` at offset 48
- total size 176 bytes, `alignof` 4

So the native byte order is:

| offset | field | bytes | meaning |
|---|---|---|---|
| 0 | `dm.x` (fp16) | 2 | block scale `d` |
| 2 | `dm.y` (fp16) | 2 | block min-offset scale `dmin` (`mn`) |
| 4 | `scales[12]` | 12 | packed 6-bit scales + mins (see §2) |
| 16 | `qh[32]` | 32 | 5th-bit plane (one bit per element) |
| 48 | `qs[128]` | 128 | low 4-bit nibbles, two elements per byte |

This matches the scalar layout in `dequant.hpp:198-201` (`scales = block+4`, `qh = block+16`, `ql = block+48`).

---

## 2. Value formula native computes per element

The scalar reference, `dequantize_q5_K` (`dequant.hpp:196-225`), computes, with `d = fp16_to_fp32(read_u16(block))`, `mn = fp16_to_fp32(read_u16(block+2))`:

```cpp
value = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1    // subblock 0
value = d2 * ((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2    // subblock 1
```

where `d1 = d*sc`, `m1 = mn*m` from `get_scale_min_k4(is+0/1)` (lines `dequant.hpp:207-211`).

**Yes: it is `d*val − m` (a min-offset / affine term), not `d*val − something-other`.** The per-element symbol is 5 bits: the nibble in `qs` (0..15) plus a 16 when the element's 5th bit is set, giving a 0..31 grid. The correct per-32-group scale term is:

```cpp
d1 * sum_low  − m1   (per group g, is = g)
d2 * sum_high − m2
```

with `sc` (the `d` multiplier, 0..63) and `m` (the min multiplier, 0..63) unpacked by `get_scale_min_k4` at `dequant.hpp:162-170`:

```cpp
if (j < 4)  { d = q[j] & 63;        m = q[j+4] & 63; }
else        { d = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4);
              m = (q[j+4] >> 4)   | ((q[j-0] >> 6) << 4); }
```

The group index `is` advances by 2 per 64-element group, so `j = is` takes values 0,2,4,6 across the four 64-element groups (= eight 32-element subblocks), all indexing the same 12 `scales` bytes.

**The dot-product form the kernel uses** is transcribed at `native_mmvq.dp.cpp:191-213` (`q5_q8_dot_impl`). Pinned to the source:

```cpp
float sumf_d = 0.0f, sumf_m = 0.0f;
for (int i = 0; i < 2; ++i) {                       // i selects the two 16-wide halves
    const int vl0i = (vl[0] >> (4*i)) & 0x0f0f0f0f;
    const int vl1i = (vl[1] >> (4*i)) & 0x0f0f0f0f;
    const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
    const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
    const int v0i = vl0i | vh0i;                   // 5-bit code, per byte lane
    const int v1i = vl1i | vh1i;
    const int dot1 = STRATA_DP4A(v0i, u[2*i],   STRATA_DP4A(v1i, u[2*i+1], 0));
    const int dot2 = STRATA_DP4A(0x01010101, u[2*i], STRATA_DP4A(0x01010101, u[2*i+1], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
}
const float2 dm5f = dm5.convert<float, automatic>();
return dm5f.x() * sumf_d - dm5f.y() * sumf_m;
```

Key points, all pinned:

- `dot1` is the integer dot of the **unsigned** 5-bit codes (0..31) against the **signed** q8 activations `u`. `dot2` is the dot of `0x01010101` against the same activations, i.e. `sum(u)` per 32-vector — this is exactly the `m * sum(a)` term, computed as a DP4A against a constant-ones vector.
- `sc[i]`, `m[i]` are the **8-bit unpacked** scale/min for this half, produced by the repack at `native_mmvq.dp.cpp:227-241` (the `aux[2]` uint16 re-assembly of the `scales[12]` bytes), so `sc` and `m` land at consecutive bytes: `sc = aux`, `m = sc + 2` (`native_mmvq.dp.cpp:245`).
- `dm5f.x()` = `d` (block scale) and `dm5f.y()` = `mn` (min scale), from `dm` (offset 0/2).
- The `i` loop pairs activation block `bq8 + bq8_offset + i` with `d8[i] = bq8i->ds[0]`, and `u[2i]`, `u[2i+1]` are the `int` words of the q8 codes (`native_mmvq.dp.cpp:246-250`).

So the **native expression** is exactly:

```
dot = d * Σ_i d8[i] * sc[i] * DP4A(code5[i], u[i])
      − mn * Σ_i d8[i] * m[i] * DP4A(0x01010101, u[i])   = the m*sum(a) term
```

i.e. **`d*val − m`, where `m`'s contribution is `mn * m * Σ(activation)`, not `mn*m*(something per-code)`.**

### Per-64 scale repacking (`scales[12]` + `dm`)

The 12 scale bytes hold 16 six-bit values (eight scale + eight min, interleaved per group), packed as in `get_scale_min_k4`. The kernel repacks them into two `uint16` words (`aux[0..1]`) at `native_mmvq.dp.cpp:227-241`, with `j = bq8_offset/2`, `jm = j&1`:

```cpp
const uint32_t s0 = scales[jm], s2 = scales[jm+2], s4 = scales[jm+4];
const uint32_t hi = uint32_t(-int32_t(j >= 2));
aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
```

Then `sc = (uint8_t*)aux`, `m = sc + 2` (`native_mmvq.dp.cpp:245`). The `hi` mask flips source half when `j >= 2`; the `0x3f3f`/`0xc0c0 >> 2`/`0x0f0f` pattern is the same 6-bit extraction as `get_scale_min_k4`, just done as 16-bit lanes. So for each 32-subblock half, the decode needs exactly one `sc` (0..63) and one `m` (0..63), plus shared `d = dm.x`, `mn = dm.y`.

---

## 3. Concrete pre-unpacked block + decode arithmetic (extends `Q6UBlock`/`Wide32`)

The Q6_K win pushed the 6-bit unpack out of the decode kernel (one-time per block) into a signed-byte block `{ float d0, d1; int8 qs[32] }`, leaving a load+dp4a decode. Q5_K has the same dense 32-wide grid but adds a per-subblock **min bucket** `m`. The natural extension pre-folds everything weight-side into signed bytes plus one per-half min-and-scale tuple, so the decode kernel does no bit unpacking and no `m*sum(a)` except a single correction per half.

### Proposed block (concrete field names)

```cpp
struct Q5UBlock {          // one 32-element subblock, pre-unpacked
    float   d0, d1;        // per-half q8 scale * block d:  d * sc_g * d8[h]   (see note)
    float   m0, m1;        // per-half min bucket in fp:   -mn * m_g           (signed, already negated)
    int8_t  qs[32];        // signed-byte code. For the affine dot to equal
                           //   d*val - m, the symbol must be stored as
                           //   qs[j] = val[j] - offset  so that
                           //   res = Σ qs[a]·u[a]  AND ALSO  m*Σu is folded in
};
```

**Folding decision (the one choice that matters).** The min term is `mn * m_g * Σ u`, i.e. proportional to the **sum of activations**, not to a per-code constant. There are two correct ways to pre-unpack it:

- **(a) sign-fold each symbol around a fixed bias `B` (recommended):** store `qs[j] = val[j] − B` with a per-subblock `B = m_g` chosen so the `0..31` grid becomes a signed grid whose `Σ qs·u` reproduces `d*val − m`. Precisely: the value is `sc_g·d·val − mn·m_g`; rewrite `val = (val − B) + B`, so
  `res = sc_g·d·Σ(val−B)·u  +  B·sc_g·d·Σu  −  mn·m_g·Σu`.
  Choose `B = m_g` and require the *signed* codes to be re-scaled so the first term alone equals the original `d*val` part. That forces `qs[j] = val[j] − m_g` **only if** `sc_g·d·val − mn·m_g` has the same `d` coefficient on both terms — it does not in general, so (a) alone is insufficient. This is the crux.

- **(b) signed codes + explicit `m*sum(a)` correction (exact, minimal, recommended):** store the signed symbol `qs[j] = val[j] − m_g` (range `−63..31` in int8, well within signed byte), keep `d0 = sc_g·d`, and handle the min term as a **single scalar correction** per half using the activation block's precomputed sum:

  ```cpp
  // decode per 32-subblock, half h ∈ {0,1}:
  s  = dp4a(qs, u_h);                     // Σ (val − m_g)·u
  d8 = bq8[half].ds[0];                   // activation's own q8_1 scale (from Q81Block)
  res_h = d_h * d8 * s                     // Q6U-style scale+dot, no bit unpack
          + (d_h * m_g − mn * m_g) * d8 * sum_u_h;   // ← the m-term correction
  ```

  where `sum_u_h = Σ u` is the activation-block sum that `Q81Block` already carries (`Q81Block::ds` is an `half2(d, sum)` from `native_quantize_q8_1_kernel`, `native_mmvq.dp.cpp:162-170`), so `sum_u_h` is **not** a new per-element cost — it is one float multiply-add per 32-int vector, hoistable once per (block, half, column).

**Correctness within fp32 rounding.** With `qs[j] = val[j] − m_g` stored as int8, `d0 = d*sc_g`, and `corr_h = (d*sc_g·m_g − mn·m_g) = m_g·(d*sc_g − mn)`, the two-integer / two-float expression is:

```
dot = d0 · d8 · Σ qs·u  +  corr · d8 · Σ u
```

Expanding: `d0·Σ(val−m_g)·u + m_g·(d·sc_g−mn)·Σu = d·sc_g·Σval·u − d·sc_g·m_g·Σu + d·sc_g·m_g·Σu − mn·m_g·Σu = d·sc_g·Σval·u − mn·m_g·Σu`, which is exactly `q5_q8_dot_impl`'s `dm5f.x()*sumf_d − dm5f.y()*sumf_m` restricted to one half. In int8 the code fits: `val` is 0..31, `m_g` 0..63, so `−63 ≤ qs ≤ 31`; the `u` side is signed int8, and `dp4a` handles signed×signed exactly in int32.

### Per 32-subblock op list (decode, no bit unpack)

For one 32-element subblock, two halves `h = 0,1`, matching `Wide32`-style two-`dp4a`-per-half decode:

| step | op | count |
|---|---|---|
| load `qs[32]` (int8×2), `u_h` (int32 word = 4×int8), `d8`, `sum_u_h` | loads | 2× (int8→int32 vectors) |
| `s_h = DP4A(qs_h, u_h, 0)` | dot | 2 dp4a |
| `sd = d_h * d8 * float(s_h)` | scale | 2 fmul (fma-able) |
| `sm = corr_h * d8 * sum_u_h` | min sum | 2 fmul + 1 fma (or 1 pre-mul `corr_h*d8`) |
| `res = sd + sm` | sum | 2 fadd |

Per subblock that is **4 dp4a + 2 int→float + 4–5 fmul/fma + 2 fadd**, with **zero per-element shifts/masks/or/sub_sat** in the decode kernel (all of `vl0i/vh0i/v0i` build at `native_mmvq.dp.cpp:197-203` and the `0x01010101` dot at `:204` are gone). The only new term vs. Q6_K's `load+dp4a` is the `corr_h * d8 * sum_u` scalar correction (cheap: it is per-32-vector, not per element).

To make `corr_h` fully weight-side (so the decode kernel loads it, not computes it), the pre-unpack block can also carry `float corr0, corr1` instead of `m0,m1`, folded once per block:

```cpp
struct Q5UBlock {
    float d0, d1;     // d * sc_g            (per half)
    float c0, c1;     // m_g * (d*sc_g − mn) (per half, pre-negated min bucket)
    int8_t qs[32];    // val[j] − m_g, signed byte
};
```

Then decode is, per half: `d_h * d8 * dp4a(qs_h,u_h) + c_h * d8 * sum_u_h` — two independent float terms, minimal and exact.

(Alternative (a), symmetric around zero: store `qs = val − 16` and put the constant `16` and `m_g` fully in the `corr` scalar — `corr_h = d·sc_g·(16 − m_g) + mn·m_g`. Same shape, only the bias constant differs; both work, the `16` form keeps `qs ∈ [−16,15]`, tighter but unnecessary.)

---

## 4. Verdict

**Yes — Q5_K pre-unpacks like Q6_K; the ~2x is plausible and the `m*sum(a)` term does not erode it.**

The decode-side cost to eliminate is the same ALU-pipe work that Q6_K removed: the per-element nibble/or/shift/`sub_sat` shown at `native_mmvq.dp.cpp:197-203`. Q5_K's extra `m` term is **not** per-element arithmetic — in the native kernel it is the `DP4A(0x01010101, u)` at `:204` plus two scalar fmas at `:209-210`, only 2 dp4a + 2 fma per 32-element block. In the pre-unpacked form that collapses to a single scalar `corr_h * d8 * sum_u` per half (the activation sum `sum_u` is already carried in `Q81Block::ds`, not recomputed). So the marginal cost added by Q5_K over Q6_K is ~1 scalar fma per 32-vector, while the win (removing the full per-element 5-bit unpack to a one-time pass) is the same structural ~2x as `Q6UBlock`. Q5_K's denser 32-wide layout and shared 12-byte scale table do not change that conclusion.
