# P0 #5 — Q4_K (ggml_type 12) pre-unpack feasibility + codegen reference

All citations are `file:line`. Sources read: `include/strata/artifact/dequant.hpp` and
`sycl/src/kernels/cuda/native_mmvq.dp.cpp` (plus `include/strata/kernels/native_mmvq.hpp:121`
for the `Q6UBlock` reference shape the prompt names). No tensor-usage figures are counted here.

---

## 1. Q4KBlock byte layout (native reads)

`native_mmvq.dp.cpp:84-88`:

```cpp
struct Q4KBlock {
    sycl::half2 dm;         // {d, dmin} — fp16 pair, FIRST 4 bytes
    uint8_t scales[12];     // 12 bytes: 8x 6-bit scale + 8x 6-bit min, packed
    uint8_t qs[128];        // 128 bytes: 256 nibbles, 2 elements per byte
};
```

Pinned by `static_assert` at `native_mmvq.dp.cpp:122-123`:

```cpp
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
```

So: `dm` at byte 0 (`d` at 0, `dmin` at 2), `scales` at byte 4, `qs` at byte 16, total 144 bytes.
This is the **opposite endian/order from Q6_K**: `Q6KBlock` (`native_mmvq.dp.cpp:90-95`) puts `d`
*last* (offset 208) with `scales[16]` int8; Q4_K puts the `{d, dmin}` fp16 pair *first* and folds
the scale/min pair into one packed 12-byte array. `dequant.hpp:157-158` documents the same order:
`fp16 d`, `fp16 dmin` (first four bytes), then `scales[12]`, `qs[128]`.

The 12 `scales` bytes hold **16 six-bit values**: 8 scales `sc[0..7]` and 8 mins `m[0..7]`,
decoded by `get_scale_min_k4` (`dequant.hpp:162-171`):

```cpp
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
```

This is NOT the Q6_K per-16 int8 `scales[16]` tile; it is a bit-packed 6-bit scale/min scheme.

---

## 2. Value formula and scale granularity

### Per-element value (dequantizer)

`dequantize_q4_K` (`dequant.hpp:172-189`) computes, per element, exactly:

```
value = d1 * nibble - m1        // low nibble, 32 elements
value = d2 * nibble - m2        // high nibble, 32 elements
```

with `d1 = d * sc`, `m1 = mn * m` (`dequant.hpp:182,185`), `d2 = d * sc`, `m2 = mn * m`
(`dequant.hpp:184,186`), where `d = fp16(block[0])`, `mn = fp16(block[2])` (`dequant.hpp:173-174`).

So per element the answer to "is it d*val - m?" is **yes, with both sides scaled by their own
6-bit factor**:

```
value[j] = d * sc[g] * nibble[j] - mn * m[g]
         = d1_g * nibble[j] - m1_g
```

where `g` is the scale group index and `nibble ∈ {0..15}` (unsigned 4-bit, NOT the signed 6-bit
`q-32` of Q6_K).

### Granularity: per-32, not per-16

`dequantize_q4_K` consumes **one (sc, m) pair per 32 elements**: the outer loop `for j in
0..256 step 64` reads two pairs (`is+0`, `is+1`, `dequant.hpp:181,183`) and each pair feeds a
32-element run (`for l in 0..32`, `dequant.hpp:185,186`). `is` advances by 2 per 64 elements
(`dequant.hpp:188`), giving **8 (sc, m) pairs over the 256-element block**, i.e. **per-32
granularity**. This is *coarser* than Q6_K, which keeps 16 scales at per-16 granularity
(`Q6KBlock.scales[16]`, `native_mmvq.dp.cpp:93`; Q6UBlock `d0,d1` = the two per-16 groups,
`native_mmvq.hpp:122-124`).

### Same formula in the dot kernel

`q4_q8_dot_impl` (`native_mmvq.dp.cpp:573-590`) is the integer-dot expression of the same math:

```cpp
float sumf_d = 0.0f, sumf_m = 0.0f;
for (int i = 0; i < 2; ++i) {                       // two 16-element halves
    const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f; // low nibbles (i=0), high (i=1)
    const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
    const int dot1 = dp4a(v1i, u[2*i+1], dp4a(v0i, u[2*i], 0));          // Σ nibble*q8
    const int dot2 = dp4a(0x01010101, u[2*i+1], dp4a(0x01010101, u[2*i], 0)); // Σ q8
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
}
return dm4f.x() * sumf_d - dm4f.y() * sumf_m;        // d*sumf_d - mn*sumf_m
```

`dm4f = dm.convert<float>()` (`native_mmvq.dp.cpp:588-589`), so `dm4f.x() = d`, `dm4f.y() = mn`.
Expanding: the weight's contribution to `Σ value·q8` is `d·(sc·dot1) − mn·(m·dot2)`, i.e.
`d·sc·Σ(nibble·q8) − mn·m·Σ(q8)` — identical to summing `(d·sc·nibble − mn·m)·q8`, which is the
dequantizer's `value[j]·q8[j]`. `d8[i]` is the *activation* Q8_1 scale `bq8i->ds[0]`
(`native_mmvq.dp.cpp:616-618`), a runtime value, NOT pre-computable.

The `sc[i]` / `m[i]` are extracted per 16-element half from `scales[]` via the `aux` repack in
`q4_q8_dot` (`native_mmvq.dp.cpp:599-611`): `sc = aux[0..1]`, `m = aux[2..3]` (2 bytes each,
`native_mmvq.dp.cpp:610-611`). The two halves belong to *adjacent* per-32 scale groups (a
`qs` byte packs one low-nibble element from group `g` and one high-nibble element from group
`g+1`), which is why one 32-element dot call touches two (sc,m) pairs even though each pair
still governs 32 contiguous elements.

---

## 3. Pre-unpacked block + decode arithmetic (extends Q6UBlock)

### Block

Extend `Q6UBlock` (`native_mmvq.hpp:121-125`: `{float d0,d1; int8_t qs[32]}`) with the second
(min) term Q4_K requires. One Q4UBlock = one 32-element subblock, **8 per 256-element Q4KBlock**:

```cpp
struct Q4UBlock {
    float ds;        // d * sc[g]     (folded weight scale, 6-bit sc)
    float ms;        // mn * m[g]     (folded min offset, 6-bit m)
    int8_t qs[32];   // nibble 0..15, one per byte (positive signed byte)
};
```

Pre-unpack (one-time, at weight load, mirrors `native_q6k_preunpack_kernel`
`native_mmvq.dp.cpp:2518-2546`): for each 32-group `g`, `ds = d * (float)sc[g]`,
`ms = mn * (float)m[g]` (folding `dequant.hpp:182,184` exactly in fp32), and
`qs[j] = (int8_t)(q[byte] >> (j&1 ? 4 : 0)) & 0x0F` for `j = 0..31` (`dequant.hpp:185-186`
nibble order). 32 bytes/32 elements = 2× the 144-byte packed block, same ratio as Q6U.

Note `qs[]` is the **unsigned nibble 0..15**, not a signed `q−16`. The block MUST keep `ds`
and `ms` as two separate floats: `ms` is not an integer multiple of the q4 grid (`mn/d` is not
integral), so the `−mn·m` offset cannot be folded into a single signed int8 per element without
losing fp32-exactness. This is the structural difference from Q6U, whose value is a pure
`d·sc·(q−32)` with no offset.

### Per-32-subblock decode ops

Given one Q4UBlock and the matching 32 Q8_1 activation bytes `x8[0..31]` with per-block scale
`ds8` (`Q81Block.ds`, `native_mmvq.dp.cpp:64-67`), the 32-element dot contribution is:

```
dot_q   = dp4a(qs[0..31], x8[0..31], 0)      // Σ nibble*x8     -> 8 dp4a
dot_1   = dp4a(ones[0..31], x8[0..31], 0)    // Σ x8 (min term)  -> 8 dp4a
acc    += ds8 * ( ds * dot_q  -  ms * dot_1)  // 2 mul + 1 sub + 1 fma
```

Per-32-op tally vs Q6U (Q6U `Wide32Q6U::apply`, `native_mmvq.dp.cpp:2562-2564`, does
`r.d * ds[0] * dp4a4(...)` — dot only, one scale, no Σx8 term):

| op (per 32 elements)      | Q6U | Q4U (pre-unpacked) |
|---------------------------|-----|--------------------|
| dp4a dot                  | 8   | 8                  |
| dp4a ones (min, Σx8)      | 0   | 8                  |
| scale mults               | 2   | 2 (ds,ms) + 2 (ds8)|
| fma/sub                   | 2   | 3 (sub + fma)      |

Because the 32 elements of one dot call span two adjacent per-32 groups, the native kernel
re-splits into two 16-element halves with `sc[0]/sc[1]`, `m[0]/m[1]`, `d8[0]/d8[1]`
(`native_mmvq.dp.cpp:576-585`); a faithful pre-unpacked kernel applies the same per-16 split:
`acc += d8[0]*(ds0·dot_q0 − ms0·dot_1_0) + d8[1]*(ds1·dot_q1 − ms1·dot_1_1)`.

### Does finer granularity force more adds than Q6U?

No — Q4_K is **coarser** (per-32, 8 pairs) than Q6_K (per-16, 16 scales), so the *scale*
multiply count is *lower*. The extra cost is **not** granularity: it is the mandatory
`−mn·m·Σq8` min term. That second term forces a full second dp4a pass against the activations
(the `0x01010101 · u` ones-dot, `native_mmvq.dp.cpp:582`), which Q6U simply does not have. The
min term doubles the dp4a count (16 vs 8 per 32 elements) independent of scale granularity.

---

## 4. Verdict on ~2x plausibility

**No — ~2x is not plausible for Q4_K.** Q6U's 2x came from deleting an *expensive* 6-bit
gather (`bfn/xor/mov` on the ALU pipe, `native_mmvq.dp.cpp:2497-2501`). Q4_K's in-kernel unpack
is only a cheap nibble shift+mask (`(v>>4i)&0x0f0f0f0f`, `native_mmvq.dp.cpp:578-579`), so
pre-unpacking removes almost nothing on the ALU side, while the kernel stays dp4a-bound at
16 dp4a/32 elements because of the irreducible min-term ones-dot. Best case is a small,
single-digit-to-~1.2x gain from skipping the nibble mask, not the ~2x Q6_K achieved.
