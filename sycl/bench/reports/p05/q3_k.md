# Q3_K (ggml_type 11) — pre-unpack feasibility & codegen reference

Scope: transcribed from `include/strata/artifact/dequant.hpp` and
`sycl/src/kernels/cuda/native_mmvq.dp.cpp` (llama.cpp 3cf03257 port) only.
Line numbers cite those two files as read in this session.

---

## 1. Q3KBlock layout and how `q3_q8_dot_impl` builds values

### Native block layout

`struct Q3KBlock` (native_mmvq.dp.cpp:72-77):

```cpp
struct Q3KBlock {
    uint8_t hmask[32];   // offset 0
    uint8_t qs[64];      // offset 32
    uint8_t scales[12];  // offset 96
    sycl::half d;        // offset 108  (fp16)
};
```

Sizes/offsets are pinned by a static_assert (native_mmvq.dp.cpp:92-94):
`sizeof(Q3KBlock)==110`, `alignof==2`, `offsetof(qs)==32`, `offsetof(scales)==96`,
`offsetof(d)==108`. Total 110 bytes per 256-element super-block.

The dequant-side reference (`dequantize_q3_K`, dequant.hpp:226-256) reads the same
layout: `hm = block` (32 bytes), `q = block + 32` (64 bytes), `d_all` = fp16 at
`block + 108` (dequant.hpp:227-229). Note `d` is LAST, matching Q6_K (not the
Q4_K/Q5_K leading d/dmin pair), and there is **no `dmin`** field.

### What the 12 scale bytes actually encode

`scales[12]` is bit-repacked into 16 signed 6-bit scale values (one per 32-element
group; 8 groups per super-block but 16 slots because the repack produces a 16-byte
`aux` of which only 8 are consumed as `int8_t`). The repack is transcribed at
dequant.hpp:231-239:

```cpp
const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
uint32_t aux[4] = {0,0,0,0};
std::memcpy(aux, block + 96, 12);           // scales[12] -> 3 of the 4 uint32s
const uint32_t tmp = aux[2];
aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
const int8_t* scales = (const int8_t*)aux;   // 16 int8 slots, first 8 used
```

The four `aux` words each hold four 6-bit fields in their low 6 bits (after repack),
reinterpreted as `int8_t`. Field `j` (0..7, consumed in order) is `scales[j] - 32`,
i.e. the 6-bit field is a **signed** scale with a `-32` bias (dequant.hpp:247-250).

### How `q3_q8_dot_impl` builds values (native_mmvq.dp.cpp:387-404)

```cpp
__dpct_inline__ float q3_q8_dot_impl(int vl, int vh, const int *__restrict__ u,
                                     const uint8_t *__restrict__ scales,
                                     int scale_offset, float d3,
                                     const float *__restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low  = isc % 8;
        const int sc_shift_low  = 4 * (isc / 8);
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = dpct::vectorized_binary<sycl::char4>(vil, vih, dpct::sub_sat());
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}
```

`q3_q8_dot` (native_mmvq.dp.cpp:411-427) supplies the operands per 32-element
group `iqs`:

- `vl = load_int_b2(w->qs, iqs)` — the 2-bit codes (four per byte).
- `vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset` — the hmask plane, shifted
  so the relevant high-mask bit lands in bit 2 of each byte lane.
- `u[i]`, `d8[i]` — the Q8_1 activation's four 8-signed-byte quads and their
  per-32 Q8 scale.
- `scale_offset = iqs - iqs%8 + (iqs%8)/4`, selecting the per-group scale field.

The `load_int_b2` helper (native_mmvq.dp.cpp:380-385) exists because the 110-byte
stride gives alternate blocks only 2-byte alignment; it does two 16-bit loads and a
little-endian combine rather than one 32-bit load.

**Decode of one element** therefore is: 2-bit code from `qs`, one hmask bit
(bit 2 position, negated so `vih` is 0 when hmask bit is set, 4 when clear),
`saturated_sub(vil, vih)` = `code - (hm? 0 : 4)` ∈ [-4, 3], then
`d8 * (dp4a(that, q8) * sc)` accumulated and finally multiplied by `d3 = d`.

---

## 2. Exact per-element value formula and scale-min handling

### Value formula (dequant.hpp:246-254)

Within each 128-element half (`n = 0, 128`), four groups `j = 0..3` each emit
32 elements — 16 from `q[l+0]` and 16 from `q[l+16]`. For a group:

```
scale_j   = scales[j] - 32          // signed 6-bit, -32 bias (int8 through aux)
dl        = d_all * scale_j
code      = (q[l] >> shift) & 3     // 2-bit code, shift = 2*j within half
corr      = (hm[l] & m) ? 0 : 4     // hmask bit: clear => subtract 4
v         = dl * (code - corr)
```

Concretely, the two inner loops (dequant.hpp:248-253):

```cpp
dl = d_all * (float)(scales[is++] - 32);
for (l = 0; l < 16; ++l)
    *y++ = dl * (float)((int8_t)((q[l+0] >> shift) & 3) - ((hm[l+0] & m) ? 0 : 4));
dl = d_all * (float)(scales[is++] - 32);
for (l = 0; l < 16; ++l)
    *y++ = dl * (float)((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
shift += 2;
m = (uint8_t)(m << 1);
```

### Full per-element closed form

For super-block element index `e` (0..255), with half `h = e/128`, in-half index
`e2 = e - 128h`, group `g = e2/32` (0..3), in-group `l = e2 - 32g`, and lane
`lo = (e2 % 32) < 16 ? 0 : 16`:

```
qeff(e) = (q[32h + lo + l] >> (2g)) & 3   minus   (hm[32h + lo + l] & (1<<g) ? 0 : 4)
v(e)    = d_all * (scale[2h + g] - 32) * qeff(e)      // qeff ∈ [-4, 3]
```

`qeff` is a 3-bit value: the 2-bit code minus a 4-offset controlled by the hmask
bit. It is **fully determined statically** — `q`, `hmask`, and the scale all live in
the weight block; nothing is per-token state.

### Scale-min handling

There is **no `dmin`**. Q3_K uses the Q2_0-style signed-scale bias, not the Q4/Q5
`d`-`m` pair (dequant.hpp:219-221 explicitly says so). The "min" role is folded into
two places:

1. The scale's `-32` bias makes it signed 6-bit (range -32..+31).
2. The hmask correction `(hm ? 0 : 4)` shifts the 2-bit code's positive-biased
   range {0,1,2,3} down to {-4,-3,-2,-1} ∪ {0,1,2,3}, centering it around 0.

The repack (`kmask1`/`kmask2`) is purely a storage compaction of the 12 bytes into
16 six-bit fields; it produces the `scales[j]` values used directly with `- 32`.
It carries no per-element state.

---

## 3. Pre-unpacked form and decode arithmetic reproducing native within fp32 rounding

### The pre-unpacked block

Define one `Q3UBlock` per 32-element group (8 per 256-super-block), mirroring the
Q6U precedent exactly but with a **single** per-32 scale (Q3_K's scale is per-32,
not per-16, so it is *simpler* than Q6U):

```cpp
struct Q3UBlock {
    float d;        // d_all * (scale[group] - 32), folded once
    int8_t qs[32];  // qeff = code - (hm? 0 : 4)  ∈ [-4, 3]
};
```

Pre-unpack (one-time, single-threaded at weight load — same shape as
`native_q6k_preunpack_kernel`, native_mmvq.dp.cpp:2518-2541):

- For each 256-super-block, for each of 8 groups `g` (half `h=g/4`, in-half `g%4`):
  - `d = d_all * (scale[g] - 32)` — computed in fp32 exactly as
    dequant.hpp:247 does.
  - For each of 32 lanes `l` (16 from `q[..+0]`, 16 from `q[..+16]`), the
    `shift = 2*(g%4)` and `m = 1<<(g%4)` give
    `qs[i] = (int8_t)(((q[off] >> shift) & 3) - ((hm[off] & m) ? 0 : 4))`.

Each `qs` byte is the full `qeff ∈ [-4,3]`, i.e. **all** bit-unpack work (2-bit
code + hmask correction + saturated subtraction) is done once. Decode never reads
`q`, `hmask`, or the 12-byte `scales` again.

### Decode arithmetic (no-bit-unpack load + dp4a)

Reuse the Q8_0 wide32 path verbatim — `Wide32Q6U` (native_mmvq.dp.cpp:2555-2563)
minus the per-16 `half`/`d0,d1` split, since Q3_K has one scale per 32:

```cpp
struct Wide32Q3U {
    using Block = Q3UBlock;
    static constexpr int LPB = 1;          // one float scale per 32 (vs Q6U's 2)
    struct W { sycl::int4 q; float d; };
    static W load(const Block* b, int l) {
        W r; r.q = load16_a2(b->qs); r.d = b->d; return r;   // single 16-byte load
    }
    static float apply(const W& r, const Q81Block* xb) {
        return r.d * (float) xb->ds[0] * (float) dp4a4(r.q, ld_q8_16(xb, 0), 0);
    }
};
```

This is **exactly** the Q8_0 `Wide32Q8` load+dp4a shape (Q6U was written to
"mirror Wide32Q8 with the two per-16 scales Q6_K keeps", native_mmvq.dp.cpp:2502-2504);
Q3_K needs only the single-scale mirror.

### Per-element / per-subblock ops (named fields)

- Block fields consumed by decode: `Q3UBlock.d` (folded fp32 scale), `Q3UBlock.qs[32]`
  (signed `qeff` bytes).
- Per 32-group ops: one `load16_a2` of `qs[32]`; one `dp4a4(r.q, q8, 0)` integer
  dot of the 32 `qeff` bytes against the Q8_1 bytes; one multiply by `r.d`; one
  multiply by `xb->ds[0]` (the Q8_1 per-32 scale); accumulate into the row sum.
- Per super-block: 8 such 32-group dot/scale/sum terms, summed to the same fp32
  partial that native accumulates.

### fp32-rounding equivalence

Native computes `d3 * Σ_i [ d8[i] * (dp4a_i * sc) ]` — i.e. `d_all` multiplies the
result **after** the integer dot is scaled by `sc`. The pre-unpacked path computes
`(d_all * sc) * Σ_i [ d8[i] * dp4a_i ]` — the two `d_all`/`sc` factors are folded
into one float once. These are the same mathematical expression re-associated, so
results agree to within a few fp32 ulp (one extra float multiply/round at the
`d_all*sc` fold). This is the **same** re-association the Q6U path already ships:
`o.d0 = d * (float) sc[...]` (native_mmvq.dp.cpp:2529-2530) followed by
`r.d * (float) xb->ds[0] * dp4a` (native_mmvq.dp.cpp:2562), accepted as "matching
within fp32 rounding" rather than bit-exact. Q3_K inherits that acceptance.

There is **no** blocker. The hypothetical "per-16 scale + state-only-recoverable
bias" does not apply: the scale is per-32 (one group), and the bias (the hmask
`-4` correction) is a static per-element weight bit, folded into `qs` at
pre-unpack time, exactly as Q6_K's 6-bit unpack is folded into `qs`.

### Memory delta

Packed: 110 bytes per 256 elements. Pre-unpacked: 8 × (4 + 32) = 288 bytes per 256,
a ~2.6× weight-buffer expansion — comparable in spirit to Q6_K's
`n_in/32 * sizeof(Q6UBlock)` mapping (native_mmvq.dp.cpp:2514-2516), and a one-time
cost, not a decode cost.

---

## 4. Verdict

**YES — worth pre-unpacking, and it is simpler than Q6_K (single per-32 scale, no per-16 split), so the ~2x decode win should carry over if the removal of `sub_sat`/shift/shift-mask ALU work removes the same pipe cost that Q6_K's pre-unpack removed (plausible ~2x).**
