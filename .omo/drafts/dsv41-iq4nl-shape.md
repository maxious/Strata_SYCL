# IQ4_NL dequant kernel — reference extraction for the SYCL dsv41 port (todo 3)

Read-only extraction. Every fact below was read from the in-repo sources named; line numbers are from
the working tree on 2026-10-25. **llama.cpp is NOT vendored in this repo** (see "Sources" at the end):
any llama.cpp line citation must be replaced by the in-repo sources listed there.

## 1. Exact IQ4_NL block layout

Source: `third_party/ggml/ggml-common.h` (lines 163-164, 446-452, 1119-1122).

- `QK4_NL` = **32** — values per block (`#define QK4_NL 32`, line 447). It is the only type in this
  port whose block is 32 values, not 256 (`ds4_iq.dp.cpp:2066`: `block_values` returns 32 for
  `kIQ4_NL`, 256 for the K-quants, 128 for the trellis types).
- `block_iq4_nl` = **18 bytes**: `ggml_half d` (2 bytes) + `uint8_t qs[QK4_NL/2]` (16 bytes).
  `static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2)` (line 452);
  `ds4_iq.dp.cpp:60` asserts `sizeof(block_iq4_nl) == 18` again, and its traits agree:
  `Ty<kIQ4_NL> { B = 18, TAB = 16, MIN = 0 }`, `kQK<kIQ4_NL> = 32` (`ds4_iq.dp.cpp:47,57`).
  ggml's index macros: `QI4_NL = QK4_NL / (4*QR4_NL)` = 4, `QR4_NL = 2` (lines 163-164).
- **Scale encoding**: exactly one fp16 scale `d` at bytes 0..1 (little-endian), applied directly to
  every value in the block. There is no integer sub-scale and no min/offset term — contrast IQ4_XS,
  whose per-32-group scale is `ls - 32` (see `dequantize_iq4_xs`). Value i = `d * kvalues_iq4nl[nibble]`.
- **Nibble layout** (`dequantize_row_iq4_nl` layout, as quoted by `dequant.hpp:131`): byte `qs[j]`,
  j = 0..15, packs TWO values — its **low nibble is element j**, its **high nibble is element j+16**.
- **Codebook** `kvalues_iq4nl` (16 int8, ggml-common.h lines 1119-1122; note the TODO there:
  "fix name to kvalues_iq4_nl") — non-linear, NOT a linear grid, so the table must be exact:

  ```
  -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
  ```

## 2. Scalar reference algorithms (verbatim)

### `include/strata/artifact/dequant.hpp`, lines 128-141 (first of the file's 2 IQ4_NL references;
the second, `dequantize_iq4_xs` at lines 276-296, reuses the same codebook)

```cpp
// ---- IQ4_NL: 32 elements from an 18-byte block. Non-linear 4-bit codebook, NOT a linear grid, so
// the table must be exact. Copied verbatim from ggml-common.h `kvalues_iq4nl` at 3cf03257:
//     -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
// Layout matches dequantize_row_iq4_nl (ggml-quants.c): low nibble is element j, high nibble j+16.
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                         1,    13,   25,  38,  53,  69,  89,  113};

inline void dequantize_iq4_nl(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint8_t* qs = block + 2;
    for (int j = 0; j < 16; ++j) {
        out[j] = d * (float)kvalues_iq4nl[qs[j] & 0x0F];
        out[j + 16] = d * (float)kvalues_iq4nl[qs[j] >> 4];
    }
}
```

### `sycl/src/dsv41/ds4_parity.cpp`, lines 110-121 — `ref_iq4_nl` (its codebook decl, lines 105-108,
is the same 16 values; a harness row is kQK = 256 values = 8 blocks of 18 bytes, `kQK` at line 27)

```cpp
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
```

## 3. Kernel shape for `dq_iq4_nl_kernel`

**Verdict: a plain vector kernel is correct. `joint_matrix` is NOT needed, at all.**
`dequant_rows` is a pure elementwise decode — 18 input bytes to 32 output values, no multiply of two
matrices anywhere in the path. `joint_matrix` in this codebase belongs only to the XMX/tensor-core
GEMM kernels (`set_tc_kernel`'s `Tiles` at `ds4_iq.dp.cpp:1006-1018`, `xmxg` in
`sycl/src/kernels/cuda/iq_kernels.dp.cpp:5800+`) that CONSUME already-dequantized F16 tiles. A
dequant kernel is memory-bound; a joint_matrix would add cost and no capability.

Shape, mirroring the existing dq_rows machinery (`ds4_iq.dp.cpp`):

- **Work per work-item**: one 18-byte block = one 32-value output run. IQ4_NL's block IS the
  32-value sub-block the machinery already speaks in: `decode<T>` (lines 383-386) takes the
  `kQK<T> == 32` branch and addresses `decode_b<T>(row + kb * Ty<T>::B, 0, ...)`.
- **Work per work-group / grid**: `kThreads = 256` (8 warps x 32, line 31); grid-stride over
  `total = n * (k / 32)` blocks; `blocks = min((total + kThreads - 1) / kThreads, 65535)` groups —
  exactly `dq_rows_kernel<T>` (lines 1939-1990) and `launch_dq_rows<T>` (lines 2008-2018).
- **Codebook**: `load_tab<T>` (lines 71-77) already special-cases `kIQ4_NL`: 4 work-items load
  `kvalues_iq4nl` as 4 u32 words into the launch's `sycl::local_accessor<uint32_t,1>` (or keep the
  16 bytes in 4 registers, as `iq_kernels.dp.cpp:357-359` does for its token path).
- **Decode**: `decode_b<kIQ4_NL>` **already exists** (lines 308-318): `tab16` 16-entry nibble lookup
  over `ld2` loads of `qs`, `s.s0 = s.s1 = 1`, `s.d = h2f(b)`, no m. (`dq_rows_kernel` pre-zeros
  `sb.m0 = sb.m1`, so `v = d * s0 * q - m0` collapses to `d * codebook`.) The fast 8-value variant
  exists too (lines 535-541).
- **Output**: write 32 values to `dst + r*k + kb*32`, BF16 via `strata::ds4::bf16_rn(v)` or F16
  `rounding_mode::rte` per the `bf16` flag (lines 1973-1985).
- **Minimal wiring (Option A)**: no new kernel is strictly required — the decode side is complete and
  `dequant_rows`'s switch (lines 2029-2041) merely lacks `case kIQ4_NL: DS4_DQ(kIQ4_NL); break;`,
  which instantiates `dq_rows_kernel<kIQ4_NL>` through `dq_rows_wrapper<T>`. The `k % 256 != 0` guard
  (line 2021) passes for the harness's k = 256 rows; relax it to `k % 32` for this type only if
  arbitrary IQ4_NL rows must be accepted.
- **Named kernel (Option B, if todo 3 wants the `dq_iq4_nl_kernel` symbol)**: same functor+launch
  pattern as `dq_rows_wrapper<T>` (lines 1993-2018), or the 4-lanes-per-block / 8-values-per-lane
  split of `iq_kernels.dp.cpp:2833-2853` (`dq_iq4_nl`: thread (ib, il) writes values [8il, 8il+8),
  consecutive lanes write consecutive 8-value runs = coalesced stores). Either is a plain vector
  kernel.

### Premise corrections (verified against the tree — read these before coding)

- There is **no `dq_wrapper` symbol** anywhere in the repo's sources. The precedent named in the task
  is `dq_rows_wrapper<T>` (`ds4_iq.dp.cpp:1993`) plus `launch_dq_rows<T>` / `dequant_rows`.
- There is **no `[[sycl::reqd_sub_group_size(16)]]` in `sycl/src/dsv41/`**. The matmul launches in
  `ds4_iq.dp.cpp` use `[[sycl::reqd_sub_group_size(32)]]` (e.g. lines 1495-1497, 1555, 1858), and
  `dq_rows_wrapper` carries no sub-group attribute at all — correct for a vector kernel.
- The `STRATA_XMX_SG16` / `STRATA_NO_DG2_XMX` guard convention lives in `sycl/CMakeLists.txt:40-41`
  (`STRATA_SYCL_AOT` matching `bmg|lnl` defines `STRATA_XMX_SG16=1` — grouped XMX kernels on
  sub-group 16, joint_matrix 8x16x16 (Xe2); matching `pvc` defines `STRATA_NO_DG2_XMX=1`) and is
  consumed in `sycl/src/kernels/cuda/iq_kernels.dp.cpp:5803-5810` (`xmxg::kSgW = 16` / `8`) and
  `sycl/src/prefill/xmx_moe.dp.cpp:19-20`. **Neither guard applies to `dq_iq4_nl_kernel`** — a
  dequant kernel is shape-agnostic across Xe2/Xe-HPC and needs no XMX guard, no sub-group-16
  requirement, and no joint_matrix.

## 4. Parity row and the refusal path (`ds4_parity.cpp:225-230`)

Today the harness fills IQ4_NL rows (`fill_block`, lines 164-168, already writes the fp16 scale at
`row[0..1]`), computes `ref_iq4_nl` (line 207), then at lines 225-230 treats a refused launch as the
pass condition:

```cpp
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
```

Required change once the decode path exists:

1. Delete the `kIQ4_NL` special case so `!launched` is a plain `FAIL (no kernel launched)` like
   every other type — refusal flips from "correct" to failure.
2. The **parity row** is then the normal compared row the other types already get: the existing loop
   (lines 235-250) compares `got[i]` against `strata::ds4::bf16_rn(ref[i])` bit-exact first, else
   counts `|got - ref| <= bf16_ulp(ref)` as near (one BF16 ulp, `bf16_ulp` at lines 172-175), else
   bad. Expected row: `IQ4_NL: X/1024 bits exact, Y within one BF16 ulp, Z outside` — 4 rows x 256
   values = 1024 values — and its counts join the `MATCH` / `NO MATCH` totals at lines 253-259.
   (`fill_block` and `ref_iq4_nl` need no change; `kIQ4_NL` is already in `types[]`, line 189.)

## 5. Files a todo-3 executor may touch

Allowed scope — `sycl/src/dsv41/` plus `sycl/CMakeLists.txt`, nothing else:

- `sycl/src/dsv41/ds4_iq.dp.cpp` — `dequant_rows` switch case / guard, or the new
  `dq_iq4_nl_kernel` + wrapper + launch if Option B is chosen. (Decode branches already exist.)
- `sycl/src/dsv41/ds4_iq.dp.hpp` — only if a declaration must change; `dequant_rows`'s signature
  does not change, so likely untouched.
- `sycl/src/dsv41/ds4_parity.cpp` — remove the refusal, get the real compared row (section 4).
- `sycl/CMakeLists.txt` — ONLY if a new `.dp.cpp` source file is added: register it in
  `strata_resolve(_dsv41 ...)` (line 259). `strata_resolve` (lines 108-126) maps a `src/dsv41/X.cu`
  name to `sycl/src/dsv41/X.dp.cpp` (FATAL_ERROR if the migrated copy is missing) and accepts a plain
  existing path as-is. If the kernel goes into `ds4_iq.dp.cpp` (recommended), no CMake change at all.

Read-only references that must NOT be touched: `third_party/ggml/ggml-common.h`,
`include/strata/artifact/dequant.hpp`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp`, `src/dsv41/*`.

## Sources (and the llama.cpp note)

**llama.cpp is NOT vendored in this repo.** (The only copy that exists is fetched at configure time
into `sycl/build-test/_deps/strata_llamacpp-src` — a build directory, not a source, and out of scope
here.) Any llama.cpp line citation — e.g. `dequantize_row_iq4_nl`, `dequantize_row_iq4_xs` in
`ggml-quants.c` — must be replaced by the in-repo sources:

- `third_party/ggml/ggml-common.h` — `QK4_NL`, `block_iq4_nl`, `kvalues_iq4nl`, `QI4_NL`/`QR4_NL`.
- `include/strata/artifact/dequant.hpp` — `dequantize_iq4_nl`, `dequantize_iq4_xs` (both transcribed
  from ggml and self-describing).
- `sycl/src/dsv41/ds4_parity.cpp` — `ref_iq4_nl`, `ref_iq4_xs`, the harness compare semantics.
- `sycl/src/dsv41/ds4_iq.dp.cpp` / `ds4_iq.dp.hpp` — `Ty<kIQ4_NL>`, `kQK`, `load_tab`, `decode_b`,
  `dq_rows_kernel`/`dq_rows_wrapper`/`launch_dq_rows`, `dequant_rows`, `block_values`/`block_bytes`.
