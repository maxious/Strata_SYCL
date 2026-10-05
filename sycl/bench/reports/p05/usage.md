# Dense-decode tensor usage by GGUF quant type

Every GGUF under `~/ComfyUI/koboldcpp` (top-level `*.gguf` plus one subdir level), parsed directly as GGUF v3 and filtered to the dense-decode tensors that `sycl/src/core/native_dense.cpp` `eligible()` matches: name starts `blk.` and ends in one of `.attn_qkv/.attn_gate/.ssm_out/.attn_q/.attn_k/.attn_v/.attn_output/.ffn_gate_shexp/.ffn_up_shexp/.ffn_down_shexp` (each with `.weight`), plus the PLE key `blk.1.ple_key.weight` when its type is Q2_0(42), IQ3_XXS(18), IQ4_XS(23) or Q8_0(8).

## Per file

| file | dense tensors | 2 Q4_0 | 6 Q5_0 | 8 Q8_0 | 10 Q2_K | 11 Q3_K | 12 Q4_K | 13 Q5_K | 14 Q6_K | 16 IQ2_XXS | 17 IQ2_XS | 18 IQ3_XXS | 20 IQ4_NL | 21 IQ3_S | 22 IQ2_S | 23 IQ4_XS | 29 IQ1_M | 30 BF16 | 42 Q2_0 | 144 custom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | 301 | 16 | 7 | 2 | 0 | 90 | 38 | 15 | 12 | 0 | 0 | 0 | 7 | 0 | 0 | 56 | 0 | 0 | 58 | 0 |
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-Q3_K_M.gguf` | 208 | 0 | 0 | 0 | 0 | 128 | 78 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Qwen3.8-27B-UD-Q2_K_XL.gguf` | 212 | 0 | 0 | 2 | 13 | 3 | 20 | 2 | 2 | 16 | 15 | 69 | 0 | 34 | 25 | 10 | 1 | 0 | 0 | 0 |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` | 212 | 0 | 0 | 2 | 3 | 4 | 32 | 22 | 3 | 0 | 0 | 15 | 2 | 41 | 1 | 87 | 0 | 0 | 0 | 0 |
| `Qwen3.8-27B-UD-Q8_K_XL.gguf` | 212 | 0 | 0 | 161 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 51 | 0 | 0 |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` | 300 | 0 | 0 | 1 | 0 | 0 | 47 | 35 | 128 | 0 | 0 | 0 | 47 | 0 | 0 | 42 | 0 | 0 | 0 | 0 |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 300 | 0 | 0 | 1 | 0 | 0 | 47 | 35 | 128 | 0 | 0 | 0 | 47 | 0 | 0 | 42 | 0 | 0 | 0 | 0 |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gemma-3-12b-it-heretic-Q4_K_M.gguf` | 192 | 0 | 0 | 0 | 0 | 0 | 168 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gemma-3-12b-it-heretic-Q4_K_M_mmproj.gguf` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gemma-3-27b-it-qat-iq4_ks.gguf` | 248 | 62 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 186 |
| `gemma-3-27b-it-qat-mix-iq3_k.gguf` | 248 | 62 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 186 |
| `mtp-q2_0.gguf` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Total by type

| ggml id | name | # tensors |
|---|---|---|
| 2 | Q4_0 | 140 |
| 6 | Q5_0 | 7 |
| 8 | Q8_0 | 169 |
| 10 | Q2_K | 16 |
| 11 | Q3_K | 225 |
| 12 | Q4_K | 430 |
| 13 | Q5_K | 111 |
| 14 | Q6_K | 297 |
| 16 | IQ2_XXS | 16 |
| 17 | IQ2_XS | 15 |
| 18 | IQ3_XXS | 84 |
| 20 | IQ4_NL | 103 |
| 21 | IQ3_S | 75 |
| 22 | IQ2_S | 26 |
| 23 | IQ4_XS | 237 |
| 29 | IQ1_M | 1 |
| 30 | BF16 | 51 |
| 42 | Q2_0 | 58 |
| 144 | custom | 372 |
| **all** | | **2433** |

`ggml_type_id` names use the classic llama.cpp enum from `ggml/include/ggml.h` (this repo, `GGML_TYPE_COUNT=43`). Type id 144 appears only in the `ik_llama.cpp` QAT K-split models; it has no name in this repo's ggml and is past the enum's range, so it is listed by id only.

## Verification

`dense tensors` sums per file to the same value as `# tensors` in the total table (the sum of the by-type counts): **2433**. Raw count line (evidence):

```
TOTAL	2433	2:140,6:7,8:169,10:16,11:225,12:430,13:111,14:297,16:16,17:15,18:84,20:103,21:75,22:26,23:237,29:1,30:51,42:58,144:372
```

## Program

`sycl/tools/gguf_count_dense_usage.py` — run with `python3 sycl/tools/gguf_count_dense_usage.py`. Parses the GGUF v3 layout this repo's `ggml/src/gguf.cpp` emits/receives: header `magic u32, version u32, tensor_count u64, kv_count u64`; metadata kv entries (`<string> key` = `u64 len + bytes`, `u32 value_type`, `<value>`; strings use u64 lengths); the metadata section flows directly into the tensor-info section (no padding there — alignment applies only before the data section); each tensor is `<string> name, u32 n_dims, u64 dims[n_dims], u32 dtype, u64 offset`. `eligible()` mirrors `native_dense.cpp`.

```python
#!/usr/bin/env python3
"""Count dense-decode tensors per GGUF quant type across all shards.

Reads GGUF v3 tensor metadata directly (the layout this repo's ggml/src/gguf.cpp
emits/receives):
  header  : magic u32 ('GGUF'), version u32, tensor_count u64, kv_count u64
  kv      : <string> key, u32 value_type, <value>   ; strings = u64 len + bytes
  tensors : <string> name, u32 n_dims, u64 dims[n_dims], u32 dtype, u64 offset
            (no per-string padding; the metadata section flows directly into the
            tensor-info section; alignment is applied only before the data section)

Filtered to the dense-decode suffixes matched by native_dense.cpp eligible().
"""
import glob, os, struct, sys

SUFFIXES = ("blk.",)
ENDS = (".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight",
        ".attn_output.weight", ".ffn_gate_shexp.weight",
        ".ffn_up_shexp.weight", ".ffn_down_shexp.weight")
PLE = "blk.1.ple_key.weight"
PLE_TYPES = {42, 18, 23, 8}  # Q2_0, IQ3_XXS, IQ4_XS, Q8_0

# ggml_type id -> name (classic llama.cpp enum from ggml/include/ggml.h)
GGML_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
    34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}

VTYPE = {0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32", 5: "INT32",
         6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY", 10: "UINT64",
         11: "INT64", 12: "FLOAT64"}


def eligible(name, dtype):
    if name == PLE:
        return dtype in PLE_TYPES
    if not name.startswith("blk."):
        return False
    return any(name.endswith(e) for e in ENDS)


def read_str(f):
    (ln,) = struct.unpack("<Q", f.read(8))
    return f.read(ln).decode("utf-8", "replace")


def read_value(f, vtype):
    if vtype in (0, 1, 7):       # u8 / i8 / bool
        return f.read(1)
    if vtype in (2, 3):          # u16 / i16
        return f.read(2)
    if vtype in (4, 5, 6):       # u32 / i32 / f32
        return f.read(4)
    if vtype in (10, 11, 12):    # u64 / i64 / f64
        return f.read(8)
    if vtype == 8:               # string
        return read_str(f)
    if vtype == 9:               # array
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            read_value(f, et)
        return None
    raise ValueError(f"unknown value type {vtype}")


def parse(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"{path}: bad magic {magic!r}"
        (version,) = struct.unpack("<I", f.read(4))
        assert version >= 3, f"{path}: version {version}"
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        alignment = 32
        for _ in range(n_kv):
            key = read_str(f)
            (vt,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment" and vt == 4:
                (alignment,) = struct.unpack("<I", f.read(4))
                continue
            read_value(f, vt)
        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            (dtype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            tensors.append((name, dtype, dims))
        return version, alignment, tensors


def main():
    root = os.path.expanduser("~/ComfyUI/koboldcpp")
    files = sorted(glob.glob(os.path.join(root, "*.gguf")) +
                   glob.glob(os.path.join(root, "*", "*.gguf")))
    grand = {}
    print(f"raw-tensors-parsed\tfiles={len(files)}")
    for path in files:
        version, alignment, tensors = parse(path)
        dense = [t for t in tensors if eligible(t[0], t[1])]
        by_type = {}
        for _, dtype, _ in dense:
            by_type[dtype] = by_type.get(dtype, 0) + 1
        for k, v in by_type.items():
            grand[k] = grand.get(k, 0) + v
        print(f"{os.path.basename(path)}\tvv{version}\talign={alignment}"
              f"\tall={len(tensors)}\tdense={len(dense)}\t"
              + ",".join(f"{d}:{by_type[d]}" for d in sorted(by_type)))
    csv = ",".join(f"{g}:{grand[g]}" for g in sorted(grand))
    print(f"TOTAL\t{sum(grand.values())}\t{csv}")


if __name__ == "__main__":
    main()
```
