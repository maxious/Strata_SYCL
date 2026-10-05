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