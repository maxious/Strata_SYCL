# Experiment 15 - DPCT sweep, round 2: register pressure (DPCT1110) and `__ldg` (DPCT1098)

Continues exp 14. Both codes were flagged there as the remaining perf leads; both are resolved here without an
engine change, one of them with a real measurement method and a named follow-up.

## DPCT1110 (46 markers): the warning does not predict register pressure

**What it says.** "The total declared local variable size in device function X exceeds 128 bytes and may cause
high register pressure." It is a source-level count of *declared locals*, not a compiler verdict, so the first
question is whether it tracks anything the hardware cares about.

**Getting the real number.** Intel's IGC will dump the final ISA, including the allocator's own spill report:

```sh
export IGC_ShaderDumpEnable=1 IGC_ForceIgnoreCaching=1 NEO_CACHE_PERSISTENT=0 SYCL_CACHE_PERSISTENT=0
IGC_DumpToCustomDir=/tmp/dump ./<bench>
grep -m1 '^//.spill size' /tmp/dump/*_simd*.asm      # bytes spilled per work-item
grep -m1 'thread_config'  /tmp/dump/*_simd*.asm      # numGRF
```

(The cache-busting variables matter: on a program-cache hit IGC dumps only the `.spv` and never runs the
backend, so the `.asm` - and the spill number - is simply absent. That cost a few runs to find.)

**Measured (B60, `numGRF=128` on every kernel):**

| kernel | DPCT1110 flagged? | spill |
|---|---|---|
| `native_mmvq_multi_kernel` (Q6KTraits) | **yes** | **0 B** |
| `native_quantize_q8_1_kernel` | no | 0 B |
| `native_mmvq_q6k_wide<4>` | **no** | **384 B** |
| `fused_gr_read_multi` (4 variants) | yes (`gr_down_*`) | 0 B |
| `s_gemv` (3 variants) | yes (`s_gemv_split_kernel`) | 0 B |
| `sampler_greedy_kernel` | yes | 0 B |
| `sampler_one_block_kernel` | yes | 0 B |
| `sampler_split_merge_kernel` | no | 0 B |
| **`sampler_split_part_kernel`** | **yes** | **2176 B** |

**So the marker is not a proxy.** Two of the three flagged kernels I could dump spill **nothing**, and the
unflagged `native_mmvq_q6k_wide` spills 384 B. "Fix the 46 flagged kernels" would have been aimed at the wrong
list; the marker says nothing about whether the allocator ran out.

**The one real spiller among the flagged.** `sampler_split_part_kernel` (`sampler.dp.cpp:728`) spills **2176 B**
per work-item, reproducibly, while its three sibling sampler kernels spill 0. It declares
`float s[kSplitPerLane]` with `kSplitPerLane = 32` - exactly 128 bytes, the threshold - plus the top-k round
lists on top.

**Parked, and why.** This kernel runs **once per generated token** (it is the default split top-k over the
vocabulary), not once per layer: 248,320 logits = 61 blocks of 4,096. The scan it must do is ~1 MB of reads per
token, about 5 us at a realistic 200 GB/s, against a ~12.8 ms decode token (78 tok/s on the B70) - so the whole
kernel is on the order of **5-15 us/token, ~0.1%**, and that is the ceiling on removing the spill, not the win
from it. (The 5-15 us is an estimate from the access volume, not a measurement: the engine's decode timer has no
sampler phase, and `sampler_parity` does not time.) Not worth a risky rewrite of a kernel whose selection order
is deliberately exact - the file's own comments are about preserving `sampler_kernel`'s ranking. Revisit only if
a sampler bench is ever wanted for another reason.

## DPCT1098 (42 markers): `__ldg` has no Xe equivalent, and nothing was lost

The marker says "The '*' expression is used instead of the `__ldg` call. These two expressions do not provide
the exact same functionality." CUDA's `__ldg` issues a read-only load (`LDG.CI`) through the texture path.

Read at the sites, the pattern is identical everywhere: the flagged load is a plain dereference of a chain that
is **already `const __restrict__`**:

```cpp
__dpct_inline__ float q8k_at(const uint8_t *__restrict__ x, long long i) {     // s_gemv.dp.cpp:63
    const float d = *(const float*) blk;                                       // <- DPCT1098
}
__dpct_inline__ int load_x_chunk(const uint8_t *__restrict__ xb, int X[8]) {    // s2_expert_grouped.dp.cpp:262
    for (int j = 0; j < 8; ++j) v[j] = *(p + j);                               // <- DPCT1098
}
```
and the host-side kernels take `const uint8_t *__restrict__ codes` / `const float *__restrict__ scales`
throughout (`kv_q4` 8, `s_gemv` 13 `const __restrict__` parameters).

`const` + `__restrict__` is the whole of what a compiler needs to issue a non-aliased read-only load, and Intel
Xe has a **unified L1** - there is no separate read-only/texture path for a `__ldg` hint to select, and neither
SYCL nor SPIR-V exposes one. So the migration dropped a CUDA-specific annotation with no Xe counterpart, and the
read-only contract is already expressed.

**Verdict: no action.** A load-level A/B would be measuring the compiler against itself.

## Where this leaves the sweep

| code | n | verdict |
|---|---|---|
| DPCT1110 | 46 | **not a register-pressure signal** (measured); one real spiller, parked with its ceiling |
| DPCT1098 | 42 | no Xe equivalent for `__ldg`; `const __restrict__` already conveys it |
| DPCT1114/1124 | 318 | dismissed in exp 14 (in-order queue holds by construction) |
| DPCT1118 | 75 | audited clean in exp 14 |
| DPCT1053 | 6 | GDN cp.async refuted in exp 14 |
| DPCT1010/1009 | 217 | **open** - `get_error_string_dummy` placeholders (hygiene) |
| DPCT1000/1001 | 128 | **open** - error handling DPCT could not rewrite (hygiene) |
| DPCT1013 | 71 | **open** - rounding-mode intrinsics (a parity audit, not a perf one) |

The two remaining *perf-flavoured* families are closed. What is left is hygiene (DPCT1010/1009, DPCT1000/1001)
and one parity audit (DPCT1013), none of which is expected to move a number.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh && cd sycl/build-b60
export IGC_ShaderDumpEnable=1 IGC_ForceIgnoreCaching=1 NEO_CACHE_PERSISTENT=0 SYCL_CACHE_PERSISTENT=0
rm -rf /tmp/dump && mkdir -p /tmp/dump
IGC_DumpToCustomDir=/tmp/dump ./sampler_parity >/dev/null 2>&1
for f in /tmp/dump/*_simd*.asm; do echo "$(grep -m1 '^//.spill size' $f) <- $(grep -m1 '^//.kernel' $f | cut -c1-110)"; done
```
