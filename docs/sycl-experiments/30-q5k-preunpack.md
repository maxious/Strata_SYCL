# Experiment 30 - pre-unpacked Q5_K decode (README P0 #5): the min-offset K-quant

## Question

P0 #5: extend the byte-layout (pre-unpacked) decode, shipped default-on for Q6_K (~2x, exp 27/28), to the remaining
dense K-quant types the engine actually loads. A mass-ulw analysis (run `p05-analysis`, reports in
sycl/bench/reports/p05/) established, by counting the real GGUF shards under ~/ComfyUI/koboldcpp, that the Flash-Next
target dense decode is: Q6_K x128 (shipped), **Q4_K x47**, **Q5_K x35**, IQ4_NL x47, IQ4_XS x42, Q8_0 x1 (already
byte). Q3_K/Q2_K/Q2_0 appear only in the non-target Qwen-27B/gemma shards. Of the used K-quants, **Q5_K** is the one
whose in-kernel 5-bit unpack is a real gather (like Q6_K); Q4_K's is just a cheap nibble shift and adds an
irreducible min ones-dp4a.

## What was built (exp 30)

- `Q5UBlock { float dsc, mn1; int8_t qs[32] }` - one 32-group, per element the unsigned 5-bit code (0..31),
  `dsc = d*sc[g]`, `mn1 = mn*m[g]` (fp32, the two halves of Q5_K's `d*val - m` affine form).
- `native_q5k_preunpack`: one-time Q5KBlock -> 8 Q5UBlock per 256 super-block, transcribing dequantize_q5_K
  (per-64 groups, `get_scale_min_k4` scale pair per 32-group, the ql low/high-nibble + qh 5th-bit gather).
- `Wide32Q5U` decode: native `q5_q8_dot_impl` minus the per-element gather - per 16-half, `ds * (dsc*dp4a(qs,u) -
  mn1*ones(u))` (the min term is the ones-dp4a, exactly what native already pays; the activation float-sum in
  `Q81Block.ds[1]` is NOT usable for it).
- Routing: `native_q5_k_mmvq` checks the shared dense-K-quant registry (same map/flag as Q6_K - each callsite routes
  by its own pointer); `native_dense` pre-unpacks Q5_K (type 13) default-on like Q6_K, `STRATA_MMVQ_PREUNPACK=0`
  opts out.

## Correctness (q5k_preunpack_parity, new ctest)

Packed oracle vs Q5U decode vs routed, random Q5_K (dm={1,1}, random scales/ql/qh): **PASS at ncols 1/4/6/8**, max-rel
~1e-7 (transcription correct). The Q6 parity still PASSes after the globals were moved earlier in the file.

## Speed (q5k_preunpack_bench, B60, two runs)

| ncols | Q5_K packed (us) | Q5U pre-unpacked (us) | ratio | Q5U GB/s |
|---|---|---|---|---|
| 1 | 15.6 / 15.6 | 12.1 / 12.1 | 1.29x | 678 |
| 2 | 20.9 / 20.9 | 17.1 / 17.1 | 1.22x | 480 |
| 4 | 30.9 / 30.9 | 26.6 / 26.6 | 1.16x | 308 |
| 6 | 41.4 / 41.3 | 36.3 / 36.3 | 1.14x | 226 |
| 8 | 51.3 / 51.3 | 45.9 / 46.0 | 1.12x | 179 |

**1.29x at the engine's primary decode (ncols=1), 1.12-1.29x across.** A real but smaller win than Q6_K's ~2x:
Q5_K's 5-bit unpack removes less ALU than Q6_K's 6-bit, and the min-term ones-dp4a is retained (it is not a new
cost - native already computes it). Stable over two runs.

## Verdict

Q5_K pre-unpack is worth it (used: 35 Flash-Next dense tensors; ~1.3x on the decode matvec; correctness-gated) and is
now default-on. The other dense types are parked with measured verdicts (see README P0 #5): **Q4_K** not worth it
(cheap nibble unpack + irreducible min ones-dp4a -> ~1.2x best); **Q3_K** ~2x-plausible but 0 tensors in the target
model; **Q2_0 / IQ4_XS** low value / LUT-bound (exp 04/11); **Q8_0** already on the byte wide32 path.
`q5k_preunpack_parity` + `q5k_preunpack_bench` stay as the regression gate and measure.

## Traps worth knowing (analysis-found)

- `Q81Block.ds` is `half2(d = amax/127, sum = warp_sum(ORIGINAL float activations))` - `ds[1]` is the FLOAT sum, NOT
  the int8-code sum the `m*sum(a)` min term needs; the decode must compute it via a ones-dp4a (native does the same).
- Q5_K scales are per-64 (two `get_scale_min_k4` pairs) but each pair governs a contiguous 32-group; group g = e/32
  of the 256 super-block.
- The shared pre-unpack registry must be declared before every callsite (Q5 at ~2110 comes before Q6 at ~2510) - the
  globals had to move to the top of the anonymous namespace.