# Experiment 14 - a DPCT migration-marker sweep: the GDN cp.async lead (refuted) and the barrier audit (clean)

## Question

The port is a DPCT (Intel DPC++ Compatibility Tool) migration of the CUDA engine, and the tool left **1,151
markers across 33 diagnostic codes** in `sycl/`. Earlier rounds resolved the barrier/shuffle families
(DPCT1065/1108/1121, committed). This sweep re-read the whole corpus for the next opportunities: which codes
hide something real - a missed fast path, a latent correctness bug, or a waste - rather than a conservative
warning about code DPCT could not prove things about?

## The corpus

| code | n | what it flags | verdict |
|---|---|---|---|
| DPCT1114 / 1124 | 221 / 97 | `cudaMemcpy` -> async memcpy **assuming an in-order queue** | dismissed - see below |
| DPCT1010 / 1009 | 110 / 107 | `get_error_string_dummy(...)` placeholder error strings | hygiene |
| DPCT1118 | 75 | SYCL group functions in **non-converged control flow** | **audited, clean (item 2)** |
| DPCT1013 | 71 | math builtins whose **rounding mode** was lost | parity audit |
| DPCT1001 / 1000 | 64 / 64 | statements DPCT "could not remove" / error-handling `if`s it could not rewrite | hygiene |
| DPCT1049 | 58 | work-group size may exceed the device limit | portability |
| DPCT1110 | 46 | device-function locals > 128 B -> register pressure | perf |
| DPCT1098 | 42 | `__ldg` -> plain `*` (read-only path lost) | perf |
| DPCT1053 | 6 | device assembly code not migrated | **the GDN lead (item 1)** |

**Dismissed without action:** the 318 `memcpy` markers (DPCT1114/1124) are the largest cluster and carry no
risk - the in-order assumption holds by construction. `strata::q_of(stream)` falls back to
`dpct::get_in_order_queue()`, and so does the second-GPU expert path (`remote_experts.cpp:148`);
`get_out_of_order_queue` exists only in the vendored `dpct/device.hpp` and is never called from `sycl/src`.
DPCT1007's three sites (`cudaGraphUpload`/`cudaInitDevice` dropped) each carry an author note explaining the
equivalent, and DPCT1053's sites in `qsa_prompt_attn`/`native_qsa_score` are behind `STRATA_PA_SM80`/NVPTX
with "unreachable on SYCL: the launcher refuses this device" - the already-known parked XMX attention.

## Item 1 - the GDN `cp.async` pipeline: real gap, but not a lever (REFUTED)

**The lead.** `sycl/src/prefill/kernels.dp.cpp` forces `#define STRATA_GDN_CP_ASYNC 0   // SYCL port: plain
copies (no cp.async)`, so `gdn_cp4`/`gdn_cp16` degrade to plain loads and `gdn_cp_commit`/`gdn_cp_wait_prev`
become no-ops. The file's own header describes what that costs on CUDA: the key-head kernel stages the next
8-token block into shared memory *while the previous block computes*, which lets a token need "2
`__syncthreads` instead of 5" and "64 blocks instead of 192", worth **1.41x on a 4080 Super**
(`src/prefill/gdn_rec_parity.cu --bench`). A SYCL port has no `cp.async`, but the overlap could be had from
register prefetching - a real, if intricate, kernel project.

**Measured first (the repo's method), with a new `gdn_rec_bench`** (B60, warm, 200 reps; the variant is chosen
once per process from the environment):

| T | `cols_pipe` (default) | `keyhead` (=1) | `cols` (PIPELINE=0) | `rec_heads` | keyhead vs default |
|---|---|---|---|---|---|
| 128 | 0.2846 ms | 1.7176 ms | 0.3064 ms | 0.4046 ms | **6.03x slower** |
| 640 | 1.1824 ms | 8.2827 ms | 1.5200 ms | 1.9086 ms | **7.00x slower** |
| 2048 | 3.6463 ms | 26.4845 ms | 4.7445 ms | 6.0129 ms | **7.26x slower** |

**And the phase it targets is 0.6% of the prompt.** The engine's own phase timer (`STRATA_PREFILL_TIMING=1`) on
a 2047-token prompt (GPU timeline 22,271 ms):

| phase | ms | share |
|---|---|---|
| gemm down | 11,187 | **50.2%** |
| dequant | 2,910 | 13.1% |
| host grouping | 2,304 | 10.3% |
| gemm gate/up | 1,590 | 7.1% |
| gdn out proj | 349 | 1.6% |
| **gdn recurrence** | **127** | **0.6%** |
| gdn conv+gates | 34 | 0.2% |

So the lead is refuted twice over: the pipelining would have to be applied to a kernel that is **6-7x slower**
than the default on Xe (its CUDA advantage was wave-quantization-specific: 64 work-groups against an SM count,
which does not transfer to a 160-EU card), and even a perfect 1.4x on the target phase is **0.16%** end to end.
The A/B also confirms the port's existing choices are the right ones: the default `cols_pipe` beats plain
`cols` (1.08-1.30x) and beats `rec_heads` (1.4-1.6x). **No code change.**

## Item 2 - the DPCT1118 barrier audit: 0 of 187 divergent (CLEAN)

**Method.** A DPCT1118 marker means DPCT saw a group function (barrier, group reduction, shuffle) and could not
prove the control flow reaching it is converged. A genuinely divergent group op is undefined behaviour and in
practice a hang or corrupted results, so the audit question is: *is any group call lexically inside a guard on
the work-item's own id?* A brace-stack parser was run over every file carrying a DPCT1118 marker: it tracks
block openings and, for each group call, reports the enclosing `if`/`while`/`for` headers, which are then
tested for thread-id expressions (`get_local_id`, `local_linear`, `tid`, `lane`, `rg`, `col`, `c`, `e`,
`continue`, `break`).

**Result: 187 group calls sit inside a block; 187 have thread-independent guards; 0 are thread-dependent.**

The sites fall into three shapes, each verified by reading the kernel:

1. **The barrier is a sibling of the conditional, not inside it.** The classic reduction:
   ```cpp
   for (int step = threads_per_row / 2; step > 0; step >>= 1) {
       if (tid < step) partial[tid] += partial[tid + step];
       item_ct1.barrier(...);            // DPCT1118 - but every thread reaches it
   }
   ```
   All threads run the same `step` ladder, so the barrier is converged. Same shape in `qsa_select`'s radix
   select (barriers after uniform-bound loops, before `if (t == 0)`), `bf16_gemv`/`s2_gemv_q8`/`s2_gemv_quads`,
   `sampler` (`if (lane == 0) { ... }` then the barrier), and `prefill/kernels.cpp`'s GDN kernels.
2. **The early return tests a work-GROUP id, so the whole group leaves together.** `bf16_gemv`,
   `s2_gemv_fast`, `s2_gemv_q8`, `s2_gemv_quads` all open with `if (o >= n_out) return;` where
   `o = item.get_group(2)` - uniform within the group.
3. **`continue` before the *next* iteration's barrier is safe.** In `fused_gr` the tile loop is
   `barrier; copy tile; barrier; if (!active) continue; compute;` - the `continue` skips only the compute, and
   every thread still arrives at the next iteration's top barrier. Likewise `if (STAGE_X) { ...copies...;
   barrier; }` in `s2_gemv_fast` is guarded by a **compile-time template parameter**.

**Corroboration:** `ctest` on the B60 passes **25 of 27**, and the two failures are documented pre-existing
ones - `ple_parity` ("required block fixtures are missing", INTEL.md:153) and `s2_expert_grouped_parity`
("NOT a data issue - a real kernel bug in the grouped path", INTEL.md:559). The suite exercises these kernels,
including the ones with the markers, and passes deterministically; a genuinely divergent group op would hang
or corrupt. **No fix needed.**

## Conclusion

- **GDN `cp.async` (DPCT1053): parked, no code.** Refuted on two independent grounds - the target phase is
  0.6% of prompt time, and the kernel that would receive the pipelining is 6-7x slower than the default on this
  hardware. Do not reopen without a different GDN kernel to pipeline.
- **DPCT1118 (barriers): audited, clean.** All 75 markers are conservative false positives, for the three
  structural reasons above. The family can be retired the way 1065/1108/1121 were - not by changing code, but
  by recording that the audit was done.
- **Still open from the sweep,** in the order I would take them: **DPCT1110** (46 sites, register pressure in
  the hot decode kernels: `fused_gr` 8, `prefill/kernels` 6, `s2_expert_grouped` 5, `qsa_select`'s
  `block_topk_kernel`/`block_scores_tc_kernel`, `native_mmvq`'s `q5_q8_dot`); **DPCT1098** (42 sites, `__ldg`
  dropped, incl. `s_gemv` and the KV paths); then hygiene - **DPCT1010/1009** (217 `get_error_string_dummy`
  placeholders) and **DPCT1000/1001** (128).

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh && cd sycl/build-b60 && ninja gdn_rec_bench
./gdn_rec_bench 640 200                                   # default (software-pipelined column kernel)
STRATA_GDN_KEYHEAD=1 ./gdn_rec_bench 640 200              # the cp.async-shaped key-head kernel
STRATA_PREFILL_TIMING=1 ./strata --pack pack/full --native "$SHARD1" --spec 2 --prefill 128 \
    --tokens-file /tmp/prompt2048.txt --max-new 4 --max-context 4096    # the phase table
ctest                                                     # 25/27; 2 documented pre-existing failures
```
