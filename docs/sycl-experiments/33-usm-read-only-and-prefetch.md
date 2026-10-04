# Experiment 33 - the two P0e items that needed no code: `device_read_only` and `prefetch`

## Question

README.sycl.md's P0e lists eight SYCL capabilities the port has not used. Two of them look like they could be
worked immediately: **F1** `sycl_ext_oneapi_usm_device_read_only` (mark the weights read-only to the device) and
**F2** `sycl_ext_oneapi_prefetch` (cooperative prefetch into a chosen cache level, to hide memory latency). This
experiment answers both with measurements rather than argument.

## F1 - `device_read_only` on the weights: measured null

**The safe targets were determined first, not assumed.** The property promises the allocation is "only possibly
modified from the host, read-only in all device code", so the two candidates are the ones no kernel writes:

- the **expert cache** (`sycl/src/core/expert_cache.cpp`): filled by host `memcpy` (`fill_slot`) and only *read*
  by the gather kernels, which merely compute pointers into it (`verify_kernels.dp.cpp`: `ptr[g] = cache_base +
  ...`). The port's `ExpertCache` has no eviction and no device-side fill.
- the **packed dense weights** (`sycl/src/core/native_dense.cpp`): allocated, then `memcpy`'d up once and
  `.wait()`ed.

Deliberately excluded: the pre-unpack output buffer (`native_q6k_preunpack` **is** a kernel that writes it) and
the q8_1 activation scratch (written per token by `native_quantize_q8_1`).

The toolchain has the extension: a standalone probe on the B60 prints
`macro SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY = 1`, the allocation is accepted, and a host `memcpy` into it plus a
kernel read give 0 wrong of 4,096 values.

### The engine A/B cannot decide it (and that is a measurement of its own)

One run per arm, greedy, 256 new tokens, cold page cache, `q2_0-single-serve` at 2,185 tokens, 2x B60:

| arm | PP (tok/s) | TG (tok/s) | drafts accepted/offered | decode cache hit |
|---|---:|---:|---|---:|
| pre-change binary | 445.3 | 32.01 | 121 / 148 | 95.5% |
| property OFF (`STRATA_USM_READ_ONLY=0`) | 447.7 | 32.07 | 120 / 145 | 95.5% |
| property ON | 448.5 | 32.57 | 121 / 142 | 95.5% |

The 1.5% TG spread is **not** attributable to the change: the three runs offered 142, 145 and 148 drafts and
accepted 121, 120 and 121. The port's decode is not run-to-run deterministic - 4.5% of the expert lookups go to
the CPU pool, and a CPU expert's rows round differently when the drafts cut the windows differently (the effect
Hardin22/Strata-DualGPU's `docs/DUAL_GPU.md` documents). So an engine-level A/B has a noise floor of roughly
+-1.5% TG here, and any claim from these three runs would be noise. The load/copy path shows the same: the
profile fill took 48.1 s with the property and 44.6 s without, the latter matching the pre-change run exactly.

**And the two arms were never even different binaries.** `sycl/tools/build.sh` was failing before it built
anything (the `set -u` trap exp 35 records: oneAPI's `compiler/.../env/vars.sh:258` reads `OCL_ICD_FILENAMES`
unbound and takes the shell with it, silently, with the log left stale), so the "property on" and "property off"
runs both executed the same engine. The three numbers above are therefore run-to-run variance of one binary, not
an A/B of the property - which is still the number worth keeping (it is the reason F1 could only be decided by
the bench), but the property's verdict rests on `mmvq_bench`, which *was* rebuilt and whose new read-only
comparison printed.

### The bench decides it: 1.000x, bit-identical

`mmvq_bench` now allocates its weights **twice** in one process - once plainly, once with
`device_read_only` - and times the same kernel on each (300 ms warmup per pointer, 400 timed calls), then
compares the two outputs value by value. Same process, same clock state, no cross-run variance:

| shape | plain | read-only | ratio | results |
|---|---:|---:|---:|---|
| 2560x2560, 4 cols (the engine's window shape) | 26.1 us / 205.6 GB/s | 26.1 us / 205.6 GB/s | **1.000x** | 0 of 10,240 differ |
| 2560x2560, 1 col (the engine's primary decode shape) | 19.1 us / 281.1 GB/s | 19.1 us / 280.9 GB/s | **1.001x** | 0 of 2,560 differ |

Both plain numbers reproduce exp 22's plateau exactly (26.1 us, 205.7 GB/s), which is the check that this
instrument is describing the same kernel the engine runs. The property changes neither the time nor a bit of the
result.

### Verdict

**Measured null.** The allocation path is back to what `HEAD` had, byte for byte - the property buys nothing on
this driver and this kernel, and this repo does not carry unmeasured code in the engine's hot allocation paths.
The instrument stays: `mmvq_bench` prints `device_read_only weights: N us = X.XXXx of plain` and the result
comparison for free, so re-testing on a new driver or a new kernel is one command.

## F2 - `prefetch`: no target in the decode set

Prefetch hides **memory latency**. The port's own profile says there is no latency-bound kernel to hide it in:

| the decode's GPU time (exp 20, 256 tokens, 15.141 s total) | GPU time | what it bound on |
|---|---:|---|
| `native_mmvq_q6k_wide_a2` | 3.713 s | **execution pipe** - exp 23: `Pipe` 16.8%, **`Send` 0.0%**, 77.3% of samples running; and at 281 GB/s of the card's 608 (exp 22) it is not at the bandwidth wall either |
| `wait_flag_ge_kernel` | 2.510 s | the device **spin** on the host's plan - not a memory kernel at all (and inflated by the collection itself, exp 20) |
| `native_gu_port<18,8>` | 1.426 s | ALU - INTEL.md: the expert dots are ALU-bound (77% XVE active) at 58.1/45.7/15.2 GB/s of expert bytes |

Where the card really does sit stalled - exp 20's `XVE Array Stalled/Idle 92.3% of elapsed time with GPU busy` -
is *between* kernels and inside the handshake spin. That is an inter-kernel and graph-scheduling cost (P0c's D1,
P0d's E2), which no in-kernel instruction can remove.

**Verdict: not implemented, because there is no target.** A prefetch added to a kernel that is 74%-of-stalls
pipe-bound cannot pay, and the third-ranked kernel's ceiling - even if a prefetch were free and perfect - is
~1.4 s of GPU time on a decode whose GPU time is 50.6% of elapsed, i.e. well under the +-1.5% noise floor
measured above. Reopen this only if a stall-reason report of the kind exp 23 ran shows a top kernel with a
**non-zero `Send`** share.

## What this leaves open

Both F1's and F2's real alternatives are the same two items the P0c/P0d sections already carry: the host gap in
the split's window (**D1**, whose instrument is `STRATA_DECODE_TIMING=1`) and the graph-concurrency rules
(**E2**). The `wait_flag_ge_kernel` at 2.510 s is the strongest evidence so far that the handshake is worth
measuring properly - with the caveat that exp 20's number is inflated by the instrumentation slowing the host's
plan writes.
