# Experiment 46 - the host-flag handshake on the mirrored-expert path: exhausted, or just slow? (measured 2026-10-05)

**Status: measured on the available config, with the design's own caveat that a small number here is not a refutation.**
Two artifacts now exist and both work: a per-window **spin-exhaustion counter** wired into the verify path's stage
table, and the **standalone doorbell spin probe** the design doc believed was missing. The probe answers the
design's question 1 (**both** happen; exhaustion dominates when the store lands mid-spin), and the engine counter puts a
number on the cost: **533.5 spin iterations per decode round, bound hit in 6 of 225 windows (2.7%)**, on a 45.2 ms
round at 21.2 tok/s.

## What was built

**(A) The counter** (per window, two mapped words: spins used, bound hits; one atomic pair per *wait*, never per spin):

| file | line | what |
|---|---|---|
| `sycl/include/strata/sycl_doorbell.hpp` | 25 | `sys_add` - a relaxed system-scope `fetch_add` beside `sys_load`/`sys_store` |
| `sycl/include/strata/sycl_doorbell.hpp` | 40 | `strata::kernels::wait_flag_set_counter(uint32_t*)` |
| `sycl/src/kernels/cuda/verify_kernels.dp.cpp` | 786, 949 | `wait_flag_ge_kernel` and `wait_flag_ge_or_kernel` take the counter and record spins + bound hits |
| same | 1001-1007 | `g_wait_ctr` + setter, following the existing `g_mirror_res` pattern (a device kernel cannot read a host global, so the wrapper passes it in) |
| `sycl/include/strata/core/verify.hpp` | 169-170, 190 | public `spin_spins` / `spin_bound`, private mapped counter |
| `sycl/src/core/verify.cpp` | 41, 386, 389, 1412-1415 | allocate, register, accumulate + reset once per window |
| `sycl/src/program/generate.cpp` | 8116 | the verify-window line now prints `device wait spins ... bound-hit ... per round` |

**(B) `sycl/probe/doorbell_spin.cpp`** (new, CMake target `doorbell_spin`, deliberately *not* a ctest so the suite
count is unchanged): the host raises a mapped flag after a controlled delay while the device spins on it with the
port's own `sys_load`, and the device reports the first-seen iteration. A delay-0 control proves the spin path itself is
not reading stale cache.

## Measured: the probe (re-run independently, 256 trials, kSpinMax 20,000)

```
delay      0 us -> first-seen 0            <- control: a pre-set flag is seen immediately
delay      5 us -> first-seen 20000  (EXHAUSTED)
delay     50 us -> first-seen 20000  (EXHAUSTED)
delay    500 us -> first-seen 401
delay   2000 us -> first-seen 1490
delay  10000 us -> first-seen 20000  (EXHAUSTED)
delay  50000 us -> first-seen 20000  (EXHAUSTED)
histogram at 1 ms delay:  first-seen min 747  mean 15791.8  max 20000  exhausted 200/256 (78%)
```

An independent second run of the same binary gave 211/256 (82%) exhausted, with 500 us and 2 ms landing in the
"seen late" bucket instead of "exhausted". **The regime is bimodal and its boundary moves run to run**: a host store
that lands while the kernel is spinning is usually never observed at all, and otherwise observed late (hundreds to
thousands of iterations, i.e. 0.4-2 ms). 20,000 iterations take ~23 ms, so an exhausted wait costs ~23 ms of dead time.

**So both hypotheses are true in different runs, and the dominant failure is exhaustion (hypothesis 1), not slow
visibility.** A "waited and saw it late" story does not fit 78-82% of trials; a "never saw it" story does not fit the
18-22% that saw it at iteration 333-1490.

## Measured: the engine, with the counter

Single card, Q2_0, `--spec 2` (so the verify window and its doorbell handshake are live), 256-token decode:

```
verify window   wait for rings 45.176 ms/round; device wait spins 533.5, bound-hit 0.0 per round
                (6 hits in 225 windows); decode 256 tokens -> 21.20 tok/s
```

- The handshake costs **~533 spin iterations per round**. At the probe's ~1.15 us per iteration that is **~0.6 ms of a
  45.2 ms round, about 1.4%**.
- The `kSpinMax` bound is hit in **6 of 225 windows (2.7%)**. Each such window burns the full ~23 ms.
- Expected value of the exhaustion tail: 0.027 x 23 ms = **~0.6 ms per round, the same order as the entire spin
  cost** - i.e. on this configuration **more than half the handshake's price is the 2.7% of windows that give up**.

## Why this is not the configuration the design wanted, and what it therefore does and does not close

The design asked for the **single-card mirrored-expert** configuration (issue 867's: 4,042 of 12,288 experts mirrored,
~3.9 ms per layer in `wait for rings`, decode rising as the spin bound falls). This box cannot run it: **serve mode
requires `--mtp DIR` containing a generated `dense.txt`**, which the on-disk MTP directory does not have (serve mode
aborts at startup with a segfault after its check). What *is* runnable is the one-shot generate path with `--spec 2`,
which exercises the same verify-window spin and is where the numbers above come from - with far more experts resident.

So, honestly:

- **Closed:** the instrument (counter + probe), the probe's bimodal answer to "exhausted or slow", and a cost for the
  handshake on the all-resident path (~1.4% of a round, half of it the 2.7% exhaustion tail).
- **Not closed:** the mirrored-expert regime where the wait was 3.9 ms per layer. The design doc's own warning applies
  in reverse - **the small number here does not refute issue 867**, it is a different placement. On this box that
  configuration is unreachable until `tools/mtp_rt.py` has produced the MTP `dense.txt`.
- **Not done:** the `kSpinMax` sweep (20,000 / 2,000 / 200 / 20). It needs the mirrored config, and sweeping the bound
  on the all-resident path would trade away headroom for nothing - the bound is hit only 2.7% of windows there, and
  issue 867's two-point result already says smaller is faster on the path where the wait is live.

## Three corrections to the design doc, with evidence

1. **`sycl/probe/doorbell.cpp` exists** - it is the six-variant handshake probe INTEL.md cites. It uses a *private*
   `atomic_ref` alias rather than `sys_load`/`sys_store`, and reports OK/FAILED + latency, not first-seen iteration.
   The new probe is its sibling `doorbell_spin.cpp` rather than a replacement, so the cited authority is intact.
2. **The spin is on the verify path, not the token-graph path.** `wait_flag_ge*_kernel` is called only from
   `verify.cpp`; the line to read is the **verify window** line (`generate.cpp:8116`), not the `token graph` line
   (`tg.flushes`). The counter is surfaced there.
3. **Both spin kernels were instrumented, not just `wait_flag_ge_or_kernel`** - the latter is only the
   `STRATA_VERIFY_DEVICE_PLAN=1` variant, so instrumenting it alone would have reported a silent zero in the default
   configuration. (Confirmed by the run above: the counter is non-zero in the default config.)

`STRATA_VERIFY_COHERENT` still does not exist in this tree, so it remains a non-untried lever rather than an omission.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh
ninja -C sycl/build-b60 doorbell_spin strata
ZE_AFFINITY_MASK=1 ./sycl/build-b60/doorbell_spin 256 1000       # probe, 256 trials at 1 ms host-store delay
./sycl/build-b60/strata --pack .../q2_0 --native ...01.gguf --ple-gguf ...02.gguf \\
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill 4096 --spec 2 \\
  --kv int8 --stream-experts --kv-resident 32768 \\
  --tokens-file sycl/bench/v1/short.ids --max-new 256 2>&1 | grep 'wait for rings'
```

Logs: `/home/maxious/exp46-harness/`, `/home/maxious/exp46-harness-verify.log`, `/home/maxious/exp46-harness/engine_counter.log`.
`ctest`: 30/30 pass with the counter in place.