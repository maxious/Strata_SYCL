# Experiment 46 - the host-flag handshake on the mirrored-expert path: is the spin exhausted, or just slow?

**Status: design.** No number below is new. The spin's cost is exp 20's, the platform limitation is INTEL.md's,
the token rates are issue 867's, and the mechanism is the code in `sycl_doorbell.hpp` and
`verify_kernels.dp.cpp`. This exists because issue 867 closed with the question open, and because our own
header carries an instruction (`kSpinMax ... experiment: 100x smaller`) that nobody ever ran.

## Question

The device waits for the host through a mapped flag:

```cpp
// sycl/include/strata/sycl_doorbell.hpp
using sys_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system>;
inline void sys_store(volatile uint32_t* p, uint32_t v) {
    sys_atomic_u32(*const_cast<uint32_t*>(p)).store(v);
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}
```

```cpp
// sycl/src/kernels/cuda/verify_kernels.dp.cpp: wait_flag_ge_or_kernel
if (strata::sys_load(skip) == value) return;
for (uint32_t spin = 0; spin < strata::kSpinMax && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
```

Issue 867 (2x B60 upstream, one card with mirrored experts) measures **~3.9 ms per layer in `wait for rings` on
its stage table**, and decode that **rises as the spin bound falls**: 25.1 tok/s at `kSpinMax` 2,000, 28.4 at 200,
same text, same binary. Its reading is "the GPU does not seem to see the host's flag store during the spin".

**Three questions, in order of how much they would change the port:**

1. Is the bound being **exhausted** (the device never observes the flag) or **satisfied late** (it observes it,
   just after a long latency)? These look identical from the outside and call for opposite remedies.
2. Is the mixed read of the host-written `skip` flag a real bug? The same file reads it **both** ways:
   `sys_load(skip)` in `wait_flag_ge_or_kernel`, but a plain `*skip == value` in `copy_or_zero_kernel` and
   `copy_i32_unless_kernel`. INTEL.md's own rule says `volatile` device loads do **not** bypass the caches on
   Intel, so the plain reads contradict the port's documented mechanism.
3. What is the right bound? The header says `kSpinMax = 20u * 1000u; // experiment: 100x smaller` - a value
   nobody swept, and issue 867's two points say smaller is faster.

## Prior evidence (and what it already settles)

- **The platform limitation is ours, already documented.** INTEL.md, "Host-mapped flags": `volatile` device
  loads do not bypass the caches on Intel, system-scope atomics do, and then plainly: **"Host-to-device
  visibility *during* a kernel stays unreliable on this platform. That is why the all-resident path waits for the
  window instead."** Issue 867's observation is the same fact seen from the other side, so this is **not** an
  unknown - it is a known limitation with a known mitigation, and the experiment is about the cost.
- **The spin is expensive, measured.** Exp 20 part 1 prices `wait_flag_ge_kernel` at **2.510 s** of a VTune run -
  the third of the three top kernels - and notes that instrumentation slows the host's plan writes until the
  device's `kSpinMax` **expires**, which is that bounded-spin design behaving as documented.
- **Exhaustion is silent.** `wait_flag_ge_or_kernel` has no post-loop check: on exhaustion it falls straight
  through to the `acq_rel` fence and proceeds as if the flag had been seen. Nothing counts it, so the engine
  cannot currently distinguish "waited and saw it" from "waited and gave up". **That is the first thing to
  instrument** - it is cheap and it decides between hypotheses 1 and 2.
- **The host's side of the same handshake was a real bug, already fixed.** Exp 19: the host's per-layer
  completion gate discarded `ext_oneapi_empty()`'s value, so any ring slower than 2 ms was called "graph
  finished". Fixed in `verify.cpp` (decode 17.4 tok/s), and now also in `session.cpp` (`ff93973`). Do not
  re-investigate the host side.
- **The all-resident path never met this.** Issue 867: it does not wait per layer, "which is why the B70 runs
  never met it". Exp 38's 2x B60 numbers are therefore **not** a reproduction path for this experiment.

## Proposed change (bench first, nothing wired before it is measured)

1. **Count exhaustion.** Add a device-side counter (per window: spins used, bound hit) beside the existing
   `tg.flushes` accounting, and surface it in the stage table next to `wait for rings`. Nothing changes
   functionally; it only makes the two hypotheses separable.
2. **Rebuild the missing probe.** INTEL.md cites `sycl/probe/doorbell.cpp` ("six variants", "measured to work")
   as the authority for choosing system-scope atomics - **that file is not in this tree.** A standalone probe
   (host writes flag -> device spins -> device reports first-seen iteration) is ~100 lines, needs no model, and
   prices the handshake alone: latency distribution, P(spin > bound), and the effect of the bound.
3. **Unify the `skip` read on `sys_load`.** If the probe shows the plain `volatile` read missing stores the
   `sys_load` one sees, that is a one-line correctness fix with a bench A/B behind it. If it shows no
   difference, that is worth recording too - it would mean the two spellings agree here and the inconsistency is
   only latent.

## Instrument and configuration

- **Reproduce on ONE B60 with the mirrored-expert config**, which is the configuration where the per-layer wait
  is live: exp 19's exact setup (Coder IQ1_M, `pack-coder`, `--expert-cache auto --stream-experts --spec 4
  --mtp ...`, greedy). Issue 867 reached 11.9 tok/s with 4,042 of 12,288 experts mirrored on one card; our 2x
  B60 all-resident config will **not** reproduce it, by design.
- **VTune GPU Hotspots** with the exp 20 recipe (`dev.xe.observation_paranoid=0`), reading
  `wait_flag_ge_kernel`'s duration against the new counters.
- **Sweep the bound**: `kSpinMax` at 20,000 (today's value), 2,000, 200, 20 - decoding tok/s **and** the
  exhaustion count, so speed and correctness trade off against each other instead of being guessed.

## Success criteria

- **Exhaustion counter non-zero on the mirrored path** -> hypothesis 1. The bound is a correctness guard being
  hit routinely; the remedy is the platform route (wait for the window, as the all-resident path already does),
  and the experiment's output is a **number for what that costs**, not a fix.
- **Counter zero, spin still slow** -> hypothesis 2's cousin: visibility works, latency does not. Then the
  measurement to make is the flag-store-to-device-observation **distribution** (PCIe 3.0 x8 in issue 867's box,
  which is a plausible contributor and worth stating as such rather than assuming).
- **`copy_or_zero_kernel`'s plain `*skip` misses stores that `sys_load(skip)` sees** -> a real correctness bug,
  fixed behind the probe's A/B, with the `wait_flag_ge_or_kernel` pre-check and the plain read made to agree.
- **Any tok/s gain from a bound sweep alone** -> take it, but record it as trading correctness headroom for
  speed, and keep the smallest bound at which the exhaustion counter stays zero on the slowest observed run.

## Honest ceiling

- This may well close as a **measurement with no fix**: INTEL.md already says in-kernel host-to-device visibility
  is unreliable here, and issue 867's own conclusion is that the fence made no difference. If the counters show
  exhaustion on every layer, the honest result is "the per-layer handshake costs X ms/layer on this platform and
  the mitigation is structural" - a number for the port, not a bug.
- Reproducing it needs the **single-card mirrored-expert** configuration. On our 2x B60 all-resident setup the
  wait is not exercised, so a null result here would be **inconclusive**, not a refutation - say which.
- Exp 20 already showed instrumentation can push the device past the bound. Counters must be cheap enough that
  measuring does not change the thing measured; that constraint is why the counter is per-window and not per-spin.
- `STRATA_VERIFY_COHERENT`, which issue 867 did not try, **does not exist in this tree** - so it is not an
  available knob here and must not be cited as an untried lever.

## Reproduce

```sh
# one B60, the config where the per-layer wait is live (exp 19's setup)
STRATA_STAGE_TRIM=1 ./sycl/build-b60/strata --serve --pack .../pack-coder \
    --expert-profile data/expert-profile-coder.bin --expert-cache auto --stream-experts \
    --spec 4 --mtp ~/ComfyUI/mtp/rt ... # then read the stage table's wait-for-rings line
# VTune with counters (exp 20's recipe)
dev.xe.observation_paranoid=0 vtune --driver memcgpu ... # look at wait_flag_ge_kernel
```