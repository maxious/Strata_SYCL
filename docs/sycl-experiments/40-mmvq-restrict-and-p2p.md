# Experiment 40 - `kernel_args_restrict` on the MMVQ (null) and peer access on the B60 pair (supported, and free)

Two small items measured on the same box, both closed with numbers rather than arguments.

## 1. `intel::kernel_args_restrict` on the Q6_K wide decode: MEASURED NULL

**The attribute is real and it compiles** - unlike the cache-control intrinsics of exp 39. A standalone kernel with
`[[intel::kernel_args_restrict]]` next to `[[sycl::reqd_sub_group_size(32)]]`, and again with the attribute alone,
both compile **and link** on icpx 2026.1 (the first attempt "failed" only because the test had no `main`). So this is
the opposite verdict to exp 39: the mechanism works, it simply has nothing to do here.

**Why it is null, and this is the point.** The MMVQ decode kernels **already declare `__restrict__` by hand** on the
pointer arguments - `native_mmvq_q6k_wide_kernel(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x, ...)`,
88 `__restrict__` occurrences in `native_mmvq.dp.cpp`, and **0** uses of the Intel attribute anywhere in the tree. The
attribute's value is telling the compiler that captured pointers do not alias; the port had already told it, per
parameter, with the C99 qualifier. The kernel is also **pipe-bound, not alias-bound** (exp 23: `Pipe` 16.8%, `Send`
0.0%), so there is no reload to eliminate.

`mmvq_bench`, best of 3 per point, 2560x2560 Q6_K:

| ncols | plain | with `kernel_args_restrict` | delta |
|---:|---:|---:|---:|
| 1 | 19.1 us | 19.1 us | 1.00x |
| 2 | 21.5 us | 21.5 us | 1.00x |
| 4 | 26.1 us | 26.1 us | 1.00x |
| 8 | 55.3 us | 55.3 us | 1.00x |

Bit-identical timings, to the 0.1 us the bench prints. The object files differ (md5 `7f0f0f41...` vs `b93fe5bb...`), so
the attribute **was** applied and changed the module - the codegen simply does not translate into time on a kernel whose
aliasing the compiler was never allowed to exploit. No register-count difference was visible in the objects
(`numGRF` strings absent from both, so IGC did not change the allocation).

**Verdict: parked, default off.** The knob is in the source as `-DSTRATA_MMVQ_RESTRICT_ATTR=1` (macro
`STRATA_MMVQ_RESTRICT`, default 0) on the two Q6_K wide launches, so a future kernel that is *not* hand-annotated can
use it without rediscovering the plumbing. It is an unchecked assertion, which is also why it must not be sprayed over
kernels that accumulate in place.

## 2. Peer access between the two B60s: SUPPORTED, and enabling it changes nothing

`p2p_bench` gained the probe (it is still the same 34 us / 7.7-8.6 GB/s hand-off measurement as exp 03, plus the peer
query and an enabled re-time). On this pair:

```
peer access: dev0->dev1 supported, dev1->dev0 supported
D0->D1 0.034 ms (262144 bytes) 7.8 GB/s | D1->D0 0.034 ms 7.7 GB/s
peer enabled: D0->D1 0.034 ms 7.8 GB/s (1.00x) | D1->D0 0.034 ms 7.7 GB/s (1.00x)
round-trip check: 0 mismatches
```

(Measured twice, both arms, same numbers to 0.1 GB/s.)

**Reading.** `can_access_peer` is **true in both directions**, and `enable_peer_access` + the identical copy is
**1.00x** - so the plain `sycl::memcpy` between the two contexts is *already* taking the peer path. That is the useful
finding, and it is the opposite of the archive's hypothesis: exp 03 concluded that raw peer-USM `memcpy` was "a silent
no-op" between the port's per-stage contexts, and the reason the hand-off costs 34 us is **not** a missing peer path.
The number to compare against is PCIe: this pair measures 13.8 GB/s host-to-device (the engine's own probe), and 7.7
GB/s device-to-device is above the host path, so the transfer is already peer-class.

**What this does and does not buy.** It removes the *peer expert tier* (`--peer-device`) from the "needs a new
transport" column: `remote_experts.cpp`'s pinned-host bounce is not obviously leaving anything on the table for the
*copy* itself. It does **not** remove the tier's benefit: the bounce exists to get bytes to a card that cannot read the
other card's allocations, and this measurement says the copy was never the slow part. Wiring the tier is still a code
change; what is now measured is that it is a plumbing change, not a bandwidth project.

## Reproduce

```sh
# item 5
cmake -S sycl -B /tmp/b-restrict -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_CXX_FLAGS=-DSTRATA_MMVQ_RESTRICT_ATTR=1 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_SYCL_PARITY=ON
ninja -C /tmp/b-restrict mmvq_bench
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./sycl/build-b60/mmvq_bench 2560 2560 1   # and 2, 4, 8
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./sycl/build-b60-restrict/mmvq_bench 2560 2560 1

# item 6 (two cards)
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./sycl/build-b60/p2p_bench 262144 200
```

## Traps this run adds

- **A missing `main` looks exactly like an unsupported attribute.** The first `kernel_args_restrict` test reported a
  link failure; the cause was `-c` vs a full link plus no `main`. Always check what the linker's complaint actually is
  before blaming the extension (this is the same lesson as exp 39, from the other side).
- **An A/B that comes back bit-identical needs a positive control.** Here the object md5s differ, which proves the
  attribute reached the compiler; without that check, "1.00x everywhere" is indistinguishable from "the flag did
  nothing" (exp 37's dead knob).
- **`sycl::memcpy` between two Level Zero contexts already uses the peer path** on this pair - probe before assuming a
  host bounce is the bottleneck.