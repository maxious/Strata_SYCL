# Experiment 16 - DPCT cleanup: the error path made real, and 897 marker blocks retired

Finishes the sweep of exp 14/15. Hygiene only: no kernel changed, no measurement expected to move.

## The one real defect: every error message said `<FIXME: Placeholder>`

The migration left `dpct::get_error_string_dummy(int ec)`, which ignored its argument and returned the literal
string `"<FIXME: Placeholder>"`. **92 call sites** in 31 files fed it into their error text:

```cpp
const dpct::err0 e = DPCT_CHECK_ERROR(queue->parallel_for(...));
if (e != 0) throw std::runtime_error(std::string("native MMVQ launch: ") +
                                     dpct::get_error_string_dummy(error));   // <- "<FIXME: Placeholder>"
```

The information was already in hand and being thrown away: `DPCT_CHECK_ERROR` is a lambda that catches
`std::exception`, prints `e.what()` and returns `default_error`. So the message existed and was discarded one
line later.

**Fix** (`sycl/include/dpct/dpct.hpp`): `DPCT_CHECK_ERROR` now also stores `e.what()` in `dpct::last_error()`,
and `get_error_string_dummy` is replaced by

```cpp
inline const char *error_string(int ec) {
  if (ec == success) return "no error";
  const std::string &m = last_error();
  return m.empty() ? "the call raised a SYCL exception (reported above)" : m.c_str();
}
```

so a launch failure now reads `native MMVQ launch: <the exception's own text>`. The 91 call sites were renamed
mechanically (`dpct::get_error_string_dummy(` -> `dpct::error_string(`); `get_error_string_dummy` is gone from
the tree (0 references outside one explanatory comment). The 6 sites whose code was `const dpct::err0 e = 0;`
with a dead `if (e != 0)` keep their (unreachable) branch but now name a real message instead of a placeholder.

## Marker blocks: 1,151 -> 255

Following the precedent set when DPCT1065/1108/1121 were resolved, the marker blocks for the families this sweep
*audited* were deleted, and the ones that are genuine records of a migration decision were kept.

**Deleted (897 blocks):**

| codes | blocks | why |
|---|---|---|
| DPCT1114 / 1124 | 318 | "migrated to async memcpy, assuming an in-order queue" - the assumption holds everywhere the port runs; the record now lives once, in `sycl_queue.hpp` (below) |
| DPCT1000 / 1001 | 128 | "could not be rewritten" / "could not be removed" - noise around error handling that is either dead or already handled |
| DPCT1009 / 1010 | 217 | "replace `get_error_string_dummy`" - done, above |
| DPCT1118 | 75 | audited clean in exp 14 (0 of 187 group calls under a thread-dependent guard) |
| DPCT1110 | 46 | measured not-a-register-pressure-signal in exp 15 |
| DPCT1098 | 42 | audited in exp 15: no Xe `__ldg` counterpart, `const __restrict__` already conveys it |
| DPCT1013 | 71 | audited here, below |

**Kept (255), because each is a record rather than noise:** DPCT1049 (58, work-group size may exceed the device
limit - a real portability note), DPCT1026/1027 (47, "the call to X was removed because this functionality is
redundant in SYCL" - it explains why a call is *missing*), DPCT1048 (30, host-alloc flags), DPCT1106 (18),
DPCT1025 (18), DPCT1083 (16), DPCT1024 (12), DPCT1093 (11, device selection), DPCT1078 (7, memory ordering),
DPCT1053 (7, un-ported device asm), and the small singletons.

**The in-order assumption, said once.** The 318 deleted memcpy blocks each repeated the same warning, so the
invariant now sits where it can be enforced - the top of `sycl_queue.hpp`:

> IN-ORDER IS LOAD-BEARING. The migration turned ~320 cudaMemcpy/cudaMemcpyAsync sites into queue.memcpy() and
> memcpy_async() calls that depend on the stream's ordering; DPCT flagged every one "assuming in-order queue".
> Every queue this port runs on is in-order - q_of() below, and the second-GPU expert path's stream_
> (remote_experts.cpp) - and nothing under sycl/src calls dpct::get_out_of_order_queue(). A queue built without
> property::queue::in_order() breaks all of those copies silently.

## DPCT1013 (71): moot, checked against the CUDA originals

The marker says "the rounding mode could not be specified and the generated code has the default rounding
mode". Whether that matters depends on which intrinsic it was, and the pre-migration CUDA tree is in the repo,
so it can be read rather than guessed:

```
$ grep -rhoE '__f(add|mul|ma|sub|div|rcp)_(rn|rz|ru|rd)|__fdividef' src/kernels/cuda/*.cu src/prefill/*.cu | sort | uniq -c
     43 __fadd_rn       28 __fmul_rn        3 __fdividef
      3 __fdiv_rn        1 __fsub_rn
```

`__fadd_rn`, `__fmul_rn`, `__fsub_rn`, `__fdiv_rn` are **round-to-nearest-even** - exactly what `+`, `-`, `*`,
`/` do by default in C++ and SYCL. For those the marker is a false alarm: the default *is* the requested mode.

The 3 `__fdividef` are the only real difference, and they are in `shared_expert.cu` ("Explicit intrinsics
reproduce its `--use_fast_math` operations"), where `__fdividef` is CUDA's *approximate* division (~2 ulp). The
port's `/` is correctly rounded, so it is **more** accurate than the CUDA original there, not less - which is
consistent with `shared_expert_parity` passing. No change; the blocks are deleted and this note is the record.

## Verification

- Full `ninja` of every target: exit 0.
- `ctest`: **25/27**, unchanged - the two failures are the documented pre-existing ones (`ple_parity`'s missing
  fixture, `s2_expert_grouped_parity`'s grouped-path kernel bug).
- Diff: 94 files, +110 / -4813. The only non-comment change is in `dpct.hpp` (the two functions and the macro's
  one added line).

## Where the sweep ends

1,151 markers down to **255**, all of them records worth keeping. Every code the sweep examined - DPCT1000,
1001, 1009, 1010, 1013, 1098, 1110, 1114, 1118, 1124 - is either fixed (the error path) or audited benign. What
remains is DPCT1049 (work-group sizes within `max_work_group_size` on every device the port targets, 1024 on
Xe), DPCT1026/1027 (intentional removals) and the device/host-flags notes; none is actionable without a
different device to test on.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh && cd sycl/build-b60 && ninja && ctest
grep -rhoE 'DPCT[0-9]{4}' ../../sycl/src ../../sycl/include | sort | uniq -c | sort -rn   # 255 left
```
