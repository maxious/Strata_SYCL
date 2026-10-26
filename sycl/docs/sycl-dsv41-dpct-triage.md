# The DPCT conversion notes in the SYCL dsv41 port

The dsv41 SYCL port was made with dpct. dpct leaves a comment wherever it had to
guess. This file is the triage of those comments. Each DPCT id gets one row: how
many notes it has, where they sit, what the class means, and what happens to it.
This file changes no code. It is the review record.

## Where the numbers come from

The counts are measured. On 2026-10-26, on branch `dsv41-dpct-w2` at `18a27e02`,
this command lists every note:

```sh
grep -rn "DPCT" sycl/src/dsv41
```

It prints 1046 lines. 776 lines carry a conversion id. The other 270 lines are
the rest of the comment blocks, `DPCT_CHECK_ERROR` calls, and two macro
definitions (`sycl/src/dsv41/ds4_kernels.dp.hpp` and `sycl/src/dsv41/ds4_mem.cpp`).
Those lines are code or comment text, so they get no row. The ids sit in 12
files. In the table, the count column is the number of grep lines that carry the
id, and the files column lists paths relative to `sycl/src/dsv41`.

## The classes

- **RESOLVED** - the code already handles the note. The row points at the line
  that does it.
- **ACTIONABLE** - the note needs a code fix. A later todo owns it and the row
  names it.
- **ACCEPTED_AS_IS** - no fix is wanted here. The row says why.
- **NOT_APPLICABLE** - the note is dpct boilerplate for a CUDA idea that this
  code path does not use.

The class decisions are static reading of the code at the cited lines
(unconfirmed static analysis). The plan behind this work settles three of them:
DPCT1098 is fixed by todo 3, DPCT1093 is fixed by todo 4, and DPCT1065 stays as
it is because a blanket barrier narrowing is a regression hazard.

## The table

| id | count | files touched | classification | reason | owner todo |
| --- | --- | --- | --- | --- | --- |
| DPCT1065 | 156 | ds4_attn_split.dp.hpp (4), ds4_gguf.dp.cpp (12), ds4_iq.dp.cpp (15), ds4_kernels.dp.cpp (52), ds4_tq.dp.cpp (7), ds4_vision.dp.cpp (3), vendor/exl3.dp.cpp (19), vendor/exl3_legacy.dp.cpp (22), vendor/sampler.dp.cpp (22) | ACCEPTED_AS_IS | The note suggests a local-only barrier as a perf hint. Some of these barriers fence host-mapped flag traffic, for example in xfer_put_kernel and xfer_get_kernel (sycl/src/dsv41/ds4_kernels.dp.cpp:2236 and :2267). A blanket narrowing is a regression hazard, so the full barriers stay. | - |
| DPCT1093 | 107 | ds4_model.dp.cpp (103), ds4_vision.dp.cpp (4) | ACTIONABLE | The note asks whether the selected device is the one meant, at every select_device and queue call. Todo 4 checks each queue-creation site against the thread that owns its device and fixes the ones that are wrong (for example sycl/src/dsv41/ds4_model.dp.cpp:2068-2077). | todo 4 |
| DPCT1124 | 101 | ds4_model.dp.cpp (98), ds4_vision.dp.cpp (3) | ACCEPTED_AS_IS | The note asks for a wait after a memcpy. Fixups rule 7d puts .wait() back on every unwaited default-queue copy (sycl/tools/fixups.py:156), and no async_memcpy site remains in sycl/src/dsv41 (measured by grep 2026-10-26). The plan accepts this class. | - |
| DPCT1118 | 83 | ds4_gguf.dp.cpp (8), ds4_iq.dp.cpp (9), ds4_kernels.dp.cpp (28), ds4_tq.dp.cpp (5), vendor/exl3.dp.cpp (7), vendor/exl3_legacy.dp.cpp (9), vendor/sampler.dp.cpp (17) | ACCEPTED_AS_IS | A convergence hint on group functions. The plan accepts this class and forbids a speculative restructure. A change needs a failing case or a measurement first. | - |
| DPCT1101 | 60 | ds4_attn_split.dp.hpp (1), ds4_gguf.dp.cpp (12), ds4_iq.dp.cpp (1), ds4_kernels.dp.cpp (20), vendor/exl3.dp.cpp (6), vendor/exl3_legacy.dp.cpp (7), vendor/moe_win.dp.cpp (4), vendor/sampler.dp.cpp (9) | ACCEPTED_AS_IS | dpct folded compile-time expressions to values and kept each expression in the comment. For example sycl/src/dsv41/ds4_kernels.dp.cpp:3238 keeps "kThreads / 32" beside its value 8, and kThreads is 256 at :29. The ds4 launch parity rows run these launches. | - |
| DPCT1110 | 45 | ds4_attn_split.dp.hpp (1), ds4_gguf.dp.cpp (7), ds4_iq.dp.cpp (8), ds4_kernels.dp.cpp (9), ds4_model.dp.cpp (8), ds4_tq.dp.cpp (4), ds4_vision.dp.cpp (1), vendor/exl3.dp.cpp (4), vendor/exl3_legacy.dp.cpp (2), vendor/sampler.dp.cpp (1) | ACCEPTED_AS_IS | A register-pressure hint. The plan forbids a speculative rewrite. A change needs a measurement first. | - |
| DPCT1013 | 39 | ds4_gguf.dp.cpp (4), ds4_iq.dp.cpp (5), ds4_kernels.dp.cpp (13), ds4_q6k.dp.hpp (1), ds4_tq.dp.cpp (1), ds4_vision.dp.cpp (1), vendor/exl3.dp.cpp (7), vendor/exl3_legacy.dp.cpp (7) | ACCEPTED_AS_IS | SYCL rounds like the CUDA default (round to nearest even). A bit-exact sycl::exp against expf is recorded at sycl/src/dsv41/ds4_gguf.dp.cpp:241-242 (measured on the B60 by the ds4_parity probe, 2026-10-25). The plan accepts this class. | - |
| DPCT1078 | 28 | ds4_gguf.dp.cpp (1), ds4_kernels.dp.cpp (14), vendor/exl3.dp.cpp (7), vendor/exl3_legacy.dp.cpp (6) | ACCEPTED_AS_IS | acq_rel matches the __threadfence() and __threadfence_system() intent of the original code (sycl/src/dsv41/ds4_kernels.dp.cpp:494). The handshake flags already take the uncached strata::sys_store and strata::sys_load path. The plan accepts this class. | - |
| DPCT1010 | 21 | ds4_gguf.dp.cpp (1), ds4_iq.dp.cpp (1), ds4_model.dp.cpp (12), ds4_vision.dp.cpp (1), vendor/exl3.dp.cpp (1), vendor/exl3_legacy.dp.cpp (2), vendor/moe_win.dp.cpp (1), vendor/sampler.dp.cpp (2) | NOT_APPLICABLE | Exception-versus-error-code boilerplate. The port keeps error codes on purpose through its launch-check pattern (sycl/src/dsv41/vendor/sampler.dp.cpp:913-919). SYCL queues throw on error, so the rewrite does not apply. | - |
| DPCT1024 | 17 | ds4_model.dp.cpp (16), ds4_vision.dp.cpp (1) | ACCEPTED_AS_IS | The note asks what consumes the old error code. Every site keeps it through DPCT_CHECK_ERROR (sycl/src/dsv41/ds4_model.dp.cpp:2625-2632). The plan accepts this class. | - |
| DPCT1049 | 15 | ds4_kernels.dp.cpp (9), ds4_vision.dp.cpp (2), vendor/sampler.dp.cpp (4) | ACCEPTED_AS_IS | A work-group size hint on the migrated launches. The plan forbids a speculative resize. A change needs a measurement first. | - |
| DPCT1048 | 13 | ds4_model.dp.cpp (12), ds4_vision.dp.cpp (1) | RESOLVED | The CUDA pinning flags are gone, and each site is allocated for its own use. The polled ring is uncached through strata::host_malloc_polled (sycl/src/dsv41/ds4_model.dp.cpp:3035) as fixups rule 7e directs (sycl/tools/fixups.py:366-373); buffers a memcpy touches stay USM on purpose. | - |
| DPCT1009 | 13 | ds4_gguf.dp.cpp (1), ds4_model.dp.cpp (8), ds4_vision.dp.cpp (1), vendor/exl3.dp.cpp (1), vendor/exl3_legacy.dp.cpp (1), vendor/moe_win.dp.cpp (1) | NOT_APPLICABLE | Exception-versus-error-code boilerplate on the same launch-check pattern (sycl/src/dsv41/vendor/exl3.dp.cpp:86-91). The port keeps error strings and codes. | - |
| DPCT1119 | 10 | vendor/exl3.dp.cpp (5), vendor/exl3_legacy.dp.cpp (5) | NOT_APPLICABLE | SYCL has no stream capture. The guard these notes sit on can never fire (sycl/src/dsv41/vendor/exl3_legacy.dp.cpp:1285-1286). The CUDA graph APIs do not apply to this code path. | - |
| DPCT1026 | 8 | ds4_kernels.dp.cpp (1), ds4_model.dp.cpp (4), vendor/exl3_legacy.dp.cpp (3) | NOT_APPLICABLE | The removed calls are CUDA-only. cudaGetLastError is redundant in SYCL, and cudaFuncSetAttribute has no SYCL setting (sycl/src/dsv41/vendor/exl3_legacy.dp.cpp:1138-1143). Local accessors size the shared memory instead. | - |
| DPCT1017 | 8 | ds4_gguf.dp.cpp (8) | ACCEPTED_AS_IS | dpct::funnelshift_r is the documented translation of __funnelshift_r in the Q6 dequant (sycl/src/dsv41/ds4_gguf.dp.cpp:417). The ds4_gguf parity rows cover the path. A change needs a measured divergence. | - |
| DPCT1001 | 7 | ds4_gguf.dp.cpp (1), ds4_model.dp.cpp (2), ds4_vision.dp.cpp (1), vendor/exl3.dp.cpp (1), vendor/exl3_legacy.dp.cpp (1), vendor/moe_win.dp.cpp (1) | NOT_APPLICABLE | The statement dpct could not remove is the port's own launch-error counter (sycl/src/dsv41/vendor/exl3.dp.cpp:95). The port wants that counter. | - |
| DPCT1106 | 6 | ds4_model.dp.cpp (6) | RESOLVED | The card-memory query is the port's own mirror: a sysman path with the SYCL API as fallback (sycl/src/dsv41/ds4_mem.cpp:99-104), called with attribution from sycl/src/dsv41/ds4_model.dp.cpp:2088. The SYCL dsv41 target builds only with oneAPI, where the extension exists. | - |
| DPCT1000 | 6 | ds4_gguf.dp.cpp (1), ds4_model.dp.cpp (1), ds4_vision.dp.cpp (1), vendor/exl3.dp.cpp (1), vendor/exl3_legacy.dp.cpp (1), vendor/moe_win.dp.cpp (1) | NOT_APPLICABLE | Error-handling boilerplate on the launch-check pattern the port keeps on purpose (sycl/src/dsv41/vendor/exl3.dp.cpp:74-95). The exception rewrite does not apply. | - |
| DPCT1098 | 5 | ds4_gguf.dp.cpp (2), ds4_kernels.dp.cpp (2), vendor/exl3.dp.cpp (1) | ACTIONABLE | A plain * load of a host-mapped payload can read a cached line and miss the host's store. Todo 3 routes every host-mapped payload read through the uncached path and records the one site that reads device-resident memory (sycl/src/dsv41/ds4_kernels.dp.cpp:1863). | todo 3 |
| DPCT1083 | 5 | ds4_iq.dp.cpp (1), ds4_kernels.dp.cpp (3), vendor/sampler.dp.cpp (1) | ACCEPTED_AS_IS | The local-memory sizes come from the port's own formulas at each launch (sycl/src/dsv41/ds4_kernels.dp.cpp:3425) and the ds4 launch parity rows run them. A change needs a measured divergence. | - |
| DPCT1027 | 5 | ds4_model.dp.cpp (4), ds4_vision.dp.cpp (1) | NOT_APPLICABLE | cublasSetWorkspace is redundant in SYCL, and SYCL has no host-memory registration (sycl/src/dsv41/ds4_model.dp.cpp:5298-5299). These CUDA calls do not exist in this code path. | - |
| DPCT1025 | 5 | ds4_model.dp.cpp (3), ds4_vision.dp.cpp (2) | NOT_APPLICABLE | The CUDA flag and priority options have no SYCL counterpart; the port creates plain in-order queues at these calls. The thread-binding question at the same calls is the DPCT1093 row. | - |
| DPCT1008 | 4 | ds4_kernels.dp.cpp (2), vendor/exl3.dp.cpp (2) | RESOLVED | __nanosleep was a spin hint only. The sync it sat beside is already carried, by the barrier at sycl/src/dsv41/ds4_kernels.dp.cpp:1839 and the fence at sycl/src/dsv41/vendor/exl3.dp.cpp:1013. | - |
| DPCT1007 | 2 | vendor/exl3.dp.cpp (2) | ACCEPTED_AS_IS | cudaOccupancyMaxActiveBlocksPerMultiprocessor has no SYCL equivalent. n = 2 is the fallback the CUDA path itself used, recorded at sycl/src/dsv41/vendor/exl3.dp.cpp:1951-1953. The plan keeps this site; a change needs a measurement through STRATA_GLM_EXL3_DECODE. | - |
| DPCT1006 | 2 | ds4_model.dp.cpp (2) | ACCEPTED_AS_IS | The query only warns about integrated GPUs. Both sites guard it so a runtime without it skips the warning (sycl/src/dsv41/ds4_model.dp.cpp:2249-2252 and :5281-5286). The Intel target has the query. | - |
| DPCT1072 | 2 | ds4_tq.dp.cpp (1), ds4_gguf.dp.cpp (1) | RESOLVED | The templated launches dpct left alone were converted by hand into wrapper classes submitted as nd_ranges (sycl/src/dsv41/ds4_tq.dp.cpp:615-622 and sycl/src/dsv41/ds4_gguf.dp.cpp:1404-1408). | - |
| DPCT1113 | 1 | ds4_kernels.dp.cpp (1) | ACCEPTED_AS_IS | The note suggests widening a local-space barrier for multi-dimensional kernels. Barrier scope stays as it is; the plan forbids a barrier-scope change, and a change needs a measurement. | - |
| DPCT1019 | 1 | vendor/exl3_legacy.dp.cpp (1) | ACCEPTED_AS_IS | The SYCL local-memory size is used only as a fit threshold for the legacy family's shared-memory rungs (sycl/src/dsv41/vendor/exl3_legacy.dp.cpp:1346-1347). A smaller reading picks a smaller rung. | - |
| DPCT1002 | 1 | ds4_model.dp.cpp (1) | NOT_APPLICABLE | Exception-versus-error-code boilerplate on a special-case break in a read loop (sycl/src/dsv41/ds4_model.dp.cpp:3874-3875). The error-code flow stays. | - |

## Totals

30 ids, 776 notes.

| classification | ids | notes |
| --- | --- | --- |
| RESOLVED | 4 | 25 |
| ACTIONABLE | 2 | 112 |
| ACCEPTED_AS_IS | 15 | 563 |
| NOT_APPLICABLE | 9 | 76 |

Both columns add up: 4 + 2 + 15 + 9 = 30 ids, and 25 + 112 + 563 + 76 = 776
notes. The evidence log records the check.

## What happens next

Todo 3 audits every host-mapped payload read and routes it through the uncached
path. Todo 4 audits every queue-creation site against the thread that owns its
device. Every other row is finished with its note left in place.

## How this table is checked

The evidence log
`.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-2-dsv41-sycl-dpct-and-xpu-test.log`
records every command and its un-piped exit code. The checks are:

- every RESOLVED row names a real file and line, and that line resolves the note
  (`RESOLVED_CITE_RC`);
- the id count and the note count equal the sums across the classes
  (`ENUM_SUM_RC`);
- no reported exit code came through a pipe (`PIPE_AUDIT_RC`);
- the SYCL dsv41 target builds and `ctest -R ds4` passes (`SYCL_BUILD_RC` and
  `SYCL_CTEST_RC`).
