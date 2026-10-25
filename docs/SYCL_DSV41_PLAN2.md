# SYCL plan for the DeepSeek engine (dsv41), round 2: what is left

Round 1 (see [SYCL_DSV41_PLAN.md](SYCL_DSV41_PLAN.md)) moved the quant kernels: `tq`, `gguf` and `iq` compile for
SYCL behind the opt-in `STRATA_SYCL_DSV41`. The measured rows are in
[SYCL_DSV41_PLAN.md](SYCL_DSV41_PLAN.md): on this PC's Intel Arc Pro B60 the parity harness matches the CPU
reference — `MATCH: 9945/13312 BF16 bits equal to the CPU reference, 3367 within one ulp, max float diff 0.000183105`
on the scalar artifact and `MATCH: 9958/13312 BF16 bits equal to the CPU reference, 3354 within one ulp, max float diff 0.000213623`
on the bmg-g21 AOT tile artifact (both measured on this PC). (An earlier round-2 run, when the harness covered
only the four decode rows, printed `MATCH: 4096/4096 BF16 bits equal to the CPU reference, 0 within one ulp, max
float diff 0` — measured on this PC.) The MXFP4 set tiles now use `joint_matrix`; the scalar fallback is kept
(measured on this PC). The IQ4_XS set tiles match too, after the round-2 defect fix — the missing barrier
between the W/X staging and the XMX B-operand pack in `sycl/src/dsv41/ds4_iq.dp.cpp:1114-1115`, described under
Phase A (measured on this PC). The engine cannot run yet.
This plan covers the rest, shaped by a research round over the local trees on this machine: `~/XeStrata` (the
Strata Xe port), Strata's own `sycl/` project, `~/llama.cpp`, and the two SYCL toolchains.

Every item below carries one of the four tags defined in [SYCL_DSV41_PLAN.md](SYCL_DSV41_PLAN.md):
`(measured on this PC)`, `(static reading)`, `(prior art)` or `(not measured yet)`.

## What this round found

**The matrix path works on this machine's chip.** A small `joint_matrix` probe (fp16 A 8x16 row-major, fp16 B
16x16 col-major, float accumulator 8x16, sub-group 16, `grf_size<256>`) compiled green — exit 0 — four ways on
this PC on 2026-10-24: oneAPI icpx 2026.2 JIT, icpx AOT for `bmg-g21`, self-built intel/llvm DPC++ 7.2.0 JIT,
and that one AOT for `bmg-g21` (prior art: the 2026-10-24 research-round record; not measured in this round's
evidence set). The stronger proof is measured on this PC: the `joint_matrix` set tiles built ahead of time for
`bmg-g21` run on the B60 and their rows match the CPU reference (the set rows in
[SYCL_DSV41_PLAN.md](SYCL_DSV41_PLAN.md); measured on this PC). The runtime header has no compile-time
specialization for `bmg_g21` (only DG2 and PVC) (static reading); kernels ask the device at run time instead,
the way XeStrata does.

**There is a pattern to copy, already in this repo.** Everything in this paragraph is `(static reading)` — read
from code, not measured here. `sycl/src/prefill/xmx_moe.dp.cpp` (lines 64-113) is the canonical shape: float
accumulator, `joint_matrix_load` from shared local memory padded by 8, K-steps of 16, `joint_matrix_mad`, store
to global with row-major layout, `grf_size<256>` properties, `[[sycl::reqd_sub_group_size(16)]]` under
`STRATA_XMX_SG16` (set by CMake when `STRATA_SYCL_AOT` is `bmg-*` or `lnl*`, `sycl/CMakeLists.txt` lines 40-41).
On Xe2 its work-groups are 128 lanes (8 sub-groups of 16). `sycl/src/kernels/cuda/iq_kernels.dp.cpp` adds
`ext_intel_packed` for packed weight tiles; `sycl/src/kernels/xmx_int8_bench.cpp` shows int8 tiles of K=32 with
int32 accumulators and a `joint_matrix_apply` step for per-block scales — that is how MXFP4/IQ scales fold in.

**XeStrata proves the same shapes in a shipping engine.** Static reading — read from that tree's code, not
measured here. Its hand-written kernels live in `~/XeStrata/src/kernels/xe/`: `xmx_gemm.cpp` (fp16/bf16, 8x16x16,
SG16, GRF256, AOT for `bmg_g21`), `iq_mmq.cpp` (int8, 8x16x32 on Xe2, grouped by expert with row-tile choice from
`max_rows`/`mean_rows`, DP4A fallback), `qsa_prompt_attn.cpp` (fp16, 16x16x16, SG16), and a per-device
capability gate built from the runtime matrix combinations query, with `STRATA_NO_BF16_MMA` /
`STRATA_NO_INT8_MMA` env switches (all static reading). It gates capability checks before launching and falls
back when the device lacks a combination (static reading).

Carried prior art, not measurements of this port: XeStrata's own `--hw-bench` produced these on this PC (AMD
Ryzen 7 5700X3D with 2× Intel Arc Pro B60) on 2026-10-17, in the contrib-llvm build xe0.1.44.1-1:

- kernel launch 5.9 µs (prior art: measured on this PC by XeStrata `--hw-bench` on 2026-10-17, contrib-llvm
  build xe0.1.44.1-1)
- 64 KiB copy 11.1 µs (prior art: same PC, same run, same build and date)
- host to device 14.44 GB/s (prior art: same PC, same run, same build and date)
- device to host 14.30 GB/s (prior art: same PC, same run, same build and date)
- two GPUs reading at once 14.2 GB/s per card (prior art: same PC, same run, same build and date)

Each number comes from XeStrata's tree with that build and that date. None of them was produced by a run in this
session, and none is a measurement of Strata's dsv41 engine on the B60. No tree we read carries a dsv41 speed
number for the B60; every speed number for this port must come from a future run on this PC (not measured yet).

**The IQ4_NL kernel we refused had a drop-in template.** Static reading — read from llama.cpp's code, not
measured here. llama.cpp (MIT) already runs IQ4_NL on SYCL: `~/llama.cpp/ggml/src/ggml-sycl/dequantize.hpp` lines
1588-1605, 32 work-items per 256-value group, 8 sub-blocks of 32 values, one fp16 scale per sub-block, table
lookup, no shared memory, no subgroup intrinsics — embarrassingly parallel (static reading). Its `kvalues_iq4nl`
table is the same 16 values as Strata's (`-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113`; read from
both trees, at `~/llama.cpp/ggml/src/ggml-common.h:1120` and `sycl/src/dsv41/ds4_parity.cpp:118-119`; static
reading). Q4_K/Q5_K/Q6_K/IQ4_XS there match the same GGUF layout Strata uses, and Strata's converted kernels
cover those four through `dq_rows` (static reading). IQ4_NL is no longer missing: it has its own kernel now, and
its row is `IQ4_NL: 1024/1024 bits exact, 0 within one BF16 ulp, 0 outside` on both artifacts (measured on this
PC). Read that row with the coverage caveat in [SYCL_DSV41_PLAN.md](SYCL_DSV41_PLAN.md): the harness gives
IQ4_NL only 4×32 random values per run (static reading). llama.cpp's MXFP4 path (`dequantize_block_mxfp4` plus
an `e8m0 → float` step that shifts the byte 23 bits) matches our E4M3/E8M0 decode approach (static reading).
Nothing in ggml-sycl covers Strata's trellis/APR format; that stays in-house (static reading). Its dual-GPU
design splits tensors by rows across devices outside the kernels; all kernel offsets are 64-bit (static
reading).

**One finding to carry: the set tiles refuse IQ4_NL, on purpose.** Static reading, not a measured result
(unconfirmed static analysis). At the merged commit `iq::set_gate_up` and `iq::set_down` return false for
`kIQ4_NL` before any tile work (`sycl/src/dsv41/ds4_iq.dp.cpp:2222` and `:2239`, both lines added by the round-2
fix commit `a8ced727`), because IQ4_NL's blocks are 32 values and not the 16-value `dq_rows` shape (the `deq8`
32-value block shape at `sycl/src/dsv41/ds4_iq.dp.cpp:435-438` and the `DS4_IQ_SWITCH` dispatch at line 1517).
The code's own comment says why: no set-tile parity row drives IQ4_NL yet. The caller turns a refusal into
`fail("set_gate_up: not covered")` or `fail("set_down: not covered")` (`sycl/src/dsv41/ds4_gguf.dp.cpp:2111` and
`:2173`), so IQ4_NL never reaches the joint-matrix set tiles. Keep the refusal until a set-tile parity row
drives IQ4_NL; only that run should re-open this route.

**The Engram is not optional for serving — static reading, not a measured result.** The four IQ2_XXS shards
alone are a supported launcher input: `strata launch --plan` accepts a shards-only folder and names the set
`DeepSeek-V4.1-Flash IST-DAS-2.00bpw` (static reading). But a launch that serves with the Engram absent
everywhere is not a configuration this code can start: the pack recipe requires the Engram tables
(`tools/dsv41/pack_gguf.py:329-340`) and the serving path reads them from disk
(`strata_control/dsv41_fit.py:185`; both static reading). Keep the Engram on disk beside the shards; it costs
SSD space, not RAM.

**The BLAS migration is small and has a precedent.** Static reading — read from code, not measured here.
`src/dsv41/ds4_model.cu` has 3 cublas calls (2 `cublasGemmEx` fp16→fp32 at lines 2910/2924, 1 `cublasSgemm` fp32
at 2937). `src/dsv41/ds4_vision.cu` has 10 more gemm call sites: 7 go through one `gemm()` wrapper at line 185
(whose own cublas call is at 188) and 2 are direct attention calls at 470/475. That is 13 call sites in total,
all plain gemms with transposes — no cublasLt, no fused epilogues (all static reading). oneMKL's SYCL `gemm` and
`gemm_batch` cover them 1:1 (type combinations float, half→float, bf16→float read from
`/opt/intel/oneapi/mkl/latest/include/oneapi/mkl/blas/`; static reading). Strata already has two precedents:
`include/strata/hip_compat/cublas_v2.h` (a macro shim that makes CUDA cublas names compile on HIP) and
`sycl/src/prefill/gemm.dp.cpp` (`dpct::blas::gemm` over oneMKL, with a 4 GiB operand staging workaround; both
static reading). `cublasSetWorkspace` has no SYCL equivalent — dpct itself marks it redundant; oneMKL manages
its workspace (static reading). The router gemv kernels (`gemv_f16`, `gemv_f32`, `gemv_f16_rows` at
`src/dsv41/ds4_kernels.cu:1921-1931`) and the vision softmax (`src/dsv41/ds4_vision.cu:126`) are custom kernels,
not BLAS calls; they need SYCL ports, not library mapping (static reading).

**Toolchain choice.** oneAPI 2026.2 stays the build for dsv41 SYCL: `icpx` prints 2026.2.0 (measured on this
PC), the oneMKL headers live in that toolkit, and the current target compiles green with it (measured on this
PC: the runs above). The self-built intel/llvm tree (DPC++ 7.2.0) compiled the matrix probe too and carries
XeStrata patches, but has no oneMKL, so it is an option for AOT builds, not the default (prior art: the
research-round record; not measured here). The AOT device name on this machine is `bmg-g21`: the tile artifact
was built with it and ran on the B60 (measured on this PC). The alias spelling `bmg_g21` appears in the driver,
and PCI addresses change between boots, so device selection must re-read PCI each start (static reading).

**Engine hygiene — this plan's own guidance, not a measurement.** No run in the round-2 evidence set stops the
engine or resets the GPU (not measured here). Treat this as guidance from the XeStrata notes to confirm on this
hardware: stop the engine with SIGTERM and wait for it to exit. Killing the engine mid-cleanup has been reported
to leave the GPU in a `PAGEFAULT → CAT error → DEVICE_LOST` state that blocks later starts. GPU reset needs root.
Use the probe pattern (`sycl-ls` plus a small device-info program) before blaming a kernel (all static reading;
not measured here).

## The phases, in order

Each phase lands as its own commit, keeps `STRATA_SYCL_DSV41` opt-in, and keeps the default CUDA/HIP path
byte-identical. The parity harness (`sycl/src/dsv41/ds4_parity.cpp`) is the gate for every kernel phase; new
numbers only from runs on this PC.

**Phase A — matrix tiles in the two tensor-core kernels.** Rewrite `mx_set_tc_kernel`
(`sycl/src/dsv41/ds4_gguf.dp.cpp:1063`) and `set_tc_kernel` (`sycl/src/dsv41/ds4_iq.dp.cpp:1023`) with
`joint_matrix` following `sycl/src/prefill/xmx_moe.dp.cpp`: dequant to fp16 into padded SLM, SG16 tiles under
`STRATA_XMX_SG16`, float accumulators, scales folded with `joint_matrix_apply` (as the int8 bench does; the
shape notes are static reading). Status: the MXFP4 and IQ4_XS set tiles are in and measured on this PC (the
set rows in [SYCL_DSV41_PLAN.md](SYCL_DSV41_PLAN.md), on the B60). The IQ4_XS tiles first matched after the round-2 defect
fix: the pack step read W/X rows staged by other threads of the same 32-value step while the only barrier sat
after the XMX B-operand pack, so staging and pack were unsynchronized and the tile's output moved between runs
until the missing barrier was added at `sycl/src/dsv41/ds4_iq.dp.cpp:1114-1115` (static reading of the code; the
moving output and the matching rows are measured on this PC). Open here: the timed run — scalar against tiles,
on the B60, both reported with the build that produced them — has not happened (not measured yet).

**Phase B — IQ4_NL.** Add a dedicated kernel for the 32-value block shape, from the llama.cpp template with
Strata's BF16 output, and drop the refusal in the `dq_rows` path (the `fail(...)` at
`sycl/src/dsv41/ds4_gguf.dp.cpp:1749`; the drop is static reading). Gate: parity against the CPU reference for
IQ4_NL rows on device. Measured on this PC: `IQ4_NL: 1024/1024 bits exact, 0 within one BF16 ulp, 0 outside` on
both artifacts, with the coverage caveat above (4×32 random values per run).

**Phase C — BLAS and helpers.** Merged and measured. `b422876c1d888c7a0b1eddb312a08d6dcd517298` ("[Engine]
cublas shim and gemv parity for dsv41 SYCL") landed the whole phase: a SYCL cublas shim header of its own at
`sycl/include/compat/cublas_v2.h`, mirroring `include/strata/hip_compat/cublas_v2.h`: `cublasGemmEx` →
`dpct::blas::gemm` (oneMKL), handle → descriptor, `SetStream` → queue, `SetWorkspace` → no-op (static reading);
the three gemv kernels and the vision softmax as SYCL kernels (`sycl/src/dsv41/ds4_gemv.dp.cpp`,
`sycl/src/dsv41/ds4_gemv.dp.hpp`, `sycl/src/dsv41/ds4_softmax.dp.cpp`); and the BLAS parity harness
(`sycl/src/dsv41/ds4_blas_parity.cpp`). The softmax port needed one device-side fix at
`sycl/src/dsv41/ds4_softmax.dp.cpp:87` — the fix record is `task-4-gemv-parity.log:104`
(`SOFTMAX_FIX_SITE=sycl/src/dsv41/ds4_softmax.dp.cpp:87 SOFTMAX_DEFECT_SIDE=device`). The gate ran on the card
and passed (measured on this PC): `RUN_RC=0` (`task-4-blas-shim.log:385`), `TILE_RUN_RC=0`
(`task-4-blas-shim.log:410`), and `CTEST_RC=0` with 2/2 tests passed on
`device: Intel(R) Arc(TM) Pro B60 Graphics` (`task-4-blas-shim.log:507-511`); the gemv parity row reads
`RUN1_MATCH: 50937/51242 values equal to the CPU reference, 305 within one BF16 ulp, max float diff 3.82063e-06` (`task-4-blas-shim.log:392`). This phase touches no engine code. Open here: the timed run — these
kernels against CPU and the tiles against the scalar loop, on the B60, reported with the build that produced
them — has not happened, so this plan states no throughput number for them.

**Phase D — engine wiring.** Build `src/dsv41/ds4_model.cu` for SYCL behind the same opt-in flag, run one small
DeepSeek pack on one B60, and measure prefill and decode throughput, reported with its build and its date.
There is no B60 baseline: that run produces the first number (not measured yet). Memory is not the blocker
the round-1 draft feared: the fit planner says the set `fits` with room
to spare, and the Engram stays on SSD (measured on this PC; the numbers are in
[SYCL_DSV41_PLAN.md](SYCL_DSV41_PLAN.md)). The blocker is the one measured there too: the documented
`strata launch --plan` path reports verdict `no_card` on this PC (measured on this PC), so that path cannot place
the model on these cards yet. Follow the SIGTERM/PCI guidance above when stopping.

**Phase E — tuning and reach.** Occupancy: replace the current one-work-group-per-launch default with measured
work-group sizes (start from 128 lanes as `xmx_moe` uses, tune with timings; the 128-lane start is static
reading). Then dual-GPU expert placement using peer access, with the `peer_experts` stub in the port as the hook
(static reading). Measure peer copy against a host relay on this hardware before choosing between them (not
measured yet); this plan states no speed number for this port. Vision, exl3 and sampler follow as separate
increments (not measured yet).

## Limits of this plan

- Measured on this PC in round 2 (the rows and facts in [SYCL_DSV41_PLAN.md](SYCL_DSV41_PLAN.md)'s measured
  section): the parity rows above, the toolchain probes, the default-off build check, and the fit measurement
  (verdict `fits`, with the `no_card` gap in the documented path).
- `(static reading)` — unconfirmed static analysis, read from code or from other projects' files, not run: the
  oneMKL type combinations, the llama.cpp kernel shapes and line citations, the cublas inventory, the XeStrata
  kernel shapes, the runtime-header observation, the IQ4_NL tile-path finding above, the catalog reading behind
  `no_card`, and the Engram findings above.
- `(prior art)` — reported from earlier runs on this PC, not in round 2's evidence set: the four-way probe
  compiles (2026-10-24) and the round-1 dpct conversion record (2026-10-24), and the XeStrata `--hw-bench`
  numbers (2026-10-17) listed above.
- The engine-stop guidance (SIGTERM and wait; the `PAGEFAULT → CAT error → DEVICE_LOST` report) is this plan's
  own guidance from the XeStrata notes (static reading; not measured here). No round-2 run confirms it.
- The `joint_matrix` set tiles are confirmed on the device (measured on this PC): the tile artifact's rows ran
  on the B60, and the IQ4_XS set rows match after the round-2 barrier fix. What is left of Phase A is the timed
  run (not measured yet).
- Phase D produces this port's first speed number (not measured yet).
