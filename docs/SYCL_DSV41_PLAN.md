# SYCL support for the DeepSeek engine (dsv41) — plan

This is the plan for running Strata's DeepSeek engine on Intel graphics cards. The engine today speaks two
languages: CUDA for NVIDIA cards and HIP for AMD cards. Intel cards speak a third language, SYCL. This branch adds
that third language for the engine's core parts, without changing anything for NVIDIA or AMD users.

## Who this is for

A person with an Intel card (an Arc Pro B60, or a future Arc card) who wants to serve a dsv41 model with Strata.
Nothing changes for anyone else: the CUDA and HIP paths stay exactly as they are. The SYCL path is opt-in, built
only when you ask for it.

## How to read the claims in this plan

Every item below says where it stands. Four tags are used:

- `(measured on this PC)` — the outcome of a run on this PC, named with its date, its build and its numbers.
- `(static reading)` — read from code or from another project's files only. Unconfirmed static analysis; not
  measured here.
- `(prior art)` — a number or a record carried from another project's own run. It keeps its source, its build
  and its date, and it is not a measurement of this port.
- `(not measured yet)` — no run exists, so no number exists yet.

This plan makes no speed claim of its own. Any speed numbers it quotes carry the `(prior art)` tag with their
source, build and date. No dsv41 tokens-per-second number has been measured on an Arc Pro B60 yet.

## What has been measured on this PC

The PC: AMD Ryzen 7 5700X3D with 2× Intel Arc Pro B60 (hardware, measured on this PC). `sycl-ls` prints
`Intel(R) Arc(TM) Pro B60 Graphics` twice, once per card (measured on this PC, 2026-10-25). The toolchain is
Intel oneAPI 2026.2 and `icpx` prints 2026.2.0 (measured on this PC). The compiler and `sycl-ls` are found only
after `source /opt/intel/oneapi/setvars.sh`; a bare probe finds nothing (exit 127 in these runs) and must not be
read as "not installed" (measured on this PC). oneMKL and oneDNN ship with the toolkit
(`/opt/intel/oneapi/mkl`, `/opt/intel/oneapi/dnnl`; measured on this PC by a byte test). This PC has no CUDA
toolkit: the compiler `nvcc` and the toolkit's install directory are both absent (measured on this PC).

The parity harness (`sycl/src/dsv41/ds4_parity.cpp`) runs the converted kernels on the card and compares the
output with a CPU reference written from the same layouts. Two builds ran on 2026-10-25 (measured on this PC):
one is the scalar artifact, and the other is compiled ahead of time for the bmg-g21 chip (Battlemage, the B60)
with the `joint_matrix` set tiles active. Both print `device: Intel(R) Arc(TM) Pro B60 Graphics`. The rows:

- scalar artifact (measured on this PC): `MATCH: 9945/13312 BF16 bits equal to the CPU reference, 3367 within one ulp, max float diff 0.000183105`
- bmg-g21 AOT tile artifact (measured on this PC): `MATCH: 9958/13312 BF16 bits equal to the CPU reference, 3354 within one ulp, max float diff 0.000213623`
- IQ4_NL on both artifacts (measured on this PC): `IQ4_NL: 1024/1024 bits exact, 0 within one BF16 ulp, 0 outside`
- MXFP4 set gate/up tiles on both artifacts (measured on this PC): `MXFP4 set gate/up tile: 4096/4096 fp16 bits exact, 0 within one BF16 ulp, 0 outside`
- MXFP4 set down tiles (measured on this PC): `742/4096 fp32 values exact, 3354 within one BF16 ulp, 0 outside` on the tile artifact
  and `729/4096 fp32 values exact, 3367 within one BF16 ulp, 0 outside` on the scalar artifact

Read the IQ4_NL row with one caveat. The harness's random-data coverage for IQ4_NL is 4×32 values per run
(static reading). Its `fill_block` helper (pre-existing, unchanged) writes only 18 bytes per IQ4_NL row: one
block of 32 values (static reading). The other 7 of 8 blocks per row are zero-filled on both sides, so most of
the 1024 values match as zeros. The row says the random part matches the CPU reference (measured on this PC).
It is not full-row coverage.

The gate is not vacuous: runs with deliberately corrupted inputs print NO MATCH, not MATCH (measured on this
PC).

One defect was found and fixed in round 2, and it is worth naming in plain words. The IQ4_XS expert-set tiles
compute each 32-value step in two stages: the threads first stage the weight and activation rows (the W/X
staging), then a pack loop fills the matrix unit's B operand from those rows (the XMX B-operand pack). The pack
reads rows that other threads of the same step stage, and the only barrier sat after the pack, so staging and
pack were unsynchronized (static reading). On the B60 the pack could pick up stale weights, and the tile's
output moved between runs until the missing barrier was added between the W/X staging and the XMX B-operand
pack at `sycl/src/dsv41/ds4_iq.dp.cpp:1114-1115` (the fix record cites this site as `:1114`; at the merged
commit the `item_ct1.barrier();` call is line 1115, under its comment at 1112-1114). The output moving before
the barrier and staying byte-identical across runs after it is measured on this PC. The same fix widened the
scalar stand-in's products to float (`sycl/src/dsv41/ds4_iq.dp.cpp:1159-1160`), so it computes the same exact
half-by-half products as the matrix tile and the CPU reference (static reading; the rows below are the
measurement). With the fix, the IQ4_XS set tiles match the CPU reference on both artifacts (measured on this
PC, 2026-10-25, the same two builds as above rebuilt with the barrier):

- IQ4_XS set gate/up tiles (measured on this PC): `8165/8192 fp16 bits exact, 27 within one BF16 ulp, 0 outside`
  on the scalar artifact, `8171/8192 fp16 bits exact, 21 within one BF16 ulp, 0 outside` on the bmg-g21 AOT tile
  artifact
- IQ4_XS set down tiles (measured on this PC): `940/8192 fp32 values exact, 7252 within one BF16 ulp, 0 outside`
  on the scalar artifact, `888/8192 fp32 values exact, 7304 within one BF16 ulp, 0 outside` on the tile artifact

One thing is still open: the timed run, scalar against tiles, has not happened (not measured yet).

Build facts, all measured on this PC in round 2:

- `STRATA_SYCL_DSV41` is OFF by default (`sycl/CMakeLists.txt:257`). With it off, the generated build lists 527
  targets and none of them names a dsv41 source: `strata_dsv41`, `src/dsv41`, `ds4_tq`, `ds4_gguf` and `ds4_iq`
  each match zero times in the target list (measured on this PC).
- The `-fp-model=precise` flag sits outside the `COMPILE_LANGUAGE:CXX` generator expression in
  `sycl/CMakeLists.txt` at lines 103-104. The build with GNU cc fails on it, not the configure: the configure
  step succeeds and the build fails on the first C file (measured on this PC: `CONFIGURE_RC=0`, `BUILD_RC=1`,
  `cc: error: unrecognized command-line option '-fp-model=precise'`). The flag must sit inside that generator
  expression. Until it moves, configure with `-DCMAKE_C_COMPILER=icx` so the C files go through Intel's
  compiler.
- The tile artifact was built with `-DSTRATA_XMX_SG16=1` and `-fsycl-targets=spir64_gen` for the `bmg-g21`
  device (measured on this PC). That build ran on the card and produced the tile rows above.

The model set on this PC fits in RAM (measured on this PC, 2026-10-25). The model directory
`/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO` is 232 GiB in total (`du -sh` prints 232G).
The four IQ2_XXS shards are 39.08, 39.39, 39.25 and 20.53 GB — 138.2 GB together — and
`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` is 110.59 GB (all sizes measured on this PC). The Engram is an SSD
file: it stays on disk and never enters the RAM math (measured on this PC; the fit arithmetic is the same with
and without it).

The fit planner — the same function the launcher uses, run on 2026-10-25 with the cards measured through clinfo
(2× Intel Arc Pro B60, 22.71 GiB of graphics memory per card) — gives the same arithmetic for all four variants
it was run for (full set, shards only, 4k, 8k): 90.1 GB of experts in RAM, 6.2 GB of dense weights (4.4 GB of
that in graphics memory) and 9.0 GB of planner overhead, 99.1 GB of RAM needed in all. RAM in the machine is
121 GiB in total (`free -h`). MemAvailable read 121,779,548 KiB — 124.69 decimal GB, 116.14 GiB — at the
measurement. The planner's verdict is `fits`, its spare is 22.4 GB in the planner's own units (it counts
decimal GB), and the spill is 0.0 GB. The per-run MemAvailable readings before each documented run were 123.05,
121.49, 121.47 and 121.46 GB. Deficit: none. Swap is 132 GiB of file-backed swap on disk with 0 used. It is not
RAM and is not counted as RAM (all of this paragraph: measured on this PC).

Placement on the cards was measured too (measured on this PC). 4,880 experts (42.0 GB) go in graphics memory:
2,440 experts, 21.0 GB and 2.2 GB of dense weights per card. The other 10,480 experts (90.1 GB) stay on the CPU
lane in RAM. As a sensitivity check, using 24.0 GiB per card instead of clinfo's 22.71 GiB (the marketed number)
still gives `fits`: 87.3 GB of CPU experts, 96.3 GB of RAM needed (measured on this PC).

One real blocker stands in the documented path. Four `strata launch --plan` runs on this PC exited 0 and printed
verdict `no_card` (measured on this PC), and the round-2 fit log's no-card mirror run reproduces it. The reason
is read from code: the Intel Arc Pro B60 (PCI `0xe211`) is not in the launcher's Intel card table
(`strata_cli/launch/catalog.py`, the `INTEL_ARC` map at lines 199-200 and the skip at 210-211; static reading,
unconfirmed static analysis), so the catalog drops the card. That is why the documented path cannot place the
model on these cards yet. The `fits` numbers above come from the same planner function run with the cards
measured through clinfo. No launcher change is proposed in this pass.

## What we researched

Everything in this section is `(static reading)` unless a line says otherwise.

**vllm-project/vllm-xpu-kernels** (read on 2026-10-24):
- License Apache-2.0, so its kernels may be vendored into Strata with a notice (static reading).
- It targets Intel's Xe2 graphics, which includes the Battlemage G21 chip (static reading; the two Arc Pro B60
  cards in this PC are measured on this PC above).
- It builds with `icpx -fsycl` and ahead-of-time compile for the card, the same way Strata's `sycl/` tree
  already builds (`-fsycl-targets=spir64_gen`) (static reading).
- It covers attention, MoE routing, DeepSeek rotary positions, and MLA flash-attention in plain SYCL, and uses
  oneDNN for big matrix multiplies (Strata's cublas equivalent) (static reading).
- It does not cover the quant types Strata's DeepSeek engine uses (IQ1_S, IQ1_M, other below-2-bit types, GGUF
  packing). Those must come from Strata's own code (static reading).

**oneapi-src/SYCLomatic** (the `dpct` converter):
- Round-1 record (prior art): dpct 21.0.0, from the 20260928 release, run on this PC on 2026-10-24. Round 2
  could not re-check the version: `dpct` is not found after `setvars.sh` in the 2026-10-25 probes (measured on
  this PC). The DPCT comments left in `sycl/src/dsv41/` show the converter ran (static reading). Treat the
  version as a record.
- Compiler for the conversion: Intel oneAPI DPC++/C++ 2026.2.0. Round 2 measured the compiler again: the probe
  prints 2026.2.0 (measured on this PC).
- It turns CUDA source into SYCL source: kernel launches become `parallel_for`, shared memory becomes
  `local_accessor`, `__syncthreads` becomes a group barrier, cuBLAS calls become oneMKL calls (static reading).
- It does not do everything. Four things needed hand fixes in Strata's code: templated kernel launches, tensor
  core tiles (WMMA), cache hints (`__ldg`/`__ldcv`, which are simply dropped), and dynamic shared memory sizes
  (prior art: the round-1 conversion record; static reading).
- It needs CUDA header files to read CUDA source. This PC has no CUDA toolkit (measured on this PC), so in
  round 1 the CUDA headers from the PyPI nvidia-cuda packages were merged into one include directory and passed
  to dpct (prior art: the round-1 record).

## What is on this branch already

Three of the engine's core quant kernels were converted with dpct and hand-fixed, in the existing `sycl/` tree
pattern (`sycl/src/dsv41/`):

| file | what it does | compiles? |
|---|---|---|
| ds4_tq.dp.cpp | ternary quant (TQ1/TQ2) decode and set | yes, `icpx -fsycl` exit 0 (measured on this PC) |
| ds4_iq.dp.cpp | IQ1_S, IQ1_M, IQ2, IQ3, IQ4 decode, gate-up and down passes | yes, `icpx -fsycl` exit 0 (measured on this PC) |
| ds4_gguf.dp.cpp | GGUF quant packing passes | yes, `icpx -fsycl` exit 0 (measured on this PC) |

All three compile with icpx 2026.2.0 on this PC (Ryzen 7 5700X3D, 2× Intel Arc Pro B60): first on 2026-10-24 in
round 1 (prior art), then again in round 2 as translation units of the parity harness whose rows are above
(measured on this PC). Compiling green means the compiler accepts the SYCL code. The rows above are what says
the output is right (measured on this PC). Nothing yet says it is fast (not measured yet).

Hand fixes applied (the four dpct gaps above):
- Templated launches became explicit work-group launches, the same pattern the rest of `sycl/` uses. Work-group
  ranges that CUDA wrote as (x, y, z) are read by SYCL as (z, y, x) (static reading).
- Shared arrays that dpct turned into wrong shapes were rebuilt as flat buffers with the sizes from the CUDA
  source (static reading).
- Tensor core tiles (WMMA) in ds4_gguf first became a plain float accumulation loop (prior art: the round-1
  record). The MXFP4 set tiles now have a SYCL tensor-core path (`joint_matrix`) with the scalar loop kept as a
  fallback; the set rows above are its parity on the B60 (measured on this PC). Whether the tiles run as fast as
  the CUDA path is not measured and is not claimed here (not measured yet).
- The CUDA occupancy query in ds4_iq became "one work-group per compute unit", which is what a persistent grid
  needs on this card (static reading).

## Next steps, in order

1. Correctness first: run each converted kernel on the Arc Pro B60 and compare its output to the CPU reference
   implementations (`src/dsv41/ds4_tq_cpu.cpp`, `src/dsv41/ds4_iq_cpu.cpp`, `src/dsv41/ds4_gguf_cpu.cpp`). Equal
   output, kept within a small tolerance, is the bar. No speed claim until this passes. Done for the rows above
   (measured on this PC), including the IQ4_XS set tiles after the round-2 barrier fix. Still open: the timed
   run (not measured yet).
2. Wire an opt-in build: a `STRATA_SYCL_DSV41` option in `sycl/CMakeLists.txt`, off by default. Done and
   measured on this PC: the option is at `sycl/CMakeLists.txt:257`, its default is OFF, and the default build
   lists no dsv41 source (measured on this PC).
3. ds4_model: the model-level passes, including its BLAS calls — 3 in `src/dsv41/ds4_model.cu` (lines 2910,
   2924, 2937) and 10 more in `src/dsv41/ds4_vision.cu`, 13 call sites in total (static reading). Those map to
   oneMKL or oneDNN; both are installed with the toolkit (measured on this PC). The mapping is no longer future
   work: `b422876c1d888c7a0b1eddb312a08d6dcd517298` merged the cublas shim `sycl/include/compat/cublas_v2.h`
   (the `include/strata/hip_compat/cublas_v2.h` pattern), the gemv ports (`sycl/src/dsv41/ds4_gemv.dp.cpp`,
   `sycl/src/dsv41/ds4_gemv.dp.hpp`) and the vision softmax port (`sycl/src/dsv41/ds4_softmax.dp.cpp`, with the
   device-side fix at `sycl/src/dsv41/ds4_softmax.dp.cpp:87`, recorded at `task-4-gemv-parity.log:104`), and the
   parity runs passed on the card (measured on this PC): `RUN_RC=0` (`task-4-blas-shim.log:385`), `TILE_RUN_RC=0`
   (`task-4-blas-shim.log:410`), `CTEST_RC=0` with 2/2 tests passed on `device: Intel(R) Arc(TM) Pro B60 Graphics`
   (`task-4-blas-shim.log:507-511`). What is left of this item is wiring the model-level passes themselves:
   Phase D in [SYCL_DSV41_PLAN2.md](SYCL_DSV41_PLAN2.md).
4. Vision, then exl3 and the sampler, in the same way (not measured yet).
5. Tensor-core timing: measure the `joint_matrix` set tiles against the scalar loop on the B60, and report both
   with the build that produced them (not measured yet). Change the default only with measured numbers, per
   CONTRIBUTING.md.
6. Benchmarks against a same-day CUDA/HIP run where the same model fits, reported per
   [docs/COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) (not measured yet).

## Rules this branch keeps

- The SYCL dsv41 path is opt-in. The CUDA and HIP paths are untouched and stay byte-identical to the release;
  each change checks its own diff scope.
- Nothing in the docs or reports claims a speed that was not measured on named hardware. The claims section
  above holds this plan's one statement about missing speed numbers.
- Vendored code from vllm-xpu-kernels (Apache-2.0) gets a license notice when it arrives.
- Converted code keeps dpct's DPCT comments until reviewed, so a reader can see what the converter changed.
