# SYCL support for the DeepSeek engine (dsv41) — plan

This is the plan for running Strata's DeepSeek engine on Intel graphics cards. The engine today speaks two
languages: CUDA for NVIDIA cards and HIP for AMD cards. Intel cards speak a third language, SYCL. This branch adds
that third language for the engine's core parts, without changing anything for NVIDIA or AMD users.

## Who this is for

A person with an Intel card (an Arc Pro B60, or a future Arc card) who wants to serve a dsv41 model with Strata.
Nothing changes for anyone else: the CUDA and HIP paths stay exactly as they are. The SYCL path is opt-in, built
only when you ask for it.

## What we researched

**vllm-project/vllm-xpu-kernels** (read on 2026-10-24; unconfirmed static analysis unless marked measured):
- License Apache-2.0, so its kernels may be vendored into Strata with a notice.
- It targets Intel's Xe2 graphics, which includes the Battlemage G21 chip (the Arc Pro B60 this work was done on).
- It builds with `icpx -fsycl` and ahead-of-time compile for the card, the same way Strata's `sycl/` tree already
  builds (`-fsycl-targets=spir64_gen`).
- It covers attention, MoE routing, DeepSeek rotary positions, and MLA flash-attention in plain SYCL, and uses
  oneDNN for big matrix multiplies (Strata's cublas equivalent).
- It does not cover the quant types Strata's DeepSeek engine uses (IQ1_S, IQ1_M, other below-2-bit types, GGUF
  packing). Those must come from Strata's own code.

**oneapi-src/SYCLomatic** (the `dpct` converter; installed and run on this PC — measured):
- Version used: dpct 21.0.0, from the 20260928 release. Compiler: Intel oneAPI DPC++/C++ 2026.2.0.
- It turns CUDA source into SYCL source: kernel launches become `parallel_for`, shared memory becomes
  `local_accessor`, `__syncthreads` becomes a group barrier, cuBLAS calls become oneMKL calls.
- It does not do everything. Four things needed hand fixes in Strata's code: templated kernel launches, tensor
  core tiles (WMMA), cache hints (`__ldg`/`__ldcv`, which are simply dropped), and dynamic shared memory sizes.
- It needs CUDA header files to read CUDA source. This PC has no CUDA toolkit, so the CUDA headers from the
  PyPI nvidia-cuda packages were merged into one include directory and passed to dpct. That worked.

## What is on this branch already

Three of the engine's core quant kernels were converted with dpct and hand-fixed, in the existing `sycl/` tree
pattern (`sycl/src/dsv41/`):

| file | what it does | compiles? |
|---|---|---|
| ds4_tq.dp.cpp | ternary quant (TQ1/TQ2) decode and set | yes, `icpx -fsycl` exit 0 |
| ds4_iq.dp.cpp | IQ1_S, IQ1_M, IQ2, IQ3, IQ4 decode, gate-up and down passes | yes, `icpx -fsycl` exit 0 |
| ds4_gguf.dp.cpp | GGUF quant packing passes | yes, `icpx -fsycl` exit 0 |

All three were compiled on 2026-10-24 with icpx 2026.2.0 on this PC (Ryzen 7 5700X3D, 2× Intel Arc Pro B60).
Compiling green means the SYCL code is accepted by the compiler. It does not yet mean the output is correct or
fast — see the checks below.

Hand fixes applied (the four dpct gaps above):
- Templated launches became explicit work-group launches, the same pattern the rest of `sycl/` uses. Work-group
  ranges that CUDA wrote as (x, y, z) are read by SYCL as (z, y, x).
- Shared arrays that dpct turned into wrong shapes were rebuilt as flat buffers with the sizes from the CUDA
  source.
- Tensor core tiles (WMMA) in ds4_gguf were replaced with a plain float accumulation loop for now. A SYCL
  tensor-core path (joint_matrix) is the later, faster form; whether it matches CUDA's speed here is not
  measured yet (unconfirmed static analysis).
- The CUDA occupancy query in ds4_iq became "one work-group per compute unit", which is what a persistent grid
  needs on this card.

## Next steps, in order

1. Correctness first: run each converted kernel on the Arc Pro B60 and compare its output to the CPU reference
   implementations (`src/dsv41/ds4_tq_cpu.cpp`, `ds4_iq_cpu.cpp`, `ds4_gguf_cpu.cpp`). Equal output, kept within a
   small tolerance, is the bar. No speed claim until this passes.
2. Wire an opt-in build: a `STRATA_SYCL_DSV41` option in `sycl/CMakeLists.txt`, off by default. With the option
   off, the build is byte-identical to today.
3. ds4_model: the model-level passes, including the 35 cuBLAS calls. Those map to oneMKL or oneDNN (oneDNN is
   installed on this PC). The `hip_compat` shim in the repo is the precedent for this mapping.
4. Vision, then exl3 and the sampler, in the same way.
5. Tensor-core speed: try `joint_matrix` (SYCL's tensor-core API) where CUDA used WMMA, and measure it against
   the scalar loop on the B60. Change the default only with measured numbers, per CONTRIBUTING.md.
6. Benchmarks against a same-day CUDA/HIP run where the same model fits, reported per docs/COMMUNITY_BENCHMARKS.md.

## Rules this branch keeps

- The SYCL dsv41 path is opt-in. The CUDA and HIP paths are untouched and stay byte-identical to the release.
- Nothing in the docs or reports claims a speed that was not measured on named hardware.
- Vendored code from vllm-xpu-kernels (Apache-2.0) gets a license notice when it arrives.
- Converted code keeps dpct's DPCT comments until reviewed, so a reader can see what the converter changed.
