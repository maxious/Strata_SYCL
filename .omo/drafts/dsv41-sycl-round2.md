---
slug: dsv41-sycl-round2
status: drafting
intent: unclear
review_required: false
pending-action: write .omo/plans/dsv41-sycl-round2.md
approach: Continue the dsv41 SYCL port on branch sycl/dsv41-sycl-support as one opt-in target, one wave per remaining PLAN2 phase, each wave in its own task-owned worktree, each gated by icpx -fsycl compile green plus a device parity run of ds4_parity on the Arc Pro B60 against the CPU references; the default CUDA/HIP build stays untouched.
---

# Draft: dsv41-sycl-round2

## Affected user and ideal state

Who this touches:
- **The owner of this PC** (one person, 2x Intel Arc Pro B60): wants to serve the dsv41 DeepSeek engine on Intel cards from Strata instead of being told the card is unsupported.
- **Strata's maintainer and reviewer** (reads the branch/PR): needs the NVIDIA/AMD path unchanged, the SYCL path opt-in, and no unmeasured performance claim.
- **The next contributor** (reads the converted sources and the plan docs): needs to see which parts are converted, which are measured, and which dpct comments still stand.

| Row | Statement | Reason |
| --- | --- | --- |
| IS-1 | With `STRATA_SYCL_DSV41` on, every dsv41 quant type the engine uses has a SYCL kernel whose output equals the CPU reference on this PC's Arc Pro B60, including the 32-value IQ4_NL block shape that round 1 refused. | An engine that cannot decode a quant type the model uses cannot run the model; a silent wrong decode is worse than a refusal. |
| IS-2 | The two tensor-core kernels (`mx_set_tc_kernel`, `set_tc_kernel`) use real Xe2 matrix tiles (`joint_matrix`) instead of scalar stand-ins, and still match the CPU reference. | The scalar stand-in is a correctness placeholder; the tile is the shape the shipping Xe engine uses, and only a device run tells whether Battlemage accepts it. |
| IS-3 | `ds4_model` and its 13 cuBLAS calls build for SYCL through a shim that mirrors `include/strata/hip_compat/cublas_v2.h`, and the gemv/softmax helpers run on device with parity. | Without the model-level passes there is no engine, only kernels; the HIP shim is the repo's own precedent for this exact mapping. |
| IS-4 | The default build (no `STRATA_SYCL_DSV41`) produces the same binaries as before this work; every new file sits behind the opt-in flag. | CONTRIBUTING.md: the default path stays byte-identical to the last release. |
| IS-5 | A first measured prefill/decode number for dsv41 on one B60 exists, or the report says plainly that it could not be measured and why. | The repo forbids unmeasured speed claims; a missing number stated as missing is honest, an invented one is not. |
| IS-6 | Stopping the engine during the work uses SIGTERM and waits, so the card never reaches `PAGEFAULT -> DEVICE_LOST`. | A lost device blocks every later run on this machine, and reset needs root. |
| IS-7 | The two plan docs say what is now measured on this PC versus what is still static reading. | The docs are the only place a reviewer learns which claims were run. |
| GAP-1 | `dequant_rows` refuses IQ4_NL; round 1 covers tq, gguf and iq types only. | An IQ4_NL model cannot decode on SYCL. |
| GAP-2 | `mx_set_tc_kernel` and `set_tc_kernel` carry float-accumulation stand-ins where CUDA used WMMA. | Correct but not the tile path; unmeasured against the tile path. |
| GAP-3 | No `ds4_model`/`ds4_vision` SYCL translation unit and no cublas shim for SYCL. | The engine entry point does not exist for SYCL. |
| GAP-4 | Parity evidence is one 4096-value BF16 comparison from round 1; IQ4_NL and gemv have none. | Acceptance without per-type parity is a claim, not evidence. |
| GAP-5 | No measured dsv41 number on Intel hardware anywhere. | Phase D's first number is the only honest baseline. |
| GAP-6 | `sycl/src/dsv41` still carries dpct markers and scalar-tile comments that no longer describe the code after Phase A. | A reader cannot tell converted-but-reviewed code from untouched dpct output. |

## Components (topology ledger)

| id | outcome (one line) | status | evidence path |
| --- | --- | --- | --- |
| C1 | Tensor-core kernels run on `joint_matrix` tiles with parity and a scalar-vs-tile timing | active | docs/SYCL_DSV41_PLAN2.md Phase A |
| C2 | IQ4_NL dequant kernel with device parity against the CPU reference | active | docs/SYCL_DSV41_PLAN2.md Phase B |
| C3 | cublas->oneMKL shim plus gemv/softmax SYCL kernels, compiling and parity-checked | active | docs/SYCL_DSV41_PLAN2.md Phase C |
| C4 | `ds4_model` builds for SYCL behind the opt-in flag and serves one small pack, with the first measured numbers | active | docs/SYCL_DSV41_PLAN2.md Phase D |
| C5 | Plan docs updated to measured-vs-static per increment | active | docs/SYCL_DSV41_PLAN.md, docs/SYCL_DSV41_PLAN2.md |
| C6 | Measured occupancy tuning and dual-GPU expert placement | deferred | docs/SYCL_DSV41_PLAN2.md Phase E: needs C1/C4 numbers first, and dual-GPU placement is its own contract change |
| C7 | Vision, exl3 and sampler SYCL ports | deferred | docs/SYCL_DSV41_PLAN2.md Phase E: separate increments after C4 runs |

## Open assumptions (announced defaults)

| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| Delivery mode | Commit per increment on the existing `sycl/dsv41-sycl-support` branch, push to `fork` at the end of each verified wave; no PR unless asked | The brief's step 9 says "commits per increment on the branch; final push"; a PR needs a scope agreement the issue thread does not yet have | Yes |
| oneAPI stays the toolchain | /opt/intel/oneapi 2026.2 `icpx` (oneMKL present), not the self-built intel/llvm tree | oneMKL only exists there and round 1 compiles green with it | Yes |
| IQ4_NL source | Port llama.cpp's `dequantize.hpp` shape (MIT, no shared memory, 32 threads/group) into Strata's BF16 output | PLAN2 already compared the tables and names it a drop-in template | Yes |
| Scalar fallback for tiles | Keep the scalar path compiled behind a macro so a device that lacks the matrix combination still decodes | XeStrata gates capability at run time; parity can then be compared scalar-vs-tile on the same box | Yes |
| Timed numbers | Report only runs produced in this session on this PC, stamped with build id and date; if Phase D cannot run for a missing pack, say so and leave the number absent | CONTRIBUTING.md forbids submitting measurements not produced here | n/a |

## Findings (cited - path:lines)

- CMake facts: option STRATA_SYCL_DSV41 OFF at sycl/CMakeLists.txt:257; strata_dsv41 target only inside if(STRATA_SYCL_DSV41) at :258-264; strata_resolve .cu to .dp.cpp at :76-84; STRATA_XMX_SG16=1 for bmg|lnl at :31-36; default sub-group 32 forced at :86; parity foreach at :269-278 has no ds4_parity entry.
- Kernel facts: ds4_gguf.dp.cpp :1052 DPCT1110, :1056 mx_set_tc_kernel, :1087 scalar-FMA stand-in comment, :1353 launch; gemv :1490 mv_launch, :1579 mv_set, :1600 x3::mv. ds4_iq.dp.cpp :1010 DPCT1110, :1014 set_tc_kernel, :1645 launch; :47 Ty<kIQ4_NL> B=18 TAB=16 MIN=0, :57 kQK 32, :2020-2039 dequant_rows switch with default return false at :2034 and no kIQ4_NL case; gemv :1144 mv_kernel, :1884 mv_nc, :2178 iq::mv.
- Parity facts: ds4_parity.cpp is hand-built; recipe is its header comment :10-12 (icpx -fsycl -std=c++20 -O2 -I sycl/include -I include -I third_party/ggml -I sycl/include/compat plus a cuda-header dir resolving to include/strata/hip_compat); main :189, types :206-208, launch :215, IQ4_NL refusal expectation :226-228, per-type print :232-233, MATCH :237 exit 0; inline refs ref_q4_K :67, ref_q5_K :83, ref_q6_K :102, ref_iq4_nl :131, ref_iq4_xs :142; no CMake target or script builds it.
- Shim facts: include/strata/hip_compat/cublas_v2.h is 63 lines (mirror shape); src/dsv41/ds4_model.cu has 11 cublas references (PLAN2 names :2910, :2924, :2937). DPCT counts 59/40/18 across the three .dp.cpp files.
- Pack facts (du -sh and ls this session): /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO = 232G, four IQ2_XXS shards 39.07/39.39/39.25/20.53 GB plus DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf 110.6 GB plus tensor-allocation/. Nothing serves: 127.0.0.1:8080 and :1234 both return 000 (curl rc 7); strata not on PATH; repo CLI strata.py plus strata_cli; built binary at /home/maxious/XeStrata/engine/strata.
- Git facts (this session): branch HEAD 52316b81 equals fork head (push is a no-op now); refs/remotes/fork stale; branch 23 behind upstream main; no PR against Niko1221/Strata; fork default branch b70-intel-arc-0139. .gitignore lacks /.omo/ and bench/results/ so commits name paths explicitly.
- Design lane aborted at 12m22s (The operation was aborted, 41 tool calls); trade-offs carried from the announced defaults above (tile-with-macro fallback, shim mirroring the HIP precedent, pack as-is, Phase E deferred); reversible, so a later review session can re-weigh them.

## Decisions (with rationale)

- Waves follow PLAN2's A->B->C->D order because C1 and C3 both touch `sycl/CMakeLists.txt` and the parity harness table, so they cannot edit concurrently; C2 is file-disjoint from C1 and runs in parallel in its own worktree.
- Every wave lands as its own commit on the integration branch, so a wave that stops keeps its verified predecessors.
- Bash: `- [~]` is used when a gate cannot be produced on this machine (e.g. no model pack for Phase D's tok/s), never to hide a failure.

## Scope IN

- Phase A tiles, Phase B IQ4_NL, Phase C shim + gemv/softmax, Phase D model wiring plus its run if a pack exists.
- Parity harness rows added for every new kernel.
- Docs truth pass per increment.
- Push of verified increments to `fork`.

## Scope OUT (Must NOT have)

- No change to the default CUDA/HIP build, no new default flag, no change to `setup.py`, `strata_cli`, `strata_control` or packaging.
- No vendored llama.cpp or vllm source copied wholesale; only the kernel shape is ported, with a notice.
- No dual-GPU expert placement, no vision/exl3/sampler port, no occupancy tuning beyond what Phase A's tile shape requires (deferred components above).
- No performance claim without a same-day run on this PC.
- No touching the untracked `bench/results/*` directories or the `.omo/` tree in commits.

## Open questions

None blocking; all forks resolved by default above and surfaced for veto in the plan's TL;DR.

## Approval gate

status: approved (bootstrap path)
The user invoked `/ulw-execute`; per the skill's bootstrap rule that counts as approval to generate the plan. Plan-consultant and plan-reviewer are gate-locked on this path, so this plan is produced without them; a later ulw-plan review session can add that rigor.
