# Experiment 20 - follow-ups: GPU Hotspots with counters, and where the eager divergence lives

Date: 2026-10-04. One Intel Arc Pro B60 (Battlemage G21), oneAPI 2026.1, `sycl/build-b60`. Coder IQ1_M,
`--expert-cache auto --stream-experts`, `--spec 4 --mtp ~/ComfyUI/koboldcpp/mtp/rt`, prompt
`9707,11,1879,1130,374,279`. Engine at the exp 19 build (the doorbell completion gate fixed).

## Part 1 - VTune GPU Hotspots with hardware counters

`dev.xe.observation_paranoid` was 1 (counters denied); set to 0 for this session with
`sudo sysctl dev.xe.observation_paranoid=0` (runtime only, not persistent). Collection:

```sh
sudo -n bash -c 'source /opt/intel/oneapi/setvars.sh; cd sycl/build-b60
/opt/intel/oneapi/vtune/2026.4/bin64/vtune -collect gpu-hotspots -result-dir /tmp/vt/hs1 \
  -finalization-mode=full -- ./strata <the flags above> --max-new 256'
```

Counters landed. Elapsed 29.9 s, **GPU Time 15.141 s (50.6% of elapsed)**; device reported as
`Battlemage G21 [Arc Pro B60]`, 160 XVEs, max core frequency 2.400 GHz, and **XVE Array Stalled/Idle 92.3% of
elapsed time with GPU busy** - i.e. when the GPU is busy, its vector engines are stalled or idle most of the time.

Top kernels by GPU time:

| kernel | GPU time |
|---|---|
| `native_mmvq_q6k_wide_a2` | 3.713 s |
| `wait_flag_ge_kernel` | 2.510 s |
| `native_gu_port<18,8>` | 1.426 s |
| `[Others]` | remainder |

`wait_flag_ge_kernel` is the **device spin** of the host/GPU handshake - the kernel that waits for the host's
plan. It is the second-largest GPU consumer in this collection.

**Caveat, and it is not small.** Under collection the decode degenerated:
`output : 23075 0 0 0 0 0 0 ...`, 256 tokens in 16,118 ms (15.88 tok/s against 20.24 without instrumentation).
That is the bounded-spin design behaving as INTEL.md documents it - instrumentation slows the host's plan writes,
the device's `kSpinMax` expires, and "with a bound the failure is a wrong window instead" of a hang. So this
collection ranks kernels and gives real hardware metrics, but its token stream is not a run of the model, and
`wait_flag_ge_kernel`'s 2.510 s is inflated by exactly that. A profile of the handshake-free path (the prefill
stages, or a bench binary) is the honest way to price the spin.

## Part 2 - where the eager divergence lives

exp 19 found that `STRATA_VERIFY_EAGER=1` (the window launched directly instead of replayed from its graph) is not
token-equivalent to the graph path. This bounds it.

**Both modes are deterministic.** Two runs each, 32 tokens:

| arm | run 1 | run 2 | decode |
|---|---|---|---|
| default (warm capture, replayed) | `23075 303 44424 ...` | identical | 20.24 tok/s |
| `STRATA_WARM_GRAPHS=0` (captured lazily) | byte-identical to default (md5 `0ed1a55dd08d`) | identical | 20.70 tok/s |
| `STRATA_VERIFY_EAGER=1` (launched directly) | `6040 3683 11 ...` (md5 `1d31ff408851`) | identical | 18.41 tok/s |

The two graph arms agree byte for byte; eager is deterministic and different, from the first token. The draft
statistics move with it (default 11 of 66 drafts accepted, 1.48 tokens/round; eager 13 of 54, 1.68).

**The launch mechanism is numerically transparent.** With the per-layer handshake disabled
(`STRATA_VERIFY_NO_HOST=1 STRATA_VERIFY_DEBUG=1`), graph and eager produce the *identical* 48-layer residual
ladder - `diff` between the two runs is empty - and the same final residual to every printed digit:

    verify dbg:   R_  n=2560 mean|.|=0.389 max|.|=3.774 nonfinite=0 first4=-0.30695 0.204523 -0.161543 0.255757

(the same line, both arms: `output : 1879`, `rc=0`). This agrees with the suite's own graph check:
`elementwise_parity` captures a gather + scale into a `command_graph`, replays it, and compares the result to the
direct launch with `memcmp` - bit-identical, and green in `ctest`.

**So the divergence requires the per-layer doorbell/plan handshake**, and is not a property of graph replay. In
the handshake path the host serves a plan per layer from its own work; in the non-handshake path nothing crosses.
The next step is therefore to instrument what crosses: the plan and miss data the host writes per ring, in each
mode.

### Two instruments that did not work, and why

- `--dump-logits` is unreachable on this configuration: the code says it outright ("a native pack never reaches
  the per-token dump site"), so the file is created empty. Localizing a numeric difference end to end therefore
  has no logits oracle on the native pack + MTP path.
- The per-layer residual ladder exists but is printed only inside the `no_host && STRATA_VERIFY_DEBUG` block, and
  that block **crashed before printing** on this configuration: `hits_.d_res` is NULL here, and the diagnostic
  memcpy'd from it. The three handshake buffers in that block are now null-guarded, which is what made the ladder
  above reachable. A diagnostic that crashes before it prints is worse than no diagnostic.

## Part 3 - the three top kernels, measured in isolation

The exp 20 collection's token stream was degenerate, so its ranking is not by itself an optimization target list. The
project's own isolated benches, which cannot be corrupted that way, say this about the same families:

| kernel (from the profile) | isolated measurement | prior art |
|---|---|---|
| `native_mmvq_q6k_wide_a2` (3.713 s) | `mmvq_bench`: q6_k 2560x2560 x 4 cols = **26.1 us, 205.7 GB/s of weights**, wide-vs-shared rel 5.449e-08 | identical to exp 00's 26.1 us / 205.7 GB/s - a stable plateau, not a regression. exp 15: this kernel **spills 384 B/work-item** at numGRF=128, the only unflagged spiller of note in the tree |
| `wait_flag_ge_kernel` (2.510 s) | not a compute kernel: `for (spin < kSpinMax && sys_load(flag) < value) strata_spin_pause();` | none in the docs; its figure is inflated by the instrumentation that produced it |
| `native_gu_port<18,8>` (1.426 s) | `native_expert_parity NATIVE_BENCH=1`: GPU grouped **58.1 / 45.7 / 15.2 GB/s of expert bytes** for 10 experts x 4 entries per layer call | INTEL.md: the expert dots are ALU-bound (77% XVE active) |

The card's streaming peak is ~608 GB/s (INTEL.md), so the Q6_K MMVQ sits at a third of it, and the grouped expert
calls - tiny (10 experts) and latency-bound rather than bandwidth-bound - sit far below it. The collection's own
headline agrees on the shape of the problem: **XVE array stalled or idle 92.3% of the time the GPU is busy**, which
is a latency story, not an instruction-count story. The project's history found the same thing for the dense
projections (memory-latency bound, 92-128 GB/s at 84-93% occupancy).

So, of the three: the MMVQ kernel is the only one with a *known defect* to attack (a measured spill) and the only
one with a cheap correctness-preserving A/B harness (`mmvq_bench` + the IGC ISA dump recipe); the spin is a protocol
cost whose fix is `STRATA_VERIFY_DEVICE_PLAN=1` (implemented, measured neutral on 1-2 GPUs), not a loop rewrite; and
`native_gu_port` has never been measured cleanly on its own, so it needs a profile before it needs an optimizer.

## Conclusion

- VTune GPU Hotspots works on this box **with hardware counters**, and the top GPU consumers of a decode run are
  the Q6_K wide MMVQ kernel (3.713 s), the handshake spin (2.510 s) and the expert gate/up kernel (1.426 s), with
  the XVE array stalled or idle 92.3% of the time the GPU is busy. Collection perturbs the handshake and corrupts
  the token stream, so kernel ranking is trustworthy and token behavior under instrumentation is not.
- The eager/graph divergence is **localized to the per-layer handshake**: with the handshake off the two modes are
  bit-identical through all 48 layers, so graph replay itself changes nothing. Still open: which bytes crossing
  that handshake differ between the two modes.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh && cd sycl/build-b60 && ninja strata
ARGS=(--pack ~/ComfyUI/koboldcpp/pack-coder
      --native ~/ComfyUI/koboldcpp/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
      --ple-gguf ~/ComfyUI/koboldcpp/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
      --expert-profile ../../data/expert-profile-coder.bin --expert-cache auto --stream-experts
      --prefill 128 --spec 4 --mtp ~/ComfyUI/koboldcpp/mtp/rt --max-context 4096
      --tokens 9707,11,1879,1130,374,279)

# the same numbers with and without the per-layer handshake (ladders must be identical)
STRATA_VERIFY_NO_HOST=1 STRATA_VERIFY_DEBUG=1 ./strata "${ARGS[@]}" --max-new 1
STRATA_VERIFY_NO_HOST=1 STRATA_VERIFY_DEBUG=1 STRATA_VERIFY_EAGER=1 ./strata "${ARGS[@]}" --max-new 1

# GPU Hotspots (counters need dev.xe.observation_paranoid=0)
sudo sysctl dev.xe.observation_paranoid=0
sudo -n /opt/intel/oneapi/vtune/2026.4/bin64/vtune -collect gpu-hotspots -result-dir /tmp/vt/hs1 \
  -finalization-mode=full -- ./strata "${ARGS[@]}" --max-new 256
sudo -n /opt/intel/oneapi/vtune/2026.4/bin64/vtune -report summary -result-dir /tmp/vt/hs1
```
