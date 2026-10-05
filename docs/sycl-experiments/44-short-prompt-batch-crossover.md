# Experiment 44 - the short-prompt batch: where an XMX matvec stops beating dequant + FP16 GEMM (measured 2026-10-05)

**Status: measured, and the answer is no.** There is **no crossover in (8, 80] where the DPAS int8 matvec wins**: the
oneMKL FP16 GEMM that the prompt path already uses is **flat in width** (155-200 us from 1 to 80 columns at
12288x2560; 9.4-10.3 us at 2560x640), so it is *already* cheaper than a matvec at every prompt width the PR's window
covers. The crossover sits at M = 8-9 and it is an artefact of this toolchain's M>1 DPAS gap (exp 42), not a property of
the matvec shape.

## Question

llama.cpp PR 29864 dispatches `mul_mat_vec_q` on 9-80 columns, where the default threshold is 8, and this port routes
**every** prompt chunk through dequant + oneMKL FP16 GEMM. At what prompt-batch width M does an XMX int8 matvec stop
beating that?

## Instrument

`xmx_mmvq_bench ... sweep` (same file as exp 42, `sweep_mode`), one process, warm 300 ms, 200 reps, per-call us,
`ZE_AFFINITY_MASK=1` on the idle card. Arms:

- **DPAS** - exp 42's kernel. For M <= 8 it is the templated `NC=M` launch; for M > 8 it is **M single-column
  launches** over the same weight tiles, because exp 42 established that int8 DPAS with a repeat count above 1 does
  not reproduce in this toolchain. That is the arm's real cost curve at 9-80 and it is stated here rather than hidden.
- **fp16GEMM** - a persistent FP16 copy of the same weights through `dpct::blas::gemm` (f32 compute), i.e. exactly
  the kernel `gemm.native` runs on the prompt path. Its dequant half is *not* included (exp 07 measured dequant at
  30.1% of prompt time separately), so this arm is the prompt path's **product** half, which is the half a matvec
  would replace.
- **preunpack** - exp 27's byte kernel, for reference at M <= 8 (the shipped kernel refuses M > 8, by design).

## Measured (per-call us)

**12288 x 2560 - the model's dominant dense down projection**

| width | preunpack | DPAS | fp16GEMM | DPAS vs fp16GEMM |
|---:|---:|---:|---:|---:|
| 1 | 52.3 | 52.3 | 200.3 | **3.83x** |
| 2 | 64.9 | 64.9 | 193.4 | 2.98x |
| 4 | 95.2 | 95.2 | 179.5 | 1.89x |
| 6 | 122.0 | 122.0 | 166.6 | 1.37x |
| 8 | 146.1 | 146.1 | 155.5 | 1.06x |
| 9 | n/a | 470.2 | 153.5 | 0.33x |
| 16 | n/a | 834.6 | 154.7 | 0.19x |
| 32 | n/a | 1670.1 | 161.4 | 0.10x |
| 64 | n/a | 3341.6 | 167.4 | 0.05x |
| 80 | n/a | 4175.8 | 181.3 | 0.04x |

**2560 x 640 - the shared expert's real gate/up shape**

| width | preunpack | DPAS | fp16GEMM | DPAS vs fp16GEMM |
|---:|---:|---:|---:|---:|
| 1 | 13.2 | 13.2 | 10.3 | 0.78x |
| 8 | 32.9 | 32.9 | 9.7 | 0.29x |
| 9 | n/a | 118.4 | 9.6 | 0.08x |
| 32 | n/a | 421.3 | 9.5 | 0.02x |
| 80 | n/a | 1052.5 | 10.0 | 0.01x |

(The DPAS column for M <= 8 equals the preunpack column's *shape* by coincidence of both being one launch per column
per tile; the values are independent measurements and both are in the table.)

## The engine side, at the widths the PR names

Three single-card generate runs, Q2_0, `--prefill 4096 --spec 2`, `STRATA_PREFILL_TIMING=1`, prompts of 9 / 32 / 80
tokens from `long.ids` (`--stop-eos`, one chunk each by construction - here the chunk *is* the subject, which is
exp 38's trap only when the question is about overlap):

| prompt | `dequant` | `gemm down` | `gemm gate/up` | dense product half |
|---:|---:|---:|---:|---:|
| 9 | 53 ms (14.4%) | 64 ms (17.4%) | 26.7% | **~44%** |
| 32 | 138 ms (21.5%) | 140 ms (21.9%) | 23.3% | ~45% |
| 80 | 209 ms (23.2%) | 199 ms (22.1%) | 23.5% | ~46% |

Two things follow. **The dense projections are not a small share at short widths** - at a 9-token prompt `gemm down`
alone is 17.4% - so exp 07's "the prompt is dequant-bound" reading extends down to 9 tokens rather than fading out.
And the **dequant/GEMM balance barely moves with width** (14.4/17.4 at 9 tokens, 23.2/22.1 at 80), which is the engine
side of the bench's flat GEMM curve: the per-weight dequant and the GEMM's fixed cost are both nearly width-independent
here. So the dense path really is worth attacking at short prompt widths - just not with a matvec.

## What this actually says

1. **The FP16 GEMM is width-insensitive in the whole 1-80 range.** 200 -> 155 us at 12288x2560 (it gets *cheaper*
   with width, as a GEMM should) and 10.3 -> 9.6 us at 2560x640. **There is no width at which a narrow matvec is
   buying anything**: the prompt path's dense projection costs the same at a 9-token chunk as at an 80-token one. The
   PR's premise - that a matvec wins in a window where the GEMM is overhead-bound - does not hold on this device,
   because oneMKL's GEMM launch overhead is already below the matvec's cost at every one of these shapes.
2. **The DPAS curve past 8 is a straight line of M launches** (52.3 us at M=1 to 4175.8 at M=80, i.e. ~52 us per
   column), which is exp 42's M>1 gap showing up as a slope, not a crossover. A real M-wide DPAS (8 queries or 8
   columns per instruction) would flatten that line; **if it did, it would still have to beat a flat 155 us**, i.e.
   ~2 us per column, which is 25x below what one M=1 DPAS pass costs today. So the honest statement is: the M>1 gap
   blocks the measurement, and the one datapoint we can take on the other side of it (a hypothetical flat DPAS curve)
   would have to be ~25x better than the M=1 pass to matter at 80 columns - while at M=1, where it *is* measurable,
   DPAS already wins 3.83x.
3. **That last point is the interesting one and it inverts the design's framing.** At **M=1** - the width decode
   actually runs - the DPAS matvec is 3.83x the GEMM at 12288x2560. The PR's window (9-80) is the wrong window for
   *this* engine's dense Q6_K, and exp 42 already showed the same shape-dependence (1.98x at 12288x2560, 0.47x at
   640x2560).

## Verdict against the design's success criteria

- "A measured M* in (8, 80] where the DPAS matvec wins below and the FP16 GEMM wins above" - **not found. The GEMM
  wins everywhere above 8, and by 3-25x.**
- "Crossover at M <= 8 -> null: the GEMM path stays for every prompt chunk" - **this is the case, and it is stronger
  than the design allowed**: the GEMM also wins at M=1 for every shape except the largest one, and the exception is a
  *decode* lever (exp 42), not a prompt lever.
- **No dispatch is worth wiring.** Nothing in production changes.

## Bounded upside, recorded so the next person does not re-run this

Even a *free* dense projection would buy what exp 07 measured: dequant 30.1% + gemm down 20.0% + gemm gate/up 8.5% =
**58.6% of prompt time**, but that is dominated by the **dequant**, which no matvec removes (the matvec still reads
quantized weights). The replaceable part here - the product half - is 28.5% of prompt time at 8,000 tokens, and this
measurement says an int8 matvec cannot take any of it at M >= 9. So **the upside of the whole PR-29864-style
"matvec for short prompt batches" idea on this engine is bounded by the dequant**, which is a different lever
(exp 07's dequant-bound finding).

## Traps

- The M > 8 arm is M separate launches, which is a property of exp 42's toolchain finding, not of the PR's design.
  Quoting the 9-80 rows as "the matvec loses" without that caveat would be wrong.
- Per-call us only; a GB/s column would be inflated by the buffer sizes (exp 27's trap).
- `--expert-cache`/layer-split pinning is irrelevant here (bench, not engine), but the **device** is not: an earlier
  run without `ZE_AFFINITY_MASK` landed on the card another measurement was using (exp 43 hit the same thing).

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh
ZE_AFFINITY_MASK=1 ./sycl/build-b60/xmx_mmvq_bench 2560 12288 200 sweep
ZE_AFFINITY_MASK=1 ./sycl/build-b60/xmx_mmvq_bench 2560 640 200 sweep
```

Logs: `/home/maxious/exp44-harness/sweep_12288x2560.log`, `sweep_640x2560.log`, `engine_{9,32,80}.log`.