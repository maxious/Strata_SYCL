# Experiment 45 - QSA block selection on XMX: the share is real and grows with context (measured 2026-10-05)

**Status: premise measured, mechanism not built.** The first deliverable this design asked for - *does the select's
share grow materially with context on a real multi-chunk prompt* - is measured: it goes from **2.7-4.1% at 8K tokens
(exp 38) to 8.0% at 32,768 and 19.0% at 131,072**. So the hypothesis holds and there is real money on the table. The
second deliverable, a DPAS scorer, is **not** written, and exp 42 explains why the obvious route is not available in
this toolchain.

## The measurement (exp 45's first deliverable)

Shipped Q2_0 pack, `--prefill 4096`, `--spec 2`, `--kv int8 --kv-resident 32768`, `--max-context 262144`,
`STRATA_PREFILL_TIMING=1`, one-shot generate, greedy, `--stop-eos`, single card. Prompts are `long.ids`' first 2,000
tokens tiled to length, and every one is **many chunks** (exp 38's trap: a prompt at or below `--prefill` is a single
chunk and reports the select at 96.7% of a stage timeline - an artefact, not a result).

| prompt tokens | chunks | prefill | GPU timeline | `qsa select` | share | `qsa attn` | tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,000 (exp 38, dual) | 7 | 14.9 s | 18.7 s | 0.5-0.8 s | 2.7-4.1% | 15.0% | 535 |
| 32,768 | 9 | 55.0 s | 54.5 s | 4.35 s | **8.0%** | 20.3% | 596 |
| 131,072 | 32 | 254.3 s | 254.0 s | 48.3 s | **19.0%** | 18.1% | 515 |
| 260,999 | 64 | 469.6 s | 469.8 s | 79.3 s | **16.9%** | 10.9% | 556 |

At 131,072 tokens the select is **48.3 seconds of a 254-second prefill**. The growth is what the design
predicted: the token-scaled phases (dequant 10.5%, gemm down 22.7%, gemm gate/up 8.2% at 261K) do not change with
context, while the select's block count is `context/4`. **8% at 32K and ~17-19% at 131-261K is the opposite of
exp 38's "still a few percent" reading, and that reading was only low because it was taken at 8K tokens.**

The last row is why this doc's framing had to change: from 131K to 261K the select's **absolute** time grows
48.3 -> 79.3 s (1.64x for 2x context) but its **share falls** 19.0% -> 16.9%, because the other context-growing phase
moves the other way (`qsa attn` 45.9 -> 51.3 s, its share 18.1% -> 10.9% as the token-scaled phases dilute it). **The
share saturates in the high teens; it does not keep climbing** - so the honest ceiling for this lever at the shipped
`--max-context 262144` is "about a sixth of prefill", not "a third". **8% at 32K and 19% at 131K is the opposite of exp 38's "still a few percent"
reading, and that reading was only low because it was taken at 8K tokens.**

(The 262,144-token point is `--max-context 262144` minus generation, i.e. 261,000 prompt tokens; that run is reported
in the table above if it completed, otherwise the 131,072 point is the longest measured.)

## Why the scorer is not written: exp 42's M>1 finding is this experiment's blocker

The select's shape is the one shape XMX is *for*: many queries (256-4,096 per chunk) x K=128 x `context/4` blocks,
triangular, with the 4 indexer heads kept separate. But the DPAS route needs **M = 16 or more rows per instruction**,
and exp 42 measured, with one-hot probes against host dot products, that **int8 DPAS with a repeat count above 1 does
not reproduce in this toolchain**: at M=2/4/8 exactly one row of the MxN result is right (16 of M*16 lanes), under both
a row-major and a K-major A layout. `dpas.hpp` itself asserts only sizes, never layouts, and the ESIMD release here has
no `permutex2var` to fix up a tile. exp 42's kernel works precisely because it issues **M=1** DPAS per column.

For the decode matvec that costs nothing (one activation column per pass). For the select it would mean **one pass per
query** - 256 to 4,096 passes instead of 16-512 - which is a different kernel, not a tuning of this one.

The feasible route exists and is *not* the one this design sketched: `joint_matrix` with `use::a, 8, 32` (M fixed at
8), which the port already uses successfully for int8 DPAS GEMM (`xmx_int8_bench.cpp`, and the IQ prompt kernels in
`iq_kernels.dp.cpp`). A select on that path would run 8 queries per pass with no padding waste at 256+ queries, four
K=32 DPAS per 128-deep dot, per-group scales applied on the vector units, and the triangular early exit. That is a
real piece of work (the epilogue is `sum_h relu(acc_h)` with the heads separate, plus the tail block left to the warp
kernel) and it is **the concrete next step**, not a dead end.

## Bounded upside, so the next person knows what is at stake

At 131K context the select is 19.0% of prefill and at 261K it is 16.9%. Removing it entirely - a physically
unreachable bound - would be a **1.23x TTFT** at 131K and **1.20x** at 261K; a 2x kernel ~1.10x and ~1.09x. At 32K it is 1.09x and ~1.04x; at exp 38's 8K it is 1.03x. **This is
a long-context lever only**: worth nothing at 8K and worth about a sixth of prefill from 131K on, which is why no
short-prompt A/B would ever see it.

## Traps this measurement had to dodge, and one it did not

- exp 38's one-chunk artefact: every prompt here is 9-33 chunks of 4,096.
- exp 37's bench trap (`capacity_cells` drives the dispatch, `context` drives the work) applies to
  `qsa_select_bench`; this experiment deliberately does **not** use that bench, because the engine-side share is the
  number that decides whether any kernel is worth building.
- exp 45's own advice - "a select measurement needs a prompt several times `--prefill`" - is satisfied by construction.
- The 262,144-token attempt failed once with `positive --max-new and --max-context must fit the prompt and
  generation`: at exactly `--max-context` the prompt plus one generated token does not fit, so the longest run
  is 260,999 tokens with `--max-new 1`. That is a config constraint, not a result.
- **The runs were single-card generate, not the dual-card serve config** (serve mode needs `--mtp DIR` with a
  `dense.txt` that this box does not have, and `--layer-split` is serve-only). The share is a per-stage GPU-timeline
  quantity, so single-card is a fair instrument for it, but the tok/s column is **not** comparable to exp 38's dual
  numbers and must not be quoted as a regression.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh
python3 -c "import pathlib;ids=pathlib.Path('sycl/bench/v1/long.ids').read_text().strip().split(',');\\
n=131072;pathlib.Path('/tmp/p.ids').write_text(','.join((ids[:2000]*((n//2000)+1))[:n]))"
STRATA_PREFILL_TIMING=1 ./sycl/build-b60/strata --pack /home/maxious/Strata-data/packs/q2_0 \\
  --native .../Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \\
  --ple-gguf .../Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \\
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill 4096 --spec 2 \\
  --max-context 262144 --kv int8 --stream-experts --kv-resident 32768 \\
  --tokens-file /tmp/p.ids --stop-eos 2>&1 | grep 'prefill timing:'
```

Logs: `/home/maxious/exp45-logs/gen32k.log`, `gen131072.log`, `gen261k.log`.