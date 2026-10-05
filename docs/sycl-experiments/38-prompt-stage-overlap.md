# Experiment 38 - the dual prompt path: the stages already overlap, and the 33-wide top-k is not a prompt regression

**Question** (archive item D4). Prefill on 2x B60 is **1.29x** of single-card, not the~2x a perfectly overlapped
pair of stages would give. Do stage 0 and stage 1 sit idle for part of each chunk, so adjacent chunks could overlap?

**Verdict. CLOSED with a number: the stages already overlap.** On the 8,000-token prompt the per-stage GPU timelines sum
to **18.66 s against a 14.94 s wall**, i.e. **1.25x of concurrency** - the chunks are already software-pipelined across
the two stages (`prefill.cpp`: "the next stage reads chunk c on a thread while this one reads chunk c + 1", a
`std::future` per chunk). The missing 0.75x is not idle time waiting for a neighbour; it is that the two stages are
*dependent* (stage 1 reads stage 0's residual from the mapped hand-off buffer), so the wall is bounded by the serial
chain of chunk boundaries, exactly as exp 34 found for the decode window. **There is no adjacent-chunk overlap left to
build.**

A second measurement rode along, because the top-k change from exp 37 needed a prompt-path regression check:

| engine | 8,000-token prompt | qsa select share of each stage timeline |
|---|---:|---:|
| `TK_PER_MAX=66` (shipped) | 535.5 tok/s (14,939 ms) | 0.2% / 2.7% / 3.6% / 2.8% / 1.2% / 4.1% / 3.3% |
| `TK_PER_MAX=33` (the old default) | 543.1 tok/s (14,731 ms) | 0.0% / 0.8% / 1.3% / 0.9% / 0.4% / 1.8% / 1.3% |

**The wider register fit is neutral on the prompt** (1.4% apart, inside the +-1.5% engine noise floor exp 33 measured).
Its 1.9-2.4x is a top-k kernel win that only shows past ~135K cells, and the prompt's chunks here are far below that.
The select *share* doubles (1.3% -> 3.3% of a stage timeline) because the same work now takes less time elsewhere in the
timeline - not because it got slower. Shipping 66 therefore risks nothing on the prompt path, which is the check exp 37
owed.

## Method

Dual Q2_0 (`strata-q2_0-dual.json`, `--layer-split auto` -> K=22), `--prefill 4096`, `--spec 2`, `--kv int8`,
`--kv-resident 32768`, 8,000-token prompt (the v1 long prompt's first 8,000 ids), `STRATA_PREFILL_TIMING=1`, greedy,
1 generated token. Two engines, same source, one `-DSTRATA_TK_PER_MAX` apart (`build-b60` = 66, `build-b60-tk33` = 33),
run back to back.

**The chunk structure is why the earlier 467-token probe misled.** With `--prefill 4096` a 2,185-token prompt is a
**single chunk**, so D4's premise (adjacent chunks) does not exist at that size. In the first probe I ran, stage 1's
timeline was 23,550 ms of which **96.7% sat in one `qsa select` interval** - an artefact of a one-chunk prompt with no
neighbour to overlap with, not evidence that selection dominates the prompt. At 8,000 tokens with 7 chunk-timelines,
select is 2.7-4.1%. **Any probe of this item needs a prompt longer than `--prefill`**, or it measures the wrong thing.

## Per-chunk breakdown (TK66 engine)

| stage timeline | wall | select | dominant phases |
|---:|---:|---:|---|
| 1,277 ms | 1,507 | 0.2% | dequant 24.5%, gemm gate/up 17.7%, gemm down 16.4%, host grouping 15.5% |
| 4,083 ms | 4,326 | 2.7% | (the 4,096-token first full chunk) |
| 2,306 ms | 2,393 | 3.6% | dequant + gemm dominate |
| 5,053 ms | 9,836 | 2.8% | wall 2x the GPU timeline: waiting on expert streaming from the SSD/host |
| 1,169 ms | 1,354 | 1.2% | |
| 2,350 ms | 2,444 | 4.1% | qsa attn 15.0%, dequant 19.3%, gemm 14.3/12.2%, host grouping 14.1% |
| 2,419 ms | 4,871 | 3.3% | wall 2x again: the host grouping / expert-stream wait |

The interesting column is wall vs GPU timeline. Two chunks sit at **wall ~2x their GPU timeline**, and that difference
is expert streaming (the config is `--stream-experts`, so a 8,000-token prompt pulls thousands of expert blobs over
PCIe) plus host grouping. **The dual prompt's real ceiling is the expert stream, not stage scheduling** - which is the
same conclusion the single-card runs reached (PP 350-580 tok/s against a 571-1,117 tok/s range elsewhere).

## What this closes and what it does not

- **Closed**: D4. Adjacent chunks already overlap (1.25x measured concurrency); the prompt is dependency-bound, not
  scheduling-bound; there is no ~1.8x sitting in stage scheduling to collect.
- **Not claimed**: that the prompt is fast. 535 tok/s at 8,000 tokens on 2x B60 with streamed experts is roughly the
  single-card number, and the wall-vs-timeline gap says why: expert streaming, not the split.
- **Follow-on this suggests**: the dual prompt's lever is the expert-stream path (the `--peer-device` tier of item 6, or
  more VRAM for experts), not the stage pipeline. That is a different item from the archive's D2/D4 pair.

## Reproduce

```sh
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu STRATA_PREFILL_TIMING=1
IDS=$(python3 -c "ids=open('sycl/bench/v1/long.ids').read().strip().split(','); print(','.join((ids[:2000]*4)[:8000]))")
ARGS=$(python3 -c "import json;print(' '.join(json.load(open('strata-q2_0-dual.json'))['args']))")
{ sleep 45; echo "GEN 1 $IDS"; sleep 240; echo QUIT; } | ./sycl/build-b60/strata $ARGS 2>&1 | grep 'prefill timing: [0-9]* tokens'
# and the same against a -DSTRATA_TK_PER_MAX=33 tree
```

## Traps this run adds

- **A prompt shorter than `--prefill` is a single chunk.** D4 is about *adjacent* chunks; measuring it at 2,185 tokens
  with `--prefill 4096` measures one chunk with no neighbour, and the phase shares are then meaningless (96.7% in one
  interval). Use a prompt several times the chunk size.
- **Two build trees for a one-define A/B need `-G Ninja` and both compilers** (`CMAKE_C_COMPILER`=icx,
  `CMAKE_CXX_COMPILER`=icpx, exp 37), or ggml's C files compile with gcc and fail on `-fp-model=precise`.
- **Wall vs GPU timeline per chunk is the useful column**, not the total: it is what separates "the GPU is busy" from
  "the host is feeding it", and here it named expert streaming as the prompt's real bound.