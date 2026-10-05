# Experiment 35 - the split's scheduling levers, measured: draft-early null, D2's ceiling, dual variance

## Question

D2 (pipelined windows) was left open in README.sycl.md with a corrected ceiling. Before building the
speculative pipeline, this session tried the one piece of the fork's scheduling work that our window structure
allows *without* speculation, and measured it: **launch the MTP draft round before the window's tokens are
emitted** (Hardin22's section 4, "the draft round is launched before the window's tokens are emitted"). It also
records a
build trap that had been silently invalidating earlier runs, and the run-to-run variance of the dual config that
governs how any of these A/Bs must be run.

## The trap first: a build that failed silently and left a stale binary

`sycl/tools/build.sh` began with `set -uo pipefail` and then sourced oneAPI's `setvars.sh`. In an environment that
does not already export `OCL_ICD_FILENAMES`, oneAPI's `compiler/.../env/vars.sh:258` reads that variable unbound, and
an unbound variable inside a *sourced* file takes the whole shell down: the script exited **rc=1 with no output**,
and because the `tee` never ran, `build.log` still held the *previous* build's contents. That reads exactly like a
clean build. It is fixed (commit `f5c84a2`) by suspending `-u` around the source alone.

Consequence, recorded honestly: this session's earlier engine builds never ran. The binary every early measurement
used was built at 03:11 from a **different (older) source revision** - the fresh one prints an extra `DONE` field
(#471's prompt-tokens-read). exp 33's F1 engine arms were therefore the *same* binary, and exp 33 and its README
item now say so; F1's verdict rests on `mmvq_bench`, which really was rebuilt. **The symptom to watch for**: a
"clean" build whose log is stale - check the binary mtime, `strings <bin> | grep <new string>`, or `ninja -n <target>`.

## The draft-early A/B: null

**Change** (behind `STRATA_DRAFT_EARLY`, default on; `=0` is today's order). In the serve window loop the draft
call used to sit *after* the emit loop. The change moves the eos scan ahead of the emit (the draft is skipped on
eos, so the scan has to lead it), launches the chain before the printing, and keeps the logprobs path
(`req_logprobs >= 0`) on the old order because `copy_logits` reads head buffers the chain writes.

**Method**: dual Q2_0 (`--layer-split auto` -> K=24: CUDA0 layers 0-23, CUDA1 24-47 + head + drafter; 12,288 cache
slots on card 0), the 2,185-token prompt, 256 greedy new tokens, `STRATA_DECODE_TIMING=1`, two arms **back to back on
the same binary**.

| | arm A (`STRATA_DRAFT_EARLY=0`) | arm B (early draft) |
|---|---:|---:|
| TG | 34.29 tok/s | 34.20 tok/s |
| PP | 571.0 tok/s | 571.8 tok/s |
| window | 54.09 ms | 54.24 ms |
| verify | 49.47 (GPU-reach wait 23.24 + host 0.12 + stage 0.60) | 49.58 (23.26 + 0.12 + 0.62) |
| commit/emit | 1.68 ms | 4.66 ms |
| draft | 2.94 ms | 0.00 ms |
| windows / avg T / tokens per window | 138 / 1.97 / 1.86 | 138 / 1.97 / 1.86 |
| `DONE` | 256 tokens, drafts 119 of 134 | 256 tokens, drafts 119 of 134 |
| VRAM hits / lookups per layer-window | 19.71 / 130,560 | 19.71 / 130,560 |

**Parity: exact.** Identical token counts, identical draft counts, identical window and T profile, identical cache
lookups. (This config decodes deterministically - 100% of decode expert lookups hit VRAM, `CPU experts 0.00` - so
unlike the single-card runs of exp 33, token-identity is a usable gate here.)

**Verdict: null.** TG moved -0.26% and the window total did not shrink (54.09 -> 54.24 ms). The timing line shows
why: `draft 2.94 -> 0.00` while `commit/emit 1.68 -> 4.66`. The chain's ~2.9 ms did not disappear, it moved to the
other bucket. A host-side reorder cannot gate device work that the drafter has already enqueued on its own graph,
so there was never any overlap to win. The change was reverted (`generate.cpp` is back to the committed state; the
hold-rate instrumentation stays).

## D2's real ceiling, and why the speculation is *safer* here than in the fork

Window n+1's tokens come from the MTP chain seeded by window n's `outv`, and `outv` only exists once stage 1's
window graph - head included - has finished. So stage 0 cannot start n+1 until then, and the only stage-1 work left
to overlap is the tail (`commit/emit` + `draft` = 7.67 ms of 59.92 ms, measured). Steady state: serial is
S0 24 + S1 28 + tail 7.7 ~ 59.9 ms; overlapped it is max(52, 35.7) = 52 ms, i.e. **~1.15x**, and ~1.13x at the
measured 90% full-hold rate (`130 windows, fully held 117 (90.0%), drafts kept 119 of 133 (89.5%)`). The fork's
larger ratios come from a window where the head and draft are ~46% of it; ours are ~13%.

What the port has that the fork did not: `capture_commit` shows the **recurrent state is published by the commit
graph**, not the window graph - `gdn_conv_commit`, `gdn_step_norm_multi` and `native_qsa_indexer_append` are all
driven by the mapped commit counts (`verify.cpp:1054+`). A mispredicted window therefore costs recomputation, not a
DeltaNet state restore, and no snapshot is needed. That makes the speculation materially safer here than in the
fork (which had to snapshot and restore on a side stream) - but ~1.13x still does not obviously pay for a change to
the one part of the port where the bounded device spin turns a stall into a wrong window rather than a hang.

## How a dual A/B has to be run: the config's run-to-run variance

Three dual runs of the **same binary, same config, same placement** (K=24, 12,288 slots, pcie_frac 0.38):

| run | TG | PP | window |
|---|---:|---:|---:|
| d2_hold2 | 31.19 tok/s | 344.7 tok/s | 59.92 ms |
| de_a (arm A) | 34.29 tok/s | 571.0 tok/s | 54.09 ms |
| de_b (arm B) | 34.20 tok/s | 571.8 tok/s | 54.24 ms |

That is a **~10% TG and ~66% PP spread between runs that differ in nothing**, while the two back-to-back arms of
the same A/B agree to 0.3%. exp 33's +-1.5% floor was measured on the *single-card* config and does not transfer.
The rule this establishes: **a single dual run cannot resolve anything below ~10%**; only same-binary,
back-to-back arms can resolve a few percent, and any cross-run comparison on this box (including against numbers
recorded in earlier sessions) must be treated as provisional until re-measured on the current binary.

## Conclusion

The split's scheduling avenue is measured out: no host gap between the stages (exp 34), host work into host-tasks
and both graphs in flight under the noise floor (exp 34), draft-before-emit null (this experiment), and the
speculative pipeline capped at ~1.13x for the riskiest path in the port. The upside that remains on the dual is in
the ~52 ms of *stage* time - decode efficiency, the P0 kernel items - not in how the two cards are sequenced.