# Experiment 41 - conversation parking on a layer split: implemented, measured, and one open question

**Question** (README item 3, archive D5). The port refused conversation parking under `--layer-split` outright, so a
server holding two long conversations re-read the non-shared prefix on every switch. ExTV's patch 07 reports
20-48 s -> 0.4-0.5 s for that switch, with 0.8-1.7 GB of snapshot per 30K tokens.

**Verdict: implemented and measured. A switch drops 1.9x-2.9x. One output difference is unexplained and is recorded
below rather than explained away.**

## The measurement

Two **prefix-disjoint** 4,000-token conversations (the v1 long prompt, and the same prompt rotated by half so its
first id differs), alternating A, B, A, B on 2x B60, greedy, split auto -> K=22. Two arms on one binary, serialized
behind a card-free check: `--conversation-cache-mib 0` vs `8192`.

| request | no parking | with parking | |
|---|---:|---:|---|
| A, 1st visit | 7367.6 ms | 7396.6 ms | full prefill either way |
| B, 1st visit | 6478.7 ms | 6627.6 ms | full prefill either way |
| **A, return** | 7021.7 ms | **3684.4 ms** | 2185 tokens reused -> **1.91x** |
| **B, return** | 6435.6 ms | **2184.9 ms** | 3278 tokens reused -> **2.95x** |

Park and restore, from the engine's own log:

```
parked  4005 tokens in 141.4 ms; snapshot 416,545,848 B
parked  4011 tokens in 107.9 ms; snapshot 298,166,048 B
restored 2185 tokens (checkpoint) in 32.4 ms
parked  4002 tokens in 126.5 ms; snapshot 416,484,768 B
restored 3278 tokens (checkpoint) in 36.1 ms
```

**Why 1.9x-2.9x and not the ~200x the raw numbers suggest.** A restore is 32-36 ms, but it only brought back a
*checkpoint prefix* (2,185 and 3,278 of 4,000 tokens); the remainder was re-read. So the switch pays
`restore + prefill(the rest)`. The copy stops being the bottleneck - which is the point - but the prefill it does not
cover still dominates. ExTV's 0.4-0.5 s comes with the whole conversation restored, not a checkpoint prefix.

## What changed in the code

| piece | what it does |
|---|---|
| `SavedStage` (`conversation_cache.hpp`) | one stage's parked state: its layer range, running state, and its own K/V |
| `SavedConversation::stages` | the later stages in stage order; **empty on a single card**, so that path is byte-identical to before |
| `with_draft` on the snapshot API | the drafter is bound to the **last** stage (`mtp.bind(last_st ...)`), so only that stage's image carries the draft layer |
| `allow_stage_parts` on the same API | a split's checkpoints carry per-stage parts and the request path needs them back; validation tolerates them only for an image that has per-stage images |
| `stage_bytes` / `stage_save` / `stage_validate` / `stage_restore` | per-stage operations run inside an `OnDevice` scope for that card - the pattern `checkpoint_at` already used |
| park estimate | sums every stage's bytes, so `make_room` cannot under-count and overfill the budget |
| restore order | **all** stages validate before **any** restores |

`ctest` 29/29 throughout.

## The open question: one token differs

Generated tokens, request by request, both arms:

```
req1: IDENTICAL (6)     req2: IDENTICAL (12)     req3: IDENTICAL (3)
req4: DIFFERENT (12 vs 12), diverging at token 2: nopark 21966, park 3575 - tokens 4-12 identical
```

The obvious explanation is that the two arms prefill different amounts (722 vs 4,000 tokens in request 4), so the
expert GEMM shapes differ and the engine's documented cache-vs-CPU rounding difference ("a reply can differ slightly
from a run without the cache") applies. **That explanation is not supported**: request 3 also had a different split
(2,185 reused) and came out bit-identical. So the cause is unknown. Two things are worth separating:

- Is the restored state itself wrong (a missing or stale per-stage page)?
- Or is it numerically equivalent but not bit-identical, and the divergence is a legitimate rounding artifact?

`STRATA_SNAPSHOT_VERIFY=1` exists and fingerprints the restored draft K/V after a synchronized restore; running the
park arm with it set is the next step, and it is the difference between "unexplained" and "diagnosed".

## What it does not yet do: per-stage K/V reuse

`ConversationKvReuse` is what makes a *rewritten* conversation cheap: after a restore the cache hands the just-used
buffers to the next park with an `unchanged_tokens` bound, and only the cells past `first_dirty` are re-copied. Under
a split that carries **only the primary's** K/V - `conversations.retain(std::move(incoming->kv), ...)` - so **every
later stage's pages are re-copied in full on each park** (`reused_kv_bytes=0` in every line above).

This is not only a compaction concern: `limit_reuse(first_dirtry)` exists for *any* rewrite of a live conversation -
regenerate, branch, or a future compaction - and under a split all of them pay a full re-copy of stages 1..N.

**Implemented, and MEASURED NULL.** `StageKvReuse` now rides along in `ConversationKvReuse`, `stage_save` takes its
stage's retained K/V and passes `unchanged_tokens` down exactly as the primary does, `stage_capture_bytes` does the
retained-K/V accounting so `make_room` is not charged for pages that are not copied, `retain()` keeps every stage and
`limit_reuse` clamps them all. On the same two-conversation run it **does not engage**: every park still reports
`reused_kv_bytes=0`, and the switch times are unchanged (3675.8 / 2182.4 ms against 3684.4 / 2184.9 ms before the
change - identical within noise). ctest 29/29.

So the plumbing is correctly shaped but inert, and **the reason is not diagnosed**. The candidates, none confirmed:
the retained image is dropped before the next park (`drop_superseded` and `make_room` run in between),
`limit_reuse(read_from)` sees a `read_from` of 0 on a conversation switch and zeroes the reuse, or `retain()`'s budget
check declines it. Until one of those is ruled in or out with a log line, treat per-stage reuse as **not working**,
and treat the "no new kernel needed" framing for compaction as unproven.

## Four defects found in this item, all in the new code

1. **`live.ids` left empty by `stage_save`.** `conversation_checkpoint_restore` refreshes the last pooled-indexer row
   only when `!c.ids.empty()`, and `checkpoint_targets` bounds the pooled rows by `ids.size()`. An empty ids
   validated vacuously and would have restored a stage with **stale pooled rows** - silently wrong, and it would
   have passed a smoke test.
2. **The `stage_parts` guard.** Removing the up-front refusal was not enough: `view_validate` rejects any view
   containing a checkpoint with `stage_parts`, and under a split every checkpoint has one, so every park still
   skipped on `checkpoint is not a live token prefix`.
3. **Stripping `stage_parts` broke the checkpoint restore.** The first fix cleared those fields to satisfy
   validation - which then made the request path fail with `c->stage_parts.size() != stages.size()` and SIGSEGV,
   *after* a 30.9 ms restore. The correct fix is to keep them in the image and relax only validation.
4. **The test measured a truncation, not a switch.** Both prompts were prefixes of each other, so the engine
   rewound instead of parking - correct behaviour that looked like a failure.

## Traps

- **Two engines on the same cards starve the split search**: an arm that started while the previous one held VRAM saw
  `CUDA1 0.00 GiB free`, the auto-split collapsed to K=34/K=47, and the process died with SIGSEGV - in an arm that
  does not touch the parking path. `perf_matrix.py` refuses to start when VRAM is held; a hand-written harness must too.
- **`drm` fd counts are not a VRAM check**: they belong to the desktop session and never go away. Use `xpu-smi`.
- **`long.ids` is 2,185 tokens and `short.ids` is a prefix of it**, so neither can supply a second conversation;
  a rotation is the cheapest way to get a prefix-disjoint one.
- **A segfault in an unrelated arm is evidence about the harness.** Both crashes reproduced with a starved split and
  vanished once the arms were serialized behind a real VRAM check.

## Reproduce

```sh
/tmp/item3_disjoint.sh                      # both arms, serialized behind a card-free check
diff <(grep '^T ' /tmp/item3c-nopark.log) <(grep '^T ' /tmp/item3c-park.log)
```

Next: `STRATA_SNAPSHOT_VERIFY=1` on the park arm to fingerprint the restored state, then per-stage K/V reuse.