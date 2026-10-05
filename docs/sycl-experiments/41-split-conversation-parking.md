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

## Per-stage K/V reuse: implemented, and the null had TWO causes (both fixed)

The reuse makes a *rewritten* conversation cheap: after a restore the cache hands the just-used buffers to the
next park with an `unchanged_tokens` bound, and only the cells past `first_dirty` are re-copied. Under a split it
has to carry **every stage's** K/V, not just the primary's, or each park re-copies stages 1..N in full - which costs
every conversation *rewrite* (regenerate, branch, compaction), not just a switch.

The first split-parking run measured `reused_kv_bytes=0` on every park and the feature was written up as inert.
That reading was incomplete. Two independent defects kept it from engaging, and only the second was visible:

1. **`conversation_snapshot_save` takes the reuse BY VALUE.** `park_current` passed `std::move(reuse)`, which
   carried `reuse.stages` into the parameter; the parameter consumes `reuse.kv` and destroys `reuse.stages` with
   it. The stage-save loop *after* that call read a moved-from (guaranteed-empty) vector, so every `stage_save`
   got an empty `StageKvReuse`. Fixed by moving `reuse.stages` out into a local **before** the save call.
2. **`conversation_snapshot_capture_bytes` decided "is this the draft?" with `i + 1 != layers`.** That holds
   single-card, where the last slot is the drafter - but a split's primary holds its own main layers *alone*
   (`with_draft=false`), so `layers` is the main-layer count and the expression validated the **last main layer**
   against the *draft's* geometry, pooled rows included. The retained K/V was rejected and the reuse silently
   dropped. This is the defect the new `reuse declined (...)` diagnostic names on its first run:
   `reuse declined (conversation snapshot: incompatible K/V geometry)`. Fixed by keying the decision off
   whether the image carries a draft at all (`!with_draft || i + 1 != layers`).

`stage_capture_bytes` carried the same expression, where a **non-final** stage has `draft == nullptr` - a null
dereference that this 2-GPU box cannot reach but a 3-GPU split would. Fixed the same way (`draft == nullptr ||
i + 1 != slots`), since that signature takes a real pointer.
3. **The estimate loop vacated `reuse.stages[i]` and appended.** It moved the entry out to price it, then
   `push_back`ed it after the hole - leaving index `i` empty - and the save loop reads `stage_reuse[i]`. With one
   later stage the vector was `[empty, full]` and **every stage re-copied in full** while the primary reused
   correctly, with nothing logged anywhere. It was found by tracing the per-slot `keep`, not by reading: the
   retain and park sides both showed the stage's vector arriving populated (`stage0 kv=8 unchanged=2182`) with
   `unchanged` correct. Fixed by assigning back in place (`i < reuse.stages.size() ? reuse.stages[i] = ... :
   push_back(...)`). `STRATA_REUSE_TRACE=1` prints the per-slot `keep`/`size` and both sides' slot counts.

The lesson worth keeping: **the silent `err.clear()` on that drop is what hid both bugs.** A declined reuse is not
an error - it just re-copies everything - so nothing said so. One line naming the reason would have found cause 2
on the first run instead of after a code review.

### Measured after the fix

Pinned expert configuration (`--pcie-frac 0.32 --adapt-swaps 0`, both arms, split auto -> K=22, 2x B60, ctest
29/29 on the binary under test). `long.ids` on this box holds 2,185 tokens, so A is 2,185 and B is 185 - disjoint
by construction, but much smaller than the 4,000-token pair above.

| request | no parking | with parking | |
|---|---:|---:|---|
| A, 1st visit | 3832 ms | 3750 ms | full read either way |
| B, 1st visit | 829 ms | 905 ms | full read either way |
| **A, return** | 3446 ms | **168 ms** | 2182 reused + 3 read -> **20.5x** |
| **B, return** | 809 ms | **172 ms** | 182 reused + 3 read -> **4.7x** |

- **Tokens identical: `T_DIFF` = 0 lines across all 48 tokens of the four requests.**
- Restore 22.6 ms / 18.7 ms; `retained=33530232` at the first restore.

**The reuse now covers the whole conversation.** After the third fix, the park that follows a restore reports
**`reused_kv_bytes=33275520` of the 33,530,232 retained - 99.2%** - with 13 slots reusing at `keep=1116160` of
`size=1124352` each (5 primary + 7 stage + the draft at `unchanged=2181`, the draft's final-cell refresh). Before
that fix it was 12,905,600, exactly the primary's 5 slots, which is what pointed at the vacated index.

(The first two fixes took the counter from 0 to 12,905,600 and the switch from 3446 ms to 168 ms; the third took
the reuse from one carve to all of it.)

## Four defects found while building this item, all in the new code

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

## The review pass: five more defects, and what the measurement actually says

Re-reading the item after it shipped found five more problems, two of them the reason the per-stage reuse never
engaged (above). The rest:

3. **`STRATA_SNAPSHOT_VERIFY` was itself broken under a split.** It fingerprinted `incoming->kv.back()` as the
   drafter's K/V, but a split image's primary carries only its own layers - so the check named as "the next step"
   for the token difference was reading a main layer. It now verifies the last stage's own draft slot, on the card
   the drafter lives on.
4. **Dead code.** `conversation_checkpoint_bytes` was declared, defined, and never called (`stage_bytes` computes
   its own estimate). Removed.
5. **`stage_bytes` undercounted the admission estimate**, and its comment claimed the per-stage ids copy did not
   exist. It does - `ConversationCheckpoint::ids` is a by-value vector, and `SavedStage::bytes()` already counted
   it - so the estimate now does too.

### The single differing token was NOT a parking defect

The earlier write-up recorded "one generated token differs (request 4, position 2)" as undiagnosed. It is
diagnosed, and it was never in the parking path:

- **The two arms were not computing the same thing even on a first visit.** With parking off and on, request 1 -
  which touches no cache code at all - diverged at token 9 (`411` vs `279`). A first visit cannot differ if
  parking is the cause.
- **The cause is the per-stage PCIe probe.** The split probes each card's link and derives `pcie_frac` from it;
  that probe returned 11.5 GB/s in one startup and 13.8 GB/s in another (same host, same cards), giving
  `pcie_frac` 0.32 vs 0.38. The share decides how much expert work is **streamed** instead of computed in place -
  a different arithmetic path, hence different logits and a flipped near-tie. `STRATA_CKPT_REREAD`'s own note
  says the same thing: with the VRAM expert set fixed the answer must match token for token.
- **The fix is to pin it:** `--pcie-frac` (giving it also skips the per-stage probes) and `--adapt-swaps 0`. An
  A/B that does not pin both is comparing two numerics, not two parking modes.
- **The park arm is deterministic**: two park runs agree token-for-token on requests 1-3. So a park-vs-nopark
  difference is systematic, never run-to-run noise.

**Closed by measurement.** With the expert configuration pinned, the no-parking and parking arms produce
**identical tokens for all four requests** (48/48, `T_DIFF` = 0). Split parking is token-exact against a full
re-read; the original "one differing token" was this probe, not the parking path.

### Traps this item cost real time

- **`long.ids` on this box holds 2,185 tokens**, so "A = ids[0:4000]" is 2,185 tokens and "B = ids[2000:6000]" is
  185 - the two overlap, so the harness must build disjoint conversations from what the file actually has, not
  from the offsets the earlier note used. Timings here are not comparable with the 4,000-token numbers above.
- **A ctest "hang" can be a driver wedge.** ctest was blocked in its event loop with no test output for 25
  minutes; the truth was `dequant_s2_parity --selftest` in **D state** on `drm_pagemap_acquire_owner` holding both
  render nodes, with `xpu-smi` hanging too. `ps | grep _test` misses it (the comm name truncates to
  `dequant_s2_pari`); use `ps --ppid <ctest-pid>`. D state cannot be killed - it needs a reboot or a driver reset.
- **A build gate that pipes the compiler into `grep` reports success on failure** (the pipeline status is grep's,
  not the compiler's), which puts a **stale binary** into the A/B. Gate on the compiler's own exit code and print
  the binary's mtime, or the freshness rule is decorative.
- **Do not edit a running bash harness.** bash reads scripts lazily by byte offset; a mid-run edit made it spawn
  rogue engines that would have starved the next arm's split search.
- **A stalled read is not always a stall**: one arm read its first prompt in 42 s where another took 3.8 s with no
  config difference. Only a reproduced stall point is evidence.

## Reproduce

```sh
/tmp/item3_disjoint.sh                      # both arms, serialized behind a card-free check
diff <(grep '^T ' /tmp/item3c-nopark.log) <(grep '^T ' /tmp/item3c-park.log)
```

Next: `STRATA_SNAPSHOT_VERIFY=1` on the park arm to fingerprint the restored state, then per-stage K/V reuse.