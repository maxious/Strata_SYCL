# Experiment 05 - decode round profile: where the 52 ms/round goes

## Question

README.sycl.md item 7 / INTEL.md "Planned" #7: *"Fewer graph nodes per decode round (~2,500 at ~5 us):
norm+rope, scores+top-k, gate+quantize fused."* The decode round on the B60 is the engine's biggest per-token
cost. Before fusing kernel pairs, measure the round and decide whether node **count** or node **work** is the
limit - fusing only helps if the round is launch-gap-bound, not kernel-bound.

## Baseline (reproducible, warm, Coder IQ1_M on one B60)

Serve mode, `--spec 4 --mtp <rt>`, 13-token prompt, 16 generated, greedy window 4 (drafts accepted 10 of 24):

    run1: decode 16 in 833 ms -> 19.2 tok/s, drafts 10/24
    run2: decode 16 in 834 ms -> 19.2 tok/s, drafts 10/24
    run3: decode 16 in 833 ms -> 19.2 tok/s, drafts 10/24

52 ms per 4-token verify window round. (Lower than INTEL.md's B70 78 tok/s because this is a B60, not a B70, and
a 42% draft-acceptance garbage-input prompt, not a code prompt at 87%.)

## Window graph node counts (STRATA_VERIFY_NODES)

    the 6-token window graph has 2625 nodes: 2619 x kernel; 6 x memcpy;
    the 4-token window graph has 2541 nodes: 2537 x kernel; 4 x memcpy;
    the 2-token window graph has 2457 nodes: 2455 x kernel; 2 x memcpy;
    the 1-token window graph has 2415 nodes: 2414 x kernel; 1 x memcpy;

## Why node fusion would not pay (measured arithmetic)

The 4-token round is ~52 ms over 2,541 nodes = **~20 us per node**. Even if every node is a kernel whose GSYL
launch gap is ~5 us (the compiler's launch-gap when idle), the 2,541 nodes paint a picture where most of the 52 ms
is the **kernel work inside the nodes**, not the gap between them: 2,541 nodes x a ~5 us launch gap would be only
~12.7 ms, leaving ~39 ms of irreducible kernel execution. So the round is **kernel-bound, not node-bound**.

Fusing a handful of pairs (norm+rope, scores+topk, gate+quantize, ~a few dozen nodes saved) is worth at most a
few hundred us per round = under 1% of 52 ms - within run-to-run noise. **Item 7 is closed with the measurement:
not implemented, per the README's own rule.**

## The real next lever (flagged, not built)

The kernel-level bench shows the multi-token dense kernels decay sharply with token count:

    fused GR read (the down/norm family): 1 tok 35.2 us (375 GB/s) -> 6 tok 116.7 us (113 GB/s)

The 6-token value (116.7 us) is at INTEL.md's documented post-improvement figure (135 -> 110 us), so it is not a
regression. But the 1->6 token decay (375 -> 113 GB/s, ~3x) is the same family of multi-row latency-bound loss
that llama.cpp's newest decode work (ESIMD wide-load DMMV/MMVQ, reorder) targets. That is the honest next
Experiment 6 candidate: the dense multi-kernel's register/occupancy at > 1 token, not graph-node fusion.

## Validity

- The round profile is reproducible: all three baseline runs at 19.2 tok/s, drafts 10/24.
- The node dump is deterministic and matches INTEL.md's documented ~2,500-node figure.
- The conclusion (kernel-bound) follows from the arithmetic (20 us/node measured work vs ~5 us launch gap).

## Conclusion

Decode on the B60 is **kernel-bound**: 52 ms/round over 2,541 nodes at ~20 us/node of real work. Fusing graph
nodes (README item 7) would save <1% and is **parked** with this measurement. The 1->6-token decay of the dense
multi-kernels (375 -> 113 GB/s) is the higher-payoff next experiment. No engine change in this experiment.