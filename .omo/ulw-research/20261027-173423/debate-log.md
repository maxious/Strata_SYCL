# Debate log
Round 1 (attacker: smol-model one-shot on lead-authored fact set; the ultrabrain skeptic lane and browser lanes died with the RPC host - recorded here as the substitute, single-pass):
- ATTACK (mild) B70->B60 transferability: +14% decode and 56->18us are author-run on B70; kept directional, never converted to B60 predictions.
- ATTACK cache-residency of the 22.2 us/expert figure -> REFUTED by bench design (F53): malloc_device, 64 distinct experts, ~419 MB streamed, header states weights come from DRAM not L2.
- SURVIVES thesis (bytes + fused-scale epilogue are the M=1 lever), port ordering, upstream evidence (PR #29864 DPAS, #29245 grouped MoE, oneDNN/oneMKL no GEMV), and the answer to 'joint_matrix trellis GEMV' (joint_matrix ranks third, for resident expert tiles).
- SURVIVES-with-caveat epilogue +7 us: single measurement, needs per-32 ablation before it becomes a design constraint.
Round 2 (lead cross-check): the libsycl-version root cause (F51) is the same class of failure our machine already dodges - oneAPI 2026.2 runtime UR 1.17.39758 advertises the matrix combos, and the repo's own sg16 kernels run (F37). No new attack landed; convergence held.
