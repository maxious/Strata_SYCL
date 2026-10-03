# Debate log

Skeptic lane failed to spawn (no category model for `ultrabrain`); per ulw-research fallback the lead ran the attack pass over primary source directly. All file:line citations were re-verified against the checkout by the lead (9/9 landed on exact cited lines; Strata native_mmvq Q2_K absence confirmed by grep).

## Round 1 — P0-L1: is reorder+ESIMD really on the decode path?
- claim: C-P0L1-1, C-P0L1-2
- attacker: lead (skeptic role) — "could the reorder+ESIMD path be dead code or prompt-only? The README claims XMX=prompt, ESIMD=decode."
- evidence: ggml-sycl.cpp dispatch (read by lead, wave-0-skeptic-dispatch.md): the reorder-ESIMD branch (L4882-4893) sits in `ggml_sycl_mul_mat`'s single/small-batch decode ladder (`should_reorder_tensor` requires `src1->ne[1] <= 8`); env `GGML_SYCL_ENABLE_ESIMD`/`GGML_SYCL_ENABLE_OPT` default to 1. dmmv.cpp:1876/2012/2046 kernels confirmed at exact lines. Mechanisms cross-confirmed by two lanes (reorder-decode + wide-load both independently cited dmmv.cpp/esimd.hpp/vecdotq.hpp).
- verdict: SUPPORTED. On the decode path, on-by-default, portable. Weakness noted: it's opt-gated (env + arch probe), so Strata must gate it too and keep default-fastest.

## Round 2 — P0-2: is the 'no llm.cpp port' verdict reliable, and is the lever actually Strata-side?
- claim: C-P0-2-1/-2/-3
- attacker: "exp 09/10/11 already concluded this — is the lane just repeating prior claims (circular)?"
- evidence: int8-dequant lane re-established from PRIMARY source with fresh greps (supports_mmq false 4084; mmq.cpp zero i-quant kernels; i-quant GEMM CUDA-only mmq-load-tiles.cuh/mmq.cu). This is primary-source, not circular. Strata-side levers (get_int_from_table_16 etc.) are code-denominated in Strata's own iq_kernels.dp.cpp.
- verdict: SUPPORTED. Not circular — primary-source fresh verification. P0-2 is correctly a Strata kernel project, not a llama.cpp port.

## Round 3 — P0-3: is the README premise ('pipe hand-off through native dev2dev') actionable?
- claim: C-P0-3-1 refuted-as-gain; C-P0-3-2; C-P0-3-3 (recommend re-scope/drop)
- attacker: "the README explicitly lists P0-3 as a lever — are you sure it's not just hard to see the gain?"
- evidence: Strata architecture confirmed by lead grep — separate SYCL context per stage (generate.cpp:2655-2680 `OnDevice on(st.dev)`), hand-off already host-pinned (malloc_host generate.cpp:4816), NO peer-access call anywhere in Strata. llama.cpp's own comment (ggml-sycl.cpp:7189): raw peer-USM q->memcpy across separate contexts is a silent no-op. exp-03 measured hand-off ~34µs/256KiB vs the 1.4× decode gain that comes from layer-split compute distribution.
- verdict: REFUTED-AS-GAIN. Direct dev2dev requires a single-context rebuild that breaks per-stage isolation, to recover a ~34µs hand-off that is not the bottleneck. The 1.4× gain already ships. P0-3 should be dropped/re-scoped (no new code; optionally adopt llama.cpp's DEV2DEV_MEMCPY host-staged framing for hardening, which Strata already effectively does).

## Round 4 — P0-4: is Q2_K the real gap, and is it worth it?
- claim: Q2_K is 'only genuinely missing dense type'
- attacker: "is Q2_K on any deployed Strata model? A gap with no user is not a P0."
- evidence: lead grep — Q2_K absent from native_mmvq.hpp supported codes AND from every README/INTEL.md; Coder reference model is IQ1_M. So Q2_K is a LATENT gap, not an active bottleneck.
- verdict: PARTIAL/RE-PRIORITIZED. Port Q2_K only if a model uses it; today the P0-4 win is proving the ESIMD infra is viable on the B60 through the already-needed P0-L1 port rather than a standalone Q2_K push. Q8_0 int4 wide-load vs Strata's load16_a2 is a cheap A/B worth keeping.
- debate-graph note: this re-prioritization reflects reality (no Q2_K user), preserving correctness.