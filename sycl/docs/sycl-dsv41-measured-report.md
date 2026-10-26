# What the SYCL build of DeepSeek V4.1 does today

Version 0.2.6, engine 0.1.51. This page gives measured numbers for the SYCL path
that runs DeepSeek V4.1 on Intel graphics cards. Each number says what it was
measured on. Nothing on this page is a guess.

## What was measured

The build was made from branch `sycl/dsv41-sycl-support` at commit `ce6218ee`.
It ran on a machine with two Intel Arc Pro B60 cards, 24 GB of graphics memory
each, with Intel oneAPI 2026.2, the Level Zero driver and the `xe` graphics
driver. The model is DeepSeek V4.1 Flash in the `DeepSeek-V4.1-Flash-1017-GSQ-RCO` pack: IQ1_M 59.45 GB, IQ2_XXS 49.05 GB, IQ2_XS 23.57 GB, Q5_K 5.33 GB and an IQ4_NL engram of 110.60 GB, with zero Q4_K tensors and 248.83 GB on disk in total (231.7 GiB; main shards 138.24 GB). It is run with
most of its experts kept in system RAM.

Every run asked the same prompt, "Reply with exactly: The capital of France is",
and was asked for 10 prompt tokens and 8 answer tokens.

## One card

One card, device 0, ran the model on 2026-10-27 and finished with exit code 0.

- Prompt: 7.92 tokens a second (10 tokens in 1262 ms, windows of 8192 tokens).
- Answer: 1.65 tokens a second (8 tokens in 4231 ms).
- The first answer token was id 671. The eight ids were
  `671,6102,294,8760,344,2619,51119,42499`.
- Read with the model's own tokenizer, the answer was:
  "The capital of France is **Paris**."
- 109 of the 1680 experts the answer asked for were worked on the CPU.

An earlier build, `ddc3c9e7`, recorded 157 of 1920 experts on the CPU lane. That
run asked for a longer answer, so the two counts describe different runs, not a
change in the model.

## Two cards

The same build and the same model, both cards (`--gpus 0,1`), two runs one after
the other on 2026-10-27. Both finished with exit code 0.

Both runs shared the layers the same way: card 0 took layers 0-20, card 1 took
layers 21-39 and the final step.

- Run 1: prompt 19.88 tokens a second (10 tokens in 503 ms); answer 2.44 tokens
  a second (8 tokens in 2873 ms).
- Run 2: prompt 38.09 tokens a second (10 tokens in 263 ms); answer 12.08 tokens
  a second (8 tokens in 579 ms).
- Both runs answered "The capital of France is **Paris**." and printed the same
  eight token ids.

Run 2 is faster than run 1 because the experts were already in RAM. Run 1 read
them from the model's files as it went; run 2 copied all 11011 of them into RAM
at load, 94.7 GB in 36.0 s.

Neither run reset a card. After each run, a journal check for card resets found
no matches (exit code 1 means nothing was found).

## The checks that must pass

- Passed: both ds4 parity tests, `ds4_parity` and `ds4_blas_parity`, pass on the Linux SYCL build (2 of
  2). Failed: the default build's test suite has one pre-existing failure, `isa_guard_check` on
  `iq512x_rows` (1 of 57). Not run: the CUDA and HIP builds were not run because no CUDA or HIP hardware
  is present on this machine.
- For the Q2_K, Q4_K and Q5_K expert shapes, the SYCL code and the CPU code
  agree: no value matched exactly (0 of 51), every value was within one BF16
  rounding step (51 of 51), and none was outside it (0 of 51). Measured on the
  SYCL build from this branch at commit `ce6218ee`.

## What is not finished

- The loud-spin control (`STRATA_SPIN_MAX=1`) prints its error, "the card never
  saw an answer", but the run ends with exit code 0 instead of 1. On the earlier
  build the same control ended with exit code 1.
- A run asked for 8 answer tokens (`--n 8`) stops before the end-of-answer token
  (id 1). The answer is still correct.
- Two fixes sit outside the files this plan may change: publishing the token ids
  from the host, and the truncation fix above. They were left for later work.
