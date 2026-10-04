# Experiment 32 - the dual-B60 "EOS bug" was not a bug: the model ends the turn on the synthetic benchy prompts

## Question

benchy v1 on 2x B60 (layer split, `--serve --layer-split auto`) decoded **1 token and stopped** on the 8,000 /
40,000 / 128,000 / 256,000-token prompts, while the same sizes on **one** B60 decoded 256. Same model, same pack,
same flags apart from the split. A layer-split correctness bug (a corrupted prompt state handing the head an
immediate end-of-turn) was the natural reading - and the cost of getting it wrong is a split that silently answers
wrong.

Measured on: Qwen3.8-Flash-Next Q2_0 (GSQ-RCO, two shards), engine 0.1.35-sycl `build-b60/strata` (commit
1fa79db63be9 + local changes), Ryzen 7 5700X3D / 121.5 GiB, 2x Arc Pro B60 32 GB in Gen4 x8 slots, one or both; the
benchy v1 config (`--prefill 4096`, `--kv int8`, `--kv-resident 32768`, `--stream-experts`, `--spec 2`, `--mtp`,
`--max-context 262144`) with and without `--serve --layer-split auto`, cold page cache, 256 greedy new tokens.

## What the split was not doing

Everything about the split checked out, and each check narrowed the search:

- **KV streaming is not involved.** A dual config without `--kv-resident` (a different split point, K=24) decodes the
  same 1 token.
- **The prompt reads fine.** PP is normal and *faster* than one GPU's on the same prompts (dual 705.7 / 944.6 tok/s
  at 8,000 / 40,000 against single 648.9 / 668.2); TTFT is honest. The failure is the first token after the prompt.
- **The turn-token prompt split is not involved.** `--turn-token -1` (one batched segment, no checkpoints) decodes
  the same 1 token.
- **The logits are not degenerate.** The serve `logprobs` request shows `<|im_end|>` at log-prob **-0.000065**
  (p ~ 1.0) with the next-best token at -9.98 - a sharp, confident distribution, not NaN or uniform garbage.
- **The residuals are byte-identical between the modes.** `STRATA_PREFILL_DUMP_R` at 2,500 tokens (every 64th
  position, the last stage's final residual rows) gives the same md5 (`8558b2d242394dc222e214f1123396c8`) for the
  serve and the one-shot runs: the prompt paths leave exactly the same state.
- **One GPU in serve mode fails identically.** `--serve` on a single B60 at 2,500 tokens also stops after 1 token
  (log-prob ~ -0.000034). The split is innocent; **serve mode** is the discriminator.

## The answer: read the first token, not the count

The one-shot runs were emitting the same end-of-turn token all along. The **output of the single-GPU one-shot runs
starts with 248046 (`<|im_end|>`) at every size from 2,500 up**:

| prompt | single one-shot, first output token | dual serve |
|---|---|---|
| 20 | 248068 (a content token) | 256 tokens |
| 2,185 (`long.ids`) | 248068 | 256 tokens |
| 2,500 | **248046 (EOS)** | 1 token, stop |
| 8,000 / 40,000 / 128,000 / 256,000 | **248046 (EOS)** | 1 token, stop |

The one-shot decoded "256 tokens" only because it **does not stop at EOS by default** (`o.stop_eos = false`,
generate.cpp) - it printed `248046 198 248045 74455 ...` (end-of-message, newline, start-of-next-turn) as the first
four of its 256 "generated" tokens. **Serve stops at EOS unconditionally** (its request loop tests `o.eos_ids` with
no gate), so the same model state is reported as 1 token. Two harness modes, two stopping rules, one model answer -
and the matrix columns looked like a correctness bug.

Why the model ends the turn: benchy v1's sizes above 2,185 are **synthetic** - `long.ids[:2000]` repeated
(`perf_matrix.prompt_file`). The v1 long prompt is one message; repeating its first 2,000 tokens makes the model
treat the (second) message as finished. This is content, not length: serve at 2,100 and 2,200 (2,000 + 100/200 of
the repeat) also ends the turn, while 2,185 - the **real** `long.ids`, which closes its message and opens a new
turn - continues, as does the 20-token prompt. (The real long prompt at 2,185 decodes 256 in serve mode on both one
and two B60s.)

## The fix (harness, not engine)

`sycl/tools/perf_matrix.py` now passes **`--stop-eos`** to the one-shot runs (native and docker), so both protocols
stop at the model's end-of-turn token and the single/dual columns measure the same thing. The generated `matrix.md`
caveats say so, and note that the v1 prompts over 2,000 tokens end the turn right away: those rows decode 1 token -
the model's answer to that text, not a failure, with TG there being one token's time.

Verified with the fixed harness: single 8,000 gives `output : 248046` and `decode 1 tokens` (parity with dual), and
dual 2,185 still gives `DONE 256` (parity on the real prompt).

No engine change: the engine computes the same state and predicts the same token in both modes. With the corrected
protocol the honest reading of the split on these prompts is the **PP/TTFT** columns, where the split wins (944.6 vs
668.2 tok/s at 40,000).

## Traps worth knowing

- **A stopping-rule asymmetry between two harness modes reads exactly like a model-correctness bug** when the model
  predicts the end-of-turn token early. The count column is not the finding; the first token is (`output :` for
  one-shot, `T <id>` / `DONE` for serve).
- **Repeated-prompt benchmarks are not neutral text.** `long.ids[:2000]` repeated makes a real model end the turn;
  a size sweep built by repetition can cross a "the model is done" boundary and look like an engine cliff.
- **`STRATA_PREFILL_DUMP_R` is the cheap state oracle** for "do two paths leave the same state": dump the final
  residual rows every 64th position from each path and compare hashes before looking for a corruption.
- **Serve's `KV streaming: 0.00% of N block reads hit VRAM` on a fresh request is normal.** A request starts with an
  empty resident map (`session_zero`), so the first window's resolve pulls every block from the pinned host copy; the
  hit rate is a symptom of the request being young, not of a broken cache.
- A serve session driven with a **second** `GEN` request (no `QUIT` between) idled without reading the prompt in the
  early repro; benchy sends one request per process so it never hit this. Left for its own experiment.
