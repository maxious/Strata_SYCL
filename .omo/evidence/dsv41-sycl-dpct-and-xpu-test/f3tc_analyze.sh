#!/bin/bash
# F3 two-card manual QA (task st_01a178c4) - analysis phase.
# Format: every CMD line is followed by its own un-piped *_RC= line (rc=$? taken immediately after the command).
E=/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test
R=$E/f3tc-receipts-analyze.txt
PY=/tmp/tokvenv/bin/python

echo "=== F3 two-card manual QA: analysis phase (task st_01a178c4) ===" > "$R"

# --- golden check: the decode tool must reproduce the prior lane's recorded decode byte-for-byte before use.
echo "CMD: $PY $E/f3_decode.py --log $E/f3qa-run1.log > $E/f3tc-golden-out.txt 2>&1" >> "$R"
$PY $E/f3_decode.py --log $E/f3qa-run1.log > $E/f3tc-golden-out.txt 2>&1
rc=$?
echo "GOLDEN_GEN_RC=$rc" >> "$R"
echo "CMD: diff $E/f3tc-golden-out.txt $E/f3qa-decode-RUN1.txt > $E/f3tc-decode-goldencheck.txt 2>&1" >> "$R"
diff $E/f3tc-golden-out.txt $E/f3qa-decode-RUN1.txt > $E/f3tc-decode-goldencheck.txt 2>&1
rc=$?
echo "GOLDEN_DIFF_RC=$rc (0 = the decode tool reproduces the prior artifact byte-identically)" >> "$R"

# --- decode every run log back to text with the pack tokenizer.
for n in base run1 run2; do
  echo "CMD: $PY $E/f3_decode.py --log $E/f3tc-run-$n.log > $E/f3tc-decode-$n.txt 2>&1" >> "$R"
  $PY $E/f3_decode.py --log $E/f3tc-run-$n.log > $E/f3tc-decode-$n.txt 2>&1
  rc=$?
  echo "DECODE_${n}_RC=$rc" >> "$R"
  cat "$E/f3tc-decode-$n.txt" >> "$R"
  echo "CMD: grep -ci paris $E/f3tc-decode-$n.txt" >> "$R"
  grep -ci paris "$E/f3tc-decode-$n.txt" >> "$R"
  rc=$?
  echo "PARIS_${n}_RC=$rc (0 = decoded text contains Paris)" >> "$R"
done

# --- repeat determinism: the same greedy invocation must emit identical ids and decode.
echo "CMD: diff $E/f3tc-decode-run1.txt $E/f3tc-decode-run2.txt > $E/f3tc-repeat-diff.txt 2>&1" >> "$R"
diff "$E/f3tc-decode-run1.txt" "$E/f3tc-decode-run2.txt" > "$E/f3tc-repeat-diff.txt" 2>&1
rc=$?
echo "REPEAT_DIFF_RC=$rc (0 = run 2 is byte-identical to run 1)" >> "$R"
echo "CMD: diff $E/f3tc-decode-base.txt $E/f3tc-decode-run1.txt > $E/f3tc-baseline-diff.txt 2>&1" >> "$R"
diff "$E/f3tc-decode-base.txt" "$E/f3tc-decode-run1.txt" > "$E/f3tc-baseline-diff.txt" 2>&1
rc=$?
echo "BASELINE_DIFF_RC=$rc (0 = the one-card baseline emits the same ids as the two-card runs)" >> "$R"

# --- per-device load lines, prompt/decode lines, hand-off lines per run.
for n in base run1 run2; do
  echo "CMD: grep -nE \"^prompt: |^decode: |^ds4: CUDA[0-9]+ |the card's waits|the hand-off's waits\" $E/f3tc-run-$n.log > $E/f3tc-keylines-$n.txt 2>&1" >> "$R"
  grep -nE "^prompt: |^decode: |^ds4: CUDA[0-9]+ |the card's waits|the hand-off's waits" "$E/f3tc-run-$n.log" > "$E/f3tc-keylines-$n.txt" 2>&1
  rc=$?
  echo "KEYLINES_${n}_RC=$rc" >> "$R"
  cat "$E/f3tc-keylines-$n.txt" >> "$R"
done

# --- XPU_TWO: the engine's own per-device lines must show both cards loaded on the two-card runs.
for n in run1 run2; do
  echo "CMD: grep -c '^ds4: CUDA0 ' $E/f3tc-run-$n.log" >> "$R"
  grep -c '^ds4: CUDA0 ' "$E/f3tc-run-$n.log" >> "$R"
  c0=$?
  echo "DEV0_LINES_${n}_GREP_RC=$c0" >> "$R"
  echo "CMD: grep -c '^ds4: CUDA1 ' $E/f3tc-run-$n.log" >> "$R"
  grep -c '^ds4: CUDA1 ' "$E/f3tc-run-$n.log" >> "$R"
  c1=$?
  echo "DEV1_LINES_${n}_GREP_RC=$c1" >> "$R"
  n0=$(grep -c '^ds4: CUDA0 ' "$E/f3tc-run-$n.log")
  n1=$(grep -c '^ds4: CUDA1 ' "$E/f3tc-run-$n.log")
  echo "CMD: [ \"$n0\" -ge 1 ] && [ \"$n1\" -ge 1 ]   (both cards' engine device lines present)" >> "$R"
  if [ "$n0" -ge 1 ] && [ "$n1" -ge 1 ]; then rc=0; else rc=1; fi
  echo "XPU_TWO_${n}_RC=$rc (CUDA0 lines=$n0, CUDA1 lines=$n1)" >> "$R"
done
echo "CMD: grep -c '^ds4: CUDA1 ' $E/f3tc-run-base.log   (baseline is one card: must be 0)" >> "$R"
grep -c '^ds4: CUDA1 ' "$E/f3tc-run-base.log" >> "$R"
rc=$?
echo "DEV1_BASE_GREP_RC=$rc (1 = the one-card baseline shows no CUDA1 line)" >> "$R"

# --- reset checks on every card run: no xe/GPU reset pattern, no level_zero backend failure, no DEVICE_LOST.
for n in base run1 run2 spin-max1; do
  echo "CMD: grep -icE 'gpu[ -]?reset|device[ -]?reset|engine[ -]?reset|gt[ -]?reset|xe[ -]?reset|page fault|reset occurred|resetting the (gpu|device|engine)' $E/f3tc-run-$n.log" >> "$R"
  grep -icE 'gpu[ -]?reset|device[ -]?reset|engine[ -]?reset|gt[ -]?reset|xe[ -]?reset|page fault|reset occurred|resetting the (gpu|device|engine)' "$E/f3tc-run-$n.log" >> "$R"
  rc=$?
  echo "RESET_${n}_RC=$rc (1 = zero reset-pattern matches, as required)" >> "$R"
  echo "CMD: grep -ic 'level_zero backend failed' $E/f3tc-run-$n.log" >> "$R"
  grep -ic 'level_zero backend failed' "$E/f3tc-run-$n.log" >> "$R"
  rc=$?
  echo "BACKEND_${n}_RC=$rc (1 = zero matches)" >> "$R"
  echo "CMD: grep -icE 'UR_RESULT_ERROR_DEVICE_LOST|DEVICE_LOST' $E/f3tc-run-$n.log" >> "$R"
  grep -icE 'UR_RESULT_ERROR_DEVICE_LOST|DEVICE_LOST' "$E/f3tc-run-$n.log" >> "$R"
  rc=$?
  echo "DEVICE_LOST_${n}_RC=$rc (1 = zero matches)" >> "$R"
done

# --- ADV-9-1 loud-failure check.
echo "CMD: grep -c 'the card never saw an answer' $E/f3tc-spin-max1.log" >> "$R"
grep -c 'the card never saw an answer' "$E/f3tc-spin-max1.log" >> "$R"
rc=$?
echo "SPIN_FAIL_LOUD_RC=$rc (0 = the bound-hit message is printed; the run must not hand off silently)" >> "$R"
grep -n 'the card never saw an answer' "$E/f3tc-spin-max1.log" | head -4 >> "$R"

# --- ADV-9-2 / ADV-9-3 error-message checks.
echo "CMD: grep -ic 'out of range' $E/f3tc-neg-badid.log" >> "$R"
grep -ic 'out of range' "$E/f3tc-neg-badid.log" >> "$R"
rc=$?
echo "NEG2_ERR_RC=$rc (0 = the error names the offending token)" >> "$R"
grep -in 'out of range' "$E/f3tc-neg-badid.log" | head -2 >> "$R"
echo "CMD: grep -icE 'error|invalid|non-numeric|parse' $E/f3tc-neg-garbage.log" >> "$R"
grep -icE 'error|invalid|non-numeric|parse' "$E/f3tc-neg-garbage.log" >> "$R"
rc=$?
echo "NEG_ERR_RC=$rc (0 = a parse error naming the offending token is printed)" >> "$R"

# --- xpu-smi per-card memory evidence across run 2.
echo "CMD: grep -c '^0, ' $E/f3tc-smi-loop.txt" >> "$R"
grep -c '^0, ' "$E/f3tc-smi-loop.txt" >> "$R"
rc=$?
echo "SMI_CARD0_ROWS_RC=$rc" >> "$R"
echo "CMD: grep -c '^1, ' $E/f3tc-smi-loop.txt" >> "$R"
grep -c '^1, ' "$E/f3tc-smi-loop.txt" >> "$R"
rc=$?
echo "SMI_CARD1_ROWS_RC=$rc" >> "$R"
echo "--- per-card memory across run 2 (index, memory.used (MiB), memory.total (MiB)) ---" >> "$R"
cat "$E/f3tc-smi-loop.txt" >> "$R"

# --- todo 8's own numbers, quoted from its log for the side-by-side (IS-7).
echo "--- todo 8 one-card numbers, quoted from task-8-...log ---" >> "$R"
grep -nE '^prompt: |^decode: ' "$E/task-8-dsv41-sycl-dpct-and-xpu-test.log" | head -8 >> "$R"
rc=$?
echo "TODO8_QUOTE_RC=$rc" >> "$R"

echo "ANALYZE_PHASE_DONE $(date -Is)" >> "$R"
