#!/bin/bash
# F3 QA analysis of the three end-state runs + two adversarial logs. Un-piped RCs, CMD lines above each.
R=/tmp/dpct_run/f3_receipts.txt
echo "=== ANALYSIS_START $(date -Is) ===" >> "$R"

for n in 1 2 3; do
  case $n in
    1) L=/tmp/dpct_run/f3_n8.log;  T=RUN1 ;;
    2) L=/tmp/dpct_run/f3_n8_r2.log; T=RUN2 ;;
    3) L=/tmp/dpct_run/f3_n8_r3.log; T=RUN3 ;;
  esac
  echo "CMD: grep -n '^prompt: ' $L > /tmp/dpct_run/f3_prompt_lines_$T.txt" >> "$R"
  grep -n '^prompt: ' "$L" > /tmp/dpct_run/f3_prompt_lines_$T.txt
  rc=$?
  echo "PROMPT_LINE_${T}_RC=$rc" >> "$R"
  cat /tmp/dpct_run/f3_prompt_lines_$T.txt >> "$R"

  echo "CMD: grep -n '^decode' $L > /tmp/dpct_run/f3_decode_lines_$T.txt" >> "$R"
  grep -n '^decode' "$L" > /tmp/dpct_run/f3_decode_lines_$T.txt
  rc=$?
  echo "DECODE_LINE_${T}_RC=$rc" >> "$R"
  cat /tmp/dpct_run/f3_decode_lines_$T.txt >> "$R"

  echo "CMD: grep -n 'CPU lane:' $L > /tmp/dpct_run/f3_cpulane_lines_$T.txt" >> "$R"
  grep -n 'CPU lane:' "$L" > /tmp/dpct_run/f3_cpulane_lines_$T.txt
  rc=$?
  echo "CPULANE_${T}_RC=$rc" >> "$R"
  cat /tmp/dpct_run/f3_cpulane_lines_$T.txt >> "$R"

  echo "CMD: /tmp/tokvenv/bin/python /tmp/dpct_run/f3_decode.py --log $L > /tmp/dpct_run/f3_decode_$T.txt" >> "$R"
  /tmp/tokvenv/bin/python /tmp/dpct_run/f3_decode.py --log "$L" > /tmp/dpct_run/f3_decode_$T.txt 2>&1
  rc=$?
  echo "DECODE_IDS_${T}_RC=$rc" >> "$R"
  cat /tmp/dpct_run/f3_decode_$T.txt >> "$R"

  echo "CMD: sed -n 's/^batch1 decoded: //p' /tmp/dpct_run/f3_decode_$T.txt > /tmp/dpct_run/f3_answer_$T.txt" >> "$R"
  sed -n 's/^batch1 decoded: //p' /tmp/dpct_run/f3_decode_$T.txt > /tmp/dpct_run/f3_answer_$T.txt
  rc=$?
  echo "ANSWER_${T}_RC=$rc" >> "$R"

  echo "CMD: grep -ci paris /tmp/dpct_run/f3_answer_$T.txt > /tmp/dpct_run/f3_paris_$T.txt" >> "$R"
  grep -ci paris /tmp/dpct_run/f3_answer_$T.txt > /tmp/dpct_run/f3_paris_$T.txt
  rc=$?
  echo "PARIS_${T}_RC=$rc" >> "$R"
  cat /tmp/dpct_run/f3_paris_$T.txt >> "$R"

  echo "CMD: grep -icE 'gpu[ -]?reset|device[ -]?reset|engine[ -]?reset|gt[ -]?reset|xe[ -]?reset|page fault|reset occurred|resetting the (gpu|device|engine)' $L > /tmp/dpct_run/f3_reset_$T.txt" >> "$R"
  grep -icE 'gpu[ -]?reset|device[ -]?reset|engine[ -]?reset|gt[ -]?reset|xe[ -]?reset|page fault|reset occurred|resetting the (gpu|device|engine)' "$L" > /tmp/dpct_run/f3_reset_$T.txt
  rc=$?
  echo "RESET_${T}_RC=$rc (1 = zero matches = no GPU reset)" >> "$R"
  cat /tmp/dpct_run/f3_reset_$T.txt >> "$R"

  echo "CMD: grep -c 'ds4: CUDA0' $L > /tmp/dpct_run/f3_dev0_$T.txt" >> "$R"
  grep -c 'ds4: CUDA0' "$L" > /tmp/dpct_run/f3_dev0_$T.txt
  rc=$?
  echo "DEV0_${T}_RC=$rc" >> "$R"
  cat /tmp/dpct_run/f3_dev0_$T.txt >> "$R"

  echo "CMD: grep -c 'ds4: CUDA1' $L > /tmp/dpct_run/f3_dev1_$T.txt" >> "$R"
  grep -c 'ds4: CUDA1' "$L" > /tmp/dpct_run/f3_dev1_$T.txt
  rc=$?
  echo "DEV1_${T}_RC=$rc (1 = zero matches = no second device)" >> "$R"
  cat /tmp/dpct_run/f3_dev1_$T.txt >> "$R"
done

echo "CMD: diff /tmp/dpct_run/f3_answer_RUN1.txt /tmp/dpct_run/f3_answer_RUN2.txt > /tmp/dpct_run/f3_repeat2_diff.txt 2>&1" >> "$R"
diff /tmp/dpct_run/f3_answer_RUN1.txt /tmp/dpct_run/f3_answer_RUN2.txt > /tmp/dpct_run/f3_repeat2_diff.txt 2>&1
rc=$?
echo "REPEAT2_RC=$rc (0 = byte-identical)" >> "$R"
echo "CMD: diff /tmp/dpct_run/f3_answer_RUN1.txt /tmp/dpct_run/f3_answer_RUN3.txt > /tmp/dpct_run/f3_repeat3_diff.txt 2>&1" >> "$R"
diff /tmp/dpct_run/f3_answer_RUN1.txt /tmp/dpct_run/f3_answer_RUN3.txt > /tmp/dpct_run/f3_repeat3_diff.txt 2>&1
rc=$?
echo "REPEAT3_RC=$rc (0 = byte-identical)" >> "$R"
echo "CMD: diff /tmp/dpct_run/f3_decode_RUN1.txt /tmp/dpct_run/f3_decode_RUN2.txt > /tmp/dpct_run/f3_ids2_diff.txt 2>&1" >> "$R"
diff /tmp/dpct_run/f3_decode_RUN1.txt /tmp/dpct_run/f3_decode_RUN2.txt > /tmp/dpct_run/f3_ids2_diff.txt 2>&1
rc=$?
echo "IDS_REPEAT2_RC=$rc (0 = byte-identical ids)" >> "$R"
echo "CMD: diff /tmp/dpct_run/f3_decode_RUN1.txt /tmp/dpct_run/f3_decode_RUN3.txt > /tmp/dpct_run/f3_ids3_diff.txt 2>&1" >> "$R"
diff /tmp/dpct_run/f3_decode_RUN1.txt /tmp/dpct_run/f3_decode_RUN3.txt > /tmp/dpct_run/f3_ids3_diff.txt 2>&1
rc=$?
echo "IDS_REPEAT3_RC=$rc (0 = byte-identical ids)" >> "$R"

echo "=== adversarial logs ===" >> "$R"
echo "CMD: grep -n 'out of range\|parse\|error' /tmp/dpct_run/f3_neg.log > /tmp/dpct_run/f3_neg_err.txt" >> "$R"
grep -n 'out of range\|parse\|error' /tmp/dpct_run/f3_neg.log > /tmp/dpct_run/f3_neg_err.txt
rc=$?
echo "NEG_ERR_GREP_RC=$rc (1 = no parse error printed)" >> "$R"
cat /tmp/dpct_run/f3_neg_err.txt >> "$R"
echo "CMD: grep -n 'out of range' /tmp/dpct_run/f3_neg2.log > /tmp/dpct_run/f3_neg2_err.txt" >> "$R"
grep -n 'out of range' /tmp/dpct_run/f3_neg2.log > /tmp/dpct_run/f3_neg2_err.txt
rc=$?
echo "NEG2_ERR_GREP_RC=$rc (0 = offending token named)" >> "$R"
cat /tmp/dpct_run/f3_neg2_err.txt >> "$R"

echo "=== default build dsv41-source greps ===" >> "$R"
echo "CMD: grep -c 'ds4_' /tmp/f3_default_build/build.ninja > /tmp/dpct_run/f3_off_grep.txt" >> "$R"
grep -c 'ds4_' /tmp/f3_default_build/build.ninja > /tmp/dpct_run/f3_off_grep.txt
rc=$?
echo "DSV41_OFF_GREP_RC=$rc (1 = count 0 = default build lists no dsv41 sources)" >> "$R"
cat /tmp/dpct_run/f3_off_grep.txt >> "$R"
echo "CMD: grep -c 'ds4_' /tmp/f3_sycl_build/build.ninja > /tmp/dpct_run/f3_on_grep.txt" >> "$R"
grep -c 'ds4_' /tmp/f3_sycl_build/build.ninja > /tmp/dpct_run/f3_on_grep.txt
rc=$?
echo "DSV41_ON_GREP_RC=$rc (0 = count > 0 = ON build lists dsv41 sources)" >> "$R"
cat /tmp/dpct_run/f3_on_grep.txt >> "$R"

echo "=== ANALYSIS_DONE $(date -Is) ===" >> "$R"
