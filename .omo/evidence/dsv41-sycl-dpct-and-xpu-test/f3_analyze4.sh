R=/tmp/dpct_run/f3_receipts.txt
L=/tmp/dpct_run/f3_n8_r4.log
T=RUN4
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
echo "CMD: diff /tmp/dpct_run/f3_answer_RUN1.txt /tmp/dpct_run/f3_answer_RUN4.txt > /tmp/dpct_run/f3_repeat4_diff.txt 2>&1" >> "$R"
diff /tmp/dpct_run/f3_answer_RUN1.txt /tmp/dpct_run/f3_answer_RUN4.txt > /tmp/dpct_run/f3_repeat4_diff.txt 2>&1
rc=$?
echo "REPEAT4_RC=$rc (0 = byte-identical)" >> "$R"
echo "CMD: diff /tmp/dpct_run/f3_decode_RUN1.txt /tmp/dpct_run/f3_decode_RUN4.txt > /tmp/dpct_run/f3_ids4_diff.txt 2>&1" >> "$R"
diff /tmp/dpct_run/f3_decode_RUN1.txt /tmp/dpct_run/f3_decode_RUN4.txt > /tmp/dpct_run/f3_ids4_diff.txt 2>&1
rc=$?
echo "IDS_REPEAT4_RC=$rc (0 = byte-identical ids)" >> "$R"
echo "CMD: grep -icE 'gpu[ -]?reset|device[ -]?reset|engine[ -]?reset|gt[ -]?reset|xe[ -]?reset|page fault|reset occurred|resetting the (gpu|device|engine)' $L > /tmp/dpct_run/f3_reset_$T.txt" >> "$R"
grep -icE 'gpu[ -]?reset|device[ -]?reset|engine[ -]?reset|gt[ -]?reset|xe[ -]?reset|page fault|reset occurred|resetting the (gpu|device|engine)' "$L" > /tmp/dpct_run/f3_reset_$T.txt
rc=$?
echo "RESET_${T}_RC=$rc (1 = zero matches = no GPU reset)" >> "$R"
cat /tmp/dpct_run/f3_reset_$T.txt >> "$R"
echo "CMD: grep -c 'ds4: CUDA1' $L > /tmp/dpct_run/f3_dev1_$T.txt" >> "$R"
grep -c 'ds4: CUDA1' "$L" > /tmp/dpct_run/f3_dev1_$T.txt
rc=$?
echo "DEV1_${T}_RC=$rc (1 = zero matches = no second device)" >> "$R"
cat /tmp/dpct_run/f3_dev1_$T.txt >> "$R"
echo "=== ANALYSIS4_DONE $(date -Is) ===" >> "$R"
