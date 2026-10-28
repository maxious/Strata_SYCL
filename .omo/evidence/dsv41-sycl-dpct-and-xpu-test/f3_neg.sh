#!/bin/bash
# F3 QA adversarial A (todo 8 failure scenario): same invocation with --ids @/tmp/q_garbage.ids
# ("hello world", genuinely malformed non-numeric tokens). Criterion: must exit non-zero and print a
# parse error naming the offending token.
R=/tmp/dpct_run/f3_receipts.txt
PG=/tmp/dpct_run/f3_pgrep_pre_neg.txt
echo "CMD: pgrep -c strata-dsv41 > /tmp/dpct_run/f3_pgrep_pre_neg.txt 2>&1" >> "$R"
pgrep -c strata-dsv41 > "$PG" 2>&1
rc=$?
echo "PGREP_NEG_RC=$rc (pgrep exits 1 when the count is 0)" >> "$R"
count=$(cat "$PG")
echo "PGREP_NEG_READBACK=$count" >> "$R"
if [ "$count" != "0" ]; then
  echo "NEG_LAUNCH_BLOCKED: card busy (count=$count)" >> "$R"
  exit 9
fi
echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/f3_sycl_build/sycl/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 --gpus 0 --ids @/tmp/q_garbage.ids --n 8 --max-context 512 --verbose > /tmp/dpct_run/f3_neg.log 2>&1'" >> "$R"
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/f3_sycl_build/sycl/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 --gpus 0 --ids @/tmp/q_garbage.ids --n 8 --max-context 512 --verbose > /tmp/dpct_run/f3_neg.log 2>&1'
rc=$?
echo "NEG_IDS_RC=$rc" >> "$R"
echo "NEG_DONE $(date -Is)" >> "$R"
