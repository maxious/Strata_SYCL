#!/bin/bash
# F3 QA adversarial B (harness-can-fail control): --ids @/tmp/q_badid.ids ("128803 999999999",
# in-range BOS plus an out-of-range id). Expected: non-zero exit with "token 999999999 out of range".
R=/tmp/dpct_run/f3_receipts.txt
PG=/tmp/dpct_run/f3_pgrep_pre_neg2.txt
echo "CMD: pgrep -c strata-dsv41 > /tmp/dpct_run/f3_pgrep_pre_neg2.txt 2>&1" >> "$R"
pgrep -c strata-dsv41 > "$PG" 2>&1
rc=$?
echo "PGREP_NEG2_RC=$rc (pgrep exits 1 when the count is 0)" >> "$R"
count=$(cat "$PG")
echo "PGREP_NEG2_READBACK=$count" >> "$R"
if [ "$count" != "0" ]; then
  echo "NEG2_LAUNCH_BLOCKED: card busy (count=$count)" >> "$R"
  exit 9
fi
echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/f3_sycl_build/sycl/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 --gpus 0 --ids @/tmp/q_badid.ids --n 8 --max-context 512 --verbose > /tmp/dpct_run/f3_neg2.log 2>&1'" >> "$R"
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/f3_sycl_build/sycl/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 --gpus 0 --ids @/tmp/q_badid.ids --n 8 --max-context 512 --verbose > /tmp/dpct_run/f3_neg2.log 2>&1'
rc=$?
echo "NEG2_RC=$rc" >> "$R"
echo "NEG2_DONE $(date -Is)" >> "$R"
