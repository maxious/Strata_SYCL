#!/bin/bash
# F3 QA run 1: the literal end-state invocation, one card (binary at its real path per caller correction:
# <build>/sycl/strata-dsv41; the w6c build root entry was a symlink). Card discipline gate embedded.
R=/tmp/dpct_run/f3_receipts.txt
PG=/tmp/dpct_run/f3_pgrep_pre_run1.txt
echo "CMD: pgrep -c strata-dsv41 > /tmp/dpct_run/f3_pgrep_pre_run1.txt 2>&1" >> "$R"
pgrep -c strata-dsv41 > "$PG" 2>&1
rc=$?
echo "PGREP_RUN1_RC=$rc (pgrep exits 1 when the count is 0)" >> "$R"
count=$(cat "$PG")
echo "PGREP_RUN1_READBACK=$count" >> "$R"
if [ "$count" != "0" ]; then
  echo "RUN1_LAUNCH_BLOCKED: card busy (count=$count)" >> "$R"
  exit 9
fi
echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/f3_sycl_build/sycl/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 --gpus 0 --ids @/tmp/q_france.ids --n 8 --max-context 512 --verbose > /tmp/dpct_run/f3_n8.log 2>&1'" >> "$R"
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/f3_sycl_build/sycl/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 --gpus 0 --ids @/tmp/q_france.ids --n 8 --max-context 512 --verbose > /tmp/dpct_run/f3_n8.log 2>&1'
rc=$?
echo "RUN_RC=$rc" >> "$R"
echo "RUN1_DONE $(date -Is)" >> "$R"
