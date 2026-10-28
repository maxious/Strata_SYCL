#!/bin/bash
# F3 QA run 4: fourth repeat of the literal end-state invocation (output log path only),
# run under the corrected xpu-smi loop sampler for a true during-window capture.
R=/tmp/dpct_run/f3_receipts.txt
PG=/tmp/dpct_run/f3_pgrep_pre_run4.txt
echo "CMD: pgrep -c strata-dsv41 > /tmp/dpct_run/f3_pgrep_pre_run4.txt 2>&1" >> "$R"
pgrep -c strata-dsv41 > "$PG" 2>&1
rc=$?
echo "PGREP_RUN4_RC=$rc (pgrep exits 1 when the count is 0)" >> "$R"
count=$(cat "$PG")
echo "PGREP_RUN4_READBACK=$count" >> "$R"
if [ "$count" != "0" ]; then
  echo "RUN4_LAUNCH_BLOCKED: card busy (count=$count)" >> "$R"
  exit 9
fi
echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/f3_sycl_build/sycl/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 --gpus 0 --ids @/tmp/q_france.ids --n 8 --max-context 512 --verbose > /tmp/dpct_run/f3_n8_r4.log 2>&1'" >> "$R"
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 SYCL_BE=PI_LEVEL_ZERO /tmp/f3_sycl_build/sycl/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 --gpus 0 --ids @/tmp/q_france.ids --n 8 --max-context 512 --verbose > /tmp/dpct_run/f3_n8_r4.log 2>&1'
rc=$?
echo "RUN4_RC=$rc" >> "$R"
echo "RUN4_DONE $(date -Is)" >> "$R"
