#!/bin/bash
# F3 two-card manual QA (task st_01a178c4) - card-run phase.
# Format: every CMD line is followed by its own un-piped *_RC= line (rc=$? taken immediately after the command).
# Gate before EVERY card launch: pgrep -c strata-dsv41 must read 0; if a card process is live, wait and
# re-read, never launch on top of it. Every run is wrapped in
# bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; <command>'.
E=/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test
R=$E/f3tc-receipts-runs.txt
B=/tmp/f3tc_bld_aot
BIN=$B/sycl/strata-dsv41
PACK=/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4

echo "=== F3 two-card manual QA: card-run phase (task st_01a178c4) ===" > "$R"
echo "binary: $B/sycl/strata-dsv41 | pack: $PACK" >> "$R"
echo "every card launch below is preceded by its pgrep gate (readback must be 0)" >> "$R"

gate() {
  local L="$1" f="$E/f3tc-pgrep-$1.txt" count rc i=0
  echo "CMD: pgrep -c strata-dsv41 > $f 2>&1" >> "$R"
  pgrep -c strata-dsv41 > "$f" 2>&1
  rc=$?
  echo "PGREP_${L}_RC=$rc (pgrep exits 1 when the count is 0)" >> "$R"
  count=$(cat "$f")
  echo "PGREP_${L}_READBACK=$count" >> "$R"
  while [ "$count" != "0" ] && [ "$i" -lt 60 ]; do
    echo "GATE_${L}_WAIT_$i: a strata-dsv41 process is live (count=$count); waiting 10 s and re-reading" >> "$R"
    sleep 10
    pgrep -c strata-dsv41 > "$f" 2>&1
    rc=$?
    count=$(cat "$f")
    i=$((i + 1))
  done
  echo "PGREP_${L}_FINAL_READBACK=$count (waits=$i)" >> "$R"
  if [ "$count" != "0" ]; then
    echo "LAUNCH_BLOCKED_${L}: card busy, no launch performed" >> "$R"
    return 9
  fi
  return 0
}

# --- baseline carrier for IS-7: one card, same build, same day (todo 9 is "two-card run against the
# --- same-day single-card baseline"); the two-card numbers are only reported against this run.
gate BASE || exit 9
echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0 --ids @/tmp/q_france.ids --n 8 --max-context 512 > $E/f3tc-run-base.log 2>&1'" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0 --ids @/tmp/q_france.ids --n 8 --max-context 512 > $E/f3tc-run-base.log 2>&1"
rc=$?
echo "RUN_BASE_RC=$rc" >> "$R"
echo "RUN_BASE_DONE $(date -Is)" >> "$R"

# --- run 1: the literal todo 9 invocation, two cards.
gate R1 || exit 9
echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0,1 --ids @/tmp/q_france.ids --n 8 --max-context 512 > $E/f3tc-run1.log 2>&1'" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0,1 --ids @/tmp/q_france.ids --n 8 --max-context 512 > $E/f3tc-run1.log 2>&1"
rc=$?
echo "RUN1_RC=$rc" >> "$R"
echo "RUN1_DONE $(date -Is)" >> "$R"

# --- run 2: the literal todo 9 invocation repeated once, with an xpu-smi memory loop across it.
gate R2 || exit 9
echo "CMD: xpu-smi --query-gpu=index,memory.used,memory.total --loop=3 --count=30 -f $E/f3tc-smi-loop.txt (backgrounded across RUN2)" >> "$R"
xpu-smi --query-gpu=index,memory.used,memory.total --loop=3 --count=30 -f "$E/f3tc-smi-loop.txt" > "$E/f3tc-smi-loop-stderr.txt" 2>&1 &
smi=$!
echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0,1 --ids @/tmp/q_france.ids --n 8 --max-context 512 > $E/f3tc-run2.log 2>&1'" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0,1 --ids @/tmp/q_france.ids --n 8 --max-context 512 > $E/f3tc-run2.log 2>&1"
rc=$?
echo "RUN2_RC=$rc" >> "$R"
wait $smi
rc=$?
echo "SMI_LOOP_RC=$rc" >> "$R"
echo "RUN2_DONE $(date -Is)" >> "$R"

echo "CMD: xpu-smi --query-gpu=index,memory.used,memory.total > $E/f3tc-smi-after.txt 2>&1" >> "$R"
xpu-smi --query-gpu=index,memory.used,memory.total > "$E/f3tc-smi-after.txt" 2>&1
rc=$?
echo "SMI_AFTER_RC=$rc" >> "$R"
cat "$E/f3tc-smi-after.txt" >> "$R"

# --- ADV-9-1: todo 9 failure scenario, --gpus 0,1 with STRATA_SPIN_MAX=1 must fail loudly, never hand off silently.
gate SPIN || exit 9
echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO STRATA_SPIN_MAX=1 $BIN --pack $PACK --gpus 0,1 --ids @/tmp/q_france.ids --n 8 --max-context 512 > $E/f3tc-spin-max1.log 2>&1'" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO STRATA_SPIN_MAX=1 $BIN --pack $PACK --gpus 0,1 --ids @/tmp/q_france.ids --n 8 --max-context 512 > $E/f3tc-spin-max1.log 2>&1"
rc=$?
echo "SPIN_RUN_RC=$rc" >> "$R"
echo "SPIN_RUN_DONE $(date -Is)" >> "$R"

# --- ADV-9-2: harness-can-fail control on the two-card surface - an out-of-range token must exit non-zero.
gate NEG2 || exit 9
echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0,1 --ids @$E/f3tc-ids-badid.ids --n 8 --max-context 512 > $E/f3tc-neg-badid.log 2>&1'" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0,1 --ids @$E/f3tc-ids-badid.ids --n 8 --max-context 512 > $E/f3tc-neg-badid.log 2>&1"
rc=$?
echo "NEG2_RUN_RC=$rc" >> "$R"
echo "NEG2_RUN_DONE $(date -Is)" >> "$R"

# --- ADV-9-3: input validation on the two-card surface - a genuinely malformed ids file must exit non-zero
# --- and name the offending token (known gap: plan row 19 / src/dsv41/ds4_main.cpp:71).
gate NEG || exit 9
echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0,1 --ids @$E/f3tc-ids-garbage.ids --n 8 --max-context 512 > $E/f3tc-neg-garbage.log 2>&1'" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0,1 --ids @$E/f3tc-ids-garbage.ids --n 8 --max-context 512 > $E/f3tc-neg-garbage.log 2>&1"
rc=$?
echo "NEG_IDS_RUN_RC=$rc" >> "$R"
echo "NEG_IDS_RUN_DONE $(date -Is)" >> "$R"

echo "CMD: pgrep -c strata-dsv41 (post-run re-read)" >> "$R"
pgrep -c strata-dsv41 > "$E/f3tc-pgrep-post.txt" 2>&1
rc=$?
echo "PGREP_POST_RC=$rc (pgrep exits 1 when the count is 0)" >> "$R"
cat "$E/f3tc-pgrep-post.txt" >> "$R"
echo "RUN_PHASE_DONE $(date -Is)" >> "$R"
