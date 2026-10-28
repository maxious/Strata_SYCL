#!/bin/bash
# F3 two-card manual QA (task st_01a178ea, plan row 9) - card-run phase on 2x Arc Pro B60.
# Format: every CMD line is followed by its own un-piped *_RC= line (rc=$? taken immediately after the command).
# Gate before EVERY card launch: pgrep -c strata-dsv41 read back must be 0; never launch on top of a live card process.
set -u
E=/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test
R=$E/f3twc-receipts-runs.txt
BIN=/tmp/f3tc_build/sycl/strata-dsv41
PACK=/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4
IDS=/tmp/q_france.ids
RP='exec queue reset detected|VM worker error|GPU fault|DEVICE_LOST|coredump'

echo "=== F3 two-card manual QA: card-run phase (task st_01a178ea) ===" > "$R"
echo "binary: $BIN | pack: $PACK | ids: $IDS" >> "$R"
echo "build identity: $(cat "$E/f3twc-build-identity.txt" 2>/dev/null)" >> "$R"
echo "HEAD: $(git -C /home/maxious/Strata rev-parse HEAD)" >> "$R"
echo "card0 = 0000:0d:00.0 (renderD128), card1 = 0000:11:00.0 (renderD129), both 8086:e211" >> "$R"
echo "PIPELINE_START $(date -Is)" >> "$R"

now(){ date +%Y-%m-%d\ %H:%M:%S; }

gate() {
  local L="$1" f="$E/f3twc-pgrep-$1.txt" count rc i=0
  echo "CMD: pgrep -c strata-dsv41 > $f 2>&1" >> "$R"
  pgrep -c strata-dsv41 > "$f" 2>&1
  rc=$?
  echo "PGREP_${L}_RC=$rc (pgrep exits 1 when the count is 0)" >> "$R"
  echo "PGREP_${L}_READBACK=$(cat "$f")" >> "$R"
  while [ "$(cat "$f")" != "0" ] && [ "$i" -lt 60 ]; do
    echo "GATE_${L}_WAIT_$i: a strata-dsv41 process is live; waiting 10 s and re-reading" >> "$R"
    sleep 10
    pgrep -c strata-dsv41 > "$f" 2>&1
    rc=$?
    i=$((i + 1))
  done
  echo "PGREP_${L}_FINAL_READBACK=$(cat "$f") (waits=$i)" >> "$R"
  [ "$(cat "$f")" = "0" ]
}

# journal window capture + reset grep for one run window
resetcheck() {
  local L="$1" s="$2" t="$3" f="$E/f3twc-journal-$1.txt"
  echo "CMD: journalctl -k --no-pager --since '$s' --until '$t' -o short-iso > $f 2>&1 (window $s .. $t)" >> "$R"
  { echo "journal window $s .. $t (physical card 1 = 0000:11:00.0)"; journalctl -k --no-pager --since "$s" --until "$t" -o short-iso; } > "$f" 2>&1
  rc=$?
  echo "JOURNAL_${L}_RC=$rc" >> "$R"
  echo "CMD: grep -cE 'xe 0000:11:00\.0:.*($RP)' $f" >> "$R"
  grep -cE "xe 0000:11:00\.0:.*($RP)" "$f" > "$E/f3twc-reset-$1.txt" 2>&1
  rc=$?
  echo "RESET_${L}_RC=$rc (1 = zero reset events on physical card 1)" >> "$R"
  echo "RESET_${L}_COUNT=$(cat "$E/f3twc-reset-$1.txt")" >> "$R"
  echo "CMD: grep -cE '$RP' $f (any device)" >> "$R"
  grep -cE "$RP" "$f" > "$E/f3twc-reset-all-$1.txt" 2>&1
  rc=$?
  echo "RESET_ALL_${L}_RC=$rc (1 = zero reset/fault events on any device)" >> "$R"
  echo "RESET_ALL_${L}_COUNT=$(cat "$E/f3twc-reset-all-$1.txt")" >> "$R"
}

run_engine() {
  local L="$1" gpus="$2" ids="$3" log="$4" extra="$5"
  gate "$L" || { echo "LAUNCH_BLOCKED_${L}: card busy, no launch performed" >> "$R"; return 9; }
  local s; s=$(now)
  echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO${extra:+ $extra} $BIN --pack $PACK --gpus $gpus --ids $ids --n 8 --max-context 512 > $log 2>&1'" >> "$R"
  bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO${extra:+ $extra} $BIN --pack $PACK --gpus $gpus --ids $ids --n 8 --max-context 512 > $log 2>&1"
  local rc=$?
  echo "RUN_${L}_RC=$rc" >> "$R"
  local t; t=$(now)
  echo "RUN_${L}_DONE $(date -Is)" >> "$R"
  resetcheck "$L" "$s" "$t"
  return $rc
}

# --- step 3: baseline one-card run (same build, same day: the IS-7 carrier for the two-card numbers)
run_engine BASE 0 "$IDS" "$E/f3twc-run-base.log" ""

# --- step 4: two literal two-card runs
run_engine TWO1 "0,1" "$IDS" "$E/f3twc-run-two1.log" ""
run_engine TWO2 "0,1" "$IDS" "$E/f3twc-run-two2.log" ""

# --- per-device load receipt in each two-card log (the engine's own device lines)
for L in TWO1 TWO2; do
  echo "CMD: grep -c 'ds4: CUDA0 layers' $E/f3twc-run-$([ "$L" = TWO1 ] && echo two1 || echo two2).log" >> "$R"
  grep -c 'ds4: CUDA0 layers' "$E/f3twc-run-$([ "$L" = TWO1 ] && echo two1 || echo two2).log" > "$E/f3twc-dev0-$L.txt" 2>&1
  rc=$?; echo "DEV0_${L}_RC=$rc (0 = CUDA0 device line present)" >> "$R"
  echo "DEV0_${L}_COUNT=$(cat "$E/f3twc-dev0-$L.txt")" >> "$R"
  echo "CMD: grep -c 'ds4: CUDA1 layers' $E/f3twc-run-$([ "$L" = TWO1 ] && echo two1 || echo two2).log" >> "$R"
  grep -c 'ds4: CUDA1 layers' "$E/f3twc-run-$([ "$L" = TWO1 ] && echo two1 || echo two2).log" > "$E/f3twc-dev1-$L.txt" 2>&1
  rc=$?; echo "DEV1_${L}_RC=$rc (0 = CUDA1 device line present)" >> "$R"
  echo "DEV1_${L}_COUNT=$(cat "$E/f3twc-dev1-$L.txt")" >> "$R"
done

# --- prompt: line completion + throughput lines, quoted from the logs
for L in BASE TWO1 TWO2; do
  lg="$E/f3twc-run-$([ "$L" = BASE ] && echo base || ([ "$L" = TWO1 ] && echo two1 || echo two2)).log"
  echo "CMD: grep -c '^prompt:' $lg" >> "$R"
  grep -c '^prompt:' "$lg" > "$E/f3twc-prompt-$L.txt" 2>&1
  rc=$?; echo "PROMPT_${L}_RC=$rc (0 = a completed prompt: line exists)" >> "$R"
  echo "PROMPT_${L}_COUNT=$(cat "$E/f3twc-prompt-$L.txt")" >> "$R"
  echo "CMD: grep -c '^decode:' $lg" >> "$R"
  grep -c '^decode:' "$lg" > "$E/f3twc-decode-$L.txt" 2>&1
  rc=$?; echo "DECODE_${L}_RC=$rc" >> "$R"
  echo "DECODE_${L}_COUNT=$(cat "$E/f3twc-decode-$L.txt")" >> "$R"
done

# --- step 6: loud-spin control - STRATA_SPIN_MAX=1 must fail loudly, never hand off silently
run_engine SPIN "0,1" "$IDS" "$E/f3twc-spin-max1.log" "STRATA_SPIN_MAX=1"
echo "CMD: grep -c 'the card never saw an answer' $E/f3twc-spin-max1.log" >> "$R"
grep -c 'the card never saw an answer' "$E/f3twc-spin-max1.log" > "$E/f3twc-spin-msg.txt" 2>&1
rc=$?; echo "SPIN_MSG_RC=$rc (0 = the loud message is printed)" >> "$R"
echo "SPIN_MSG_COUNT=$(cat "$E/f3twc-spin-msg.txt")" >> "$R"

# --- ADV-9-2: card-busy gate must refuse a launch while a card process is live (probe during a real run)
gate TWO3 || exit 9
s=$(now)
echo "CMD: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0,1 --ids $IDS --n 8 --max-context 512 > $E/f3twc-run-two3.log 2>&1' (backgrounded for the gate probe)" >> "$R"
bash -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO $BIN --pack $PACK --gpus 0,1 --ids $IDS --n 8 --max-context 512 > $E/f3twc-run-two3.log 2>&1" &
eng=$!
i=0
while [ "$i" -lt 30 ]; do
  pgrep -c strata-dsv41 > "$E/f3twc-pgrep-PROBE.txt" 2>&1
  rc=$?
  c=$(cat "$E/f3twc-pgrep-PROBE.txt")
  if [ "$c" != "0" ]; then break; fi
  sleep 1
  i=$((i + 1))
done
echo "PROBE_RC=$rc (pgrep exits 0 when the count is >0)" >> "$R"
echo "PROBE_READBACK=$c (after $i one-second re-reads)" >> "$R"
echo "CMD: pgrep -c strata-dsv41 > $E/f3twc-pgrep-PROBE.txt 2>&1  (gate probe while the card process is live)" >> "$R"
pgrep -c strata-dsv41 > "$E/f3twc-pgrep-PROBE.txt" 2>&1
rc=$?; echo "PGATE_PROBE_RC=$rc" >> "$R"
echo "PGATE_PROBE_READBACK=$(cat "$E/f3twc-pgrep-PROBE.txt")" >> "$R"
if [ "$(cat "$E/f3twc-pgrep-PROBE.txt")" != "0" ]; then
  echo "LAUNCH_BLOCKED_PROBE: gate refused a second launch while a card process was live" >> "$R"
  echo "PGATE_BLOCK_RC=0 (gate detects a busy card)" >> "$R"
else
  echo "PGATE_BLOCK_RC=1 (gate did not detect a busy card)" >> "$R"
fi
wait $eng
rc=$?
echo "RUN_TWO3_RC=$rc" >> "$R"
t=$(now)
resetcheck TWO3 "$s" "$t"

# --- ADV-9-3: harness-can-fail control on the two-card surface: out-of-range token id
printf '128803 999999999\n' > "$E/f3twc-ids-badid.ids"
run_engine NEG2 "0,1" "$E/f3twc-ids-badid.ids" "$E/f3twc-neg-badid.log" ""
echo "CMD: grep -c 'out of range' $E/f3twc-neg-badid.log" >> "$R"
grep -c 'out of range' "$E/f3twc-neg-badid.log" > "$E/f3twc-neg2-err.txt" 2>&1
rc=$?; echo "NEG2_ERR_RC=$rc (0 = the error names the offending token)" >> "$R"
echo "NEG2_ERR_COUNT=$(cat "$E/f3twc-neg2-err.txt")" >> "$R"

# --- ADV-9-4: input-validation control on the two-card surface: genuinely malformed ids (non-numeric)
printf 'hello world\n' > "$E/f3twc-ids-garbage.ids"
run_engine NEG "0,1" "$E/f3twc-ids-garbage.ids" "$E/f3twc-neg-garbage.log" ""

# --- ADV-9-5: out-of-range GPU ordinal on the two-card surface
run_engine NEGGPU "0,2" "$IDS" "$E/f3twc-neg-gpu2.log" ""
echo "CMD: grep -ciE 'gpu|device|ordinal|invalid|out of range' $E/f3twc-neg-gpu2.log" >> "$R"
grep -ciE 'gpu|device|ordinal|invalid|out of range' "$E/f3twc-neg-gpu2.log" > "$E/f3twc-neggpu-err.txt" 2>&1
rc=$?; echo "NEGGPU_ERR_RC=$rc" >> "$R"
echo "NEGGPU_ERR_COUNT=$(cat "$E/f3twc-neggpu-err.txt")" >> "$R"

# --- ADV-9-6: the reset detector must be able to fail: same pattern over a window known to contain resets
echo "CMD: journalctl -k --no-pager --since '2026-10-27 01:41:00' --until '2026-10-27 01:42:30' -o short-iso > $E/f3twc-journal-CTRL.txt 2>&1 (a window with recorded resets)" >> "$R"
{ echo "journal window 2026-10-27 01:41:00 .. 2026-10-27 01:42:30 (known reset window)"; journalctl -k --no-pager --since '2026-10-27 01:41:00' --until '2026-10-27 01:42:30' -o short-iso; } > "$E/f3twc-journal-CTRL.txt" 2>&1
rc=$?; echo "JOURNAL_CTRL_RC=$rc" >> "$R"
echo "CMD: grep -cE 'xe 0000:11:00\.0:.*($RP)' $E/f3twc-journal-CTRL.txt" >> "$R"
grep -cE "xe 0000:11:00\.0:.*($RP)" "$E/f3twc-journal-CTRL.txt" > "$E/f3twc-reset-CTRL.txt" 2>&1
rc=$?; echo "RESET_CTRL_RC=$rc (0 = detector finds resets; must NOT be 1)" >> "$R"
echo "RESET_CTRL_COUNT=$(cat "$E/f3twc-reset-CTRL.txt")" >> "$R"

# --- xpu-smi state after all runs (supplementary; the load receipt is the engine's device lines)
echo "CMD: xpu-smi --query-gpu=index,memory.used,memory.total > $E/f3twc-smi-after.txt 2>&1" >> "$R"
xpu-smi --query-gpu=index,memory.used,memory.total > "$E/f3twc-smi-after.txt" 2>&1
rc=$?; echo "SMI_AFTER_RC=$rc" >> "$R"
cat "$E/f3twc-smi-after.txt" >> "$R"

echo "CMD: pgrep -c strata-dsv41 (post-run re-read)" >> "$R"
pgrep -c strata-dsv41 > "$E/f3twc-pgrep-post.txt" 2>&1
rc=$?; echo "PGREP_POST_RC=$rc" >> "$R"
echo "PGREP_POST_READBACK=$(cat "$E/f3twc-pgrep-post.txt")" >> "$R"
echo "RUN_PHASE_DONE $(date -Is)" >> "$R"
echo "PIPELINE_DONE $(date -Is)"
