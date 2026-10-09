#!/bin/bash
cd /home/maxious/strata-upstream-baseline || exit 9
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u
export STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 STRATA_STAGER_THREADS=12 STRATA_RING_TIMEOUT_S=8
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export STRATA_NATIVE=1
export STRATA_VERIFY_TRACE=1 STRATA_VERIFY_EAGER=1 STRATA_PREFILL_ALLOC_DBG=1 STRATA_PLAN_DEBUG=1 STRATA_MIRROR_DEBUG=1
export STRATA_UR_ALLOC_TRACE=1
export LD_PRELOAD=/home/maxious/xe-dumps/ur-alloc-trace.so

BIN=${BIN:-/home/maxious/strata-upstream-baseline/sycl/build-test/strata}
OUT=/home/maxious/xe-dumps/run-$(date +%H%M%S)
mkdir -p "$OUT"
LOG="$OUT/engine.log"
FAULTS="$OUT/faults.txt"
FIFO=/tmp/engine-in.$$
PROMPTS=/home/maxious/strata-upstream-baseline/sycl/benchy-results/fix-test2/prompts
IDS=$(cat "$PROMPTS/20.ids" 2>/dev/null | tr -d '\n')
if [ -z "$IDS" ]; then echo "NO IDS FILE in $PROMPTS"; exit 8; fi

sudo dmesg -C 2>/dev/null
rm -f "$LOG" "$FIFO"
mkfifo "$FIFO"

( echo 'OFF'; exit 0; ) 2>/dev/null &  # placeholder to keep the block open
(
  while :; do
    for p in /sys/class/drm/card*/device/devcoredump/data; do
      [ -e "$p" ] || continue
      sudo cp "$p" "$OUT/devcd-tmp.bin" 2>/dev/null
      sz=$(sudo stat -c %s "$OUT/devcd-tmp.bin" 2>/dev/null || echo 0)
      if [ "${sz:-0}" -gt 1000 ]; then
        sudo mv "$OUT/devcd-tmp.bin" "$OUT/devcd-live.bin"
        echo "devcoredump banked $sz bytes"
        break
      fi
    done
    sleep 0.2
  done
) &
GRABBER=$!
(
  timeout 600 "$BIN" --serve --pack /home/maxious/Strata-data/packs/q2_0 \
    --native /home/maxious/ComfyUI/koboldcpp/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --ple-gguf /home/maxious/ComfyUI/koboldcpp/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
    --expert-profile /home/maxious/Strata_SYCL/data/expert-profile.bin --mtp /home/maxious/Strata-data/mtp/rt \
    --expert-cache auto --prefill 4096 --spec 2 --spec-min-p 0.5 --max-context 32768 --kv int8 \
    --stream-experts --vram-reserve-mib 2048 --layer-split auto --split-device 1 \
    < "$FIFO" > "$LOG" 2>&1
  echo "ENGINE_RC=$?" >> "$LOG"
  echo "__ENGINE_DONE__" >> "$LOG"
) &
ENGINE_WRAPPER=$!

(
  exec 3>"$FIFO"
  for i in $(seq 1 300); do
    grep -q 'session is up' "$LOG" 2>/dev/null && break
    sleep 1
  done
  printf 'GEN 256 %s\n' "$IDS" >&3
  for i in $(seq 1 300); do
    grep -qE '^DONE [0-9]|__ENGINE_DONE__' "$LOG" 2>/dev/null && break
    sleep 1
  done
  printf 'QUIT\n' >&3 2>/dev/null || true
  exec 3>&-
) &
FEEDER=$!

for i in $(seq 1 240); do
  kill -0 $ENGINE_WRAPPER 2>/dev/null || break
  sleep 1
done
if kill -0 $ENGINE_WRAPPER 2>/dev/null; then
  echo "still running after the budget; stopping it" | tee -a "$LOG"
  sudo pkill -x strata
fi
wait $ENGINE_WRAPPER 2>/dev/null
kill $FEEDER 2>/dev/null
kill $GRABBER 2>/dev/null
sudo /home/maxious/xe-dumps/capture.sh "cd-$(date +%H%M%S)" 2>&1 | head -8

sudo dmesg > "$FAULTS" 2>/dev/null
echo "=== run dir: $OUT"
echo "=== engine result ==="
grep -E 'ENGINE_RC|DONE [0-9]|decode [0-9]+ tokens|verify:|took|error' "$LOG" | tail -8
echo "=== takes logged: $(grep -c 'prefill-alloc' "$LOG")"
echo "=== faults this run: $(grep -c 'Faulted Address' "$FAULTS")  (full blocks in $FAULTS)"
grep -E 'Faulted Address|EngineClass|Error:|ASID|FaultLevel|SRCID' "$FAULTS" | head -24
rm -f "$FIFO"
