#!/bin/bash
# F3 two-card manual QA (task st_01a178c4) - chained pipeline: wait for the AOT build marker (direct
# node:fs state reads), then the card-run phase, then the analysis phase. No product file is touched.
E=/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test
export PATH=/home/maxious/.omo/agent/bundled-bun/daff063de3f967ea:$PATH

echo "PIPELINE_START $(date -Is)" > "$E/f3tc-pipeline.log"

# step 1: the AOT build phase must have recorded AOT_PHASE_DONE (binary linked at <build>/sycl/strata-dsv41).
bun -e 'import { readFileSync } from "node:fs"; const p="/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f3tc-receipts-build-aot.txt"; const sab=new SharedArrayBuffer(4); const deadline=Date.now()+2400000; let ok=false; while(Date.now()<deadline){ try{ if(readFileSync(p,"utf8").includes("AOT_PHASE_DONE")){ ok=true; break; } }catch(e){} Atomics.wait(new Int32Array(sab),0,0,3000); } console.log("BUILD_MARKER="+(ok?1:0)); process.exit(ok?0:1);' >> "$E/f3tc-pipeline.log" 2>&1
rc=$?
echo "BUILD_MARKER_RC=$rc $(date -Is)" >> "$E/f3tc-pipeline.log"
[ "$rc" = "0" ] || { echo "PIPELINE_ABORTED $(date -Is)" >> "$E/f3tc-pipeline.log"; exit 9; }

# step 2: the card runs (pgrep-gated; baseline, two literal two-card runs, spin control, two adversarial runs).
bash "$E/f3tc_runs.sh"
rc=$?
echo "RUNS_SH_RC=$rc $(date -Is)" >> "$E/f3tc-pipeline.log"

# step 3: decode, key lines, per-device lines, reset checks, adversarial checks, todo-8 quote.
bash "$E/f3tc_analyze.sh"
rc=$?
echo "ANALYZE_SH_RC=$rc $(date -Is)" >> "$E/f3tc-pipeline.log"
echo "PIPELINE_DONE $(date -Is)" >> "$E/f3tc-pipeline.log"
