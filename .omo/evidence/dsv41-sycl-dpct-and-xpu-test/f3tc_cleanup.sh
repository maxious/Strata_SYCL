#!/bin/bash
# F3 two-card manual QA (task st_01a178c4) - cleanup phase.
# The cleanup receipt is node:fs existsSync === false on each scratch build dir (never ls/test/stat exits).
E=/home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test
R=$E/f3tc-receipts-cleanup.txt
export PATH=/home/maxious/.omo/agent/bundled-bun/daff063de3f967ea:$PATH

echo "=== F3 two-card manual QA: cleanup (task st_01a178c4) ===" > "$R"
echo "CMD: rm -rf /tmp/f3tc_bld_aot /tmp/f3tc_bld_def" >> "$R"
rm -rf /tmp/f3tc_bld_aot /tmp/f3tc_bld_def
rc=$?
echo "CLEANUP_RC=$rc" >> "$R"

echo "CMD: bun -e node:fs existsSync check of /tmp/f3tc_bld_aot and /tmp/f3tc_bld_def > $E/f3tc-cleanup-receipt.txt 2>&1" >> "$R"
bun -e 'import { existsSync } from "node:fs"; const ps = ["/tmp/f3tc_bld_aot", "/tmp/f3tc_bld_def"]; let all = true; for (const p of ps) { const e = existsSync(p); console.log(p + ": existsSync === " + e); if (e !== false) all = false; } console.log("CLEANUP_RECEIPT=node:fs existsSync === " + all); process.exit(all ? 0 : 1);' > "$E/f3tc-cleanup-receipt.txt" 2>&1
rc=$?
echo "BUILD_DIRS_GONE_RC=$rc (0 = every scratch build dir reports node:fs existsSync === false)" >> "$R"
cat "$E/f3tc-cleanup-receipt.txt" >> "$R"
echo "CLEANUP_DONE $(date -Is)" >> "$R"
