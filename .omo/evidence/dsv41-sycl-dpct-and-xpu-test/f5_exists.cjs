// F5 cleanup audit probe helper (audit tooling, kept beside the audit as evidence).
// Usage: bun /home/maxious/Strata/.omo/evidence/dsv41-sycl-dpct-and-xpu-test/f5_exists.cjs <path>
// Prints node:fs existsSync for <path>, then exits 0 when the path is ABSENT and 1 when it EXISTS,
// so the un-piped exit code of the printed command is itself the absence receipt.
const fs = require("node:fs");
const p = process.argv[2];
if (!p) {
  console.error("usage: f5_exists.cjs <path>");
  process.exit(64);
}
const e = fs.existsSync(p);
console.log("existsSync(" + p + ") === " + e);
process.exit(e ? 1 : 0);
