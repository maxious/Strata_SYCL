# Todo 3 attempt-7 brief: formal gate suite under the kernel's F16 input contract (plan dsv41-trellis-sg16-perf)

Worktree: /home/maxious/Strata-wt/dsv41-trellis-w2 (branch dsv41-trellis-w2, HEAD c0e7439c, git clean). Never edit /home/maxious/Strata. Never touch ~/llama.cpp. Never push. Every strata/icpx/binary invocation inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...'. Repro env: ZES_ENABLE_SYSMAN=1 STRATA_DS4_NO_LIGHT=1.

## Established by attempt 6 (do not re-litigate)
The kernel is bit-exact against the repo dequantizer reference (strata::dequantize_iq4_nl + F16 plane bits hand-check). The reported AOT k=128 defect is NOT confirmed; failure_class harness-recipe. Five harness bugs were fixed in attempt 6 (fp16_to_f32 sign-shift leak, subnormal integer-cast branch, rndf range, c.wgu single-matrix overwrite, ref h stored by value not bits). Under the F16-contract oracle: G1 AOT gate/up 65441/65536 exact, maxdiff 0; down 3-4 outside maxdiff 0; G3/G4/G5 PASS both flavors; G7 flavor-identical. The brief-letter gate criteria produce bad>0 because ulp(0)=0 at ±0 references and sub-ulp residuals tally as "outside" — the CRITERIA are the remaining defect, not the kernel.

## What to do
Build a formal gate driver in the worktree (parity/gate harness, e.g. extend ds4_parity.cpp / gate driver) that:
1. Quantizes oracle inputs to the kernel's F16 input contract (bits, not values).
2. Applies the plan's documented tolerance: abs_err <= 32*2^-24*sum_abs + 4*ulp_f32(ref) for tile sums, per-element one-BF16-ulp bound for gate/up/down outputs; at ref==0 treat the bound as the subnormal floor (2^-24), not ulp(0)=0.
3. Runs G1-G7 on BOTH flavors: AOT build (-DSTRATA_XMX_SG16=1, cmake bmg-g21) and JIT build, fresh builds at HEAD.
4. Includes the corrupt-scale-byte negative control: it MUST fail (the gate suite must be able to fail).
5. Shapes M=2,15,16,17,31,32,33,160; one-hot and dense k=128 probes.

## Must NOT
Do NOT edit the kernel unless a device-side defect is proven by a corrected-oracle run (attempt 6 proved none). Do NOT weaken tolerances to force PASS — only fix the ±0-ulp tally artifact and the F16-contract oracle, exactly as above. Never state a throughput number for a prompt that did not complete. Report only un-piped exit codes.

## Acceptance
One gate-suite invocation RC 0 printing MATCH lines for G1-G7 on both flavors, plus the negative control failing as expected. Commit ONLY harness/gate-driver changes if gates pass under corrected criteria: '[Engine] formal sg16 gate tolerances under F16 input contract' on dsv41-trellis-w2 (never push). Kernel commit only if a device defect is proven and fixed.

## Evidence
Append every CMD+un-piped RC to /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-3-attempt7.log ending VERDICT PASS or VERDICT FAIL with failure_class (kernel-behavior | harness-recipe | runtime-stack | gate-criteria). Never end a turn before the VERDICT line exists. rm -rf your scratch dir with ls absence proof RC 2.

## Yield
Keyed JSON {"verdict","failure_class","evidence","repro","confidence","coverage_limits"}.
