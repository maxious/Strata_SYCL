# G016-e8 Quality Gate Review (final)

## Verdict
APPROVE

## Scope label (agreed)
Version-gating mechanism proven externally on bmg_g31 (S6 single-variable experiment);
bmg_g21-specific 2025.3 failure is unverified locally (no 2025.3 libsycl on this machine).

## C001 — happy: sg16 joint_matrix + DPAS run on Arc Pro B60 — PASS
- M=160 block: device "Intel(R) Arc(TM) Pro B60 Graphics, sub-group 16", BENCH_M160_RC=0.
- f16 joint_matrix 15.1 TFLOPs + i8 DPAS 30.9 TOPS + i8blk 20.2 TOPS (gate+up),
  15.3/29.6/19.6 (down). Completed run on target hardware, matches S2/F35.

## C003 — regression: decode M=1 numbers on same device — PASS
- M=1 block: same B60, BENCH_M1_RC=0; gate+up f16 22.3 / i8 11.1 / i8blk 17.9 us,
  down 11.9 / 5.9 / 9.2 us. Matches S1 (22.2/11.2 f16; run-to-run drift immaterial).

## C002 — edge: availability version-gated — PASS under scope label
- 2026.2 passes, locally: corrected header capture shows dpas.hpp and matrix.hpp exist
  (rc=0, with file listings), libsycl.so.9.0.0 resolved, icpx 2026.2.0, both benches RC=0.
- 2025.3 fails: mechanism proven by S6 single-variable experiment on bmg_g31 (2025.3.2
  FAIL -> 2026 nightly PASS, byte-identical driver/IGC). Not run on bmg_g21 (no local
  2025.3 runtime); the transcript now labels this explicitly as unverified, not asserted.
- Earlier rc=2/2/0 header block is explained: wrong include/oneapi/experimental paths;
  superseded by the corrected-path capture.

## Throughput-number audit
- All locally-stated throughput traces to completed runs (BENCH_*_RC=0). External figures
  (PR #644 B70, dkhaldi README) are labeled author/README, not our measurements. OK.

## Blockers
(none)

## Residual caveats (non-blocking)
- bmg_g21 2025.3-fail stays unverified; keep the scope label when citing C002, and do not
  state "2025.3 fails on B60" as a measured result.
