# Verification economics
- Cheapest decisive test for feasibility: compile+run Strata's own xmx_int8_bench at M=160 and M=1 (2 builds, 2 runs, ~2 min). Settled C1/C2 without any web source.
- Header availability settled by 4 ls calls, not docs reading (C1).
- Author-reported numbers (Q3,Q4,Q5) were NOT re-run: they are B70 hardware we do not have; kept as author-reported, never converted to Strata predictions.
- oneDNN/gemmstone internals were read from source at pinned SHAs, not from blog summaries (C8).
- Negative claims ('trellis absent', 'no GPU GEMV primitive') need a second independent observer before they enter verified-claims: R1b L4 assigned.
- Naive bw.cpp was cheap but weak (F41, low confidence) - not used for any claim beyond 'scalar naive is far off'.
