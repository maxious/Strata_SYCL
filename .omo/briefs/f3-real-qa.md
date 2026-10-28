# F3 — Real manual QA (dsv41-trellis-sg16-perf)

Run the real surface from the base worktree /home/maxious/Strata-wt/dsv41-trellis-base (branch dsv41-trellis-int, HEAD after todo-5 merge). Steps, each with un-piped RC printed directly above its output:
1. Build/serve: inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ...' run strata launch --family deepseek41 --model ist-das-2.00bpw --yes --no-tui (env ZES_ENABLE_SYSMAN=1 STRATA_DS4_NO_LIGHT=1; keep 127.0.0.1). If the full flavor link fails with the known pre-existing undefined CPU-kernel refs, fall back to the AOT bmg-g21 build path used by the evidence logs and say so.
2. curl the OpenAI-compatible endpoint with temperature 0, a fixed short prompt, max_tokens 32, twice; the two completions must be byte-identical (diff RC=0). Report the exact token text. Never state throughput unless a prompt completed.
3. Kill-switch check: relaunch or restart with STRATA_DS4_NO_STAGEA=1 and confirm the route log shows legacy path (grep the server log; RC printed).
4. strata launch --stop; pgrep -x strata-dsv41 → RC=1 (GPU idle).
Do not modify repo files. End with exactly one line: VERDICT APPROVED or VERDICT REJECTED + reason. Use real tools inside eval; never XML-as-text; never end before the VERDICT line.
