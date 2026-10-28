ROLE: you are the executor of this unit; do it yourself; do NOT delegate.
UNIT: plan todo 9 - decode probe D-x3: window crossover at T=8 and T=16 for win_experts.
WORKTREE: supplied in the dispatch message. NEVER touch /home/maxious/Strata.
READ FIRST: win_experts ds4_gguf.dp.cpp:2076, kLightRows=16 ds4_iq.dp.cpp:1316-1317, light path 1321 and 1473.
DO: measure tile vs light at T=8 and T=16, capture us per T with rc=0 lines, and verify the window-verification bits are unchanged. Adopt only if faster and parity-clean; a kLightRows change is a separate decision and out of scope here.
VERIFY: measured us per T, rc=0; window-verification bits identical before/after.
CONSTRAINTS: every build/run inside bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; <command>'. Report only un-piped exit codes, each on its own named RC line. NEVER state a throughput number unless a prompt actually completed - a number from a failed or partial run is a defect. The two B60 cards are exclusive: this unit runs alone, it must stop any server it started before returning (strata launch --stop, PID check) and record the teardown receipt.
MANUAL QA: the real surface is a completed API prompt. Start the server, send one fixed prompt via curl -i to the OpenAI-compatible endpoint at temperature 0, capture the HTTP status and the completion text plus the engine's own tok/s line into the evidence log. A dry run or a plan printout is NOT evidence.
CLEANUP: stop every server/process you started, remove temp dirs, and record the teardown receipts.
ARTIFACT: /home/maxious/Strata/.omo/evidence/dsv41-trellis-sg16-perf/task-9-dsv41-trellis-sg16-perf.log ending with VERDICT PASS or VERDICT FAIL.
COMMIT: no product change expected; title '[Benchmark] window crossover probe' only if a source knob was added.
RETURN a DoneClaim JSON: task, changed_files, tests, manual_qa, cleanup, risks.
SERVE RECIPE (verified by an explore lane and re-probed by the orchestrator this session - use it, do not rediscover):
- pack: /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 (metadata pack: ds4.txt, engram.bin, expert-profile.bin, tensors.txt, tokenizer/; the 232G of IQ2_XXS + Engram GGUF shards sit in the parent dir)
- build the engine from YOUR worktree: cmake -S sycl -B <dir> -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON   (target: strata-dsv41, sycl/CMakeLists.txt:297-318)
- run: bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ZES_ENABLE_SYSMAN=1 <build>/sycl/strata-dsv41 --pack <pack> --gpus 0 --ids @<ids-file> --n 8 --max-context 512 --verbose'
- tok/s lines to capture verbatim: 'prompt: %zu tokens in %.0f ms (%.2f tok/s, windows of %d)' (src/dsv41/ds4_main.cpp:1380) and 'decode: %zu tokens in %.0f ms (%.2f tok/s); CPU lane: %lld of %lld experts; the cache moved %lld' (ds4_main.cpp:1445); serve mode uses :922-924
- the AVX512 expert-kernel exit at src/kernels/cpu/expert_layout.cpp:346-356 is NOT called by ds4_main.cpp, so this Ryzen host serves; no workaround needed
- --stream-experts (sycl/src/program/generate.cpp:945,2034) exists if the xe driver needs it
- A stale binary at /home/maxious/Strata/build-sycl-dsv41/strata-dsv41 exists from the main checkout: do NOT use it for evidence, build your own from your worktree so the numbers belong to this branch.
