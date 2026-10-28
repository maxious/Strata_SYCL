# Row 15 gate review - 83417a6f hyper-connect / sparse-attn kernel copies

Date: 2026-10-26. Branch: sycl/dsv41-sycl-support. Build tree: /tmp/r15v_build (clean, compiles with
TMPDIR=/home/maxious/tmpbuild). Gated commit 83417a6f33727d86e654c4788f72bee17f10193d: 6 files,
1197 insertions, 1529 deletions, all under sycl/.

## Gate 1 - configure and build

SYCL_CONFIGURE_RC=0
bash -c 'mkdir -p /home/maxious/tmpbuild; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; TMPDIR=/home/maxious/tmpbuild cmake -S sycl -B /tmp/r15v_build -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_DSV41=ON -DSTRATA_SYCL_DSV41_EXE=ON'

SYCL_BUILD_RC=0
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; TMPDIR=/home/maxious/tmpbuild cmake --build /tmp/r15v_build'

All targets reached: the full build ran 260/260 edges (last line `[260/260] Linking CXX executable strata`), and a
verification re-run of the same build command reached `[53/53] Linking CXX executable strata`. Zero FAILED lines in
both: `grep -c FAILED` on the verification build log = 0, and the full build exited 0 (ninja exits non-zero at the
first FAILED edge, so reaching 260/260 with exit 0 is zero failures by construction).

## Gate 2 - one definition per kernel

nm /tmp/r15v_build/libstrata_dsv41.a | grep -E ' T .*(hc_pre_norm|hc_mix_scale|sparse_attn)'

```
0000000000000050 T _ZN6strata3ds411hc_pre_normEPKfS2_S2_iiifPfPN4sycl3_V15queueE
0000000000000390 T _ZN6strata3ds411sparse_attnEPKfPKtPKvPKiiRKNS0_8AttnRowsES2_iiifPfPN4sycl3_V15queueEiiPKNS0_11AttnScratchE
0000000000000200 T _ZN6strata3ds412hc_mix_scaleEPKfiiifPfPN4sycl3_V15queueE
```

Exactly one T each: hc_pre_norm 1, hc_mix_scale 1, sparse_attn 1.

The same three names grep to zero in the old copies' object files:

- /tmp/r15v_build/CMakeFiles/strata-dsv41.dir/src/dsv41/ds4_kernels.dp.cpp.o: hc_pre_norm 0, hc_mix_scale 0,
  sparse_attn 0 (0 occurrences total)
- /tmp/r15v_build/CMakeFiles/strata_dsv41.dir/src/dsv41/ds4_launch.dp.cpp.o: hc_pre_norm 0, hc_mix_scale 0,
  sparse_attn 0 (0 occurrences total)

The one home is sycl/src/dsv41/ds4_attn_hc.dp.cpp, built as ds4_attn_hc.dp.cpp.o into libstrata_dsv41.a.
(ds4_kernels.dp.cpp is compiled only into the strata-dsv41 executable target; ds4_launch.dp.cpp only into the
library - hence the two different object directories above.)

## Gate 3 - ctest

SYCL_CTEST_RC=0
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ctest --test-dir /tmp/r15v_build -R ds4 --output-on-failure'

```
1/2 Test #1: ds4_parity ....... Passed  0.27 sec
2/2 Test #2: ds4_blas_parity .. Passed  0.33 sec
100% tests passed, 0 tests failed out of 2
```

Both named suites pass. Re-run after both linkers had settled: 2/2 passed again (ds4_parity 0.27 s,
ds4_blas_parity 0.31 s).

## Gate 4 - runtime

pgrep -c strata-dsv41 read 0 immediately before the launch (0 waits).

RUN_RC=0
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO /tmp/r15v_build/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 --gpus 0 --ids @/tmp/q_france.ids --n 8 --max-context 512'

prompt: line (quoted from the run):

```
prompt: 10 tokens in 262 ms (38.22 tok/s, windows of 8192)
```

decoded text (quoted from the run's output line, 8 tokens):

```
671,6102,294,8760,344,2619,51119,42499,
```

(The decoded line is printed without a trailing newline and the next log line follows it inline in the log; the
token sequence above is exactly what the run produced.) Output was captured to /home/maxious/tmpbuild/r15v_run.log;
the exit code was echoed un-piped.

## Gate 5 - scope

git show --name-only 83417a6f lists only sycl/ paths:

```
sycl/CMakeLists.txt
sycl/src/dsv41/ds4_attn_hc.dp.cpp
sycl/src/dsv41/ds4_kernels.dp.cpp
sycl/src/dsv41/ds4_launch.dp.cpp
sycl/src/dsv41/ds4_launch.dp.hpp
sycl/src/dsv41/ds4_numerics.dp.hpp
```

Subject starts with [Engine]: `[Engine] dsv41 SYCL: give the hyper-connect and sparse-attn kernel copies one home`.
Diffstat matches the row: 6 files changed, 1197 insertions(+), 1529 deletions(-).

## Notes and assumptions

- /tmp/r15v_build held a stale tree from an earlier attempt and was removed before configuring, so Gate 1 is a clean
  full build. /tmp/r16v6_build (another row's tree) was never touched.
- Working tree HEAD is d337362b ("[Docs] dsv41 SYCL: measured report for the ported engine"), one commit after
  83417a6f. Its only path is sycl/docs/sycl-dsv41-measured-report.md, and the build-relevant diff 83417a6f..HEAD is
  empty, so what was built and run is the gated commit's code.
- Configure and build ran as one chain with both RCs echoed un-piped between the halves; nothing was piped through
  tee, so every RC is the producing command's own exit code.
- The verification re-run overlapped the first build's last edge and re-linked the 53 executables from identical
  inputs; both linkers exited 0 and ctest was re-run green after they settled.

## Cleanup

`rm -rf /tmp/r15v_build` (this row's tree only). This row never issued a removal against /tmp/r16v6_build (only
`ls -d` checks; that tree disappeared during this run when its own row cleaned up). Proven gone after removal:
`node:fs` `existsSync('/tmp/r15v_build') === false` -> `true`.

VERDICT confirmed
