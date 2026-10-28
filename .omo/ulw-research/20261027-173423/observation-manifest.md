# Observation manifest
| observation_id | source | evidence_layer | observer group | independence basis | observer | observed_at | valid_at | artifact | quote/anchor | contamination |
| O1 | /home/maxious/Strata git HEAD | local git | lead | direct command | lead | 2026-10-27T17:34 | 2026-10-27 | wave-1-launch.md | HEAD aa09dc07, branch sycl/dsv41-sycl-support | none |
| O2 | /opt/intel/oneapi listing | local fs | lead | direct listing | lead | 2026-10-27T17:34 | 2026-10-27 | wave-1-launch.md | dnnl component present; libdnnl.so.3.13 | none |
| O3 | sycl/src/dsv41/ds4_gemv.dp.cpp:3-4 | local code | lead | grep | lead | 2026-10-27T17:34 | 2026-10-27 | wave-1-launch.md | gemv_f16_kernel :372, gemv_f16_rows :398, gemv_f32 :438 | none |
| O4 | sycl/docs/sycl-dsv41-measured-report.md | local doc | lead | prior read | lead | 2026-10-27 | 2026-10-25 build ce6218ee | measured-report | decode 1.65 tok/s 1-card; 12.08 tok/s 2-card | prior-session read |
| O5 | .omo/boulder.json:174 | local state | lead | grep | lead | 2026-10-27T17:34 | 2026-10-27 | wave-1-launch.md | missing barrier ... ds4_iq.dp.cpp:1114 | agent-authored state file |
| O6 | /opt/intel/oneapi/compiler/2026.2 include tree | local fs | lead | find, un-piped rc=0 | lead | 2026-10-27 | 2026-10-27 | jm-probe compile.log | no joint_matrix.hpp shipped by icpx 2026.2.0; ext/intel/experimental has esimd only | none |
| O7 | /tmp/w5_jm_research/jm-probe compile.log | local compile | lead | icpx rc=1 un-piped | lead | 2026-10-27 | 2026-10-27 | compile.log | fatal error: joint_matrix.hpp file not found | none |
| O8 | sycl-ls output | local runtime | lead | direct | lead | 2026-10-27 | driver 26.35.39758.10 | run.log | 2x Arc Pro B60 on level_zero UR V2 1.17.39758+10 | none |
| O9 | grep -c joint_matrix per sycl/src/dsv41/*.dp.cpp | local code | lead | un-piped rc=0 | lead | 2026-10-27 | HEAD aa09dc07 | ds4_gguf.dp.cpp 16, ds4_iq.dp.cpp 13, ds4_gemv.dp.cpp 0, ds4_softmax.dp.cpp 0 | none |
| O10 | ds4_gguf.dp.cpp:1152-1163 | local code | lead | read | lead | 2026-10-27 | HEAD aa09dc07 | legacy API jm::joint_matrix<sycl::sub_group,float,use::accumulator,8,16> + comment 'WMMA replaced by scalar FMA (joint_matrix/XMX is the next phase)' | none |
| O11 | /opt/intel/oneapi/dpcpp-ct/2025.3/include/dpct/dpct.hpp | local fs | lead | grep -c (rc pending) | lead | 2026-10-27 | dpcpp-ct 2025.3 | candidate shim for legacy joint_matrix API | pending verification |
EOF; echo APPEND_RC=$?
| O6 | /opt/intel/oneapi/compiler/2026.2 include tree | local fs | lead | find rc=0 | lead | 2026-10-27 | 2026-10-27 | jm-probe | icpx 2026.2.0 ships no sycl/ext/intel/experimental/joint_matrix.hpp | none |
| O7 | /tmp/w5_jm_research/jm-probe compile.log | local compile | lead | icpx rc=1 | lead | 2026-10-27 | 2026-10-27 | compile.log | legacy intel::experimental joint_matrix include not found | none |
| O8 | sycl-ls | local runtime | lead | direct | lead | 2026-10-27 | driver 26.35.39758.10 | r2.log | 2x Arc Pro B60 level_zero UR V2 1.17.39758+10 | none |
| O9 | grep -c joint_matrix per sycl/src/dsv41/*.dp.cpp | local code | lead | rc=0 | lead | 2026-10-27 | HEAD aa09dc07 | ds4_gguf 16, ds4_iq 13, ds4_gemv 0, ds4_softmax 0 | none |
| O10 | ds4_gguf.dp.cpp:1108,1152-1163 | local code | lead | read | lead | 2026-10-27 | HEAD aa09dc07 | namespace jm = sycl::ext::oneapi::experimental::matrix; comment 'WMMA replaced by scalar FMA (joint_matrix/XMX is the next phase)' | none |
| O11 | dpct.hpp dpcpp-ct 2025.3 | local fs | lead | grep -c rc=1 | lead | 2026-10-27 | dpcpp-ct 2025.3 | 0 joint_matrix occurrences - shim does NOT supply it | none |
| O12 | ext/oneapi/experimental/matrix.hpp | local fs+compile | lead | pending in this cell | lead | 2026-10-27 | icpx 2026.2.0 | unified matrix API probe result below | none |
| O13 | /tmp/w5_jm_research/xmx/run.log | local run | lead | RUN_RC=0 | lead | 2026-10-27 | icpx 2026.2.0, B60 sg16 | f16 69.4us 15.1 TFLOPs / i8 34.0us 30.9 TOPS / i8blk 52.0us 20.1 TOPS gate+up; down 34.2/17.7/26.7 us | none |
| O14 | icpx -fsycl -DSTRATA_XMX_SG16=1 xmx_int8_bench.cpp | local compile | lead | BUILD_RC=0 | lead | 2026-10-27 | icpx 2026.2.0 | XMX sg16 compiles for Battlemage | none |
| O15 | /tmp/w5_jm_research/xmx/run_m1.log | local run | lead | RUN_RC=0 | lead | 2026-10-27 | B60 sg16 M=1 | f16 22.2us / i8 11.1us / i8blk 17.9us gate+up; down 11.2/5.9/9.2us | none |
| O16 | /tmp/w5_jm_research/bw/r3.log | local run | lead | RUN_RC=0 | lead | 2026-10-27 | B60 | dot 0.7ms 9GB/s naive scalar, read pass 42.9ms first-touch | none |
| O17 | github ggml fused-gemm.cpp @ d81d0d8f | lane transcript | R1b | completed | R1b | 2026-10-27 | pinned SHA | FG_BK32 KSPLIT4 MAXN64, 8 combos, no M=1 path | none |
| O18 | openvino fully_connected_kernel_gemv.cpp + ternocl_int2_architecture.md + moe_3gemm_swiglu_mlp.cl | lane transcript | R3b | completed | R3b | 2026-10-27 | master pinned | dedicated GEMV sg16, M<=8 tiles, MoE gemv n2x | none |
| O19 | yiliu30.github.io/viz/joint_matrix_debug.html | webfetch | lead | fetched | lead | 2026-10-27 | B70 libsycl .so.8 FAIL / .so.9 PASS | none |
| O20 | xmx_int8_bench.cpp header + malloc_device lines | local grep | lead | GREP_RC=0 | lead | 2026-10-27 | DRAM-not-L2 by design | none |
