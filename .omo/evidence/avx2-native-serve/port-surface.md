# dsv41 SYCL port surface - measured 2026-10-26, icpx 2026.2.0, HEAD e2d89a94466c61fc2388a7550b446f536225e360

Front-end target: sycl/CMakeLists.txt STRATA_SYCL_DSV41_EXE (OFF by default). With STRATA_SYCL_DSV41=ON and the exe option ON: CFG_RC=0, PARITY_BUILD_RC=0, EXE_RC=1.
Link log: 93 undefined references resolving to 37 distinct symbols, all methods of strata::ds4::Model.
ds4_model.hpp: 199 lines, class Model at line 104. ds4_model.cu: 9450 lines, 510 lines mention cuda / exl3 / __global / gemv / softmax / wmma; 38 Model:: definitions, each a thin delegation (1-16 lines) over Model::Impl.
Default path unchanged: at HEAD 76b3c960 the SYCL tree builds (CFG_RC=0, BUILD_RC=0) and ctest -R ds4 passes 2/2 (CTEST_RC=0).
