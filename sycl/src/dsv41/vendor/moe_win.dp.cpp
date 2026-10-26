// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/dsv41/vendor/moe_win.cu - the window's expert grouping kernel.
// Extracted from Project Maya's src/kernels/cuda/glm_fast.cu (MIT, see third_party/project-maya/NOTICE.md).
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/dsv41/moe_win.hpp"
#include <cstdio>
#include <cstdlib>

namespace strata::dsv41k::glmf {
namespace {
constexpr int kWinEnt = kMaxWinTokens * 8;

// the entries grouped by expert slot (in order of first appearance), each group's members in entry order (t * 8 + i)
// (the tables in shared memory: as locals they were a 2.6 KB stack a thread, which the driver reserves for every thread
// the device can hold - ~420 MB on a V100 - and a launch failed "out of memory" once the expert pool had the rest)
void moe_win_group_kernel(const MoeWin w,
                          unsigned long long *gp, int *cnt, int *gof, int *fill) {

    if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) !=
        0) return;
    int ng = 0;
    for (int t = 0; t < w.T; ++t)
        for (int i = 0; i < 8; ++i) {
            const int e = t * 8 + i;
            gof[e] = -1;
            const unsigned long long p = i < w.k ? w.plan_ptr[e] : 0ull;
            if (p == 0ull) continue;
            int g = 0;
            while (g < ng && gp[g] != p) ++g;
            if (g == ng) {
                gp[ng] = p;
                cnt[ng++] = 0;
            }
            ++cnt[g];
            gof[e] = g;
        }
    fill[0] = 0;
    for (int g = 0; g < ng; ++g) {
        w.grp_ptr[g] = gp[g];
        w.grp_start[g] = fill[g];
        fill[g + 1] = fill[g] + cnt[g];
    }
    w.grp_start[ng] = fill[ng];
    for (int e = 0; e < w.T * 8; ++e)
        if (gof[e] >= 0) w.grp_ent[fill[gof[e]]++] = e;
    *w.n_groups = ng;
}
}  // namespace

void moe_win_group(const MoeWin &w, dpct::queue_ptr s) {
        s->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:183: 'kWinEnt' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<unsigned long long, 1> gp_acc_ct1(
                    sycl::range<1>(128 /*kWinEnt*/), cgh);
                /*
                DPCT1101:184: 'kWinEnt' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<int, 1> cnt_acc_ct1(
                    sycl::range<1>(128 /*kWinEnt*/), cgh);
                /*
                DPCT1101:185: 'kWinEnt' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<int, 1> gof_acc_ct1(
                    sycl::range<1>(128 /*kWinEnt*/), cgh);
                /*
                DPCT1101:186: 'kWinEnt + 1' expression was replaced with a
                value. Modify the code to use the original expression, provided
                in comments, if it is correct.
                */
                sycl::local_accessor<int, 1> fill_acc_ct1(
                    sycl::range<1>(129 /*kWinEnt + 1*/), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, 32),
                                      sycl::range<3>(1, 1, 32)),
                    [=](sycl::nd_item<3> item_ct1) {
                            moe_win_group_kernel(
                                w,
                                gp_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get(),
                                cnt_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get(),
                                gof_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get(),
                                fill_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get());
                    });
        });
    /*
    DPCT1010:140: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1000:139: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009:141: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        std::fprintf(stderr, "dsv41: moe_win_group: %s\n",
                     dpct::get_error_string_dummy(e));
        /*
        DPCT1001:138: The statement could not be removed.
        */
        std::abort();
    }
}
}  // namespace strata::dsv41k::glmf
