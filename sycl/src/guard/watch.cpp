
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/guard/watch.hpp"

#include <cstdint>
#include <cstdio>
#include <atomic>

namespace strata::guard {

namespace {
std::atomic<Watch*> g_watch{nullptr};
const bool g_installed = [] {
    std::set_terminate([] {
        if (Watch* w = g_watch.load()) w->dump(stderr);
        std::abort();
    });
    return true;
}();
}

Watch* watch() noexcept { return g_watch.load(); }
void install(Watch* w) noexcept { g_watch.store(w); }

}
