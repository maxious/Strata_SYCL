// xmx_probe_test: the XMX capability probe (llama.cpp's gpu_has_xmx / ext_intel_matrix) over every visible GPU.
//   xmx_probe_test [--allow-non-xmx]   # default: exit 1 if any GPU lacks XMX
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <string>
int main(int argc, char** argv) {
    const bool expect_xmx = argc < 2 || std::string(argv[1]) != "--allow-non-xmx";
    int n_gpu = 0, fail = 0;
    for (auto const& dev : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        ++n_gpu;
        const bool xmx = strata::gpu_has_xmx(dev);
        std::string name = dev.get_info<sycl::info::device::name>();
        std::printf("device %d: %s xmx=%d (ext_intel_matrix)\n", n_gpu - 1, name.c_str(), (int) xmx);
        if (expect_xmx && !xmx) { std::printf("  FAIL: GPU has no XMX\n"); ++fail; }
    }
    std::printf("%d GPU device(s), %d without XMX\n", n_gpu, fail);
    return fail == 0 ? 0 : 1;
}