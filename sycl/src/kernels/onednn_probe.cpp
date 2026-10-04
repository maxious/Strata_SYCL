// onednn_probe: does oneDNN offer a JIT-optimized (non-ref) matmul on this GPU?  (README P0b #1 ground check.)
//   Mirrors llama.cpp #28985's ggml_sycl_dnnl_detect_optimized_gemm: ask oneDNN which matmul implementation it
//   would pick for a small f16 problem and print impl_info_str().  "ref" means oneDNN has no JIT kernel for this
//   device and its fallback would be < our SYCL kernels (llama.cpp disables oneDNN there).
//   onednn_probe [--allow-none]     # default: exit non-zero when any GPU picks only reference impls
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <cstdio>
#include <string>

static bool probe(const sycl::device& dev) {
    using dt = dnnl::memory::data_type;
    const std::string name = dev.get_info<sycl::info::device::name>();
    const bool xmx = strata::gpu_has_xmx(dev);
    try {
        const sycl::queue q(dev);
        const dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        const dnnl::memory::dims dims    = { 1, 64, 64 };
        const dnnl::memory::dims strides = { 64 * 64, 64, 1 };
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        const dnnl::matmul::primitive_desc pd(
            eng,
            dnnl::memory::desc(dims, dt::f16, strides),
            dnnl::memory::desc(dims, dt::f16, strides),
            dnnl::memory::desc(dims, dt::f32, strides), attr);
        const std::string impl     = pd.impl_info_str();
        const bool optimized       = impl.find("ref") == std::string::npos;
        std::printf("device %s: xmx=%d oneDNN-f16-matmul=\"%s\" -> %s\n", name.c_str(), (int)xmx,
                    impl.c_str(), optimized ? "JIT/optimized" : "REFERENCE (not optimized)");
        return optimized;
    } catch (const std::exception& e) {
        std::printf("device %s: xmx=%d oneDNN probe FAILED: %s\n", name.c_str(), (int)xmx, e.what());
        return false;
    }
}

int main(int argc, char** argv) {
    const bool expect = argc < 2 || std::string(argv[1]) != "--allow-none";
    int n = 0, ok = 0;
    for (auto const& dev : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        ++n;
        if (probe(dev)) ++ok;
    }
    std::printf("%d GPU(s), %d with an optimized oneDNN matmul\n", n, ok);
    return (n == 0 || (expect && ok < n)) ? 1 : 0;
}