// include/strata/sycl_queue.hpp - the SYCL port's one addition to the engine's API surface.
//
// Every launcher takes `void* stream`, a cudaStream_t where null means the default stream. dpct migrates the
// cast to `(dpct::queue_ptr) stream` and dereferences it, so a null stream is a null sycl::queue* and a crash.
// q_of() is that cast with CUDA's null-stream meaning restored: the default in-order queue.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

namespace strata {
inline sycl::queue* q_of(const void* stream) {
    return stream ? (sycl::queue*) stream : &dpct::get_in_order_queue();
}

/// XMX (DPAS / ext_intel_matrix) capability probe, the SYCL analog of llama.cpp's gpu_has_xmx()
/// (ggml/sycl/common.cpp -> sycl::aspect::ext_intel_matrix). Gates Strata's XMX kernel paths so they are only
/// reachable on a card whose runtime advertises matrix units, instead of purely by env flag.
inline bool gpu_has_xmx(sycl::device const& dev) {
    return dev.has(sycl::aspect::ext_intel_matrix);
}
inline bool gpu_has_xmx(sycl::queue const& q) {
    return gpu_has_xmx(q.get_device());
}
inline bool gpu_has_xmx(sycl::queue const* q) {
    return q && gpu_has_xmx(*q);
}
}  // namespace strata
