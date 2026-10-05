// include/strata/sycl_queue.hpp - the SYCL port's one addition to the engine's API surface.
//
// Every launcher takes `void* stream`, a cudaStream_t where null means the default stream. dpct migrates the
// cast to `(dpct::queue_ptr) stream` and dereferences it, so a null stream is a null sycl::queue* and a crash.
// q_of() is that cast with CUDA's null-stream meaning restored: the default in-order queue.
//
// IN-ORDER IS LOAD-BEARING. The migration turned ~320 cudaMemcpy/cudaMemcpyAsync sites into queue.memcpy() and
// memcpy_async() calls that depend on the stream's ordering; DPCT flagged every one "assuming in-order queue".
// Every queue this port runs on is in-order - q_of() below, and the second-GPU expert path's stream_
// (remote_experts.cpp) - and nothing under sycl/src calls dpct::get_out_of_order_queue(). A queue built without
// property::queue::in_order() breaks all of those copies silently.
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
