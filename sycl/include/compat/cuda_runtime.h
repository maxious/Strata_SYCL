// Port-only stand-in for CUDA's cuda_runtime.h, for compiling the converted dsv41 files (sycl/src/dsv41/)
// on a machine with no CUDA toolkit. It is reached only by the opt-in strata_dsv41 target
// (STRATA_SYCL_DSV41=ON, docs/SYCL_DSV41_PLAN.md): include/compat sits on that target's include path alone,
// so nothing else in the build sees it. The converted code carries dpct's queue_ptr everywhere it used a
// stream; an unconverted header (strata/dsv41/exl3.hpp) still names cudaStream_t in its declarations, and
// this is that name. Strata's CUDA sources never include this file - nvcc takes the real header.
//
// The runtime calls below are what dpct's ds4_model draft still makes by CUDA names (the gap list measured
// against the draft: events, device selection, copies, pinned memory, stream waits).  They map onto one SYCL
// queue; cudaEventElapsedTime reports cudaErrorNotReady because the shim owns no timing - nothing here invents
// a measurement.  blockIdx / threadIdx are a launch-context stand-in: dpct left pred_proj_k's body
// unconverted, and the zeroed ids compile those reads.  No real launch goes through the stand-in.
#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstddef>

typedef dpct::queue_ptr cudaStream_t;
typedef dpct::event_ptr cudaEvent_t;
typedef int cudaError_t;
constexpr cudaError_t cudaErrorSuccess = 0;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorNotReady = 600;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};
constexpr unsigned cudaStreamDefault = 0x00;
constexpr unsigned cudaStreamNonBlocking = 0x01;
constexpr unsigned cudaEventDefault = 0x00;
constexpr unsigned cudaEventBlockingSync = 0x01;
constexpr unsigned cudaEventDisableTiming = 0x02;
constexpr unsigned cudaEventInterprocess = 0x04;
constexpr unsigned cudaHostRegisterDefault = 0x00;
constexpr unsigned cudaHostRegisterPortable = 0x01;
constexpr unsigned cudaHostRegisterMapped = 0x02;

struct cuda_dim3 {
    unsigned x, y, z;
};
inline constexpr cuda_dim3 blockIdx{0, 0, 0};
inline constexpr cuda_dim3 threadIdx{0, 0, 0};

namespace strata::ds4::compat {
inline sycl::queue& q() {
    static sycl::queue dq{sycl::default_selector_v};
    return dq;
}
}  // namespace strata::ds4::compat

inline const char* cudaGetErrorString(cudaError_t) { return "ds4 cuda_runtime compat on a SYCL queue"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) {
    if (d) *d = 0;
    return cudaSuccess;
}
inline cudaError_t cudaGetDeviceCount(int* n) {
    if (n) *n = 1;
    return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() {
    strata::ds4::compat::q().wait();
    return cudaSuccess;
}
inline cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total) {
    const size_t t = (size_t) strata::ds4::compat::q().get_device().get_info<sycl::info::device::global_mem_size>();
    if (total) *total = t;
    if (free_bytes) *free_bytes = t;
    return cudaSuccess;
}
inline cudaError_t cudaMalloc(void** p, size_t n) {
    if (p == nullptr) return 1;
    *p = sycl::malloc_device(n, strata::ds4::compat::q());
    return *p != nullptr ? cudaSuccess : 1;
}
template <typename T>
inline cudaError_t cudaMalloc(T** p, size_t n) {
    return cudaMalloc(reinterpret_cast<void**>(p), n);
}
inline cudaError_t cudaFree(void* p) {
    sycl::free(p, strata::ds4::compat::q());
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    strata::ds4::compat::q().memcpy(dst, src, n).wait();
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t s = nullptr) {
    if (s != nullptr) s->memcpy(dst, src, n);
    else strata::ds4::compat::q().memcpy(dst, src, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void* p, int v, size_t n) {
    strata::ds4::compat::q().memset(p, v, n).wait();
    return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t s = nullptr) {
    if (s != nullptr) s->memset(p, v, n);
    else strata::ds4::compat::q().memset(p, v, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                                cudaMemcpyKind) {
    for (size_t r = 0; r < height; ++r)
        strata::ds4::compat::q().memcpy((char*) dst + r * dpitch, (const char*) src + r * spitch, width).wait();
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                     size_t height, cudaMemcpyKind, cudaStream_t s = nullptr) {
    sycl::queue& qq = s != nullptr ? *s : strata::ds4::compat::q();
    for (size_t r = 0; r < height; ++r)
        qq.memcpy((char*) dst + r * dpitch, (const char*) src + r * spitch, width);
    return cudaSuccess;
}
inline cudaError_t cudaHostAlloc(void** p, size_t n, unsigned) {
    if (p == nullptr) return 1;
    *p = sycl::malloc_host(n, strata::ds4::compat::q());
    return *p != nullptr ? cudaSuccess : 1;
}
inline cudaError_t cudaFreeHost(void* p) {
    sycl::free(p, strata::ds4::compat::q());
    return cudaSuccess;
}
inline cudaError_t cudaHostRegister(void*, size_t, unsigned) { return cudaSuccess; }
inline cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }
inline cudaError_t cudaHostGetDevicePointer(void** d, void* h, unsigned) {
    if (d) *d = h;
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    if (s != nullptr) s->wait();
    return cudaSuccess;
}
inline cudaError_t cudaStreamQuery(cudaStream_t s) {
    if (s != nullptr) s->wait();
    return cudaSuccess;
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t e, unsigned = 0) {
    if (e != nullptr) e->wait();
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    if (s != nullptr) s->wait();
    return cudaSuccess;
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) {
    if (s == nullptr) return 1;
    *s = new sycl::queue{strata::ds4::compat::q().get_context(), strata::ds4::compat::q().get_device()};
    return cudaSuccess;
}
inline cudaError_t cudaEventQuery(cudaEvent_t e) {
    if (e == nullptr) return 1;
    return e->get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete
               ? cudaSuccess
               : cudaErrorNotReady;
}
inline cudaError_t cudaEventCreate(cudaEvent_t* e) {
    if (e == nullptr) return 1;
    *e = new sycl::event{};
    return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { return cudaEventCreate(e); }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = nullptr) {
    if (e == nullptr) return 1;
    auto mark = [](sycl::handler& cgh) { cgh.single_task([]() {}); };
    *e = s != nullptr ? s->submit(mark) : strata::ds4::compat::q().submit(mark);
    return cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    if (e != nullptr) e->wait();
    return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t e) {
    delete e;
    return cudaSuccess;
}
inline cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t) { return cudaErrorNotReady; }

// CUDA-only qualifiers nvcc understands and icpx does not; the unconverted ggml headers use them on
// declarations only, so empty is right for the SYCL build.
#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
