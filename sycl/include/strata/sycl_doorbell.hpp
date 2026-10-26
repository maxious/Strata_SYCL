// include/strata/sycl_doorbell.hpp - the SYCL port: host<->device flag traffic that must bypass the GPU's caches.
//
// Strata's decode loop is a handshake through host-mapped memory: the GPU rings a sequence number the host polls,
// and a one-thread kernel spins on a flag the host writes. In CUDA those are `volatile` loads and stores, which
// nvcc turns into cache-bypassing accesses. A `volatile` in SYCL device code carries no such meaning on Intel
// GPUs: the spin read its first value from L3 forever (measured: the GPU at 100% and the host seeing no ring).
// Atomic loads and stores with system scope are the accesses that go to memory, so every side of the handshake
// goes through these two.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <string>
#include "strata/sycl_queue.hpp"

namespace strata {
// STRATA_SYCL_A770_FAST=1 (opt-in, experimental, PR #1710): the A770 router and 64-bit paired payload transfers.
inline bool a770_fast() {
    static const bool on = [] { const char* v = std::getenv("STRATA_SYCL_A770_FAST"); return v != nullptr && v[0] == '1'; }();
    return on;
}
using sys_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system>;

// Payload pairs use 64-bit transactions; flags remain 32-bit. Keep the same
// cache-bypass contract while transferring two floats per PCIe transaction.
template<class T> inline T sys_load_mapped(const T* p) {
    using read_props = decltype(sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::read_hint<sycl::ext::intel::experimental::cache_control<
            sycl::ext::intel::experimental::cache_mode::uncached,
            sycl::ext::oneapi::experimental::cache_level::L1,
            sycl::ext::oneapi::experimental::cache_level::L3>>));
    sycl::ext::oneapi::experimental::annotated_ptr<T, read_props> u(const_cast<T*>(p));
    return u[0];
}
inline sycl::float4 load_mapped_float4(const sycl::float4* p) {
    const uint64_t* words = reinterpret_cast<const uint64_t*>(p);
    const auto lo = sycl::bit_cast<sycl::float2>(sys_load_mapped(words));
    const auto hi = sycl::bit_cast<sycl::float2>(sys_load_mapped(words + 1));
    return sycl::float4(lo.x(), lo.y(), hi.x(), hi.y());
}

// A system-scope atomic load of host USM is still served from the GPU's cache on an Arc Pro B60 (NEO 26.31,
// oneAPI 2026.1.1): a device spin never sees the host's store and runs to the spin bound every time (bounded host/GPU
// ping-pong: 263 ms per round trip, 168 of 200 waits hit the bound). An explicit uncached L1+L3 read hint goes to
// memory every time: 2.4 us per round trip, 0 of 200. The acquire fence keeps the compiler from hoisting the load
// out of a spin loop. -DSTRATA_DOORBELL_ATOMIC_LOAD restores the atomic load.
#ifndef STRATA_DOORBELL_ATOMIC_LOAD
using doorbell_uncached_read = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::read_hint<sycl::ext::intel::experimental::cache_control<
        sycl::ext::intel::experimental::cache_mode::uncached,
        sycl::ext::oneapi::experimental::cache_level::L1, sycl::ext::oneapi::experimental::cache_level::L3>>));
inline uint32_t sys_load(const volatile uint32_t* p) {
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, doorbell_uncached_read> u(const_cast<uint32_t*>(p));
    return u[0];
}
#else
inline uint32_t sys_load(const volatile uint32_t* p) {
    return sys_atomic_u32(*const_cast<uint32_t*>(p)).load();
}
#endif
// A flag the copy engine wrote into DEVICE memory (a queue's memcpy).  The read has to go to memory every time: the
// copy's write is in memory, and the card's cache line for the flag is stale (the flag's slot is used again and again,
// and the copy engine's write does not refresh the card's line).  Measured on an Arc Pro B60, 2026-10-26, on a slot
// the card had already cached: a device-scope atomic read the slot's value from 64 requests earlier 1,000,000 times
// over (the wait never saw the copy), while the uncached read goes to memory every time.
template <class T> inline T dev_load(const T* p) {
    return sys_load_mapped(p);
}

// The answer's wait needs this many reads to be credible: the CPU lane's cold expert read takes ~40,000 reads (the
// disk, ~0.1 s, at the 2.4 us one measured in this header).  A wait that ran to a bound below it had no chance - the
// operator's bound was too tight - and the engine ends the run rather than going on with a missing answer.
constexpr uint32_t kAnswerCredibleReads = 40000;

// The host-mapped record of a wait: the reads it made, the last value it read, whether it ran to its bound, and which
// wait it was.  The device writes it (the device's write to host memory is the direction that works: the host reads the
// polled memory and sees it).  The card's own memory cannot be read back reliably: the host's copy of it is a DMA, and
// the card's writes sit in its cache until they are written back (measured on an Arc Pro B60, 2026-10-26: 1 of 64
// records readable).
struct AnsRec {
    unsigned reads = 0;    // the reads the wait made
    unsigned last = 0;     // the last value it read
    unsigned fault = 0;    // 1: the wait ran to its bound
    unsigned site = 0;     // which wait it was
};

// The answer's slot in device memory: the flag the host's copy engine wrote (the card spins on it) and the pointers to
// the host-mapped record of the wait and to the host-mapped word the wait marks when it runs to its bound.  The slot is
// a cache line of its own: the copy engine's write covers the whole line (measured: a 4-byte copy clears the rest of
// the line), so the host copies the slot's bytes - then nothing else in the line is touched and no other slot's flag is.
struct AnsSlot {
    unsigned seq = 0;                       // the flag: the seq the answer carries
    unsigned pad0 = 0;
    AnsRec* host_rec = nullptr;             // the host-mapped record of the wait
    unsigned* host_fault = nullptr;         // the host-mapped word the wait marks on a bound hit (null: none)
    char pad[40];                           // (the slot is a 64-byte cache line of its own)
};
// The answer's wait sites: the token path's wait (one token a window) and the window path's wait (a window of
// tokens).  Both spin on the same kind of flag, in device memory.
enum : unsigned { kSpinAnswerToken = 1, kSpinAnswerWin = 2 };

// The decode's answer wait: the card waits for the CPU lane's part (the experts it does not hold), so the bound has
// to outlast a cold expert read (the disk: ~0.1 s, ~40,000 reads at the 2.4 us one measured in this header) and stay
// under the driver's 5 s job timeout (a longer spin resets the GT).  STRATA_SPIN_MAX overrides it, as in
// strata::spin_max().
inline uint32_t answer_spin_max() {
    static const uint32_t v = [] {
        if (const char* e = std::getenv("STRATA_SPIN_MAX")) {
            const long n = std::atol(e);
            if (n > 0) return (uint32_t) n;
        }
        return 1000000u;   // ~2.4 s at the measured 2.4 us per uncached read
    }();
    return v;
}

// A bounded wait's record (the hand-off's waits, whose flags are host-mapped): the reads it made, the last value it
// read, the value it waited for, and the request's seq.  The device writes them; the host reads them back.
struct SpinRec {
    unsigned count = 0;    // the reads the wait made
    unsigned last = 0;     // the last value it read
    unsigned target = 0;   // the value it waited for
    unsigned seq = 0;      // the request's seq
    unsigned site = 0;     // which wait it was
};
// The wait sites: the hand-off's get (the card waits for the host's copy of a peer's rows), the readers' ack, and the
// tensor-parallel all-reduce's wait (a member waits for the others' partials).
enum : unsigned { kSpinXferGet = 1, kSpinFlags = 2, kSpinAllReduce = 3 };
constexpr int kSpinRecs = 64;   // the records a device keeps (the newest ones)
struct SpinBuf {
    unsigned next = 0;       // the next record's index (a device atomic)
    SpinRec rec[kSpinRecs];
};
inline void spin_record(SpinBuf* buf, unsigned site, unsigned seq, unsigned target, unsigned count, unsigned last) {
    if (buf == nullptr) return;
    sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device> next(buf->next);
    const unsigned i = next.fetch_add(1u);
    SpinRec& r = buf->rec[i % kSpinRecs];
    r.count = count;
    r.last = last;
    r.target = target;
    r.seq = seq;
    r.site = site;
}

// The store side of the handshake (device -> host) needs the same hint. A system-scope atomic store, even followed by
// a release fence, is served from the GPU's cache on an Arc (xe) and reaches the host only when the kernel ends
// (measured on an A770: 482 ms into a 482 ms kernel), so a host that polls the ring mid-window never sees it; that is
// why the decode window had to wait for the whole graph (STRATA_VERIFY_NO_HOST) and every expert had to be
// device-resident. The uncached L1+L3 write hint reaches memory at once (44 ms, the launch latency).
// -DSTRATA_DOORBELL_ATOMIC_STORE restores the atomic store.
#ifndef STRATA_DOORBELL_ATOMIC_STORE
using doorbell_uncached_write = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::write_hint<sycl::ext::intel::experimental::cache_control<
        sycl::ext::intel::experimental::cache_mode::uncached,
        sycl::ext::oneapi::experimental::cache_level::L1, sycl::ext::oneapi::experimental::cache_level::L3>>));
inline void sys_store(volatile uint32_t* p, uint32_t v) {
    sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, doorbell_uncached_write> u(const_cast<uint32_t*>(p));
    u[0] = v;
    // Ordering, not visibility: the value is already in memory. The release fence keeps a later payload store (the
    // experts' ids the host reads after the ring) after the ring, the same contract the atomic store had.
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}
// The doorbell's PAYLOAD (activations, expert ids, routing weights) is the other half of the handshake: a plain store
// to host USM reaches the host only when the kernel ends on an Arc (xe), so the host would spin on the ring and then
// read a stale payload. Same uncached L1+L3 hint as sys_store, for float/int32.
template<class T> inline void sys_store_mapped(T* p, T v) {
    sycl::ext::oneapi::experimental::annotated_ptr<T, doorbell_uncached_write> u(p);
    u[0] = v;
}
#else
inline void sys_store(volatile uint32_t* p, uint32_t v) {
    sys_atomic_u32(*const_cast<uint32_t*>(p)).store(v);
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}
template<class T> inline void sys_store_mapped(T* p, T v) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system>(*p).store(v);
}
#endif
// plain = true: an ordinary store. On the i915 driver (Arc A750) a plain store to host USM reaches the host in time
// and the uncached hint only slows the publish (decode -11% on an A750, 4K prompt, 5 rounds); on xe it must be hinted.
// STRATA_DOORBELL_PLAIN=0/1 overrides. The launchers read this on the host and pass it to the kernel.
template<class T> inline void sys_store_mapped(bool plain, T* p, T v) {
    if (plain) *p = v; else sys_store_mapped(p, v);
}
inline bool doorbell_plain_payload() {
    static const bool v = [] {
        if (const char* e = std::getenv("STRATA_DOORBELL_PLAIN")) return e[0] == '1';
        return intel_gpu_driver() == "i915";
    }();
    return v;
}

// Every device spin is bounded. An unbounded spin that never sees its flag is not a hang of one process: the
// xe driver times the queue out, resets the GT node by node (a window graph has 2,366 of them), and the card
// stays wedged until a reboot - measured twice. With a bound the failure is a wrong window instead, which the
// verifier's checks catch. ~2 M host-memory reads is a few seconds at PCIe latency.
// The bound is per device, chosen at run time (spin_max below): 20,000 reads on a card under the xe driver (the B-series
// on Linux), whose device-plan windows only spin in a failure and where a longer spin is what makes the driver reset the
// GT; 2,000,000 (a few seconds) elsewhere. A B-series card with no sysfs driver (Windows / OpenCL) gets 20,000 too (#1473,
// measured +46-67% with --mtp), found from the device's architecture or name. On an Arc A-series (i915) the CPU computes the experts the card does not hold and the GPU waits for it at
// every layer: 20,000 reads is a few tens of milliseconds, the first request after a start (cold pages, slow CPU
// layers) is slower than that, the GPU gave up, went on with the experts' outputs missing, and the answer was token 0
// ("!!!!!") or the engine crashed in the CPU pool on the garbage routing it read next.
// STRATA_SPIN_MAX=<reads> overrides both; a build's -DSTRATA_SYCL_SPIN_MAX=<reads> (CMake) fixes one bound for every
// device. The device functions take the bound as an argument: the launchers pass spin_max(queue).
// Without a sysfs driver name (Windows, OpenCL) a B-series card is recognised from the device itself (intel_gpu_gen):
// its window waits are the xe kind (expected to expire, so 20,000 reads), EXCEPT in a run that drafts without the MTP
// layer (--spec N, no --mtp): the suffix drafter's own waits have to complete there, and 20,000 reads left it empty-handed
// (0 drafts, 21.8 vs 68.8 tok/s, #1473). generate calls spin_drafts_need_long_bound(true) for such a run, before any
// launch. A Linux xe card keeps 20,000 in every mode (unchanged).
inline std::atomic<bool>& spin_long_drafts_flag() { static std::atomic<bool> f{false}; return f; }
inline void spin_drafts_need_long_bound(bool on) { spin_long_drafts_flag() = on; }
inline uint32_t spin_max(const sycl::queue& q) {
    static std::mutex mu;
    static std::unordered_map<sycl::device, uint32_t> cache;
    const sycl::device d = q.get_device();
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(d);
    if (it != cache.end()) return it->second;
    uint32_t v;
    if (const char* e = std::getenv("STRATA_SPIN_MAX"); e && std::atol(e) > 0) v = (uint32_t) std::atol(e);
#ifdef STRATA_SYCL_SPIN_MAX
    else v = STRATA_SYCL_SPIN_MAX;
#else
    else if (intel_gpu_driver() == "xe") v = 20000u;
    else if (intel_gpu_driver().empty() && intel_gpu_gen(d) == IntelGpuGen::Battlemage)
        v = spin_long_drafts_flag().load() ? 2000000u : 20000u;
    else v = 2000000u;
#endif
    cache.emplace(d, v);
    return v;
}
}  // namespace strata
