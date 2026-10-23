// src/core/gguf_expert_source.cpp - see the header. Plain C++, no device code.
#include "strata/core/gguf_expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <utility>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <drm/drm.h>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <thread>

namespace strata::core {

namespace {
constexpr size_t kRing = 512;   // blobs alive at once: the prompt path holds a layer's worth of streamed experts

// xe P2P carve-out (valarauca PR 1, kernel 7.3.0-rc7-xe-perf-pr1-p2p): DRM_IOCTL_XE_P2P_ALLOC hands out a chunk of
// VRAM the CPU maps and the GPU dereferences at local-VRAM speed - measured on an Arc Pro B60: 364-408 GB/s
// kernel reads against 4-9.4 GB/s for USM host, byte-verified against a buffered read of the same range
// (~/src/pr1-p2p-test). STRATA_MIRROR_CARVEOUT=1 puts the expert mirror there instead of pinned host memory, so
// spilled experts are read on the card instead of over PCIe. The uapi is the PR's own; on a kernel without it the
// ioctl fails and every chunk falls back to USM host, with one line said.
#ifndef PR1_DRM_XE_P2P_ALLOC
#define PR1_DRM_XE_P2P_ALLOC 0x10
struct pr1_drm_xe_p2p_alloc {
    uint64_t extensions;
    uint64_t size;
    uint32_t flags;
    uint32_t pad;
    uint64_t reserved[2];
};
#define PR1_DRM_IOCTL_XE_P2P_ALLOC \
    DRM_IOW(DRM_COMMAND_BASE + PR1_DRM_XE_P2P_ALLOC, struct pr1_drm_xe_p2p_alloc)
#endif

std::string read_small_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return {};
    char buf[128] = {0};
    const ssize_t n = ::read(fd, buf, sizeof buf - 1);
    ::close(fd);
    if (n <= 0) return {};
    std::string s(buf, (size_t) n);
    const size_t e = s.find_last_not_of("\n \t");
    return e == std::string::npos ? std::string() : s.substr(0, e + 1);
}

std::string pci_of_queue() {
    try {
        sycl::queue& q = dpct::get_in_order_queue();
        if (!q.get_device().has(sycl::aspect::ext_intel_pci_address)) return {};
        std::string s = q.get_device().get_info<sycl::ext::intel::info::device::pci_address>();
        const size_t b = s.find_first_not_of("[] \t");
        const size_t e = s.find_last_not_of("[] \t");
        return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
    } catch (...) {
        return {};
    }
}

struct Carveout {
    int fd = -1;             // the render node of the device the queue runs on
    uint64_t left = 0;       // p2pmem/available, minus what this mirror took
    bool dead = false;       // no carve-out or a failed alloc: stop trying
    std::string pci;
};

Carveout carveout_setup() {
    Carveout c;
    const char* v = std::getenv("STRATA_MIRROR_CARVEOUT");
    if (!v || !*v || !std::strcmp(v, "0")) return c;
    // The carve-out is VRAM: fast for the GPU's direct mirror reads (the STRATA_VERIFY_NO_HOST device plan,
    // measured 364-408 GB/s) and poison for a CPU reader - and under STRATA_VERIFY_NO_HOST=0 the expert pool
    // serves misses by reading the mirror pointer, which then goes over the PCIe BAR to WC VRAM (measured on
    // 2x Arc Pro B60, Gyro-M dual 32K: decode 36-41 tok/s with the mirror in RAM, 1.2-5.7 tok/s in the
    // carve-out).  So the carve-out mirror only runs with the no-host plan.
    const char* nh = std::getenv("STRATA_VERIFY_NO_HOST");
    if (nh && *nh && std::strcmp(nh, "0") == 0) {
        std::fprintf(stderr, "strata generate: STRATA_MIRROR_CARVEOUT is off: STRATA_VERIFY_NO_HOST=0 has the CPU "
                             "pool read the mirror, and the carve-out is VRAM; the mirror uses pinned host memory\n");
        c.dead = true;
        return c;
    }
    c.pci = pci_of_queue();
    if (c.pci.empty()) {
        std::fprintf(stderr, "strata generate: STRATA_MIRROR_CARVEOUT: the queue's PCI address is unknown; "
                             "the mirror uses pinned host memory\n");
        c.dead = true;
        return c;
    }
    const std::string avail = read_small_file("/sys/bus/pci/devices/" + c.pci + "/p2pmem/available");
    if (avail.empty()) {
        std::fprintf(stderr, "strata generate: STRATA_MIRROR_CARVEOUT: no xe P2P carve-out on %s (the kernel's "
                             "p2p_carveout_mb, valarauca PR 1); the mirror uses pinned host memory\n", c.pci.c_str());
        c.dead = true;
        return c;
    }
    c.left = (uint64_t) std::strtoull(avail.c_str(), nullptr, 10);
    if (DIR* d = ::opendir("/sys/class/drm")) {
        while (struct dirent* de = ::readdir(d)) {
            if (std::strncmp(de->d_name, "renderD", 7) != 0) continue;
            char link[256];
            const std::string node = std::string("/sys/class/drm/") + de->d_name + "/device";
            const ssize_t n = ::readlink(node.c_str(), link, sizeof link - 1);
            if (n <= 0) continue;
            link[n] = 0;
            const char* slash = std::strrchr(link, '/');
            if (slash == nullptr || c.pci != slash + 1) continue;
            const std::string dev = std::string("/dev/dri/") + de->d_name;
            c.fd = ::open(dev.c_str(), O_RDWR | O_CLOEXEC);
            if (c.fd >= 0)
                std::fprintf(stderr, "strata generate: xe P2P carve-out on %s (%s): %llu MiB available\n",
                             c.pci.c_str(), dev.c_str(), (unsigned long long) (c.left >> 20));
            break;
        }
        ::closedir(d);
    }
    if (c.fd < 0 || c.left < (2ull << 20)) {
        if (c.fd >= 0) ::close(c.fd);
        if (!c.dead)
            std::fprintf(stderr, "strata generate: STRATA_MIRROR_CARVEOUT: no usable carve-out on %s; "
                                 "the mirror uses pinned host memory\n", c.pci.c_str());
        c.fd = -1;
        c.left = 0;
        c.dead = true;
    }
    return c;
}

// One page-multiple chunk of `bytes` out of the carve-out, mmap'd; null with errno set on any failure.
uint8_t* carveout_chunk(const Carveout& c, uint64_t bytes) {
    struct pr1_drm_xe_p2p_alloc a = {};
    a.size = bytes;
    const int fd = ::ioctl(c.fd, PR1_DRM_IOCTL_XE_P2P_ALLOC, &a);
    if (fd < 0) return nullptr;
    void* m = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    const int saved = errno;
    ::close(fd);                                        // the mapping holds its own reference
    if (m == MAP_FAILED) { errno = saved; return nullptr; }
    return (uint8_t*) m;
}
}

GgufExpertSource::~GgufExpertSource() { close(); }

void GgufExpertSource::close() {
    for (size_t i = 0; i < mirror_chunks_.size(); ++i)
        if (mirror_chunks_[i]) {
            if (i < mirror_chunk_p2p_.size() && mirror_chunk_p2p_[i])
                ::munmap(mirror_chunks_[i], (size_t) mirror_chunk_bytes_[i]);
            else sycl::free(mirror_chunks_[i], dpct::get_in_order_queue());
        }
    mirror_chunks_.clear();
    mirror_chunk_p2p_.clear();
    mirror_chunk_bytes_.clear();
    mirror_bytes_ = 0; mirror_p2p_bytes_ = 0; mirror_ptr_.clear(); layer_first_.clear();
    for (int fd : fds_) if (fd >= 0) ::close(fd);
    fds_.clear(); names_.clear(); layer_fd_.clear(); ring_.clear(); ring_key_.clear(); where_.clear();
    ring_next_ = 0;
}

bool GgufExpertSource::open(const std::string& shard1, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || lay.gguf_off.size() < (size_t) (3 * n_layers)) {
        err = "--stream-experts needs a native (IQ) pack whose native_experts.txt carries the GGUF tensor offsets";
        return false;
    }
    for (int64_t l = 0; l < n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            if (lay.gguf_off[(size_t) (3 * l + r)] == 0) {
                err = "--stream-experts: layer " + std::to_string(l) + " has no GGUF offset for its experts";
                return false;
            }
    shard_ = shard1;
    const size_t cut = shard1.find_last_of("/\\");
    dir_ = cut == std::string::npos ? std::string() : shard1.substr(0, cut + 1);
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    layer_fd_.assign((size_t) (3 * n_layers), -1);
    for (int64_t l = 0; l < n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            if (fd_of(l, r, err) < 0) { close(); return false; }
    ring_.resize(kRing);
    for (auto& b : ring_) b.resize((size_t) lay.max_blob);
    ring_key_.assign(kRing, -1);
    return true;
}

// The file of role `role` (0 gate, 1 up, 2 down) of `layer`. ExpertLayout::gguf_file is per layer AND role
// (`3 * layer + role`) since upstream 0.1.31; indexing it by layer alone read a shard-2 layer from shard 1 (Swift 1.5,
// whose layers 13-47 are in shard 2: garbage IQ1_M scales, NaN logits - 2026-10-01).
int GgufExpertSource::fd_of(int64_t layer, int role, std::string& err) {
    const size_t i = (size_t) (3 * layer + role);
    if (layer_fd_[i] >= 0) return fds_[(size_t) layer_fd_[i]];
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::string name = shard_;
    if (lay.gguf_file.size() > i && !lay.gguf_file[i].empty()) name = dir_ + lay.gguf_file[i];
    for (size_t k = 0; k < names_.size(); ++k)
        if (names_[k] == name) { layer_fd_[i] = (int) k; return fds_[k]; }
    const int fd = ::open(name.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { err = "--stream-experts: cannot open " + name; return -1; }
    names_.push_back(name);
    fds_.push_back(fd);
    layer_fd_[i] = (int) fds_.size() - 1;
    return fd;
}

const uint8_t* GgufExpertSource::blob(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || ring_.empty()) return nullptr;
    if (!mirror_ptr_.empty()) {
        const uint8_t* p = mirror_ptr_[(size_t) (layer * n_expert_ + expert)];
        if (p != nullptr) return p;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    // the arena loader's gather, for one expert: [gate | up | down] from the three tensors
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    const int64_t key = ((int64_t) layer << 20) | expert;
    size_t slot;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = where_.find(key);
        if (it != where_.end()) { ++reads_; return ring_[it->second].data(); }   // still resident
        slot = ring_next_;
        ring_next_ = (ring_next_ + 1) % ring_.size();
        if (ring_key_[slot] >= 0) where_.erase(ring_key_[slot]);
        ring_key_[slot] = key;
        where_[key] = slot;
    }
    std::vector<uint8_t>& buf = ring_[slot];
    for (int r = 0; r < 3; ++r) {
        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];
        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;
        uint64_t done = 0;
        while (done < per[r]) {
            const ssize_t n = ::pread(fd, buf.data() + at[r] + done, (size_t) (per[r] - done), (off_t) (src + done));
            if (n <= 0) { std::lock_guard<std::mutex> lk(mu_); where_.erase(key); ring_key_[slot] = -1; return nullptr; }
            done += (uint64_t) n;
        }
    }
    ++reads_;
    return buf.data();
}

int64_t GgufExpertSource::mirror(const std::vector<std::pair<int64_t, int64_t>>& pairs, uint64_t cap, int threads,
                                 std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<std::pair<int64_t, int64_t>> take;
    uint64_t total = 0;
    for (const auto& [l, e] : pairs) {
        if (l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) continue;
        const uint64_t b = (lay.bytes[(size_t) l] + 255) / 256 * 256;
        if (total + b > cap) break;
        take.push_back({l, e});
        total += b;
    }
    if (take.empty()) return 0;
    // Alchemist (Arc A-series) refuses a single USM host allocation past ~3 GiB (docs/INTEL.md, A750), so the mirror
    // is built from chunks under that. Battlemage and the CUDA/HIP builds never hit the limit, but chunking is neutral
    // for them. One chunk per (up to) 2 GiB of experts.
    constexpr uint64_t kChunk = 2ull << 30;
    std::vector<uint8_t*> chunks;
    std::vector<uint8_t> chunks_p2p;                        // 1: an xe carve-out chunk (munmap), 0: USM host (sycl::free)
    std::vector<uint64_t> chunk_bytes;
    // Free every chunk unless the mirror is handed over to the object: covers an allocation failure, a failed read and
    // any exception (a container allocation, a std::thread that cannot be created) on the way out.
    struct ChunkCleanup {
        std::vector<uint8_t*>& v;
        std::vector<uint8_t>& p2p;
        std::vector<uint64_t>& bytes;
        bool armed = true;
        ~ChunkCleanup() {
            if (!armed) return;
            for (size_t i = 0; i < v.size(); ++i)
                if (v[i]) {
                    if (i < p2p.size() && p2p[i]) ::munmap(v[i], (size_t) bytes[i]);
                    else sycl::free(v[i], dpct::get_in_order_queue());
                }
        }
    } cleanup{chunks, chunks_p2p, chunk_bytes};
    Carveout carve = carveout_setup();
    std::vector<uint8_t*> ptrs(take.size(), nullptr);
    uint64_t placed = 0;
    uint8_t* cur = nullptr;
    uint64_t cur_off = 0, cur_size = 0;
    for (size_t j = 0; j < take.size(); ++j) {
        const uint64_t b = (lay.bytes[(size_t) take[j].first] + 255) / 256 * 256;
        if (cur == nullptr || cur_off + b > cur_size) {
            // never smaller than one blob, so a single blob past kChunk still fits its chunk
            const uint64_t want = std::max(b, std::min<uint64_t>(kChunk, total - placed));
            uint8_t* nc = nullptr;
            uint64_t got = 0;
            bool is_p2p = false;
            if (!carve.dead && carve.left >= std::max<uint64_t>(b, 2ull << 20)) {
                got = std::max(b, std::min<uint64_t>(want, carve.left & ~4095ull));
                nc = carveout_chunk(carve, got);
                if (nc != nullptr) {
                    is_p2p = true;
                    carve.left -= got;
                } else {
                    carve.dead = true;
                    std::fprintf(stderr, "strata generate: xe P2P carve-out chunk of %llu MiB failed (%s): the rest "
                                         "of the mirror uses pinned host memory\n",
                                 (unsigned long long) (got >> 20), std::strerror(errno));
                }
            }
            if (nc == nullptr) {
                got = want;
                try {
                    nc = (uint8_t*) sycl::malloc_host((size_t) got, dpct::get_in_order_queue());
                } catch (const sycl::exception& ex) {
                    err = std::string("mirror: ") + ex.what();
                }
                if (nc == nullptr) {
                    if (err.empty())
                        err = "mirror: no pinned host memory for " + std::to_string(got >> 20) + " MiB";
                    return -1;
                }
            }
            chunks.push_back(nc);
            chunks_p2p.push_back(is_p2p ? 1 : 0);
            chunk_bytes.push_back(got);
            cur = nc; cur_off = 0; cur_size = got;
        }
        ptrs[j] = cur + cur_off;
        cur_off += b;
        placed += b;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::vector<std::thread> ts;
    try {
        for (int i = 0; i < std::max(1, threads); ++i)
            ts.emplace_back([&] {
                for (size_t j; (j = next.fetch_add(1)) < take.size() && !bad.load();) {
                    const auto [l, e] = take[j];
                    if (!read_into(l, e, ptrs[j], (size_t) lay.bytes[(size_t) l])) bad.store(true);
                }
            });
    } catch (...) {
        for (auto& th : ts) if (th.joinable()) th.join();   // join before the cleanup frees, and before unwinding
        throw;
    }
    for (auto& th : ts) th.join();
    if (bad.load()) {
        err = "mirror: reading an expert from the GGUF failed";
        return -1;
    }
    for (size_t i = 0; i < mirror_chunks_.size(); ++i)
        if (mirror_chunks_[i]) {
            if (i < mirror_chunk_p2p_.size() && mirror_chunk_p2p_[i])
                ::munmap(mirror_chunks_[i], (size_t) mirror_chunk_bytes_[i]);
            else sycl::free(mirror_chunks_[i], dpct::get_in_order_queue());
        }
    mirror_chunks_ = std::move(chunks);
    mirror_chunk_p2p_ = std::move(chunks_p2p);
    mirror_chunk_bytes_ = std::move(chunk_bytes);
    mirror_p2p_bytes_ = 0;
    for (size_t i = 0; i < mirror_chunk_p2p_.size(); ++i)
        if (mirror_chunk_p2p_[i]) mirror_p2p_bytes_ += mirror_chunk_bytes_[i];
    cleanup.armed = false;                                  // the chunks now belong to the object
    mirror_bytes_ = total;
    mirror_ptr_.assign((size_t) (n_layers_ * n_expert_), nullptr);
    layer_first_.assign((size_t) n_layers_, nullptr);
    for (size_t j = 0; j < take.size(); ++j) {
        const auto [l, e] = take[j];
        mirror_ptr_[(size_t) (l * n_expert_ + e)] = ptrs[j];
        if (layer_first_[(size_t) l] == nullptr) layer_first_[(size_t) l] = ptrs[j];
    }
    return (int64_t) take.size();
}

bool GgufExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (mirror_ptr_.empty() || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    return mirror_ptr_[(size_t) (layer * n_expert_ + expert)] != nullptr;
}

const uint8_t* GgufExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (mirror_ptr_.empty() || layer < 0 || layer >= n_layers_) return nullptr;
    if (expert >= 0 && expert < n_expert_) {
        const uint8_t* p = mirror_ptr_[(size_t) (layer * n_expert_ + expert)];
        if (p != nullptr) return p;
    }
    return layer_first_[(size_t) layer];
}

bool GgufExpertSource::read_into(int64_t layer, int64_t expert, uint8_t* dst, size_t bytes) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || dst == nullptr) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    if (bytes < blob) return false;
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    for (int r = 0; r < 3; ++r) {
        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];
        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;
        uint64_t done = 0;
        while (done < per[r]) {
            const ssize_t n = ::pread(fd, dst + at[r] + done, (size_t) (per[r] - done), (off_t) (src + done));
            if (n <= 0) return false;
            done += (uint64_t) n;
        }
    }
    return true;
}

}  // namespace strata::core
