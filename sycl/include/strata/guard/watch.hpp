#pragma once
#include <cstdint>
#include <cstdio>

#define STRATA_WATCH_NB 1
#if STRATA_WATCH_NB
#define WATCH_ENABLE 1
#else
#define WATCH_ENABLE 0
#endif

namespace strata::guard {

// record layout: [0] = (n << 32) | bad_count; [1] = first failing pointer or 0; [2] = cache_base;
// [3] = blob; [4] = n_slots; then the group pointer + entry start pairs
constexpr int kWatchWords = 64;

struct Watch {
    unsigned long long* mem_ = nullptr;
    int slots_ = 0;

    void arm(unsigned long long* mem, int slots) { mem_ = mem; slots_ = slots; }

    // the window prologue's mapped copies, recorded by their launchers (the args are what the kernel receives);
    // the last six such records are the faulting copy if the prologue is where the wild pointer goes out.
    // bank: 0 = the primary card's records, 1 = the split's (+64), so both cards' records survive each other
    void prologue(int id, unsigned long long dst, unsigned long long src, unsigned long long n,
                  int bank = 0) {
        if (mem_ == nullptr) return;
        // bank 0 = the primary card, 6 slots at 98..103; bank -1 = the split card, 4 slots at 94..97;
        // bank 1 = unused here (the plan watch's split bank).  Each bank keeps its own last record.
        volatile unsigned long long* r =
            mem_ + (size_t) (bank == -1 ? 94 + (id % 4) : 98 + (id % 6)) * kWatchWords;
        r[2] = dst;
        r[3] = src;
        r[4] = n;
        const bool bad = (dst >> 47) == 0 || (src >> 47) != 0;
        if (bad) { r[1] += 1; r[58] = dst; r[59] = src; }
    }

    __attribute__((noinline)) void dump(std::FILE* f) const {
        if (f == nullptr) return;
        std::fprintf(f, "strata guard watch: %d slots\n", slots_);
        for (int s = 0; mem_ != nullptr && s < slots_; ++s) {
            const unsigned long long* r = mem_ + (size_t) s * kWatchWords;
            const unsigned long long n = r[0] & 0xffffffffull, bad = r[0] >> 32;
            if (n == 0 && bad == 0 && r[2] == 0) continue;
            std::fprintf(f, "  slot %3d: %llu groups, %llu bad; cache_base=%#llx blob=%#llx n_slots=%#llx",
                         s, n, bad, r[2], r[3], r[4]);
            if (bad != 0) std::fprintf(f, "\n            BAD ptr=%#llx slot=%lld", r[58], (long long) r[59]);
            for (unsigned long long i = 0; i < n && 5 + i * 2 + 1 < (unsigned long long) kWatchWords; ++i)
                std::fprintf(f, "\n            grp %llu: %#llx (start %llu)", i, r[5 + i * 2], r[6 + i * 2]);
            std::fprintf(f, "\n");
        }
        std::fflush(f);
    }
};

Watch* watch() noexcept;
void install(Watch* w) noexcept;

}
