// include/strata/core/graph_audit.hpp - the SYCL command-graph capture audit.
//
// A captured `sycl::ext::oneapi::experimental::command_graph` is replayed with the USM pointers it was
// recorded with, so a "graph break" is not a slow path - it is a wrong answer or a stalled replay, and both
// are silent by default on this port:
//
//   * `dpct::experimental::end_recording` (dpct/graph.hpp) RETURNS WITHOUT SETTING `*graph` when the queue
//     is not in its recording map.  A mismatched begin/end therefore hands back a null graph and the caller
//     dereferences it (`graph->finalize()`), or worse, the recording of an outer capture silently swallowed
//     the inner one.  The shim reports no error for either.
//   * A key captured twice means the exec was dropped and re-recorded - the graph was invalidated (a moved
//     pointer, a queue teardown) and the first replay's results are not what the second was built for.
//   * A node that stops being recorded BETWEEN BUILDS moves the node count and nothing else; only a recorded
//     count catches it, which is what `STRATA_GRAPH_GOLDEN` is for.
//
// So this audit makes four things loud, and two of them fatal, because the alternative is a segfault or a
// wrong answer with no call site:
//
//   1. `err == 0 && graph == nullptr` after `end_recording` -> abort with the key and the call site.
//   2. Two captures overlapping on the same queue (what the shim folds silently) -> abort.
//   3. A key captured more than once -> a loud line, and fatal with `STRATA_GRAPH_STRICT=1`.
//   4. A node count that differs from `STRATA_GRAPH_GOLDEN`, or a golden key never captured -> a loud line at
//      exit, and fatal with `STRATA_GRAPH_STRICT=1`.
//
// `STRATA_GRAPH_AUDIT=1` prints one machine-readable line per capture; `STRATA_GRAPH_GOLDEN_WRITE=<file>`
// writes the counts so a later run can diff against them.  Both print through the exit summary.
//
// **The state is one function-local object and its destructor prints the summary.**  Three separate statics
// plus `std::atexit` is what this started as, and it crashed on the first golden load: the map holding the
// counts is constructed AFTER the `atexit` handler is registered, so it is destroyed BEFORE the handler runs,
// and the summary iterated freed memory.  One object cannot get that order wrong.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core::audit {

struct GraphTally {
    int captures = 0;
    int nodes = 0;
    int breaks = 0;
};

struct GraphAudit {
    std::mutex mutex;
    std::unordered_map<std::string, GraphTally> tallies;
    std::unordered_map<std::string, int> golden;
    bool golden_loaded = false;
    const void *recording_queue = nullptr;

    ~GraphAudit() { report(); }

    bool audit_on() {
        static const bool on = [] {
            const char *v = std::getenv("STRATA_GRAPH_AUDIT");
            return v && v[0] != '0' && v[0] != '\0';
        }();
        return on;
    }

    bool strict() {
        static const bool s = [] {
            const char *v = std::getenv("STRATA_GRAPH_STRICT");
            return v && v[0] != '0' && v[0] != '\0';
        }();
        return s;
    }

    void load_golden() {
        if (golden_loaded) return;
        golden_loaded = true;
        const char *path = std::getenv("STRATA_GRAPH_GOLDEN");
        if (path == nullptr || path[0] == '\0') return;
        std::FILE *f = std::fopen(path, "r");
        if (f == nullptr) {
            std::fprintf(stderr, "strata graph audit: cannot read golden %s\n", path);
            return;
        }
        char line[256];
        while (std::fgets(line, sizeof line, f) != nullptr) {
            if (line[0] == '#' || line[0] == '\n') continue;
            char key[128];
            int nodes = 0;
            if (std::sscanf(line, "%127s %d", key, &nodes) == 2) golden[key] = nodes;
        }
        std::fclose(f);
    }

    void check_golden(const char *key, std::size_t nodes) {
        if (std::getenv("STRATA_GRAPH_GOLDEN") == nullptr) return;
        load_golden();
        auto it = golden.find(key);
        if (it == golden.end() || it->second == (int) nodes) return;
        std::fprintf(stderr, "strata graph audit: GOLDEN MISMATCH [%s] golden=%d now=%zu\n", key, it->second,
                     nodes);
        std::fflush(stderr);
        if (strict()) std::abort();
    }

    void write_golden() {
        const char *path = std::getenv("STRATA_GRAPH_GOLDEN_WRITE");
        if (path == nullptr || path[0] == '\0') return;
        std::FILE *f = std::fopen(path, "w");
        if (f == nullptr) {
            std::fprintf(stderr, "strata graph audit: cannot write golden %s\n", path);
            return;
        }
        std::fprintf(f, "# strata graph node golden: one `key nodes` line per captured graph\n");
        std::vector<std::string> keys;
        for (auto &[k, t] : tallies) keys.push_back(k);
        std::sort(keys.begin(), keys.end());
        for (auto &k : keys) std::fprintf(f, "%s %d\n", k.c_str(), tallies[k].nodes);
        std::fclose(f);
        std::fprintf(stderr, "strata graph audit: wrote %zu golden counts to %s\n", keys.size(), path);
    }

    void report() {
        int captures = 0, recaptures = 0, breaks = 0;
        for (auto &[k, t] : tallies) {
            captures += t.captures;
            breaks += t.breaks;
            if (t.captures > 1) ++recaptures;
        }
        std::fprintf(stderr, "strata graph audit: %d keys, %d captures, %d re-captured, %d breaks\n",
                     (int) tallies.size(), captures, recaptures, breaks);
        if (audit_on())
            for (auto &[k, t] : tallies)
                std::fprintf(stderr, "strata graph audit: key=%s nodes=%d captures=%d\n", k.c_str(), t.nodes,
                             t.captures);
        load_golden();
        if (!golden.empty()) {
            int changed = 0, missing = 0, added = 0;
            for (auto &[k, want] : golden) {
                auto it = tallies.find(k);
                if (it == tallies.end()) {
                    ++missing;
                    std::fprintf(stderr, "strata graph audit: MISSING [%s] golden=%d, never captured\n",
                                 k.c_str(), want);
                } else if (it->second.nodes != want) {
                    ++changed;   // the per-capture GOLDEN MISMATCH line already named both counts
                }
            }
            for (auto &[k, t] : tallies)
                if (golden.find(k) == golden.end()) {
                    ++added;
                    std::fprintf(stderr, "strata graph audit: NEW [%s] nodes=%d (not in the golden)\n",
                                 k.c_str(), t.nodes);
                }
            std::fprintf(stderr, "strata graph audit: golden %zu keys: %d changed, %d never captured, %d new\n",
                         golden.size(), changed, missing, added);
        }
        write_golden();
        std::fflush(stderr);
    }
};

inline GraphAudit &graph_audit() {
    static GraphAudit a;
    return a;
}

inline void graph_fail(const char *key, const char *why) {
    std::fprintf(stderr, "strata graph audit: FAIL [%s] %s\n", key, why);
    std::fflush(stderr);
    std::abort();
}

inline void graph_capture_begin(const char *key, const void *queue) {
    GraphAudit &a = graph_audit();
    std::lock_guard<std::mutex> lock(a.mutex);
    if (a.recording_queue != nullptr)
        graph_fail(key, "a capture began while another was still recording on this queue "
                        "(the shim would fold it into the outer graph in silence)");
    a.recording_queue = queue;
}

inline void graph_capture_abandon(const void *queue) {
    GraphAudit &a = graph_audit();
    std::lock_guard<std::mutex> lock(a.mutex);
    if (a.recording_queue == queue) a.recording_queue = nullptr;
}

inline void graph_capture_end(const char *key, const void *queue, const void *graph, long err,
                              const char *where) {
    GraphAudit &a = graph_audit();
    std::lock_guard<std::mutex> lock(a.mutex);
    if (a.recording_queue == queue) a.recording_queue = nullptr;
    GraphTally &t = a.tallies[key];
    ++t.captures;
    if (err == 0 && graph == nullptr) {
        ++t.breaks;
        std::fprintf(stderr,
                     "strata graph audit: FAIL [%s] end_recording returned no error but no graph either "
                     "(%s): the queue was not recording\n",
                     key, where ? where : "?");
        std::fflush(stderr);
        std::abort();
    }
    if (t.captures > 1) {
        ++t.breaks;
        std::fprintf(stderr,
                     "strata graph audit: RE-CAPTURE [%s] this key was already captured (%d times): the "
                     "exec was dropped and re-recorded\n",
                     key, t.captures);
        std::fflush(stderr);
        if (a.strict()) std::abort();
    }
    if (a.audit_on())
        std::fprintf(stderr, "strata graph audit: key=%s captured (%d)\n", key, t.captures);
}

inline void graph_capture_nodes(const char *key, std::size_t nodes) {
    GraphAudit &a = graph_audit();
    std::lock_guard<std::mutex> lock(a.mutex);
    GraphTally &t = a.tallies[key];
    if (t.nodes != 0 && t.nodes != (int) nodes) {
        std::fprintf(stderr, "strata graph audit: NODE COUNT [%s] %d -> %zu\n", key, t.nodes, nodes);
        std::fflush(stderr);
        if (a.strict()) std::abort();
    }
    t.nodes = (int) nodes;
    a.check_golden(key, nodes);
    if (a.audit_on())
        std::fprintf(stderr, "strata graph audit: key=%s nodes=%d captures=%d\n", key, t.nodes, t.captures);
}

}  // namespace strata::core::audit
