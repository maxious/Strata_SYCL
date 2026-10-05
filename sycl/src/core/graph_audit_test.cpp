// src/core/graph_audit_test.cpp - the capture audit's own gate.
//
// Each assertion in strata/core/graph_audit.hpp must be SHOWN to fire.  A detector that quietly stops firing has
// to fail this test, not pass it by finding nothing to check, so the negative cases run in a forked child with
// stderr on a pipe: the assertion's whole contract is an abort() plus a named line.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/graph_audit.hpp"

#if defined(_WIN32)

int main() {
    std::fprintf(stderr, "graph_audit_test: needs POSIX fork; skipped\n");
    return 0;
}

#else

#include <cstdio>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace audit = strata::core::audit;

namespace {

bool child_stderr_has(const char *expect, void (*fn)()) {
    int fds[2];
    if (pipe(fds) != 0) return false;
    const pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], 2);
        close(fds[1]);
        fn();
        _exit(0);
    }
    close(fds[1]);
    std::string out;
    char buf[512];
    ssize_t n;
    while ((n = read(fds[0], buf, sizeof buf)) > 0) out.append(buf, (size_t) n);
    close(fds[0]);
    int st = 0;
    if (waitpid(pid, &st, 0) < 0) return false;
    return out.find(expect) != std::string::npos;
}

void child_null_graph() {
    audit::graph_capture_end("t.null", (const void *) 1, nullptr, 0, "graph_audit_test");
}

void child_overlapping_begin() {
    audit::graph_capture_begin("t.overlap.a", (const void *) 1);
    audit::graph_capture_begin("t.overlap.b", (const void *) 1);
}

void child_recapture() {
    setenv("STRATA_GRAPH_STRICT", "1", 1);
    audit::graph_capture_begin("t.recap", (const void *) 1);
    audit::graph_capture_end("t.recap", (const void *) 1, (const void *) 0x10, 0, "graph_audit_test");
    audit::graph_capture_begin("t.recap", (const void *) 1);
    audit::graph_capture_end("t.recap", (const void *) 1, (const void *) 0x11, 0, "graph_audit_test");
}

void child_node_shift() {
    setenv("STRATA_GRAPH_STRICT", "1", 1);
    audit::graph_capture_nodes("t.nodes", 10);
    audit::graph_capture_nodes("t.nodes", 11);
}

bool audit_accepts_a_real_capture() {
    sycl::queue q{sycl::property::queue::in_order{}};
    int *p = sycl::malloc_device<int>(1, q);
    if (p == nullptr) return false;
    audit::graph_capture_begin("t.real", &q);
    dpct::experimental::begin_recording(&q);
    q.submit([&](sycl::handler &h) {
        h.single_task([=]() { *p = 1; });
    });
    dpct::experimental::command_graph_ptr graph = nullptr;
    dpct::experimental::end_recording(&q, &graph);
    size_t nn = 0;
    if (graph) dpct::experimental::get_nodes(graph, nullptr, &nn);
    const bool ok = graph != nullptr && nn == 1;
    if (graph) {
        audit::graph_capture_end("t.real", &q, graph, 0, "graph_audit_test");
        audit::graph_capture_nodes("t.real", nn);
        delete (graph);
    } else {
        audit::graph_capture_abandon(&q);
    }
    sycl::free(p, q);
    return ok;
}

}  // namespace

int main() {
    int failed = 0;
    auto check = [&](const char *name, bool ok) {
        std::fprintf(stderr, "%s: %s\n", ok ? "PASS" : "FAIL", name);
        if (!ok) ++failed;
    };
    check("a real capture is accepted",
          audit_accepts_a_real_capture());
    check("a null graph with no error aborts",
          child_stderr_has("end_recording returned no error but no graph", child_null_graph));
    check("an overlapping begin aborts",
          child_stderr_has("began while another was still recording", child_overlapping_begin));
    check("a re-capture is fatal under STRATA_GRAPH_STRICT",
          child_stderr_has("RE-CAPTURE", child_recapture));
    check("a moved node count is fatal under STRATA_GRAPH_STRICT",
          child_stderr_has("NODE COUNT", child_node_shift));
    // The trap behind exp 19: `DPCT_CHECK_ERROR` evaluates its argument as a STATEMENT and returns
    // `dpct::success`, so it cannot be used to read a bool back.  The doorbell wait did exactly that - `q`
    // was always `success`, `q != 1` was always true - and abandoned a layer after 2 ms of no progress,
    // which stopped every decode round.  If this macro ever starts returning its expression's value, the
    // completion gate in verify.cpp has to be revisited.
    {
        bool value = false;
        const dpct::err0 e = DPCT_CHECK_ERROR(value);
        check("DPCT_CHECK_ERROR discards its expression's value", e == dpct::success && !value);
    }
    if (failed) {
        std::fprintf(stderr, "graph_audit_test: %d failed\n", failed);
        return 1;
    }
    std::fprintf(stderr, "graph_audit_test: all assertions fire\n");
    return 0;
}

#endif
