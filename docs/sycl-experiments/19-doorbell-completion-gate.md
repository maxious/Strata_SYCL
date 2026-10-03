# Experiment 19 - the doorbell completion gate: one discarded bool stopped every decode round

Date: 2026-10-04. Hardware: **one Intel Arc Pro B60 (Battlemage G21)**, oneAPI 2026.1. Model: **Coder IQ1_M**,
`pack-coder`, `--expert-profile data/expert-profile-coder.bin --expert-cache auto --stream-experts`, `--spec 4
--mtp ~/ComfyUI/koboldcpp/mtp/rt`, greedy, 6-token prompt `9707,11,1879,1130,374,279`.

Experiment 18 left the engine unable to finish a decode round: every arm died with `verify: layer N never rang`,
with N moving between runs, and the audit showed the capture layer was clean (23 exactly-once captures, 0 breaks).
This closes that.

## The bug

`verify.cpp`'s per-layer wait polls the host-mapped sequence number and, every ~2 ms, asked whether the queue had
emptied so it could tell "the device never rang" from "the device is still working":

```cpp
const dpct::err0 q = DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
if (q != 1 && *seq < want) { err = "... never rang (" + (q == 0 ? "graph finished" : ...) + ")"; return false; }
```

`DPCT_CHECK_ERROR` is not a value-returning wrapper. It is (from `sycl/include/dpct/dpct.hpp`):

```cpp
#define DPCT_CHECK_ERROR(expr) [&]() { try { expr; return dpct::success; } catch (...) { ...; return dpct::default_error; } }()
```

`expr;` is evaluated as a **statement and its value is discarded**. So `q` was `dpct::success` on every call, the
emptiness result of `ext_oneapi_empty()` was never read, and:

- `q != 1` was **always true**, so the guard was not a completion gate at all;
- `q == 0` always held, so the message always said `"graph finished"` - a lie;
- the host abandoned a layer roughly **2 ms** after it last saw the sequence advance, and the enclosing 20 s
  timeout - the actual backstop - never ran.

For a ~40 ms window graph over a dozen-plus layers, a >2 ms gap between rings is ordinary. That is why the failure
landed at layer 1, 10, 12, 14, 28 across runs: it was the first layer whose ring happened to take longer than 2 ms.

The `-48` node drift that exp 18 flagged as unexplained is unrelated, and is resolved below.

## The fix

Read the emptiness as its own statement, and keep the true completion semantics - only a genuinely finished graph
with a missing ring is an error; otherwise the wait continues to the 20 s bound:

```cpp
bool idle = false;
try { idle = cs_->ext_oneapi_empty(); }
catch (const std::exception &e) { err = "verify: layer " + ... + " never rang (" + e.what() + ")"; return false; }
if (idle && *seq < want) { err = "verify: layer " + ... + " never rang (the graph finished, host seq=... want=...)"; return false; }
```

`graph_audit_test` now also asserts the trap itself (`DPCT_CHECK_ERROR` returns `dpct::success` for a `false`
argument), so a future DPCT change that gives the macro value semantics fails a test instead of silently
re-breaking this gate.

## After the fix: the engine decodes

    decode                   16 tokens in 918.4 ms  ->  17.42 tok/s
    prefill                  5 tokens in 304.6 ms  ->  16.42 tok/s  (time to first token 481.3 ms)
    output  : 23075 303 44424 75021 980 57536 11 198 220 694 279 15167 4874 958 641 48

`rc=0`, and the text is not a fixed point (exp 18's `STRATA_VERIFY_NO_HOST=1` workaround produced `1879 13 1879 13`;
this is a different, correct path). The audit reports 23 keys, 23 captures, 0 re-captured, 0 breaks.

### Token parity across capture modes (32 tokens, same everything else)

| arm | output | decode |
|---|---|---|
| default (warm capture, replayed) | `23075 303 44424 ... 3118 18 17` | 20.24 tok/s |
| `STRATA_WARM_GRAPHS=0` (captured lazily) | **identical** to default | 20.70 tok/s |
| `STRATA_VERIFY_EAGER=1` (window launched directly) | **different from the first token** | 18.41 tok/s |

The two graph arms are byte-identical, which is what a healthy capture layer looks like: the same graph, captured
at a different time, replayed to the same result. The eager arm is **not** token-equivalent, and it does not look
like the near-tie flip this port has measured before (INTEL.md: "identical for 146 tokens, then a near-tie after a
comma goes the other way") - it diverges at token 1. The draft statistics differ with it (default: 11 of 66 drafts
accepted, 1.48 tokens/round; eager: 13 of 54, 1.68), so the difference enters through the speculative loop, not
through a missing node in a window graph. **Open, and flagged rather than claimed**: `STRATA_VERIFY_EAGER` is an
optimization/debug escape hatch, and it should either be token-equivalent to the graph path or say why it is not.

## The golden node counts

`sycl/src/core/graph_audit.hpp` gained `STRATA_GRAPH_GOLDEN=<file>` (compare) and `STRATA_GRAPH_GOLDEN_WRITE=<file>`
(write). The file is one `key nodes` line per capture; a count that differs prints `GOLDEN MISMATCH [key] golden=N
now=M` as it is captured, a golden key that is never captured prints `MISSING [key]`, a key absent from the golden
prints `NEW [key]`, and the exit summary counts all three. `STRATA_GRAPH_STRICT=1` makes a mismatch fatal.

Measured end to end on two configurations, each written then checked in a second run:

    gen   : wrote 23 golden counts to sycl/bench/graph-golden/coder-iq1_m.txt
    check : golden 23 keys: 0 changed, 0 never captured, 0 new
    bad   : GOLDEN MISMATCH [verify.window6] golden=9999 now=2577
            MISSING [verify.ghost] golden=4242, never captured
            golden 24 keys: 1 changed, 1 never captured, 0 new

(The first version of this crashed: it held the counts in a function-local static and printed from a `std::atexit`
handler, and the map was constructed *after* the handler was registered, so it was destroyed *before* the handler
ran. The state is now one object whose destructor prints the summary, which cannot get that order wrong.)

### The exp 18 drift, resolved

Windowing the same run with `STRATA_VERIFY_DEVICE_PLAN=1`:

| window | without device plan | with device plan | exp 05 (INTEL.md) |
|---|---|---|---|
| 1 token | 2367 | **2415** | 2415 |
| 2 tokens | 2409 | **2457** | 2457 |
| 4 tokens | 2493 | **2541** | 2541 |
| 6 tokens | 2577 | **2625** | 2625 |

Exact match. The `-48` was **one device-plan node per 48 layers**: exp 05 ran with `STRATA_VERIFY_DEVICE_PLAN=1`
(as INTEL.md's canonical recipe does) and the exp 18 arms did not. Not a regression, not a lost node - a
configuration difference, and now two goldens (`coder-iq1_m.txt`, `coder-iq1_m-device-plan.txt`) record both.

## Validity

- The fix is one statement: `DPCT_CHECK_ERROR` discarded the value, so the emptiness test never ran. The macro's
  own definition is quoted above; `graph_audit_test` asserts the behaviour.
- Control: pre-fix, seven configurations failed at layer 1/10/12/14/28; post-fix, all three A/B arms return `rc=0`
  and the two graph arms are byte-identical.
- `ctest`: **27/27 passed** (the capture audit's five assertions plus the new trap assertion included).
- The node counts are the engine's own `get_nodes` on the finalized graph, printed by the audit.

## Conclusion

The decode stall was not the graph and not host-mapped-write visibility. It was a completion gate that never read
its own condition, because `DPCT_CHECK_ERROR` throws away the value of the expression it is given - and the error
message it produced (`"graph finished"`) described the opposite of the state it was in. With the gate fixed the
engine decodes at 17-20 tok/s on this box, the capture layer still reports zero breaks, and the node counts are
now pinned by two goldens. Open: `STRATA_VERIFY_EAGER` is not token-equivalent to the graph path.

## Reproduce

```sh
source /opt/intel/oneapi/setvars.sh && cd sycl/build-b60 && ninja strata graph_audit_test
./graph_audit_test && ctest -R graph_audit_test

SHARD=~/ComfyUI/koboldcpp/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
S2=~/ComfyUI/koboldcpp/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
ARGS=(--pack ~/ComfyUI/koboldcpp/pack-coder --native "$SHARD" --ple-gguf "$S2"
      --expert-profile ../../data/expert-profile-coder.bin --expert-cache auto --stream-experts
      --prefill 128 --spec 4 --mtp ~/ComfyUI/koboldcpp/mtp/rt --max-context 4096
      --tokens 9707,11,1879,1130,374,279 --max-new 32)

./strata "${ARGS[@]}"                                                    # 20.2 tok/s, output A
STRATA_WARM_GRAPHS=0 ./strata "${ARGS[@]}"                               # identical output A
STRATA_VERIFY_EAGER=1 ./strata "${ARGS[@]}"                              # different output

STRATA_GRAPH_GOLDEN_WRITE=/tmp/g.txt ./strata "${ARGS[@]}"               # write the golden
STRATA_GRAPH_GOLDEN=/tmp/g.txt ./strata "${ARGS[@]}"                     # 0 changed, 0 never captured, 0 new
STRATA_VERIFY_DEVICE_PLAN=1 STRATA_GRAPH_GOLDEN=sycl/bench/graph-golden/coder-iq1_m-device-plan.txt \
    ./strata "${ARGS[@]}"                                                # the DEVICE_PLAN counts match exp 05
```
