# Experiment 39 - the MMVQ cache policy cannot be set in source on this toolchain: the extension does not link

**Question** (README item 7, archive's untested-not-refuted cache policy). exp 22 swept
`-cl-load/store-cache-default` 13 ways and got a null - but the option **never reached the compiler** on a SYCL/SPIR-V
input, so nothing was measured. The archive's rule for a retry: set it in the source, and confirm it applied by
re-dumping the ISA. `sycl_ext_intel_cache_controls` is the in-source route (a `read_hint` property on the launch).

**Verdict. CLOSED as a toolchain limitation, measured three ways.** On icpx **2026.1**, every spelling of the property
compiles and then fails at the **SPIR-V device link** with the same error:

```
InvalidLlvmModule: Invalid LLVM module: CacheControlLoadINTEL requires exactly 2 extra operands
icpx: error: llvm-spirv command failed with exit code 11
```

So there is nothing to sweep on this compiler: the MMVQ cache policy is **not expressible**, which makes exp 22's null
a *permanent* result rather than an untested one. The knob is kept in the source (`-DSTRATA_MMVQ_CACHE_MODE`, default
0 = no hints, the shipping path) so a fixed compiler can measure it without rediscovering the plumbing; modes 1-3 do not
build here.

## What was tried

| route | spelling | result |
|---|---|---|
| launch property, spec's two-entry form | `read_hint<cache_control<streaming, L1>, cache_control<cached, L2, L3>>` | compiles, **device link fails** |
| launch property, one entry | `read_hint<cache_control<streaming, L1>>` | compiles, **device link fails** |
| `annotated_ptr` on the pointer argument | `annotated_ptr<const float, properties<...read_hint...>>` | compiles, **device link fails** |
| one entry, standalone `nd_range` kernel (minimal repro) | same | compiles, **device link fails** |

The error is emitted **once per kernel**, so it is the property lowering and not one bad launch. The minimal repro is
~12 lines and fails the same way, which is what makes this a compiler verdict rather than a port bug.

Two spellings that are *not* the property and do work, recorded so nobody re-derives them: `properties` takes a
**type list** (`properties<detail::properties_type_list<property_value<...>>>`; `properties_t` does not exist in this
release and `properties{...}` with a bare `read_hint` is the form the spec shows but is not what the header wants), and
the property-carrying `parallel_for` overload is deprecated in favour of a launch-config object or a kernel functor's
`get(properties_tag)` - the deprecation is only a warning and does not affect the link failure.

## Consequence for the archive's other cache items

- **exp 22's sweep stays refuted**, now for a stronger reason: not "the option never arrived" but "there is no in-source
  spelling that links either".
- The GRF/register knobs (`sycl_ext_intel_maximum_registers`, `sycl_ext_intel_grf_size`) and `private_alloca` are
  unaffected - they are ordinary compile-time attributes/qualifiers, not intrinsics, and exp 37's `TK_PER_MAX` build
  (a template argument) links fine. Item 2's shipping change is the live demonstration that register-width control works
  on this compiler even though cache control does not.

## Reproduce

```sh
# the minimal repro that shows it is the compiler, not the port
cat > /tmp/ct.cpp <<'EOF'
#include <sycl/sycl.hpp>
namespace ce = sycl::ext::intel::experimental; namespace cl = sycl::ext::oneapi::experimental;
namespace px = sycl::ext::oneapi::experimental;
using p1 = px::properties<px::detail::properties_type_list<px::property_value<
    ce::read_hint_key, ce::cache_control<ce::cache_mode::streaming, cl::cache_level::L1>>>>;
void k(const float* a, float* b, sycl::queue& q) {
  q.parallel_for<class kk>(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)), p1{},
    [=](sycl::nd_item<1>) { b[0] = a[0]; });
}
EOF
icpx -fsycl /tmp/ct.cpp -o /tmp/ct    # -> InvalidLlvmModule: CacheControlLoadINTEL requires exactly 2 extra operands

# the port's own knob (mode 0 ships; 1-3 do not link here)
cmake -S sycl -B /tmp/b-cache1 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_CXX_FLAGS=-DSTRATA_MMVQ_CACHE_MODE=1 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_SYCL_PARITY=ON
ninja -C /tmp/b-cache1 mmvq_bench       # -> device link fails, once per kernel
```

## Traps this run adds

- **A compile that succeeds is not a link that succeeds.** Every cache-hint spelling compiled cleanly; the failure only
  appears at the SPIR-V device link, so a build that is "only linking" can still be the thing that is broken. Check the
  link step before concluding an extension is unsupported.
- **The property is a type parameter of the launch**, which is why it is a build-time switch (`-D...`) rather than an
  env var: the property list has to be part of the `parallel_for`'s type.
- The `properties` template takes `detail::properties_type_list<...>`, not a pack of values, and `properties_t` is not
  exported in 2026.1 - both are easy to get wrong and produce a wall of template errors that hides the real issue.