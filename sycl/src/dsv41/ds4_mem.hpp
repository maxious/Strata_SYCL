// src/dsv41/ds4_mem.hpp - free / total card memory on an Intel GPU.
// dpct's get_memory_info asks for sycl::aspect::ext_intel_free_memory and reports 0 free when the driver has no
// such aspect (an Arc on the xe driver, #1545), so the residency planner sizes the expert lane from 0 bytes.
// Mirrors llama.cpp's ggml/src/ggml-sycl/mem.cpp (PR #27968, MIT): Level Zero sysman first, the SYCL aspect after.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>

namespace strata::dsv41 {

// Returns false when sysman did not answer and the value came from the aspect or the global memory size.
bool get_memory_size(::sycl::device dev, size_t& free_bytes, size_t& total_bytes);

}  // namespace strata::dsv41
