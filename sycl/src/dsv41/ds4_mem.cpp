// src/dsv41/ds4_mem.cpp - the Level Zero sysman memory query.
// Mirrors llama.cpp's ggml/src/ggml-sycl/mem.cpp (PR #27968, MIT license): sum the device's memory modules'
// free and size, and fall back to the SYCL aspect, then to the global memory size, when sysman does not answer.
#define DPCT_COMPAT_RT_VERSION 12080
#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

#include "ds4_mem.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

namespace strata::dsv41 {

namespace {

constexpr ::sycl::backend kL0Backend = ::sycl::backend::ext_oneapi_level_zero;

bool query_free_memory_by_ze(::sycl::device dev, size_t& free_bytes, size_t& total_bytes) {
    free_bytes = 0;
    total_bytes = 0;

    uint32_t module_count = 0;
    try {
        if (zesInit(0) != ZE_RESULT_SUCCESS) {
            std::cerr << "strata: zesInit failed; sysman free-memory query unavailable\n";
        }
        if (dev.get_platform().get_backend() != kL0Backend) {
            total_bytes = dev.get_info<::sycl::info::device::global_mem_size>();
            free_bytes = total_bytes;
            return false;
        }
        ze_device_handle_t ze_dev = ::sycl::get_native<kL0Backend>(dev);
        if (ze_dev == nullptr) {
            total_bytes = dev.get_info<::sycl::info::device::global_mem_size>();
            free_bytes = total_bytes;
            return false;
        }
        if (zesDeviceEnumMemoryModules(ze_dev, &module_count, nullptr) != ZE_RESULT_SUCCESS || module_count == 0) {
            total_bytes = dev.get_info<::sycl::info::device::global_mem_size>();
            free_bytes = total_bytes;
            return false;
        }
        std::vector<zes_mem_handle_t> modules(module_count);
        if (zesDeviceEnumMemoryModules(ze_dev, &module_count, modules.data()) != ZE_RESULT_SUCCESS ||
            module_count == 0) {
            total_bytes = dev.get_info<::sycl::info::device::global_mem_size>();
            free_bytes = total_bytes;
            return false;
        }
        for (uint32_t i = 0; i < module_count; ++i) {
            zes_mem_state_t state = {};
            state.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext = nullptr;
            if (zesMemoryGetState(modules[i], &state) != ZE_RESULT_SUCCESS) {
                continue;
            }
            free_bytes += state.free;
            total_bytes += state.size;
        }
        if (total_bytes == 0) {
            total_bytes = dev.get_info<::sycl::info::device::global_mem_size>();
            free_bytes = total_bytes;
            return false;
        }
        return true;
    } catch (const ::sycl::exception& e) {
        std::cerr << "strata: Level Zero memory query failed: " << e.what() << "\n";
        total_bytes = dev.get_info<::sycl::info::device::global_mem_size>();
        free_bytes = total_bytes;
        return false;
    }
}

bool get_memory_size_by_sycl_api(::sycl::device dev, size_t& free_bytes, size_t& total_bytes) {
    total_bytes = dev.get_info<::sycl::info::device::global_mem_size>();
#if defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105
    if (dev.has(::sycl::aspect::ext_intel_free_memory)) {
        try {
            free_bytes = dev.get_info<::sycl::ext::intel::info::device::free_memory>();
            return true;
        } catch (const ::sycl::exception&) {
            free_bytes = total_bytes;
            return false;
        }
    }
#endif
    free_bytes = total_bytes;
    return true;
}

}  // namespace

bool get_memory_size(::sycl::device dev, size_t& free_bytes, size_t& total_bytes) {
    if (!query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
        return get_memory_size_by_sycl_api(dev, free_bytes, total_bytes);
    }
    return true;
}

}  // namespace strata::dsv41
