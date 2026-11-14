#pragma once

// Per-backend capability flags (Device::supports_*). Each backend derives them from what its own
// API reports for the device, so the flags of one physical device may differ between its Setups.
// The readers take function tables so the derivation can be exercised without a vendor runtime.

#include <vulkan/vulkan_core.h>
#include <ze_api.h>
#include <ze_ddi.h>

#include <cstdint>
#include <string_view>

#include "gpgpu/device.hpp"

namespace gpgpu::backends {

// --- CUDA ---

void set_cuda_capabilities(int compute_capability_major, int compute_capability_minor, Device& out);

// --- HIP ---

// `gcn_arch_name` is hipDeviceProp_t::gcnArchName (e.g. "gfx90a:sramecc+:xnack-"); empty when the
// runtime could not report it.
void set_rocm_capabilities(std::string_view gcn_arch_name, Device& out);

// --- Level Zero ---

// `get_module_properties` may be null; the device then reports no capability.
void read_level_zero_capabilities(ze_pfnDeviceGetModuleProperties_t get_module_properties,
                                  ze_device_handle_t device,
                                  Device& out);

// --- Vulkan ---

struct VulkanCapabilityFns {
    PFN_vkGetPhysicalDeviceFeatures          vkGetPhysicalDeviceFeatures;
    PFN_vkGetPhysicalDeviceFeatures2         vkGetPhysicalDeviceFeatures2;         // optional
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties; // optional
};

// fp16, int8 and tensor_cores need vkGetPhysicalDeviceFeatures2 and a device API version of at
// least 1.1; without them only fp64 is reported.
void read_vulkan_capabilities(const VulkanCapabilityFns& fns,
                              VkPhysicalDevice device,
                              std::uint32_t device_api_version,
                              Device& out);

} // namespace gpgpu::backends
