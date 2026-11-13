#pragma once

// Per-backend device identity. Each backend reads its devices' PCI address, PCI ids, UUID and
// type here, and the probe and the native-handle lookups in gpgpu/vendor.hpp both go through the
// same reader, so the id a lookup compares against is produced by exactly the code that produced
// Device::id() during query().

#include <CL/cl.h>
#include <CL/cl_function_types.h>
#include <vulkan/vulkan_core.h>
#include <ze_api.h>
#include <ze_ddi.h>

#include <cstdint>
#include <optional>

#include "gpgpu/device.hpp"
#include "calling_convention.hpp"

namespace gpgpu::backends {

// --- CUDA (Driver API, declared by hand; see calling_convention.hpp) ---

using CUresult = int;
using CUdevice = int;

using PFN_cuDeviceGetCount     = CUresult (GPGPU_STDCALL*)(int*);
using PFN_cuDeviceGet          = CUresult (GPGPU_STDCALL*)(CUdevice*, int);
using PFN_cuDeviceGetAttribute = CUresult (GPGPU_STDCALL*)(int*, int, CUdevice);
using PFN_cuDeviceGetUuid      = CUresult (GPGPU_STDCALL*)(unsigned char*, CUdevice);

struct CudaIdentityFns {
    PFN_cuDeviceGetCount     cuDeviceGetCount;
    PFN_cuDeviceGet          cuDeviceGet;
    PFN_cuDeviceGetAttribute cuDeviceGetAttribute;
    PFN_cuDeviceGetUuid      cuDeviceGetUuid; // optional
};

void read_cuda_identity(const CudaIdentityFns& fns, CUdevice device, int ordinal, Device& out);
std::optional<int> find_cuda_ordinal(const CudaIdentityFns& fns, const Device& target);

// --- HIP (Runtime API, declared by hand; see calling_convention.hpp) ---

using hipError_t  = int;
using hipDevice_t = int;

struct HipUuid {
    char bytes[16];
};

using PFN_hipGetDeviceCount    = hipError_t (GPGPU_STDCALL*)(int*);
using PFN_hipDeviceGet         = hipError_t (GPGPU_STDCALL*)(hipDevice_t*, int);
using PFN_hipDeviceGetPCIBusId = hipError_t (GPGPU_STDCALL*)(char*, int, hipDevice_t);
using PFN_hipDeviceGetUuid     = hipError_t (GPGPU_STDCALL*)(HipUuid*, hipDevice_t);

struct HipIdentityFns {
    PFN_hipGetDeviceCount    hipGetDeviceCount;
    PFN_hipDeviceGet         hipDeviceGet;         // optional: the ordinal is the handle without it
    PFN_hipDeviceGetPCIBusId hipDeviceGetPCIBusId; // optional
    PFN_hipDeviceGetUuid     hipDeviceGetUuid;     // optional
};

void read_hip_identity(const HipIdentityFns& fns, hipDevice_t device, int ordinal, Device& out);
std::optional<int> find_hip_ordinal(const HipIdentityFns& fns, const Device& target);

// --- Level Zero ---

// `pci_properties` may be null (a loader predating ZE_extension_pci_properties).
// `ordinal` is the device's position across all drivers, used only by the backend-specific id.
void read_level_zero_identity(ze_pfnDevicePciGetPropertiesExt_t pci_properties,
                              ze_device_handle_t device,
                              const ze_device_properties_t& properties,
                              std::uint32_t ordinal,
                              Device& out);

// --- OpenCL ---

void read_opencl_identity(clGetDeviceInfo_fn get_device_info, cl_device_id device, Device& out);

// Platforms the probe does not report (Microsoft's OpenCLOn12 / D3D12 mapping layer).
bool is_skipped_opencl_platform(clGetPlatformInfo_fn get_platform_info, cl_platform_id platform);

// --- Vulkan ---

struct VulkanIdentityFns {
    PFN_vkGetPhysicalDeviceProperties        vkGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceProperties2       vkGetPhysicalDeviceProperties2;       // optional
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties; // optional
};

// The PCI address and UUID need Vulkan 1.1 (vkGetPhysicalDeviceProperties2 and a device API
// version of at least 1.1); without them the device gets the backend-specific id.
void read_vulkan_identity(const VulkanIdentityFns& fns, VkPhysicalDevice device, Device& out);

} // namespace gpgpu::backends
