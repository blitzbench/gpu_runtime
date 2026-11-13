#include "gpgpu/vendor.hpp"

#include <string>
#include <vector>

#include "backends/device_identity.hpp"
#include "gpgpu/backend.hpp"
#include "platform/lib_loader.hpp"
#include "search_paths.hpp"

namespace gpgpu::vendor {

namespace {

// The loader handle is deliberately leaked: entry points stay valid for the process lifetime and
// tables returned earlier must never dangle.
platform::LibHandle* load_library(BackendId backend) noexcept {
    try {
        std::string error;
        auto lib = platform::try_load(search::candidates(backend), error);
        if (!lib.is_open()) return nullptr;
        return new platform::LibHandle(std::move(lib));
    } catch (...) {
        return nullptr;
    }
}

#define GPGPU_RESOLVE(member)                                                                  \
    table.member = reinterpret_cast<decltype(table.member)>(resolve(#member));                 \
    if (!table.member) complete = false;

// Resolved but never required (see the member's documentation in gpgpu/vendor.hpp).
#define GPGPU_RESOLVE_OPTIONAL(member)                                                         \
    table.member = reinterpret_cast<decltype(table.member)>(resolve(#member));

template <typename Resolver>
bool resolve_all(OpenClFns& table, Resolver resolve) noexcept {
    bool complete = true;
    GPGPU_RESOLVE(clGetPlatformIDs)
    GPGPU_RESOLVE(clGetPlatformInfo)
    GPGPU_RESOLVE(clGetDeviceIDs)
    GPGPU_RESOLVE(clGetDeviceInfo)
    GPGPU_RESOLVE(clCreateContext)
    GPGPU_RESOLVE(clReleaseContext)
    GPGPU_RESOLVE(clCreateCommandQueueWithProperties)
    GPGPU_RESOLVE(clReleaseCommandQueue)
    GPGPU_RESOLVE(clFlush)
    GPGPU_RESOLVE(clFinish)
    GPGPU_RESOLVE(clCreateBuffer)
    GPGPU_RESOLVE(clReleaseMemObject)
    GPGPU_RESOLVE(clCreateProgramWithSource)
    GPGPU_RESOLVE(clBuildProgram)
    GPGPU_RESOLVE(clGetProgramBuildInfo)
    GPGPU_RESOLVE(clReleaseProgram)
    GPGPU_RESOLVE(clCreateKernel)
    GPGPU_RESOLVE(clSetKernelArg)
    GPGPU_RESOLVE(clReleaseKernel)
    GPGPU_RESOLVE(clEnqueueNDRangeKernel)
    GPGPU_RESOLVE(clEnqueueReadBuffer)
    GPGPU_RESOLVE(clEnqueueWriteBuffer)
    GPGPU_RESOLVE(clEnqueueCopyBuffer)
    GPGPU_RESOLVE(clEnqueueMapBuffer)
    GPGPU_RESOLVE(clEnqueueUnmapMemObject)
    GPGPU_RESOLVE(clWaitForEvents)
    GPGPU_RESOLVE(clGetEventProfilingInfo)
    GPGPU_RESOLVE(clReleaseEvent)
    return complete;
}

template <typename Resolver>
bool resolve_all(VulkanFns& table, Resolver resolve) noexcept {
    bool complete = true;
    GPGPU_RESOLVE_OPTIONAL(vkEnumerateInstanceVersion)
    GPGPU_RESOLVE(vkCreateInstance)
    return complete;
}

template <typename Resolver>
bool resolve_all(VulkanInstanceFns& table, Resolver resolve) noexcept {
    bool complete = true;
    GPGPU_RESOLVE(vkDestroyInstance)
    GPGPU_RESOLVE(vkEnumeratePhysicalDevices)
    GPGPU_RESOLVE(vkGetPhysicalDeviceProperties)
    GPGPU_RESOLVE_OPTIONAL(vkGetPhysicalDeviceProperties2)
    GPGPU_RESOLVE(vkGetPhysicalDeviceFeatures)
    GPGPU_RESOLVE_OPTIONAL(vkGetPhysicalDeviceFeatures2)
    GPGPU_RESOLVE(vkGetPhysicalDeviceMemoryProperties)
    GPGPU_RESOLVE(vkGetPhysicalDeviceQueueFamilyProperties)
    GPGPU_RESOLVE(vkEnumerateDeviceExtensionProperties)
    GPGPU_RESOLVE(vkCreateDevice)
    GPGPU_RESOLVE(vkGetDeviceProcAddr)
    return complete;
}

template <typename Resolver>
bool resolve_all(VulkanDeviceFns& table, Resolver resolve) noexcept {
    bool complete = true;
    GPGPU_RESOLVE(vkDestroyDevice)
    GPGPU_RESOLVE(vkGetDeviceQueue)
    GPGPU_RESOLVE(vkQueueSubmit)
    GPGPU_RESOLVE(vkQueueWaitIdle)
    GPGPU_RESOLVE(vkAllocateMemory)
    GPGPU_RESOLVE(vkFreeMemory)
    GPGPU_RESOLVE(vkMapMemory)
    GPGPU_RESOLVE(vkUnmapMemory)
    GPGPU_RESOLVE(vkCreateBuffer)
    GPGPU_RESOLVE(vkDestroyBuffer)
    GPGPU_RESOLVE(vkGetBufferMemoryRequirements)
    GPGPU_RESOLVE(vkBindBufferMemory)
    GPGPU_RESOLVE(vkCreateShaderModule)
    GPGPU_RESOLVE(vkDestroyShaderModule)
    GPGPU_RESOLVE(vkCreateDescriptorSetLayout)
    GPGPU_RESOLVE(vkDestroyDescriptorSetLayout)
    GPGPU_RESOLVE(vkCreatePipelineLayout)
    GPGPU_RESOLVE(vkDestroyPipelineLayout)
    GPGPU_RESOLVE(vkCreateComputePipelines)
    GPGPU_RESOLVE(vkDestroyPipeline)
    GPGPU_RESOLVE(vkCreateDescriptorPool)
    GPGPU_RESOLVE(vkDestroyDescriptorPool)
    GPGPU_RESOLVE(vkAllocateDescriptorSets)
    GPGPU_RESOLVE(vkUpdateDescriptorSets)
    GPGPU_RESOLVE(vkCreateQueryPool)
    GPGPU_RESOLVE(vkDestroyQueryPool)
    GPGPU_RESOLVE(vkGetQueryPoolResults)
    GPGPU_RESOLVE(vkCreateCommandPool)
    GPGPU_RESOLVE(vkDestroyCommandPool)
    GPGPU_RESOLVE(vkAllocateCommandBuffers)
    GPGPU_RESOLVE(vkBeginCommandBuffer)
    GPGPU_RESOLVE(vkEndCommandBuffer)
    GPGPU_RESOLVE(vkResetCommandBuffer)
    GPGPU_RESOLVE(vkCmdBindPipeline)
    GPGPU_RESOLVE(vkCmdBindDescriptorSets)
    GPGPU_RESOLVE(vkCmdPushConstants)
    GPGPU_RESOLVE(vkCmdDispatch)
    GPGPU_RESOLVE(vkCmdCopyBuffer)
    GPGPU_RESOLVE(vkCmdPipelineBarrier)
    GPGPU_RESOLVE(vkCmdResetQueryPool)
    GPGPU_RESOLVE(vkCmdWriteTimestamp)
    return complete;
}

template <typename Resolver>
bool resolve_all(LevelZeroFns& table, Resolver resolve) noexcept {
    bool complete = true;
    GPGPU_RESOLVE(zeInit)
    GPGPU_RESOLVE(zeDriverGet)
    GPGPU_RESOLVE(zeDeviceGet)
    GPGPU_RESOLVE(zeDeviceGetProperties)
    GPGPU_RESOLVE(zeDeviceGetModuleProperties)
    GPGPU_RESOLVE(zeDeviceGetCommandQueueGroupProperties)
    GPGPU_RESOLVE(zeContextCreate)
    GPGPU_RESOLVE(zeContextDestroy)
    GPGPU_RESOLVE(zeCommandQueueCreate)
    GPGPU_RESOLVE(zeCommandQueueDestroy)
    GPGPU_RESOLVE(zeCommandQueueExecuteCommandLists)
    GPGPU_RESOLVE(zeCommandQueueSynchronize)
    GPGPU_RESOLVE(zeCommandListCreate)
    GPGPU_RESOLVE(zeCommandListDestroy)
    GPGPU_RESOLVE(zeCommandListClose)
    GPGPU_RESOLVE(zeCommandListReset)
    GPGPU_RESOLVE(zeCommandListAppendBarrier)
    GPGPU_RESOLVE(zeCommandListAppendMemoryCopy)
    GPGPU_RESOLVE(zeCommandListAppendWriteGlobalTimestamp)
    GPGPU_RESOLVE(zeCommandListAppendLaunchKernel)
    GPGPU_RESOLVE(zeEventPoolCreate)
    GPGPU_RESOLVE(zeEventPoolDestroy)
    GPGPU_RESOLVE(zeEventCreate)
    GPGPU_RESOLVE(zeEventDestroy)
    GPGPU_RESOLVE(zeEventHostReset)
    GPGPU_RESOLVE(zeEventQueryKernelTimestamp)
    GPGPU_RESOLVE(zeMemAllocDevice)
    GPGPU_RESOLVE(zeMemAllocHost)
    GPGPU_RESOLVE(zeMemFree)
    GPGPU_RESOLVE(zeModuleCreate)
    GPGPU_RESOLVE(zeModuleDestroy)
    GPGPU_RESOLVE(zeModuleBuildLogGetString)
    GPGPU_RESOLVE(zeModuleBuildLogDestroy)
    GPGPU_RESOLVE(zeKernelCreate)
    GPGPU_RESOLVE(zeKernelDestroy)
    GPGPU_RESOLVE(zeKernelSetGroupSize)
    GPGPU_RESOLVE(zeKernelSetArgumentValue)
    GPGPU_RESOLVE_OPTIONAL(zeDevicePciGetPropertiesExt)
    return complete;
}

#undef GPGPU_RESOLVE_OPTIONAL
#undef GPGPU_RESOLVE

const OpenClFns* load_opencl() noexcept {
    const platform::LibHandle* lib = load_library(BackendId::OpenCL);
    if (!lib) return nullptr;
    static OpenClFns table{};
    const bool complete = resolve_all(table, [lib](const char* name) { return lib->resolve(name); });
    return complete ? &table : nullptr;
}

const VulkanFns* load_vulkan() noexcept {
    const platform::LibHandle* lib = load_library(BackendId::Vulkan);
    if (!lib) return nullptr;
    static VulkanFns table{};
    table.vkGetInstanceProcAddr =
        platform::resolve_as<PFN_vkGetInstanceProcAddr>(*lib, "vkGetInstanceProcAddr");
    if (!table.vkGetInstanceProcAddr) return nullptr;
    const PFN_vkGetInstanceProcAddr gipa = table.vkGetInstanceProcAddr;
    const bool complete = resolve_all(table, [gipa](const char* name) {
        return reinterpret_cast<void*>(gipa(VK_NULL_HANDLE, name));
    });
    return complete ? &table : nullptr;
}

const LevelZeroFns* load_level_zero() noexcept {
    const platform::LibHandle* lib = load_library(BackendId::OneAPI);
    if (!lib) return nullptr;
    static LevelZeroFns table{};
    const bool complete = resolve_all(table, [lib](const char* name) { return lib->resolve(name); });
    return complete ? &table : nullptr;
}

// cuInit is part of loading: every Driver API call after it needs an initialised driver, and
// calling it again is harmless.
const backends::CudaIdentityFns* load_cuda() noexcept {
    const platform::LibHandle* lib = load_library(BackendId::CUDA);
    if (!lib) return nullptr;
    using PFN_cuInit = backends::CUresult (GPGPU_STDCALL*)(unsigned int);
    const auto cuInit = platform::resolve_as<PFN_cuInit>(*lib, "cuInit");
    static backends::CudaIdentityFns table{};
    table.cuDeviceGetCount = platform::resolve_as<backends::PFN_cuDeviceGetCount>(*lib, "cuDeviceGetCount");
    table.cuDeviceGet = platform::resolve_as<backends::PFN_cuDeviceGet>(*lib, "cuDeviceGet");
    table.cuDeviceGetAttribute =
        platform::resolve_as<backends::PFN_cuDeviceGetAttribute>(*lib, "cuDeviceGetAttribute");
    table.cuDeviceGetUuid = platform::resolve_as<backends::PFN_cuDeviceGetUuid>(*lib, "cuDeviceGetUuid");
    if (!cuInit || !table.cuDeviceGetCount || !table.cuDeviceGet || !table.cuDeviceGetAttribute) {
        return nullptr;
    }
    if (cuInit(0) != 0) return nullptr;
    return &table;
}

const backends::HipIdentityFns* load_hip() noexcept {
    const platform::LibHandle* lib = load_library(BackendId::ROCm);
    if (!lib) return nullptr;
    static backends::HipIdentityFns table{};
    table.hipGetDeviceCount = platform::resolve_as<backends::PFN_hipGetDeviceCount>(*lib, "hipGetDeviceCount");
    table.hipDeviceGet = platform::resolve_as<backends::PFN_hipDeviceGet>(*lib, "hipDeviceGet");
    table.hipDeviceGetPCIBusId =
        platform::resolve_as<backends::PFN_hipDeviceGetPCIBusId>(*lib, "hipDeviceGetPCIBusId");
    table.hipDeviceGetUuid = platform::resolve_as<backends::PFN_hipDeviceGetUuid>(*lib, "hipDeviceGetUuid");
    return table.hipGetDeviceCount ? &table : nullptr;
}

} // namespace

const OpenClFns* opencl() noexcept {
    static const OpenClFns* const table = load_opencl();
    return table;
}

const VulkanFns* vulkan() noexcept {
    static const VulkanFns* const table = load_vulkan();
    return table;
}

const LevelZeroFns* level_zero() noexcept {
    static const LevelZeroFns* const table = load_level_zero();
    return table;
}

VkPhysicalDevice find_vulkan_device(const VulkanInstanceFns& fns, VkInstance instance,
                                    const Device& target) noexcept {
    try {
        std::uint32_t count = 0;
        if (fns.vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) {
            return VK_NULL_HANDLE;
        }
        std::vector<VkPhysicalDevice> devices(count);
        if (fns.vkEnumeratePhysicalDevices(instance, &count, devices.data()) < VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        devices.resize(count);
        const backends::VulkanIdentityFns identity_fns{fns.vkGetPhysicalDeviceProperties,
                                                       fns.vkGetPhysicalDeviceProperties2,
                                                       fns.vkEnumerateDeviceExtensionProperties};
        for (VkPhysicalDevice pd : devices) {
            Device candidate;
            backends::read_vulkan_identity(identity_fns, pd, candidate);
            if (candidate.id() == target.id()) return pd;
        }
    } catch (...) {
        return VK_NULL_HANDLE;
    }
    return VK_NULL_HANDLE;
}

bool find_level_zero_device(const LevelZeroFns& fns, const Device& target,
                            ze_driver_handle_t& driver, ze_device_handle_t& device) noexcept {
    try {
        std::uint32_t n_drivers = 0;
        if (fns.zeDriverGet(&n_drivers, nullptr) != ZE_RESULT_SUCCESS || n_drivers == 0) return false;
        std::vector<ze_driver_handle_t> drivers(n_drivers);
        if (fns.zeDriverGet(&n_drivers, drivers.data()) != ZE_RESULT_SUCCESS) return false;
        std::uint32_t ordinal = 0;
        for (ze_driver_handle_t drv : drivers) {
            std::uint32_t n_devices = 0;
            if (fns.zeDeviceGet(drv, &n_devices, nullptr) != ZE_RESULT_SUCCESS || n_devices == 0) continue;
            std::vector<ze_device_handle_t> devices(n_devices);
            if (fns.zeDeviceGet(drv, &n_devices, devices.data()) != ZE_RESULT_SUCCESS) continue;
            for (ze_device_handle_t dev : devices) {
                const std::uint32_t device_ordinal = ordinal++;
                ze_device_properties_t props{};
                props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                if (fns.zeDeviceGetProperties(dev, &props) != ZE_RESULT_SUCCESS) continue;
                Device candidate;
                backends::read_level_zero_identity(fns.zeDevicePciGetPropertiesExt, dev, props,
                                                   device_ordinal, candidate);
                if (candidate.id() == target.id()) {
                    driver = drv;
                    device = dev;
                    return true;
                }
            }
        }
    } catch (...) {
        return false;
    }
    return false;
}

cl_device_id find_opencl_device(const OpenClFns& fns, const Device& target) noexcept {
    try {
        cl_uint n_platforms = 0;
        if (fns.clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) {
            return nullptr;
        }
        std::vector<cl_platform_id> platforms(n_platforms);
        if (fns.clGetPlatformIDs(n_platforms, platforms.data(), nullptr) != CL_SUCCESS) return nullptr;
        for (cl_platform_id platform : platforms) {
            if (backends::is_skipped_opencl_platform(fns.clGetPlatformInfo, platform)) continue;
            cl_uint n_devices = 0;
            if (fns.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &n_devices) != CL_SUCCESS ||
                n_devices == 0) {
                continue;
            }
            std::vector<cl_device_id> devices(n_devices);
            if (fns.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, n_devices, devices.data(), nullptr) !=
                CL_SUCCESS) {
                continue;
            }
            for (cl_device_id dev : devices) {
                Device candidate;
                backends::read_opencl_identity(fns.clGetDeviceInfo, dev, candidate);
                if (candidate.id() == target.id()) return dev;
            }
        }
    } catch (...) {
        return nullptr;
    }
    return nullptr;
}

std::optional<int> find_cuda_device(const Device& target) noexcept {
    try {
        static const backends::CudaIdentityFns* const table = load_cuda();
        if (!table) return std::nullopt;
        return backends::find_cuda_ordinal(*table, target);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<int> find_hip_device(const Device& target) noexcept {
    try {
        static const backends::HipIdentityFns* const table = load_hip();
        if (!table) return std::nullopt;
        return backends::find_hip_ordinal(*table, target);
    } catch (...) {
        return std::nullopt;
    }
}

bool load_instance_fns(const VulkanFns& global, VkInstance instance, VulkanInstanceFns& out) noexcept {
    out = VulkanInstanceFns{};
    if (!global.vkGetInstanceProcAddr || instance == VK_NULL_HANDLE) return false;
    return resolve_all(out, [&global, instance](const char* name) {
        return reinterpret_cast<void*>(global.vkGetInstanceProcAddr(instance, name));
    });
}

bool load_device_fns(const VulkanInstanceFns& instance_fns, VkDevice device, VulkanDeviceFns& out) noexcept {
    out = VulkanDeviceFns{};
    if (!instance_fns.vkGetDeviceProcAddr || device == VK_NULL_HANDLE) return false;
    return resolve_all(out, [&instance_fns, device](const char* name) {
        return reinterpret_cast<void*>(instance_fns.vkGetDeviceProcAddr(device, name));
    });
}

} // namespace gpgpu::vendor
