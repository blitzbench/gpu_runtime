#include "gpgpu/vendor.hpp"

#include <string>

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

// Resolved but never required: absent on a loader or instance older than Vulkan 1.1.
#define GPGPU_RESOLVE_OPTIONAL(member)                                                         \
    table.member = reinterpret_cast<decltype(table.member)>(resolve(#member));

template <typename Resolver>
bool resolve_all(OpenClFns& table, Resolver resolve) noexcept {
    bool complete = true;
    GPGPU_RESOLVE(clGetPlatformIDs)
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
