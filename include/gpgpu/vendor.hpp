#pragma once

// Vendor runtime entry points bound at run time.
//
// A consumer that calls OpenCL or Vulkan through these tables carries no link-time dependency on
// libOpenCL / libvulkan: the loader library is dlopen'd once per process from the same candidate
// paths the backend probes use, and every entry point is resolved by symbol. A process therefore
// starts on a machine without the vendor runtime and simply sees a null table.
//
// The vendored Khronos headers supply the types and constants only. VK_NO_PROTOTYPES is defined on
// this library's public interface so that a direct vk* call in a consumer fails to compile instead
// of silently re-introducing a link dependency. The OpenCL headers declare prototypes, so a direct
// cl* call only fails at link time, and only as long as nothing else in the link pulls in
// libOpenCL.

#include <CL/cl.h>
#include <CL/cl_function_types.h>
#include <vulkan/vulkan_core.h>

namespace gpgpu::vendor {

// Every OpenCL entry point the bundled tasks call. All members are non-null in a table returned by
// opencl().
struct OpenClFns {
    clGetPlatformIDs_fn                    clGetPlatformIDs;
    clGetDeviceIDs_fn                      clGetDeviceIDs;
    clGetDeviceInfo_fn                     clGetDeviceInfo;
    clCreateContext_fn                     clCreateContext;
    clReleaseContext_fn                    clReleaseContext;
    clCreateCommandQueueWithProperties_fn  clCreateCommandQueueWithProperties;
    clReleaseCommandQueue_fn               clReleaseCommandQueue;
    clFlush_fn                             clFlush;
    clFinish_fn                            clFinish;
    clCreateBuffer_fn                      clCreateBuffer;
    clReleaseMemObject_fn                  clReleaseMemObject;
    clCreateProgramWithSource_fn           clCreateProgramWithSource;
    clBuildProgram_fn                      clBuildProgram;
    clGetProgramBuildInfo_fn               clGetProgramBuildInfo;
    clReleaseProgram_fn                    clReleaseProgram;
    clCreateKernel_fn                      clCreateKernel;
    clSetKernelArg_fn                      clSetKernelArg;
    clReleaseKernel_fn                     clReleaseKernel;
    clEnqueueNDRangeKernel_fn              clEnqueueNDRangeKernel;
    clEnqueueReadBuffer_fn                 clEnqueueReadBuffer;
    clEnqueueWriteBuffer_fn                clEnqueueWriteBuffer;
    clEnqueueCopyBuffer_fn                 clEnqueueCopyBuffer;
    clEnqueueMapBuffer_fn                  clEnqueueMapBuffer;
    clEnqueueUnmapMemObject_fn             clEnqueueUnmapMemObject;
    clWaitForEvents_fn                     clWaitForEvents;
    clGetEventProfilingInfo_fn             clGetEventProfilingInfo;
    clReleaseEvent_fn                      clReleaseEvent;
};

// Vulkan global-level entry points: the ones vkGetInstanceProcAddr resolves without an instance.
// In a table returned by vulkan() every member is non-null except vkEnumerateInstanceVersion,
// which a Vulkan 1.0 loader does not export; a null there means the instance version is 1.0.
struct VulkanFns {
    PFN_vkGetInstanceProcAddr      vkGetInstanceProcAddr;
    PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion;
    PFN_vkCreateInstance           vkCreateInstance;
};

// Vulkan instance-level entry points, resolved for one VkInstance by load_instance_fns. Every
// member is required except vkGetPhysicalDeviceProperties2 and vkGetPhysicalDeviceFeatures2
// (Vulkan 1.1 core): an instance created for apiVersion 1.0 may leave those two null, and a caller
// that did not request 1.1 or later must check them before calling.
struct VulkanInstanceFns {
    PFN_vkDestroyInstance                          vkDestroyInstance;
    PFN_vkEnumeratePhysicalDevices                 vkEnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties              vkGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceProperties2             vkGetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceFeatures                vkGetPhysicalDeviceFeatures;
    PFN_vkGetPhysicalDeviceFeatures2               vkGetPhysicalDeviceFeatures2;
    PFN_vkGetPhysicalDeviceMemoryProperties        vkGetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties   vkGetPhysicalDeviceQueueFamilyProperties;
    PFN_vkEnumerateDeviceExtensionProperties       vkEnumerateDeviceExtensionProperties;
    PFN_vkCreateDevice                             vkCreateDevice;
    PFN_vkGetDeviceProcAddr                        vkGetDeviceProcAddr;
};

// Vulkan device-level entry points, resolved for one VkDevice by load_device_fns through
// vkGetDeviceProcAddr so calls bypass the loader trampoline.
struct VulkanDeviceFns {
    PFN_vkDestroyDevice                 vkDestroyDevice;
    PFN_vkGetDeviceQueue                vkGetDeviceQueue;
    PFN_vkQueueSubmit                   vkQueueSubmit;
    PFN_vkQueueWaitIdle                 vkQueueWaitIdle;
    PFN_vkAllocateMemory                vkAllocateMemory;
    PFN_vkFreeMemory                    vkFreeMemory;
    PFN_vkMapMemory                     vkMapMemory;
    PFN_vkUnmapMemory                   vkUnmapMemory;
    PFN_vkCreateBuffer                  vkCreateBuffer;
    PFN_vkDestroyBuffer                 vkDestroyBuffer;
    PFN_vkGetBufferMemoryRequirements   vkGetBufferMemoryRequirements;
    PFN_vkBindBufferMemory              vkBindBufferMemory;
    PFN_vkCreateShaderModule            vkCreateShaderModule;
    PFN_vkDestroyShaderModule           vkDestroyShaderModule;
    PFN_vkCreateDescriptorSetLayout     vkCreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout    vkDestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout          vkCreatePipelineLayout;
    PFN_vkDestroyPipelineLayout         vkDestroyPipelineLayout;
    PFN_vkCreateComputePipelines        vkCreateComputePipelines;
    PFN_vkDestroyPipeline               vkDestroyPipeline;
    PFN_vkCreateDescriptorPool          vkCreateDescriptorPool;
    PFN_vkDestroyDescriptorPool         vkDestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets        vkAllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets          vkUpdateDescriptorSets;
    PFN_vkCreateQueryPool               vkCreateQueryPool;
    PFN_vkDestroyQueryPool              vkDestroyQueryPool;
    PFN_vkGetQueryPoolResults           vkGetQueryPoolResults;
    PFN_vkCreateCommandPool             vkCreateCommandPool;
    PFN_vkDestroyCommandPool            vkDestroyCommandPool;
    PFN_vkAllocateCommandBuffers        vkAllocateCommandBuffers;
    PFN_vkBeginCommandBuffer            vkBeginCommandBuffer;
    PFN_vkEndCommandBuffer              vkEndCommandBuffer;
    PFN_vkResetCommandBuffer            vkResetCommandBuffer;
    PFN_vkCmdBindPipeline               vkCmdBindPipeline;
    PFN_vkCmdBindDescriptorSets         vkCmdBindDescriptorSets;
    PFN_vkCmdPushConstants              vkCmdPushConstants;
    PFN_vkCmdDispatch                   vkCmdDispatch;
    PFN_vkCmdCopyBuffer                 vkCmdCopyBuffer;
    PFN_vkCmdPipelineBarrier            vkCmdPipelineBarrier;
    PFN_vkCmdResetQueryPool             vkCmdResetQueryPool;
    PFN_vkCmdWriteTimestamp             vkCmdWriteTimestamp;
};

// Loaded once per process from the same candidate paths the backend probes use; the loader library
// stays mapped for the lifetime of the process. Returns nullptr when the library or any required
// symbol is absent. Thread-safe; never throws, never terminates.
const OpenClFns* opencl() noexcept;
const VulkanFns* vulkan() noexcept;

// Resolve the instance-level table for `instance`. Returns false when any member is missing; the
// members that did resolve are still set, so a caller can tear the instance down.
bool load_instance_fns(const VulkanFns& global, VkInstance instance, VulkanInstanceFns& out) noexcept;

// Resolve the device-level table for `device`. Returns false when any member is missing; the
// members that did resolve are still set, so a caller can tear the device down.
bool load_device_fns(const VulkanInstanceFns& instance_fns, VkDevice device, VulkanDeviceFns& out) noexcept;

} // namespace gpgpu::vendor
