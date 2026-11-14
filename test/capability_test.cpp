#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "backends/device_capabilities.hpp"
#include "gpgpu/device.hpp"

// Fake vendor enumerations for the capability readers. A native handle is the address of the fake
// device's table entry, as in lookup_test.cpp.

namespace {

using gpgpu::Device;

struct Flags {
    bool fp16;
    bool fp64;
    bool int8;
    bool tensor_cores;
};

void expect_flags(const Device& d, Flags expected) {
    EXPECT_EQ(d.supports_fp16(), expected.fp16);
    EXPECT_EQ(d.supports_fp64(), expected.fp64);
    EXPECT_EQ(d.supports_int8(), expected.int8);
    EXPECT_EQ(d.supports_tensor_cores(), expected.tensor_cores);
}

// --- Vulkan -----------------------------------------------------------------------------------

struct FakeVkDevice {
    bool shader_float64;
    bool shader_float16;
    bool shader_int8;
    bool cooperative_matrix;
    std::vector<std::string> extensions;
};

FakeVkDevice g_vk_device;
// Every sType chained into vkGetPhysicalDeviceFeatures2, in chain order.
std::vector<VkStructureType> g_vk_chained;

const FakeVkDevice& vk_fake(VkPhysicalDevice pd) {
    return *reinterpret_cast<const FakeVkDevice*>(pd);
}

VkPhysicalDevice vk_handle() {
    return reinterpret_cast<VkPhysicalDevice>(&g_vk_device);
}

VKAPI_ATTR void VKAPI_CALL fake_vkGetPhysicalDeviceFeatures(VkPhysicalDevice pd,
                                                            VkPhysicalDeviceFeatures* out) {
    *out = VkPhysicalDeviceFeatures{};
    out->shaderFloat64 = vk_fake(pd).shader_float64 ? VK_TRUE : VK_FALSE;
}

VKAPI_ATTR void VKAPI_CALL fake_vkGetPhysicalDeviceFeatures2(VkPhysicalDevice pd,
                                                             VkPhysicalDeviceFeatures2* out) {
    const FakeVkDevice& d = vk_fake(pd);
    fake_vkGetPhysicalDeviceFeatures(pd, &out->features);
    for (auto* next = static_cast<VkBaseOutStructure*>(out->pNext); next; next = next->pNext) {
        g_vk_chained.push_back(next->sType);
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES) {
            auto* f = reinterpret_cast<VkPhysicalDeviceShaderFloat16Int8Features*>(next);
            f->shaderFloat16 = d.shader_float16 ? VK_TRUE : VK_FALSE;
            f->shaderInt8 = d.shader_int8 ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR) {
            reinterpret_cast<VkPhysicalDeviceCooperativeMatrixFeaturesKHR*>(next)->cooperativeMatrix =
                d.cooperative_matrix ? VK_TRUE : VK_FALSE;
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL fake_vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice pd, const char*, std::uint32_t* count, VkExtensionProperties* out) {
    const auto& extensions = vk_fake(pd).extensions;
    if (!out) {
        *count = static_cast<std::uint32_t>(extensions.size());
        return VK_SUCCESS;
    }
    for (std::uint32_t i = 0; i < *count && i < extensions.size(); ++i) {
        out[i] = VkExtensionProperties{};
        std::strncpy(out[i].extensionName, extensions[i].c_str(), VK_MAX_EXTENSION_NAME_SIZE - 1);
    }
    return VK_SUCCESS;
}

gpgpu::backends::VulkanCapabilityFns vk_fns(bool with_features2) {
    return {fake_vkGetPhysicalDeviceFeatures,
            with_features2 ? fake_vkGetPhysicalDeviceFeatures2 : nullptr,
            fake_vkEnumerateDeviceExtensionProperties};
}

Device read_vk(const gpgpu::backends::VulkanCapabilityFns& fns, std::uint32_t api) {
    Device d;
    gpgpu::backends::read_vulkan_capabilities(fns, vk_handle(), api, d);
    return d;
}

class VulkanCapabilities : public ::testing::Test {
protected:
    void SetUp() override {
        g_vk_chained.clear();
        g_vk_device = FakeVkDevice{true, true, true, true, {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME}};
    }
};

TEST_F(VulkanCapabilities, ReadsEveryFlagFromItsFeature) {
    expect_flags(read_vk(vk_fns(true), VK_API_VERSION_1_3), {true, true, true, true});
}

TEST_F(VulkanCapabilities, MissingFeaturesClearTheirFlags) {
    // An integrated GPU without double precision or matrix units (Iris Xe).
    g_vk_device = FakeVkDevice{false, true, true, false, {}};
    expect_flags(read_vk(vk_fns(true), VK_API_VERSION_1_3), {true, false, true, false});

    g_vk_device = FakeVkDevice{true, false, false, false, {}};
    expect_flags(read_vk(vk_fns(true), VK_API_VERSION_1_3), {false, true, false, false});
}

TEST_F(VulkanCapabilities, CooperativeMatrixNeedsTheExtensionAndTheFeature) {
    g_vk_device.extensions.clear();
    EXPECT_FALSE(read_vk(vk_fns(true), VK_API_VERSION_1_3).supports_tensor_cores());

    g_vk_device.extensions = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
    g_vk_device.cooperative_matrix = false;
    EXPECT_FALSE(read_vk(vk_fns(true), VK_API_VERSION_1_3).supports_tensor_cores());
}

TEST_F(VulkanCapabilities, ChainsOnlyStructuresTheDeviceKnows) {
    g_vk_device.extensions.clear();
    read_vk(vk_fns(true), VK_API_VERSION_1_3);
    EXPECT_EQ(g_vk_chained,
              std::vector<VkStructureType>{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES});

    g_vk_chained.clear();
    g_vk_device.extensions = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
    read_vk(vk_fns(true), VK_API_VERSION_1_1);
    EXPECT_EQ(g_vk_chained,
              std::vector<VkStructureType>{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR});
}

TEST_F(VulkanCapabilities, Float16Int8ExtensionStandsInForVulkan12) {
    g_vk_device.extensions = {VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME};
    expect_flags(read_vk(vk_fns(true), VK_API_VERSION_1_1), {true, true, true, false});
}

TEST_F(VulkanCapabilities, Vulkan11WithoutExtensionsReportsOnlyFp64) {
    g_vk_device.extensions.clear();
    expect_flags(read_vk(vk_fns(true), VK_API_VERSION_1_1), {false, true, false, false});
    EXPECT_TRUE(g_vk_chained.empty());
}

TEST_F(VulkanCapabilities, WithoutFeatures2OnlyFp64IsReported) {
    expect_flags(read_vk(vk_fns(false), VK_API_VERSION_1_3), {false, true, false, false});
    expect_flags(read_vk(vk_fns(true), VK_API_VERSION_1_0), {false, true, false, false});
    EXPECT_TRUE(g_vk_chained.empty());
}

// --- Level Zero -------------------------------------------------------------------------------

struct FakeZeDevice {
    ze_result_t result;
    ze_device_module_flags_t flags;
};

FakeZeDevice g_ze_device;

ze_result_t ZE_APICALL fake_zeDeviceGetModuleProperties(ze_device_handle_t h,
                                                        ze_device_module_properties_t* out) {
    const auto& d = *reinterpret_cast<const FakeZeDevice*>(h);
    if (d.result != ZE_RESULT_SUCCESS) return d.result;
    out->flags = d.flags;
    // fp64flags describes fp64 rounding and denormal behaviour, not the capability. INF_NAN shares
    // its bit with ZE_DEVICE_MODULE_FLAG_FP64, so reading the wrong field reports fp64.
    out->fp64flags = ZE_DEVICE_FP_FLAG_INF_NAN;
    return ZE_RESULT_SUCCESS;
}

Device read_ze(ze_pfnDeviceGetModuleProperties_t fn) {
    Device d;
    gpgpu::backends::read_level_zero_capabilities(fn, reinterpret_cast<ze_device_handle_t>(&g_ze_device), d);
    return d;
}

TEST(LevelZeroCapabilities, ReadsModuleFlags) {
    g_ze_device = {ZE_RESULT_SUCCESS, ZE_DEVICE_MODULE_FLAG_FP16 | ZE_DEVICE_MODULE_FLAG_FP64};
    expect_flags(read_ze(fake_zeDeviceGetModuleProperties), {true, true, true, false});

    // Iris Xe: fp16, no fp64.
    g_ze_device = {ZE_RESULT_SUCCESS, ZE_DEVICE_MODULE_FLAG_FP16 | ZE_DEVICE_MODULE_FLAG_DP4A};
    expect_flags(read_ze(fake_zeDeviceGetModuleProperties), {true, false, true, false});

    g_ze_device = {ZE_RESULT_SUCCESS, ZE_DEVICE_MODULE_FLAG_FP64};
    expect_flags(read_ze(fake_zeDeviceGetModuleProperties), {false, true, true, false});
}

TEST(LevelZeroCapabilities, Int8DoesNotDependOnDp4a) {
    g_ze_device = {ZE_RESULT_SUCCESS, 0};
    EXPECT_TRUE(read_ze(fake_zeDeviceGetModuleProperties).supports_int8());
}

TEST(LevelZeroCapabilities, UnreadablePropertiesReportOnlyInt8) {
    g_ze_device = {ZE_RESULT_ERROR_UNINITIALIZED, ZE_DEVICE_MODULE_FLAG_FP16 | ZE_DEVICE_MODULE_FLAG_FP64};
    expect_flags(read_ze(fake_zeDeviceGetModuleProperties), {false, false, true, false});
    expect_flags(read_ze(nullptr), {false, false, true, false});
}

// --- CUDA -------------------------------------------------------------------------------------

Device cuda(int major, int minor) {
    Device d;
    gpgpu::backends::set_cuda_capabilities(major, minor, d);
    return d;
}

TEST(CudaCapabilities, Fp16StartsAtComputeCapability53) {
    EXPECT_FALSE(cuda(5, 2).supports_fp16());
    EXPECT_TRUE(cuda(5, 3).supports_fp16());
    EXPECT_TRUE(cuda(6, 0).supports_fp16());
    EXPECT_TRUE(cuda(8, 9).supports_fp16());
}

TEST(CudaCapabilities, FlagsByArchitecture) {
    expect_flags(cuda(5, 0), {false, true, true, false});
    expect_flags(cuda(6, 0), {true, true, true, false});
    expect_flags(cuda(6, 1), {true, true, true, false});
    expect_flags(cuda(7, 0), {true, true, true, true});
    expect_flags(cuda(8, 9), {true, true, true, true});
}

TEST(CudaCapabilities, UnknownComputeCapabilityReportsNothing) {
    expect_flags(cuda(0, 0), {false, false, false, false});
}

// --- ROCm -------------------------------------------------------------------------------------

Device rocm(const char* arch) {
    Device d;
    gpgpu::backends::set_rocm_capabilities(arch, d);
    return d;
}

TEST(RocmCapabilities, MatrixCoresByArchitecture) {
    for (const char* arch : {"gfx908", "gfx90a:sramecc+:xnack-", "gfx940", "gfx942:sramecc+:xnack-", "gfx950",
                             "gfx1100", "gfx1103", "gfx1151", "gfx1200", "gfx1201"}) {
        SCOPED_TRACE(arch);
        expect_flags(rocm(arch), {true, true, true, true});
    }
    for (const char* arch : {"gfx803", "gfx900", "gfx906:sramecc+:xnack-", "gfx1010", "gfx1030", "gfx1036"}) {
        SCOPED_TRACE(arch);
        expect_flags(rocm(arch), {true, true, true, false});
    }
}

TEST(RocmCapabilities, UnknownArchitectureKeepsArithmeticFlags) {
    expect_flags(rocm(""), {true, true, true, false});
}

} // namespace
