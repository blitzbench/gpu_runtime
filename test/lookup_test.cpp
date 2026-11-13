#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <CL/cl_ext.h>

#include "backends/device_identity.hpp"
#include "gpgpu/vendor.hpp"

// Fake vendor enumerations. Each fake driver keeps its devices in a static table; a native handle
// is the address of the device's table entry. The lookups are exercised through the same function
// tables a consumer passes, and every expectation is phrased in the id a probe would produce.

namespace {

using gpgpu::Device;
using gpgpu::DeviceType;
using gpgpu::PciAddress;

// --- Vulkan -----------------------------------------------------------------------------------

struct FakeVkDevice {
    VkPhysicalDeviceProperties props;
    bool has_pci_ext;
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci;
    std::uint8_t uuid[VK_UUID_SIZE];
};

std::vector<FakeVkDevice> g_vk_devices;
VkResult g_vk_enumerate_result = VK_SUCCESS;
bool g_vk_pci_chained_without_extension = false;

const FakeVkDevice& vk_fake(VkPhysicalDevice pd) {
    return *reinterpret_cast<const FakeVkDevice*>(pd);
}

VkPhysicalDevice vk_handle(std::size_t i) {
    return reinterpret_cast<VkPhysicalDevice>(&g_vk_devices[i]);
}

VKAPI_ATTR VkResult VKAPI_CALL fake_vkEnumeratePhysicalDevices(VkInstance, std::uint32_t* count,
                                                               VkPhysicalDevice* out) {
    if (g_vk_enumerate_result != VK_SUCCESS) return g_vk_enumerate_result;
    if (!out) {
        *count = static_cast<std::uint32_t>(g_vk_devices.size());
        return VK_SUCCESS;
    }
    for (std::uint32_t i = 0; i < *count && i < g_vk_devices.size(); ++i) out[i] = vk_handle(i);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_vkGetPhysicalDeviceProperties(VkPhysicalDevice pd,
                                                              VkPhysicalDeviceProperties* out) {
    *out = vk_fake(pd).props;
}

VKAPI_ATTR void VKAPI_CALL fake_vkGetPhysicalDeviceProperties2(VkPhysicalDevice pd,
                                                               VkPhysicalDeviceProperties2* out) {
    const FakeVkDevice& d = vk_fake(pd);
    out->properties = d.props;
    for (auto* next = static_cast<VkBaseOutStructure*>(out->pNext); next; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES) {
            std::memcpy(reinterpret_cast<VkPhysicalDeviceIDProperties*>(next)->deviceUUID, d.uuid,
                        VK_UUID_SIZE);
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT) {
            if (!d.has_pci_ext) g_vk_pci_chained_without_extension = true;
            auto* pci = reinterpret_cast<VkPhysicalDevicePCIBusInfoPropertiesEXT*>(next);
            pci->pciDomain = d.pci.pciDomain;
            pci->pciBus = d.pci.pciBus;
            pci->pciDevice = d.pci.pciDevice;
            pci->pciFunction = d.pci.pciFunction;
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL fake_vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice pd, const char*, std::uint32_t* count, VkExtensionProperties* out) {
    const bool has = vk_fake(pd).has_pci_ext;
    if (!out) {
        *count = has ? 1 : 0;
        return VK_SUCCESS;
    }
    if (has && *count >= 1) {
        out[0] = VkExtensionProperties{};
        std::strncpy(out[0].extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME, VK_MAX_EXTENSION_NAME_SIZE - 1);
    }
    return VK_SUCCESS;
}

FakeVkDevice vk_device(std::uint32_t vendor, std::uint32_t device, VkPhysicalDeviceType type,
                       std::uint32_t api, bool pci_ext, PciAddress address, std::uint8_t uuid_seed) {
    FakeVkDevice d{};
    d.props.vendorID = vendor;
    d.props.deviceID = device;
    d.props.deviceType = type;
    d.props.apiVersion = api;
    d.props.pipelineCacheUUID[0] = 0xcc;
    d.has_pci_ext = pci_ext;
    d.pci.pciDomain = address.domain;
    d.pci.pciBus = address.bus;
    d.pci.pciDevice = address.device;
    d.pci.pciFunction = address.function;
    for (std::uint8_t i = 0; i < VK_UUID_SIZE; ++i) d.uuid[i] = static_cast<std::uint8_t>(uuid_seed + i);
    return d;
}

gpgpu::vendor::VulkanInstanceFns vk_fns(bool with_properties2) {
    gpgpu::vendor::VulkanInstanceFns fns{};
    fns.vkEnumeratePhysicalDevices = fake_vkEnumeratePhysicalDevices;
    fns.vkGetPhysicalDeviceProperties = fake_vkGetPhysicalDeviceProperties;
    fns.vkGetPhysicalDeviceProperties2 = with_properties2 ? fake_vkGetPhysicalDeviceProperties2 : nullptr;
    fns.vkEnumerateDeviceExtensionProperties = fake_vkEnumerateDeviceExtensionProperties;
    return fns;
}

Device read_vk(const gpgpu::vendor::VulkanInstanceFns& fns, std::size_t i) {
    Device d;
    gpgpu::backends::read_vulkan_identity(
        {fns.vkGetPhysicalDeviceProperties, fns.vkGetPhysicalDeviceProperties2,
         fns.vkEnumerateDeviceExtensionProperties},
        vk_handle(i), d);
    return d;
}

class VulkanLookup : public ::testing::Test {
protected:
    void SetUp() override {
        g_vk_enumerate_result = VK_SUCCESS;
        g_vk_pci_chained_without_extension = false;
        g_vk_devices = {
            vk_device(0x8086, 0xa7a0, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, VK_API_VERSION_1_3, true,
                      PciAddress{0, 0, 2, 0}, 0x10),
            vk_device(0x10de, 0x2820, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, VK_API_VERSION_1_3, true,
                      PciAddress{0, 1, 0, 0}, 0x20),
            vk_device(0x10005, 0x0000, VK_PHYSICAL_DEVICE_TYPE_CPU, VK_API_VERSION_1_3, false,
                      PciAddress{}, 0x30),
        };
    }
};

TEST_F(VulkanLookup, IdentityUsesPciBusInfo) {
    const Device d = read_vk(vk_fns(true), 1);
    EXPECT_EQ(d.id(), "pci-0000:01:00.0");
    ASSERT_TRUE(d.pci_address().has_value());
    EXPECT_EQ(*d.pci_address(), (PciAddress{0, 1, 0, 0}));
    EXPECT_EQ(d.pci_vendor_id(), std::optional<std::uint16_t>(0x10de));
    EXPECT_EQ(d.pci_device_id(), std::optional<std::uint16_t>(0x2820));
    EXPECT_EQ(d.type(), DeviceType::Discrete);
}

TEST_F(VulkanLookup, CpuRendererIsFlaggedAndKeyedByUuid) {
    const Device d = read_vk(vk_fns(true), 2);
    EXPECT_EQ(d.type(), DeviceType::Cpu);
    EXPECT_FALSE(d.pci_address().has_value());
    EXPECT_FALSE(d.pci_vendor_id().has_value());
    EXPECT_EQ(d.id(), "uuid-30313233-3435-3637-3839-3a3b3c3d3e3f");
}

TEST_F(VulkanLookup, PciBusInfoIsNotChainedWithoutTheExtension) {
    const Device d = read_vk(vk_fns(true), 2);
    EXPECT_FALSE(g_vk_pci_chained_without_extension);
    EXPECT_FALSE(d.pci_address().has_value());
}

TEST_F(VulkanLookup, EnumerationFailureFindsNothing) {
    const Device target = read_vk(vk_fns(true), 1);
    g_vk_enumerate_result = VK_ERROR_INITIALIZATION_FAILED;
    EXPECT_EQ(gpgpu::vendor::find_vulkan_device(vk_fns(true), VK_NULL_HANDLE, target), VK_NULL_HANDLE);
}

TEST_F(VulkanLookup, NoDevicesFindsNothing) {
    const Device target = read_vk(vk_fns(true), 1);
    g_vk_devices.clear();
    EXPECT_EQ(gpgpu::vendor::find_vulkan_device(vk_fns(true), VK_NULL_HANDLE, target), VK_NULL_HANDLE);
}

TEST_F(VulkanLookup, SameGpuOnTwoDriversResolvesTheFirst) {
    g_vk_devices.push_back(g_vk_devices[1]);
    g_vk_devices.back().props.driverVersion = 2;
    const auto fns = vk_fns(true);
    ASSERT_EQ(read_vk(fns, 3).id(), read_vk(fns, 1).id());
    EXPECT_EQ(gpgpu::vendor::find_vulkan_device(fns, VK_NULL_HANDLE, read_vk(fns, 3)), vk_handle(1));
}

TEST_F(VulkanLookup, WithoutProperties2FallsBackToBackendId) {
    const Device d = read_vk(vk_fns(false), 0);
    EXPECT_FALSE(d.pci_address().has_value());
    EXPECT_FALSE(d.uuid().has_value());
    EXPECT_EQ(d.id().rfind("vk-8086:a7a0-cc000000-", 0), 0u) << d.id();
}

TEST_F(VulkanLookup, Vulkan10DeviceFallsBackToBackendId) {
    g_vk_devices[0].props.apiVersion = VK_API_VERSION_1_0;
    const Device d = read_vk(vk_fns(true), 0);
    EXPECT_FALSE(d.pci_address().has_value());
    EXPECT_EQ(d.id().rfind("vk-", 0), 0u);
}

TEST_F(VulkanLookup, FindsDeviceByProbeId) {
    const auto fns = vk_fns(true);
    for (std::size_t i = 0; i < g_vk_devices.size(); ++i) {
        EXPECT_EQ(gpgpu::vendor::find_vulkan_device(fns, VK_NULL_HANDLE, read_vk(fns, i)), vk_handle(i));
    }
}

TEST_F(VulkanLookup, FindsDeviceIdentifiedByAnotherBackend) {
    Device cuda;
    cuda.set_pci_address(PciAddress{0, 1, 0, 0});
    cuda.set_id("pci-0000:01:00.0");
    EXPECT_EQ(gpgpu::vendor::find_vulkan_device(vk_fns(true), VK_NULL_HANDLE, cuda), vk_handle(1));
}

TEST_F(VulkanLookup, UnknownDeviceIsNotFound) {
    Device other;
    other.set_id("pci-0000:05:00.0");
    EXPECT_EQ(gpgpu::vendor::find_vulkan_device(vk_fns(true), VK_NULL_HANDLE, other), VK_NULL_HANDLE);
}

// --- Level Zero ------------------------------------------------------------------------------

struct FakeZeDevice {
    ze_device_properties_t props;
    bool has_pci;
    ze_pci_address_ext_t address;
};

std::vector<FakeZeDevice> g_ze_devices;
int g_ze_driver_tag = 0;
ze_result_t g_ze_driver_result = ZE_RESULT_SUCCESS;

ze_device_handle_t ze_handle(std::size_t i) {
    return reinterpret_cast<ze_device_handle_t>(&g_ze_devices[i]);
}

ze_result_t ZE_APICALL fake_zeDriverGet(std::uint32_t* count, ze_driver_handle_t* out) {
    if (g_ze_driver_result != ZE_RESULT_SUCCESS) return g_ze_driver_result;
    if (!out) {
        *count = 1;
        return ZE_RESULT_SUCCESS;
    }
    out[0] = reinterpret_cast<ze_driver_handle_t>(&g_ze_driver_tag);
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL fake_zeDeviceGet(ze_driver_handle_t, std::uint32_t* count, ze_device_handle_t* out) {
    if (!out) {
        *count = static_cast<std::uint32_t>(g_ze_devices.size());
        return ZE_RESULT_SUCCESS;
    }
    for (std::uint32_t i = 0; i < *count && i < g_ze_devices.size(); ++i) out[i] = ze_handle(i);
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL fake_zeDeviceGetProperties(ze_device_handle_t h, ze_device_properties_t* out) {
    *out = reinterpret_cast<const FakeZeDevice*>(h)->props;
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL fake_zeDevicePciGetPropertiesExt(ze_device_handle_t h, ze_pci_ext_properties_t* out) {
    const auto* d = reinterpret_cast<const FakeZeDevice*>(h);
    if (!d->has_pci) return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
    out->address = d->address;
    return ZE_RESULT_SUCCESS;
}

FakeZeDevice ze_device(std::uint32_t device_id, ze_device_type_t type, bool integrated, bool has_pci,
                       PciAddress address, std::uint8_t uuid_seed) {
    FakeZeDevice d{};
    d.props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    d.props.type = type;
    d.props.vendorId = 0x8086;
    d.props.deviceId = device_id;
    d.props.flags = integrated ? ZE_DEVICE_PROPERTY_FLAG_INTEGRATED : 0;
    for (std::uint8_t i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; ++i) {
        d.props.uuid.id[i] = static_cast<std::uint8_t>(uuid_seed ? uuid_seed + i : 0);
    }
    d.has_pci = has_pci;
    d.address = ze_pci_address_ext_t{address.domain, address.bus, address.device, address.function};
    return d;
}

gpgpu::vendor::LevelZeroFns ze_fns(bool with_pci) {
    gpgpu::vendor::LevelZeroFns fns{};
    fns.zeDriverGet = fake_zeDriverGet;
    fns.zeDeviceGet = fake_zeDeviceGet;
    fns.zeDeviceGetProperties = fake_zeDeviceGetProperties;
    fns.zeDevicePciGetPropertiesExt = with_pci ? fake_zeDevicePciGetPropertiesExt : nullptr;
    return fns;
}

Device read_ze(const gpgpu::vendor::LevelZeroFns& fns, std::size_t i) {
    Device d;
    gpgpu::backends::read_level_zero_identity(fns.zeDevicePciGetPropertiesExt, ze_handle(i),
                                              g_ze_devices[i].props, static_cast<std::uint32_t>(i), d);
    return d;
}

class LevelZeroLookup : public ::testing::Test {
protected:
    void SetUp() override {
        g_ze_driver_result = ZE_RESULT_SUCCESS;
        g_ze_devices = {
            ze_device(0xa7a0, ZE_DEVICE_TYPE_GPU, true, true, PciAddress{0, 0, 2, 0}, 0x80),
            ze_device(0x56a0, ZE_DEVICE_TYPE_GPU, false, false, PciAddress{}, 0x40),
            ze_device(0x56a0, ZE_DEVICE_TYPE_GPU, false, false, PciAddress{}, 0),
        };
    }
};

TEST_F(LevelZeroLookup, IdentityUsesPciProperties) {
    const Device d = read_ze(ze_fns(true), 0);
    EXPECT_EQ(d.id(), "pci-0000:00:02.0");
    EXPECT_EQ(d.type(), DeviceType::Integrated);
    EXPECT_EQ(d.pci_vendor_id(), std::optional<std::uint16_t>(0x8086));
    EXPECT_EQ(d.pci_device_id(), std::optional<std::uint16_t>(0xa7a0));
}

TEST_F(LevelZeroLookup, WithoutPciExtensionUsesUuid) {
    const Device d = read_ze(ze_fns(false), 0);
    EXPECT_FALSE(d.pci_address().has_value());
    EXPECT_EQ(d.id(), "uuid-80818283-8485-8687-8889-8a8b8c8d8e8f");
}

TEST_F(LevelZeroLookup, WithoutPciAndUuidUsesOrdinal) {
    const Device d = read_ze(ze_fns(true), 2);
    EXPECT_EQ(d.id(), "ze-8086:56a0-2");
    EXPECT_EQ(d.type(), DeviceType::Discrete);
}

TEST_F(LevelZeroLookup, FindsEveryDeviceByProbeId) {
    const auto fns = ze_fns(true);
    for (std::size_t i = 0; i < g_ze_devices.size(); ++i) {
        ze_driver_handle_t driver = nullptr;
        ze_device_handle_t device = nullptr;
        ASSERT_TRUE(gpgpu::vendor::find_level_zero_device(fns, read_ze(fns, i), driver, device));
        EXPECT_EQ(device, ze_handle(i));
        EXPECT_EQ(driver, reinterpret_cast<ze_driver_handle_t>(&g_ze_driver_tag));
    }
}

TEST_F(LevelZeroLookup, DriverEnumerationFailureFindsNothing) {
    const Device target = read_ze(ze_fns(true), 0);
    g_ze_driver_result = ZE_RESULT_ERROR_UNINITIALIZED;
    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    EXPECT_FALSE(gpgpu::vendor::find_level_zero_device(ze_fns(true), target, driver, device));
    EXPECT_EQ(device, nullptr);
}

TEST_F(LevelZeroLookup, NoDevicesFindsNothing) {
    const Device target = read_ze(ze_fns(true), 0);
    g_ze_devices.clear();
    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    EXPECT_FALSE(gpgpu::vendor::find_level_zero_device(ze_fns(true), target, driver, device));
}

TEST_F(LevelZeroLookup, UnknownDeviceLeavesOutputsUntouched) {
    Device other;
    other.set_id("pci-0000:05:00.0");
    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    EXPECT_FALSE(gpgpu::vendor::find_level_zero_device(ze_fns(true), other, driver, device));
    EXPECT_EQ(driver, nullptr);
    EXPECT_EQ(device, nullptr);
}

// --- OpenCL ----------------------------------------------------------------------------------

// The PCIe layout of cl_device_topology_amd, as AMD's drivers write it.
struct AmdTopology {
    cl_uint type;
    char unused[17];
    char bus;
    char device;
    char function;
};

constexpr cl_device_info kNvBusId    = 0x4008;
constexpr cl_device_info kNvSlotId   = 0x4009;
constexpr cl_device_info kNvDomainId = 0x400A;
constexpr cl_device_info kAmdTopology = 0x4037;

struct FakeClDevice {
    cl_device_type type;
    cl_bool unified;
    cl_uint vendor_id;
    std::string extensions;
    cl_device_pci_bus_info_khr pci;
    std::uint8_t uuid[CL_UUID_SIZE_KHR];
    cl_uint nv_bus;
    cl_uint nv_slot;
    std::optional<cl_uint> nv_domain;
    AmdTopology amd;
};

struct FakeClPlatform {
    std::string name;
    std::vector<FakeClDevice> devices;
};

std::vector<FakeClPlatform> g_cl_platforms;
cl_int g_cl_platform_result = CL_SUCCESS;

cl_int CL_API_CALL fake_clGetPlatformIDs(cl_uint n, cl_platform_id* out, cl_uint* count) {
    if (g_cl_platform_result != CL_SUCCESS) return g_cl_platform_result;
    if (count) *count = static_cast<cl_uint>(g_cl_platforms.size());
    for (cl_uint i = 0; out && i < n && i < g_cl_platforms.size(); ++i) {
        out[i] = reinterpret_cast<cl_platform_id>(&g_cl_platforms[i]);
    }
    return CL_SUCCESS;
}

cl_int CL_API_CALL fake_clGetPlatformInfo(cl_platform_id p, cl_platform_info key, std::size_t size,
                                          void* value, std::size_t* size_ret) {
    const auto* platform = reinterpret_cast<const FakeClPlatform*>(p);
    const std::string text = key == CL_PLATFORM_NAME ? platform->name : std::string("OpenCL 3.0 fake");
    if (size_ret) *size_ret = text.size() + 1;
    if (value && size >= text.size() + 1) std::memcpy(value, text.c_str(), text.size() + 1);
    return CL_SUCCESS;
}

cl_int CL_API_CALL fake_clGetDeviceIDs(cl_platform_id p, cl_device_type, cl_uint n, cl_device_id* out,
                                       cl_uint* count) {
    auto* platform = reinterpret_cast<FakeClPlatform*>(p);
    if (count) *count = static_cast<cl_uint>(platform->devices.size());
    for (cl_uint i = 0; out && i < n && i < platform->devices.size(); ++i) {
        out[i] = reinterpret_cast<cl_device_id>(&platform->devices[i]);
    }
    return CL_SUCCESS;
}

template <typename T>
cl_int copy_out(const T& v, std::size_t size, void* value, std::size_t* size_ret) {
    if (size_ret) *size_ret = sizeof(T);
    if (value) {
        if (size < sizeof(T)) return CL_INVALID_VALUE;
        std::memcpy(value, &v, sizeof(T));
    }
    return CL_SUCCESS;
}

cl_int CL_API_CALL fake_clGetDeviceInfo(cl_device_id dev, cl_device_info key, std::size_t size,
                                        void* value, std::size_t* size_ret) {
    const auto* d = reinterpret_cast<const FakeClDevice*>(dev);
    switch (key) {
        case CL_DEVICE_TYPE:                return copy_out(d->type, size, value, size_ret);
        case CL_DEVICE_HOST_UNIFIED_MEMORY: return copy_out(d->unified, size, value, size_ret);
        case CL_DEVICE_VENDOR_ID:           return copy_out(d->vendor_id, size, value, size_ret);
        case CL_DEVICE_PCI_BUS_INFO_KHR:    return copy_out(d->pci, size, value, size_ret);
        case CL_DEVICE_UUID_KHR:            return copy_out(d->uuid, size, value, size_ret);
        case kNvBusId:                      return copy_out(d->nv_bus, size, value, size_ret);
        case kNvSlotId:                     return copy_out(d->nv_slot, size, value, size_ret);
        case kNvDomainId:
            return d->nv_domain ? copy_out(*d->nv_domain, size, value, size_ret) : CL_INVALID_VALUE;
        case kAmdTopology:                  return copy_out(d->amd, size, value, size_ret);
        case CL_DEVICE_EXTENSIONS:
            if (size_ret) *size_ret = d->extensions.size() + 1;
            if (value && size >= d->extensions.size() + 1) {
                std::memcpy(value, d->extensions.c_str(), d->extensions.size() + 1);
            }
            return CL_SUCCESS;
        default:
            return CL_INVALID_VALUE;
    }
}

FakeClDevice cl_device(cl_device_type type, cl_bool unified, cl_uint vendor, std::string extensions,
                       PciAddress address, std::uint8_t uuid_seed) {
    FakeClDevice d{};
    d.type = type;
    d.unified = unified;
    d.vendor_id = vendor;
    d.extensions = std::move(extensions);
    d.pci = cl_device_pci_bus_info_khr{address.domain, address.bus, address.device, address.function};
    for (std::uint8_t i = 0; i < CL_UUID_SIZE_KHR; ++i) d.uuid[i] = static_cast<std::uint8_t>(uuid_seed + i);
    return d;
}

gpgpu::vendor::OpenClFns cl_fns() {
    gpgpu::vendor::OpenClFns fns{};
    fns.clGetPlatformIDs = fake_clGetPlatformIDs;
    fns.clGetPlatformInfo = fake_clGetPlatformInfo;
    fns.clGetDeviceIDs = fake_clGetDeviceIDs;
    fns.clGetDeviceInfo = fake_clGetDeviceInfo;
    return fns;
}

cl_device_id cl_handle(std::size_t platform, std::size_t device) {
    return reinterpret_cast<cl_device_id>(&g_cl_platforms[platform].devices[device]);
}

Device read_cl(std::size_t platform, std::size_t device) {
    Device d;
    gpgpu::backends::read_opencl_identity(fake_clGetDeviceInfo, cl_handle(platform, device), d);
    return d;
}

class OpenClLookup : public ::testing::Test {
protected:
    void SetUp() override {
        g_cl_platform_result = CL_SUCCESS;
        g_cl_platforms = {
            {"OpenCLOn12", {cl_device(CL_DEVICE_TYPE_GPU, CL_FALSE, 0x10de, "cl_khr_pci_bus_info",
                                      PciAddress{0, 1, 0, 0}, 0)}},
            {"NVIDIA CUDA", {cl_device(CL_DEVICE_TYPE_GPU, CL_FALSE, 0x10de,
                                       "cl_khr_fp64 cl_khr_pci_bus_info", PciAddress{0, 1, 0, 0}, 0)}},
            {"Portable Computing Language",
             {cl_device(CL_DEVICE_TYPE_CPU, CL_TRUE, 0x10006, "cl_khr_device_uuid", PciAddress{}, 0x50),
              cl_device(CL_DEVICE_TYPE_GPU, CL_TRUE, 0x8086, "", PciAddress{}, 0)}},
        };
    }
};

TEST_F(OpenClLookup, IdentityUsesKhrPciBusInfo) {
    const Device d = read_cl(1, 0);
    EXPECT_EQ(d.id(), "pci-0000:01:00.0");
    EXPECT_EQ(d.type(), DeviceType::Discrete);
    EXPECT_EQ(d.pci_vendor_id(), std::optional<std::uint16_t>(0x10de));
    EXPECT_FALSE(d.pci_device_id().has_value());
}

TEST_F(OpenClLookup, ExtensionMustBeAdvertised) {
    g_cl_platforms[1].devices[0].extensions = "cl_khr_pci_bus_info_extra";
    const Device d = read_cl(1, 0);
    EXPECT_FALSE(d.pci_address().has_value());
    EXPECT_EQ(d.id().rfind("ocl-10de-", 0), 0u) << d.id();
}

TEST_F(OpenClLookup, CpuDeviceIsFlaggedAndKeyedByUuid) {
    const Device d = read_cl(2, 0);
    EXPECT_EQ(d.type(), DeviceType::Cpu);
    EXPECT_FALSE(d.pci_vendor_id().has_value());
    EXPECT_EQ(d.id(), "uuid-50515253-5455-5657-5859-5a5b5c5d5e5f");
}

TEST_F(OpenClLookup, DeviceWithoutPciOrUuidGetsBackendId) {
    const Device d = read_cl(2, 1);
    EXPECT_EQ(d.type(), DeviceType::Integrated);
    EXPECT_EQ(d.id().rfind("ocl-8086-", 0), 0u) << d.id();
}

TEST_F(OpenClLookup, SkipsCompatibilityPlatform) {
    EXPECT_EQ(gpgpu::vendor::find_opencl_device(cl_fns(), read_cl(1, 0)), cl_handle(1, 0));
}

TEST_F(OpenClLookup, FindsEveryDeviceByProbeId) {
    EXPECT_EQ(gpgpu::vendor::find_opencl_device(cl_fns(), read_cl(2, 0)), cl_handle(2, 0));
    EXPECT_EQ(gpgpu::vendor::find_opencl_device(cl_fns(), read_cl(2, 1)), cl_handle(2, 1));
}

FakeClDevice nv_device(std::string extensions, std::optional<cl_uint> domain) {
    FakeClDevice d = cl_device(CL_DEVICE_TYPE_GPU, CL_FALSE, 0x10de, std::move(extensions), PciAddress{}, 0);
    d.nv_bus = 0x41;
    d.nv_slot = 0x0B;
    d.nv_domain = domain;
    return d;
}

FakeClDevice amd_device(cl_uint topology_type) {
    FakeClDevice d = cl_device(CL_DEVICE_TYPE_GPU, CL_FALSE, 0x1002, "cl_amd_device_attribute_query",
                               PciAddress{}, 0);
    d.amd.type = topology_type;
    d.amd.bus = static_cast<char>(0xc3);
    d.amd.device = 0;
    d.amd.function = 1;
    return d;
}

TEST_F(OpenClLookup, NvidiaSlotSplitsIntoDeviceAndFunction) {
    g_cl_platforms = {{"NVIDIA CUDA", {nv_device("cl_nv_device_attribute_query", 2)}}};
    const Device d = read_cl(0, 0);
    ASSERT_TRUE(d.pci_address().has_value());
    EXPECT_EQ(*d.pci_address(), (PciAddress{2, 0x41, 1, 3}));
    EXPECT_EQ(d.id(), "pci-0002:41:01.3");
}

TEST_F(OpenClLookup, NvidiaWithoutDomainQueryAssumesDomainZero) {
    g_cl_platforms = {{"NVIDIA CUDA", {nv_device("cl_nv_device_attribute_query", std::nullopt)}}};
    EXPECT_EQ(read_cl(0, 0).id(), "pci-0000:41:01.3");
}

TEST_F(OpenClLookup, NvidiaQueryNeedsTheExtension) {
    g_cl_platforms = {{"NVIDIA CUDA", {nv_device("cl_khr_fp64", 0)}}};
    EXPECT_FALSE(read_cl(0, 0).pci_address().has_value());
}

TEST_F(OpenClLookup, AmdPcieTopologyGivesAddressInDomainZero) {
    g_cl_platforms = {{"AMD Accelerated Parallel Processing", {amd_device(1)}}};
    const Device d = read_cl(0, 0);
    EXPECT_EQ(d.id(), "pci-0000:c3:00.1");
    EXPECT_EQ(d.pci_vendor_id(), std::optional<std::uint16_t>(0x1002));
}

TEST_F(OpenClLookup, AmdNonPcieTopologyGivesNoAddress) {
    g_cl_platforms = {{"AMD Accelerated Parallel Processing", {amd_device(2)}}};
    const Device d = read_cl(0, 0);
    EXPECT_FALSE(d.pci_address().has_value());
    EXPECT_EQ(d.id().rfind("ocl-1002-", 0), 0u) << d.id();
}

TEST_F(OpenClLookup, AmdTopologyNeedsTheExtension) {
    g_cl_platforms = {{"AMD Accelerated Parallel Processing", {amd_device(1)}}};
    g_cl_platforms[0].devices[0].extensions = "";
    EXPECT_FALSE(read_cl(0, 0).pci_address().has_value());
}

TEST_F(OpenClLookup, SameGpuOnTwoPlatformsResolvesTheFirst) {
    g_cl_platforms.push_back(g_cl_platforms[1]);
    g_cl_platforms.back().name = "rusticl";
    ASSERT_EQ(read_cl(3, 0).id(), read_cl(1, 0).id());
    EXPECT_EQ(gpgpu::vendor::find_opencl_device(cl_fns(), read_cl(3, 0)), cl_handle(1, 0));
}

TEST_F(OpenClLookup, PlatformEnumerationFailureFindsNothing) {
    const Device target = read_cl(1, 0);
    g_cl_platform_result = CL_PLATFORM_NOT_FOUND_KHR;
    EXPECT_EQ(gpgpu::vendor::find_opencl_device(cl_fns(), target), nullptr);
}

TEST_F(OpenClLookup, NoPlatformsFindsNothing) {
    const Device target = read_cl(1, 0);
    g_cl_platforms.clear();
    EXPECT_EQ(gpgpu::vendor::find_opencl_device(cl_fns(), target), nullptr);
}

TEST_F(OpenClLookup, UnknownDeviceIsNotFound) {
    Device other;
    other.set_id("pci-0000:05:00.0");
    EXPECT_EQ(gpgpu::vendor::find_opencl_device(cl_fns(), other), nullptr);
}

// --- CUDA ------------------------------------------------------------------------------------

struct FakeCuDevice {
    int domain, bus, slot, integrated;
    bool has_uuid;
    bool has_pci = true;
};

std::vector<FakeCuDevice> g_cu_devices;

gpgpu::backends::CUresult GPGPU_STDCALL fake_cuDeviceGetCount(int* n) {
    *n = static_cast<int>(g_cu_devices.size());
    return 0;
}

gpgpu::backends::CUresult GPGPU_STDCALL fake_cuDeviceGet(gpgpu::backends::CUdevice* d, int ordinal) {
    *d = 100 + ordinal;
    return 0;
}

gpgpu::backends::CUresult GPGPU_STDCALL fake_cuDeviceGetAttribute(int* v, int attribute,
                                                                 gpgpu::backends::CUdevice d) {
    const FakeCuDevice& dev = g_cu_devices[static_cast<std::size_t>(d - 100)];
    switch (attribute) {
        case 18: *v = dev.integrated; return 0;
        case 33: *v = dev.bus; return dev.has_pci ? 0 : 1;
        case 34: *v = dev.slot; return dev.has_pci ? 0 : 1;
        case 50: *v = dev.domain; return dev.has_pci ? 0 : 1;
        default: return 1;
    }
}

gpgpu::backends::CUresult GPGPU_STDCALL fake_cuDeviceGetUuid(unsigned char* out, gpgpu::backends::CUdevice d) {
    const FakeCuDevice& dev = g_cu_devices[static_cast<std::size_t>(d - 100)];
    if (!dev.has_uuid) return 1;
    for (int i = 0; i < 16; ++i) out[i] = static_cast<unsigned char>(0xa0 + i);
    return 0;
}

const gpgpu::backends::CudaIdentityFns kCudaFns{fake_cuDeviceGetCount, fake_cuDeviceGet,
                                                fake_cuDeviceGetAttribute, fake_cuDeviceGetUuid};

TEST(CudaLookup, IdentityUsesPciAttributes) {
    g_cu_devices = {{0, 0x41, 0, 0, true}, {0, 1, 0, 0, true}};
    Device d;
    gpgpu::backends::read_cuda_identity(kCudaFns, 101, 1, d);
    EXPECT_EQ(d.id(), "pci-0000:01:00.0");
    EXPECT_EQ(d.type(), DeviceType::Discrete);
    EXPECT_EQ(d.pci_vendor_id(), std::optional<std::uint16_t>(0x10de));
    EXPECT_TRUE(d.uuid().has_value());
}

TEST(CudaLookup, WithoutPciAttributesUsesUuidThenOrdinal) {
    g_cu_devices = {{0, 0, 0, 1, true, false}, {0, 0, 0, 0, false, false}};
    Device with_uuid;
    gpgpu::backends::read_cuda_identity(kCudaFns, 100, 0, with_uuid);
    EXPECT_FALSE(with_uuid.pci_address().has_value());
    EXPECT_FALSE(with_uuid.pci_vendor_id().has_value());
    EXPECT_EQ(with_uuid.id(), "uuid-a0a1a2a3-a4a5-a6a7-a8a9-aaabacadaeaf");
    EXPECT_EQ(with_uuid.type(), DeviceType::Integrated);

    Device bare;
    gpgpu::backends::read_cuda_identity(kCudaFns, 101, 1, bare);
    EXPECT_EQ(bare.id(), "cuda-1");
    EXPECT_EQ(gpgpu::backends::find_cuda_ordinal(kCudaFns, bare), std::optional<int>(1));
}

TEST(CudaLookup, FindsOrdinalOfProbeId) {
    g_cu_devices = {{0, 0x41, 0, 0, true}, {0, 1, 0, 0, true}};
    Device target;
    target.set_id("pci-0000:01:00.0");
    EXPECT_EQ(gpgpu::backends::find_cuda_ordinal(kCudaFns, target), std::optional<int>(1));
    target.set_id("pci-0000:02:00.0");
    EXPECT_FALSE(gpgpu::backends::find_cuda_ordinal(kCudaFns, target).has_value());
}

// --- HIP -------------------------------------------------------------------------------------

std::vector<std::string> g_hip_bus_ids;

gpgpu::backends::hipError_t GPGPU_STDCALL fake_hipGetDeviceCount(int* n) {
    *n = static_cast<int>(g_hip_bus_ids.size());
    return 0;
}

gpgpu::backends::hipError_t GPGPU_STDCALL fake_hipDeviceGetPCIBusId(char* out, int len,
                                                                   gpgpu::backends::hipDevice_t d) {
    const std::string& bdf = g_hip_bus_ids[static_cast<std::size_t>(d)];
    if (bdf.empty()) return 1;
    std::snprintf(out, static_cast<std::size_t>(len), "%s", bdf.c_str());
    return 0;
}

const gpgpu::backends::HipIdentityFns kHipFns{fake_hipGetDeviceCount, nullptr, fake_hipDeviceGetPCIBusId,
                                              nullptr};

TEST(HipLookup, IdentityParsesBusIdString) {
    g_hip_bus_ids = {"0000:03:00.0", ""};
    Device d;
    gpgpu::backends::read_hip_identity(kHipFns, 0, 0, d);
    EXPECT_EQ(d.id(), "pci-0000:03:00.0");
    EXPECT_EQ(d.pci_vendor_id(), std::optional<std::uint16_t>(0x1002));

    Device no_pci;
    gpgpu::backends::read_hip_identity(kHipFns, 1, 1, no_pci);
    EXPECT_EQ(no_pci.id(), "hip-1");
}

TEST(HipLookup, MalformedBusIdGivesNoAddress) {
    g_hip_bus_ids = {"0000:03:00", "not-a-bus-id"};
    for (int i = 0; i < 2; ++i) {
        Device d;
        gpgpu::backends::read_hip_identity(kHipFns, i, i, d);
        EXPECT_FALSE(d.pci_address().has_value());
        EXPECT_FALSE(d.pci_vendor_id().has_value());
        EXPECT_EQ(d.id(), "hip-" + std::to_string(i));
    }
}

TEST(HipLookup, FindsOrdinalOfProbeId) {
    g_hip_bus_ids = {"0000:03:00.0", "0000:83:00.0"};
    Device target;
    target.set_id("pci-0000:83:00.0");
    EXPECT_EQ(gpgpu::backends::find_hip_ordinal(kHipFns, target), std::optional<int>(1));
}

} // namespace
