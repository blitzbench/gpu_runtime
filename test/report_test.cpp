#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "backends/probe.hpp"
#include "gpgpu/runtime.hpp"
#include "identity.hpp"

namespace {

using gpgpu::Backend;
using gpgpu::BackendId;
using gpgpu::BackendStatus;
using gpgpu::Device;
using gpgpu::DeviceType;
using gpgpu::PciAddress;
using gpgpu::Preference;
using gpgpu::Vendor;
using gpgpu::backends::ProbeResult;

constexpr std::uint8_t kLlvmpipeUuid[16] = {0x6d, 0x65, 0x73, 0x61, 0x32, 0x36, 0x2e, 0x30,
                                            0x2e, 0x38, 0x2d, 0x31, 0x75, 0x62, 0x75, 0x00};
constexpr std::uint8_t kOtherUuid[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

Device pci_device(Vendor vendor, DeviceType type, PciAddress address, std::string name) {
    Device d;
    d.set_vendor(vendor).set_type(type).set_name(std::move(name)).set_pci_address(address);
    gpgpu::identity::assign_id(d, "unused");
    return d;
}

Device uuid_device(Vendor vendor, DeviceType type, const std::uint8_t (&uuid)[16], std::string name) {
    Device d;
    d.set_vendor(vendor).set_type(type).set_name(std::move(name));
    d.set_uuid(*gpgpu::identity::uuid_from(uuid));
    gpgpu::identity::assign_id(d, "unused");
    return d;
}

ProbeResult probe(BackendId id, std::vector<Device> devices) {
    ProbeResult p;
    p.backend = Backend(id, "1.0", "/lib", BackendStatus::Available, std::nullopt);
    p.devices = std::move(devices);
    return p;
}

const PciAddress kNvidia{0, 1, 0, 0};
const PciAddress kIntel{0, 0, 2, 0};

Device rtx() { return pci_device(Vendor::Nvidia, DeviceType::Discrete, kNvidia, "RTX 4070"); }
Device iris() { return pci_device(Vendor::Intel, DeviceType::Integrated, kIntel, "Iris Xe"); }
Device llvmpipe() { return uuid_device(Vendor::Unknown, DeviceType::Cpu, kLlvmpipeUuid, "llvmpipe"); }

// The machine of M0-38: Intel iGPU + NVIDIA dGPU with CUDA, Level Zero, OpenCL and Vulkan, and
// Mesa's llvmpipe on Vulkan. Probe order matches Runtime::query().
gpgpu::Report laptop() {
    std::vector<ProbeResult> probes;
    probes.push_back(probe(BackendId::CUDA, {rtx()}));
    probes.push_back(probe(BackendId::OneAPI, {iris()}));
    probes.push_back(probe(BackendId::OpenCL, {rtx(), iris()}));
    probes.push_back(probe(BackendId::Vulkan, {iris(), rtx(), llvmpipe()}));
    return gpgpu::backends::assemble_report(std::move(probes));
}

TEST(AssembleReport, SameGpuAcrossBackendsCollapsesToOnePreferredSetup) {
    const auto report = laptop();
    EXPECT_EQ(report.size(), 7u);
    EXPECT_EQ(report.devices().size(), 3u);

    const auto preferred = report.preferred();
    ASSERT_EQ(preferred.size(), 3u);
    EXPECT_EQ(preferred[0].device.id(), "pci-0000:01:00.0");
    EXPECT_EQ(preferred[0].backend.id(), BackendId::CUDA);
    EXPECT_EQ(preferred[0].preferred, Preference::Native);
    EXPECT_EQ(preferred[1].device.id(), "pci-0000:00:02.0");
    EXPECT_EQ(preferred[1].backend.id(), BackendId::OneAPI);
    EXPECT_EQ(preferred[1].preferred, Preference::Native);
    EXPECT_EQ(preferred[2].device.name(), "llvmpipe");
    EXPECT_EQ(preferred[2].backend.id(), BackendId::Vulkan);
    EXPECT_EQ(preferred[2].preferred, Preference::Fallback);
}

TEST(AssembleReport, NonPreferredSetupsOfTheSameGpuAreOther) {
    const auto report = laptop();
    for (const gpgpu::Setup& s : report.setups_for("pci-0000:01:00.0")) {
        EXPECT_EQ(s.preferred, s.backend.id() == BackendId::CUDA ? Preference::Native : Preference::Other);
    }
    EXPECT_EQ(report.setups_for("pci-0000:01:00.0").size(), 3u);
    EXPECT_EQ(report.setups_for("pci-0000:00:02.0").size(), 3u);
}

TEST(AssembleReport, CpuRendererIsFlaggedNotDropped) {
    const auto report = laptop();
    std::size_t cpu = 0;
    for (const gpgpu::Setup& s : report.preferred()) {
        if (s.device.type() == DeviceType::Cpu) {
            ++cpu;
            EXPECT_EQ(s.device.name(), "llvmpipe");
        }
    }
    EXPECT_EQ(cpu, 1u);
}

TEST(AssembleReport, DeviceWithoutPciAddressStaysSeparate) {
    std::vector<ProbeResult> probes;
    probes.push_back(probe(BackendId::OpenCL, {rtx()}));
    probes.push_back(probe(BackendId::Vulkan,
                           {uuid_device(Vendor::Nvidia, DeviceType::Discrete, kOtherUuid, "RTX 4070")}));
    const auto report = gpgpu::backends::assemble_report(std::move(probes));

    ASSERT_EQ(report.devices().size(), 2u);
    EXPECT_EQ(report.devices()[1].id(), "uuid-01020304-0506-0708-090a-0b0c0d0e0f10");
    const auto preferred = report.preferred();
    ASSERT_EQ(preferred.size(), 2u);
    EXPECT_EQ(preferred[0].preferred, Preference::Fallback);
    EXPECT_EQ(preferred[1].preferred, Preference::Fallback);
}

TEST(AssembleReport, WithoutNativeBackendOpenClBeatsVulkan) {
    std::vector<ProbeResult> probes;
    probes.push_back(probe(BackendId::OpenCL, {iris()}));
    probes.push_back(probe(BackendId::Vulkan, {iris()}));
    const auto report = gpgpu::backends::assemble_report(std::move(probes));

    const auto preferred = report.preferred();
    ASSERT_EQ(preferred.size(), 1u);
    EXPECT_EQ(preferred[0].backend.id(), BackendId::OpenCL);
    EXPECT_EQ(preferred[0].preferred, Preference::Fallback);
    EXPECT_EQ(report.by_backend(BackendId::Vulkan)[0].preferred, Preference::Other);
}

TEST(AssembleReport, VulkanOnlyDeviceFallsBackToVulkan) {
    std::vector<ProbeResult> probes;
    probes.push_back(probe(BackendId::Vulkan, {rtx()}));
    const auto preferred = gpgpu::backends::assemble_report(std::move(probes)).preferred();
    ASSERT_EQ(preferred.size(), 1u);
    EXPECT_EQ(preferred[0].preferred, Preference::Fallback);
}

TEST(AssembleReport, NativePicksPrecedeFallbackPicks) {
    std::vector<ProbeResult> probes;
    probes.push_back(probe(BackendId::OpenCL, {iris()}));
    probes.push_back(probe(BackendId::Vulkan, {llvmpipe()}));
    probes.push_back(probe(BackendId::CUDA, {rtx()}));
    const auto preferred = gpgpu::backends::assemble_report(std::move(probes)).preferred();
    ASSERT_EQ(preferred.size(), 3u);
    EXPECT_EQ(preferred[0].backend.id(), BackendId::CUDA);
    EXPECT_EQ(preferred[1].backend.id(), BackendId::OpenCL);
    EXPECT_EQ(preferred[2].backend.id(), BackendId::Vulkan);
}

TEST(AssembleReport, KeepsEveryBackendIncludingEmptyOnes) {
    std::vector<ProbeResult> probes;
    probes.push_back(probe(BackendId::CUDA, {}));
    probes.push_back(probe(BackendId::Vulkan, {rtx()}));
    const auto report = gpgpu::backends::assemble_report(std::move(probes));
    EXPECT_EQ(report.backends().size(), 2u);
    EXPECT_EQ(report.size(), 1u);
}

} // namespace
