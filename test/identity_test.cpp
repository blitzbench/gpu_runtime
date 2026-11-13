#include <gtest/gtest.h>

#include <cstdint>

#include "gpgpu/device.hpp"
#include "identity.hpp"

namespace {

using gpgpu::Device;
using gpgpu::DeviceUuid;
using gpgpu::PciAddress;

constexpr std::uint8_t kUuidBytes[16] = {0x10, 0xf8, 0x33, 0xea, 0x3c, 0x3c, 0x37, 0x78,
                                         0x60, 0xfb, 0xf5, 0x61, 0xdc, 0x40, 0x37, 0xd3};

TEST(PciAddress, FormatsCanonicalLowerCaseForm) {
    EXPECT_EQ(gpgpu::to_string(PciAddress{0, 1, 0, 0}), "0000:01:00.0");
    EXPECT_EQ(gpgpu::to_string(PciAddress{0x10, 0xab, 0x1f, 7}), "0010:ab:1f.7");
}

TEST(PciAddress, ComparesAllFourFields) {
    EXPECT_EQ((PciAddress{0, 1, 0, 0}), (PciAddress{0, 1, 0, 0}));
    EXPECT_NE((PciAddress{0, 1, 0, 0}), (PciAddress{0, 1, 0, 1}));
    EXPECT_NE((PciAddress{0, 1, 0, 0}), (PciAddress{1, 1, 0, 0}));
}

TEST(AssignId, PciAddressWinsOverUuid) {
    Device d;
    d.set_pci_address(PciAddress{0, 1, 0, 0});
    d.set_uuid(*gpgpu::identity::uuid_from(kUuidBytes));
    gpgpu::identity::assign_id(d, "vk-backend");
    EXPECT_EQ(d.id(), "pci-0000:01:00.0");
}

TEST(AssignId, UuidWinsOverBackendId) {
    Device d;
    d.set_uuid(*gpgpu::identity::uuid_from(kUuidBytes));
    gpgpu::identity::assign_id(d, "vk-backend");
    EXPECT_EQ(d.id(), "uuid-10f833ea-3c3c-3778-60fb-f561dc4037d3");
}

TEST(AssignId, BackendIdOnlyWithoutPciAndUuid) {
    Device d;
    gpgpu::identity::assign_id(d, "ocl-8086-0x1234");
    EXPECT_EQ(d.id(), "ocl-8086-0x1234");
}

TEST(UuidFrom, AllZeroMeansNotReported) {
    constexpr std::uint8_t zero[16] = {};
    EXPECT_FALSE(gpgpu::identity::uuid_from(zero).has_value());
    ASSERT_TRUE(gpgpu::identity::uuid_from(kUuidBytes).has_value());
    EXPECT_EQ((*gpgpu::identity::uuid_from(kUuidBytes))[0], 0x10);
}

TEST(ParsePciAddress, AcceptsCanonicalForm) {
    const auto a = gpgpu::identity::parse_pci_address("0000:03:00.0");
    ASSERT_TRUE(a.has_value());
    EXPECT_EQ(*a, (PciAddress{0, 3, 0, 0}));
    const auto b = gpgpu::identity::parse_pci_address("0001:C1:1F.7");
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(*b, (PciAddress{1, 0xc1, 0x1f, 7}));
}

TEST(ParsePciAddress, DomainDefaultsToZero) {
    const auto a = gpgpu::identity::parse_pci_address("03:00.1");
    ASSERT_TRUE(a.has_value());
    EXPECT_EQ(*a, (PciAddress{0, 3, 0, 1}));
}

TEST(ParsePciAddress, RejectsMalformedText) {
    EXPECT_FALSE(gpgpu::identity::parse_pci_address("").has_value());
    EXPECT_FALSE(gpgpu::identity::parse_pci_address("0000:03:00").has_value());
    EXPECT_FALSE(gpgpu::identity::parse_pci_address("0000:03:20.0").has_value());
    EXPECT_FALSE(gpgpu::identity::parse_pci_address("0000:03:00.8").has_value());
    EXPECT_FALSE(gpgpu::identity::parse_pci_address("0000:103:00.0").has_value());
    EXPECT_FALSE(gpgpu::identity::parse_pci_address("zz:00.0").has_value());
}

} // namespace
