#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "gpgpu/device.hpp"

namespace gpgpu::identity {

// Sets Device::id() from the identity fields already on `device`, by the rule documented on
// Device::id(). `backend_id` is used only when neither a PCI address nor a UUID is known.
void assign_id(Device& device, std::string_view backend_id);

// "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", lower-case hex.
std::string format_uuid(const std::uint8_t (&bytes)[16]);

// A UUID of all zero bytes means "not reported"; returns nullopt for it.
std::optional<DeviceUuid> uuid_from(const std::uint8_t (&bytes)[16]) noexcept;

// Parses the canonical "DDDD:BB:DD.F" form (as hipDeviceGetPCIBusId prints it). The domain may
// be omitted ("BB:DD.F"), which means domain 0.
std::optional<PciAddress> parse_pci_address(std::string_view text) noexcept;

} // namespace gpgpu::identity
