#include "identity.hpp"

#include <algorithm>
#include <cstdio>

namespace gpgpu::identity {

namespace {

std::optional<std::uint32_t> parse_hex(std::string_view text, std::size_t max_digits) noexcept {
    if (text.empty() || text.size() > max_digits) return std::nullopt;
    std::uint32_t value = 0;
    for (char c : text) {
        std::uint32_t digit = 0;
        if (c >= '0' && c <= '9') digit = static_cast<std::uint32_t>(c - '0');
        else if (c >= 'a' && c <= 'f') digit = static_cast<std::uint32_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') digit = static_cast<std::uint32_t>(c - 'A' + 10);
        else return std::nullopt;
        value = value * 16 + digit;
    }
    return value;
}

} // namespace

void assign_id(Device& device, std::string_view backend_id) {
    if (const auto& pci = device.pci_address()) {
        device.set_id("pci-" + to_string(*pci));
    } else if (const auto& uuid = device.uuid()) {
        std::uint8_t bytes[16];
        std::copy(uuid->begin(), uuid->end(), bytes);
        device.set_id("uuid-" + format_uuid(bytes));
    } else {
        device.set_id(std::string(backend_id));
    }
}

std::string format_uuid(const std::uint8_t (&u)[16]) {
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                  u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

std::optional<DeviceUuid> uuid_from(const std::uint8_t (&bytes)[16]) noexcept {
    DeviceUuid uuid{};
    std::copy(std::begin(bytes), std::end(bytes), uuid.begin());
    if (std::all_of(uuid.begin(), uuid.end(), [](std::uint8_t b) { return b == 0; })) {
        return std::nullopt;
    }
    return uuid;
}

std::optional<PciAddress> parse_pci_address(std::string_view text) noexcept {
    PciAddress address;
    const auto dot = text.rfind('.');
    if (dot == std::string_view::npos) return std::nullopt;
    const auto function = parse_hex(text.substr(dot + 1), 1);
    if (!function || *function > 7) return std::nullopt;
    text = text.substr(0, dot);

    const auto last_colon = text.rfind(':');
    if (last_colon == std::string_view::npos) return std::nullopt;
    const auto device = parse_hex(text.substr(last_colon + 1), 2);
    if (!device || *device > 0x1f) return std::nullopt;
    text = text.substr(0, last_colon);

    std::optional<std::uint32_t> domain = 0;
    const auto first_colon = text.rfind(':');
    if (first_colon != std::string_view::npos) {
        domain = parse_hex(text.substr(0, first_colon), 8);
        text = text.substr(first_colon + 1);
    }
    const auto bus = parse_hex(text, 2);
    if (!domain || !bus) return std::nullopt;

    address.domain = *domain;
    address.bus = *bus;
    address.device = *device;
    address.function = *function;
    return address;
}

} // namespace gpgpu::identity
