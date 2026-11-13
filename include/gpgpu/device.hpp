#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace gpgpu {

enum class Vendor : std::uint8_t {
    Nvidia,
    AMD,
    Intel,
    Apple,
    Qualcomm,
    ARM,
    Unknown,
};

std::string_view to_string(Vendor v) noexcept;

// Best-effort vendor classification from a raw vendor string or PCI vendor ID.
Vendor classify_vendor(std::string_view raw) noexcept;
Vendor classify_vendor_pci(std::uint32_t pci_vendor_id) noexcept;

// PCI bus address (domain:bus:device.function) of a device attached over PCI.
struct PciAddress {
    std::uint32_t domain{0};
    std::uint32_t bus{0};
    std::uint32_t device{0};
    std::uint32_t function{0};
};

bool operator==(const PciAddress& a, const PciAddress& b) noexcept;
bool operator!=(const PciAddress& a, const PciAddress& b) noexcept;

// Canonical "DDDD:BB:DD.F" form, lower-case hex, as Linux sysfs and lspci -D print it.
std::string to_string(const PciAddress& address);

// Device UUID as reported by the backend (Vulkan deviceUUID, Level Zero / CUDA / OpenCL UUID).
// UUIDs are only guaranteed to be stable within one backend; across backends only the PCI address
// identifies a device.
using DeviceUuid = std::array<std::uint8_t, 16>;

enum class DeviceType : std::uint8_t {
    Discrete,
    Integrated,
    Virtual,
    Cpu,      // a software implementation on the host CPU (Mesa llvmpipe, pocl, ...)
    Unknown,
};

std::string_view to_string(DeviceType t) noexcept;

class Device {
public:
    Device() = default;

    // --- stable identity ---
    // Identifies the physical device. Every backend derives it by the same rule, so a device seen
    // by several backends has the same id in each of them whenever its PCI address is known:
    //   1. "pci-DDDD:BB:DD.F" when the backend reports the PCI address;
    //   2. otherwise "uuid-<8-4-4-4-12 hex>" when it reports a non-zero device UUID;
    //   3. otherwise a backend-specific id that only identifies the device within that backend.
    // Use the lookup helpers in gpgpu/vendor.hpp to find a backend's native handle for a Device;
    // never re-derive the id format.
    [[nodiscard]] const std::string& id() const noexcept { return id_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] Vendor vendor() const noexcept { return vendor_; }
    [[nodiscard]] const std::string& vendor_string() const noexcept { return vendor_string_; }

    // --- PCI and kind ---
    // Absent when the device is not on PCI or the backend cannot report it.
    [[nodiscard]] const std::optional<PciAddress>& pci_address() const noexcept { return pci_address_; }
    [[nodiscard]] std::optional<std::uint16_t> pci_vendor_id() const noexcept { return pci_vendor_id_; }
    [[nodiscard]] std::optional<std::uint16_t> pci_device_id() const noexcept { return pci_device_id_; }
    [[nodiscard]] const std::optional<DeviceUuid>& uuid() const noexcept { return uuid_; }
    [[nodiscard]] DeviceType type() const noexcept { return type_; }

    // --- memory ---
    [[nodiscard]] std::optional<std::uint64_t> memory() const noexcept { return memory_; }
    [[nodiscard]] std::optional<std::uint64_t> memory_bandwidth_gbps() const noexcept { return memory_bandwidth_gbps_; }
    [[nodiscard]] std::optional<std::uint64_t> l2_cache_bytes() const noexcept { return l2_cache_bytes_; }
    [[nodiscard]] std::optional<std::uint32_t> memory_bus_width_bits() const noexcept { return memory_bus_width_bits_; }

    // --- compute units ---
    [[nodiscard]] std::optional<std::uint32_t> cores() const noexcept { return cores_; }
    [[nodiscard]] std::optional<std::uint32_t> compute_units() const noexcept { return compute_units_; }
    [[nodiscard]] std::optional<std::uint32_t> subgroup_size() const noexcept { return subgroup_size_; }
    [[nodiscard]] std::optional<std::uint32_t> max_workgroup_size() const noexcept { return max_workgroup_size_; }
    [[nodiscard]] std::optional<std::uint32_t> shared_memory_per_block() const noexcept { return shared_memory_per_block_; }

    // --- clocks ---
    [[nodiscard]] std::optional<std::uint32_t> frequency() const noexcept { return frequency_; }

    // --- topology / form ---
    [[nodiscard]] std::optional<bool> integrated() const noexcept { return integrated_; }
    [[nodiscard]] std::optional<std::uint32_t> pcie_gen() const noexcept { return pcie_gen_; }
    [[nodiscard]] std::optional<std::uint32_t> pcie_width() const noexcept { return pcie_width_; }

    // --- driver ---
    [[nodiscard]] const std::optional<std::string>& driver_version() const noexcept { return driver_version_; }

    // --- feature flags ---
    [[nodiscard]] bool supports_fp16() const noexcept { return fp16_; }
    [[nodiscard]] bool supports_fp64() const noexcept { return fp64_; }
    [[nodiscard]] bool supports_int8() const noexcept { return int8_; }
    [[nodiscard]] bool supports_tensor_cores() const noexcept { return tensor_cores_; }

    // Builder methods are public so each backend can populate the value type
    // without forcing a deep friend graph. The Device is only mutated by its
    // producing backend during the query() call.
    Device& set_id(std::string v) { id_ = std::move(v); return *this; }
    Device& set_name(std::string v) { name_ = std::move(v); return *this; }
    Device& set_vendor(Vendor v) { vendor_ = v; return *this; }
    Device& set_vendor_string(std::string v) { vendor_string_ = std::move(v); return *this; }
    Device& set_pci_address(PciAddress v) { pci_address_ = v; return *this; }
    Device& set_pci_vendor_id(std::uint16_t v) { pci_vendor_id_ = v; return *this; }
    Device& set_pci_device_id(std::uint16_t v) { pci_device_id_ = v; return *this; }
    Device& set_uuid(const DeviceUuid& v) { uuid_ = v; return *this; }
    Device& set_type(DeviceType v) { type_ = v; return *this; }
    Device& set_memory(std::uint64_t v) { memory_ = v; return *this; }
    Device& set_memory_bandwidth_gbps(std::uint64_t v) { memory_bandwidth_gbps_ = v; return *this; }
    Device& set_l2_cache_bytes(std::uint64_t v) { l2_cache_bytes_ = v; return *this; }
    Device& set_memory_bus_width_bits(std::uint32_t v) { memory_bus_width_bits_ = v; return *this; }
    Device& set_cores(std::uint32_t v) { cores_ = v; return *this; }
    Device& set_compute_units(std::uint32_t v) { compute_units_ = v; return *this; }
    Device& set_subgroup_size(std::uint32_t v) { subgroup_size_ = v; return *this; }
    Device& set_max_workgroup_size(std::uint32_t v) { max_workgroup_size_ = v; return *this; }
    Device& set_shared_memory_per_block(std::uint32_t v) { shared_memory_per_block_ = v; return *this; }
    Device& set_frequency(std::uint32_t v) { frequency_ = v; return *this; }
    Device& set_integrated(bool v) { integrated_ = v; return *this; }
    Device& set_pcie_gen(std::uint32_t v) { pcie_gen_ = v; return *this; }
    Device& set_pcie_width(std::uint32_t v) { pcie_width_ = v; return *this; }
    Device& set_driver_version(std::string v) { driver_version_ = std::move(v); return *this; }
    Device& set_fp16(bool v) { fp16_ = v; return *this; }
    Device& set_fp64(bool v) { fp64_ = v; return *this; }
    Device& set_int8(bool v) { int8_ = v; return *this; }
    Device& set_tensor_cores(bool v) { tensor_cores_ = v; return *this; }

private:
    std::string id_;
    std::string name_;
    Vendor vendor_{Vendor::Unknown};
    std::string vendor_string_;

    std::optional<PciAddress> pci_address_;
    std::optional<std::uint16_t> pci_vendor_id_;
    std::optional<std::uint16_t> pci_device_id_;
    std::optional<DeviceUuid> uuid_;
    DeviceType type_{DeviceType::Unknown};

    std::optional<std::uint64_t> memory_;
    std::optional<std::uint64_t> memory_bandwidth_gbps_;
    std::optional<std::uint64_t> l2_cache_bytes_;
    std::optional<std::uint32_t> memory_bus_width_bits_;

    std::optional<std::uint32_t> cores_;
    std::optional<std::uint32_t> compute_units_;
    std::optional<std::uint32_t> subgroup_size_;
    std::optional<std::uint32_t> max_workgroup_size_;
    std::optional<std::uint32_t> shared_memory_per_block_;

    std::optional<std::uint32_t> frequency_;

    std::optional<bool> integrated_;
    std::optional<std::uint32_t> pcie_gen_;
    std::optional<std::uint32_t> pcie_width_;

    std::optional<std::string> driver_version_;

    bool fp16_{false};
    bool fp64_{false};
    bool int8_{false};
    bool tensor_cores_{false};
};

} // namespace gpgpu
