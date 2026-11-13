#include "gpgpu/runtime.hpp"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "backends/probe.hpp"
#include "preference.hpp"

namespace gpgpu {

namespace backends {

namespace {

// The identity fields of one physical device, gathered from every backend that reports it.
struct MergedIdentity {
    std::optional<std::uint16_t> pci_vendor_id;
    std::optional<std::uint16_t> pci_device_id;
    std::optional<DeviceUuid> uuid;
    DeviceType type{DeviceType::Unknown};
};

void fill_missing_identity(Device& device, const MergedIdentity& merged) {
    if (!device.pci_vendor_id() && merged.pci_vendor_id) device.set_pci_vendor_id(*merged.pci_vendor_id);
    if (!device.pci_device_id() && merged.pci_device_id) device.set_pci_device_id(*merged.pci_device_id);
    if (!device.uuid() && merged.uuid) device.set_uuid(*merged.uuid);
    if (device.type() == DeviceType::Unknown) device.set_type(merged.type);
}

} // namespace

Report assemble_report(std::vector<ProbeResult> probes) {
    std::unordered_map<std::string, std::vector<BackendId>> per_device;
    std::unordered_map<std::string, MergedIdentity> identity;
    for (const auto& p : probes) {
        for (const auto& dev : p.devices) {
            per_device[dev.id()].push_back(p.backend.id());
            MergedIdentity& merged = identity[dev.id()];
            if (!merged.pci_vendor_id) merged.pci_vendor_id = dev.pci_vendor_id();
            if (!merged.pci_device_id) merged.pci_device_id = dev.pci_device_id();
            if (!merged.uuid) merged.uuid = dev.uuid();
            if (merged.type == DeviceType::Unknown) merged.type = dev.type();
        }
    }

    std::vector<Setup> setups;
    std::vector<Backend> backends_out;
    backends_out.reserve(probes.size());
    // A backend can expose one physical device twice (two OpenCL platforms, two Vulkan drivers).
    // The lookups in gpgpu/vendor.hpp resolve the first of them, so only that one may be picked.
    std::set<std::pair<std::string, BackendId>> classified;

    for (auto& p : probes) {
        backends_out.push_back(p.backend);
        for (auto& dev : p.devices) {
            Setup s;
            s.device = std::move(dev);
            fill_missing_identity(s.device, identity[s.device.id()]);
            s.backend = p.backend;
            s.preferred = classified.emplace(s.device.id(), p.backend.id()).second
                              ? pref::classify(s.device.vendor(), p.backend.id(), per_device[s.device.id()])
                              : Preference::Other;
            setups.push_back(std::move(s));
        }
    }

    return Report(std::move(setups), std::move(backends_out));
}

} // namespace backends

Report Runtime::query() {
    std::vector<backends::ProbeResult> probes;
    probes.reserve(6);
    probes.push_back(backends::probe_cuda());
    probes.push_back(backends::probe_rocm());
    probes.push_back(backends::probe_oneapi());
    probes.push_back(backends::probe_opencl());
    probes.push_back(backends::probe_vulkan());
    probes.push_back(backends::probe_metal());
    return backends::assemble_report(std::move(probes));
}

} // namespace gpgpu
