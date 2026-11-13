// ROCm/HIP probe. We dlopen libamdhip64.so (amdhip64.dll on Windows) and
// resolve a deliberately small set of stable entry points. The hipDeviceProp_t
// struct layout has shifted between ROCm releases, so we avoid it entirely.

#include "probe.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>

#include "../identity.hpp"
#include "../platform/lib_loader.hpp"
#include "../search_paths.hpp"
#include "device_identity.hpp"

namespace gpgpu::backends {

namespace {

constexpr hipError_t hipSuccess = 0;

using PFN_hipDeviceGetName     = hipError_t (GPGPU_STDCALL*)(char*, int, hipDevice_t);
using PFN_hipDeviceTotalMem    = hipError_t (GPGPU_STDCALL*)(std::size_t*, hipDevice_t);
using PFN_hipDriverGetVersion  = hipError_t (GPGPU_STDCALL*)(int*);
using PFN_hipRuntimeGetVersion = hipError_t (GPGPU_STDCALL*)(int*);

bool ok(hipError_t r) { return r == hipSuccess; }

std::string format_version(int v) {
    // ROCm encodes versions as MMMmmpp (e.g. 60002033 -> "6.0.2").
    int major = v / 10000000;
    int minor = (v / 100000) % 100;
    int patch = (v / 1000) % 100;
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d.%d.%d", major, minor, patch);
    return buf;
}

constexpr std::uint16_t kAmdPciVendorId = 0x1002;

hipDevice_t device_for_ordinal(const HipIdentityFns& fns, int ordinal) {
    hipDevice_t device = ordinal;
    if (fns.hipDeviceGet) {
        hipDevice_t resolved = 0;
        if (ok(fns.hipDeviceGet(&resolved, ordinal))) device = resolved;
    }
    return device;
}

} // namespace

void read_hip_identity(const HipIdentityFns& fns, hipDevice_t device, int ordinal, Device& out) {
    if (fns.hipDeviceGetPCIBusId) {
        char bdf[32] = {0};
        if (ok(fns.hipDeviceGetPCIBusId(bdf, sizeof(bdf), device))) {
            if (auto address = identity::parse_pci_address(bdf)) {
                out.set_pci_address(*address);
                out.set_pci_vendor_id(kAmdPciVendorId);
            }
        }
    }

    if (fns.hipDeviceGetUuid) {
        HipUuid raw{};
        if (ok(fns.hipDeviceGetUuid(&raw, device))) {
            std::uint8_t bytes[16];
            std::memcpy(bytes, raw.bytes, sizeof(bytes));
            if (auto uuid = identity::uuid_from(bytes)) out.set_uuid(*uuid);
        }
    }

    identity::assign_id(out, "hip-" + std::to_string(ordinal));
}

std::optional<int> find_hip_ordinal(const HipIdentityFns& fns, const Device& target) {
    int count = 0;
    if (!ok(fns.hipGetDeviceCount(&count))) return std::nullopt;
    for (int i = 0; i < count; ++i) {
        Device candidate;
        read_hip_identity(fns, device_for_ordinal(fns, i), i, candidate);
        if (candidate.id() == target.id()) return i;
    }
    return std::nullopt;
}

ProbeResult probe_rocm() {
    ProbeResult out;

    std::string loader_err;
    auto lib = platform::try_load(search::candidates(BackendId::ROCm), loader_err);
    if (!lib.is_open()) {
        out.backend = make_missing(BackendId::ROCm, std::move(loader_err));
        return out;
    }
    const std::string lib_path = lib.path();

    auto hipGetDeviceCount    = platform::resolve_as<PFN_hipGetDeviceCount>(lib, "hipGetDeviceCount");
    auto hipDeviceGet         = platform::resolve_as<PFN_hipDeviceGet>(lib, "hipDeviceGet");
    auto hipDeviceGetName     = platform::resolve_as<PFN_hipDeviceGetName>(lib, "hipDeviceGetName");
    auto hipDeviceTotalMem    = platform::resolve_as<PFN_hipDeviceTotalMem>(lib, "hipDeviceTotalMem");
    auto hipDriverGetVersion  = platform::resolve_as<PFN_hipDriverGetVersion>(lib, "hipDriverGetVersion");
    auto hipRuntimeGetVersion = platform::resolve_as<PFN_hipRuntimeGetVersion>(lib, "hipRuntimeGetVersion");
    auto hipDeviceGetPCIBusId = platform::resolve_as<PFN_hipDeviceGetPCIBusId>(lib, "hipDeviceGetPCIBusId");
    auto hipDeviceGetUuid     = platform::resolve_as<PFN_hipDeviceGetUuid>(lib, "hipDeviceGetUuid");

    if (!hipGetDeviceCount) {
        out.backend = Backend(BackendId::ROCm, /*version*/ {}, lib_path,
                              BackendStatus::InitFailed,
                              std::string("hipGetDeviceCount not exported"));
        return out;
    }

    std::string version_str;
    if (hipRuntimeGetVersion) {
        int v = 0;
        if (ok(hipRuntimeGetVersion(&v))) version_str = format_version(v);
    }
    if (version_str.empty() && hipDriverGetVersion) {
        int v = 0;
        if (ok(hipDriverGetVersion(&v))) version_str = format_version(v);
    }

    int count = 0;
    if (!ok(hipGetDeviceCount(&count)) || count <= 0) {
        out.backend = Backend(BackendId::ROCm, version_str, lib_path,
                              BackendStatus::NoDevices, std::nullopt);
        return out;
    }

    const HipIdentityFns identity_fns{hipGetDeviceCount, hipDeviceGet, hipDeviceGetPCIBusId,
                                      hipDeviceGetUuid};
    for (int i = 0; i < count; ++i) {
        Device d;
        d.set_vendor(Vendor::AMD).set_vendor_string("Advanced Micro Devices, Inc.");

        const hipDevice_t dev = device_for_ordinal(identity_fns, i);

        if (hipDeviceGetName) {
            char name_buf[256] = {0};
            if (ok(hipDeviceGetName(name_buf, sizeof(name_buf), dev))) {
                d.set_name(name_buf);
            }
        }

        if (hipDeviceTotalMem) {
            std::size_t bytes = 0;
            if (ok(hipDeviceTotalMem(&bytes, dev)) && bytes > 0) d.set_memory(bytes);
        }

        if (!version_str.empty()) d.set_driver_version(version_str);

        read_hip_identity(identity_fns, dev, i, d);

        // ROCm GPUs all expose FP64; FP16 and INT8 are supported on every GCN
        // generation from Vega onward. We cannot easily distinguish older
        // hardware without a properties struct, so flag them as supported.
        d.set_fp16(true);
        d.set_fp64(true);
        d.set_int8(true);
        d.set_tensor_cores(false); // matrix cores exist on CDNA but identification is fragile.

        out.devices.push_back(std::move(d));
    }

    BackendStatus status = out.devices.empty() ? BackendStatus::NoDevices
                                                : BackendStatus::Available;
    out.backend = Backend(BackendId::ROCm, std::move(version_str), lib_path,
                          status, std::nullopt);
    return out;
}

} // namespace gpgpu::backends
