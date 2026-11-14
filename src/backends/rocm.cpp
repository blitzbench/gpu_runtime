// ROCm/HIP probe. We dlopen libamdhip64.so (amdhip64.dll on Windows) and
// resolve a deliberately small set of stable entry points. The hipDeviceProp_t
// struct layout has shifted between ROCm releases; the only use of it is the
// architecture name, read through the layout-versioned hipGetDevicePropertiesR0600.

#include "probe.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../identity.hpp"
#include "../platform/lib_loader.hpp"
#include "../search_paths.hpp"
#include "device_capabilities.hpp"
#include "device_identity.hpp"

namespace gpgpu::backends {

namespace {

constexpr hipError_t hipSuccess = 0;

using PFN_hipDeviceGetName     = hipError_t (GPGPU_STDCALL*)(char*, int, hipDevice_t);
using PFN_hipDeviceTotalMem    = hipError_t (GPGPU_STDCALL*)(std::size_t*, hipDevice_t);
using PFN_hipDriverGetVersion  = hipError_t (GPGPU_STDCALL*)(int*);
using PFN_hipRuntimeGetVersion = hipError_t (GPGPU_STDCALL*)(int*);
using PFN_hipGetDevicePropertiesR0600 = hipError_t (GPGPU_STDCALL*)(void*, int);

// hipDeviceProp_tR0600, frozen since ROCm 6.0: 1472 bytes, with char gcnArchName[256] at offset
// 1160. A runtime predating ROCm 6.0 does not export the R0600 entry point.
constexpr std::size_t kDevicePropR0600Size = 1472;
constexpr std::size_t kGcnArchNameOffset = 1160;
constexpr std::size_t kGcnArchNameSize = 256;

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

std::string gcn_arch_name(PFN_hipGetDevicePropertiesR0600 get_properties, int ordinal) {
    if (!get_properties) return {};
    std::vector<unsigned char> props(kDevicePropR0600Size, 0);
    if (!ok(get_properties(props.data(), ordinal))) return {};
    const char* name = reinterpret_cast<const char*>(props.data() + kGcnArchNameOffset);
    return std::string(name, ::strnlen(name, kGcnArchNameSize));
}

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.substr(0, prefix.size()) == prefix;
}

} // namespace

void set_rocm_capabilities(std::string_view gcn_arch_name, Device& out) {
    // Every architecture ROCm supports executes fp64, fp16 and 8-bit integer arithmetic. Matrix
    // cores are the MFMA units of CDNA (gfx908, gfx90a, gfx94x, gfx950) and the WMMA instructions
    // of RDNA3 and RDNA4 (gfx11xx, gfx12xx).
    out.set_fp16(true);
    out.set_fp64(true);
    out.set_int8(true);
    const std::string_view arch = gcn_arch_name.substr(0, gcn_arch_name.find(':'));
    out.set_tensor_cores(arch == "gfx908" || arch == "gfx90a" || starts_with(arch, "gfx94") ||
                         starts_with(arch, "gfx95") || starts_with(arch, "gfx11") ||
                         starts_with(arch, "gfx12"));
}

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
    auto hipGetDeviceProps    = platform::resolve_as<PFN_hipGetDevicePropertiesR0600>(lib, "hipGetDevicePropertiesR0600");

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

        set_rocm_capabilities(gcn_arch_name(hipGetDeviceProps, i), d);

        out.devices.push_back(std::move(d));
    }

    BackendStatus status = out.devices.empty() ? BackendStatus::NoDevices
                                                : BackendStatus::Available;
    out.backend = Backend(BackendId::ROCm, std::move(version_str), lib_path,
                          status, std::nullopt);
    return out;
}

} // namespace gpgpu::backends
