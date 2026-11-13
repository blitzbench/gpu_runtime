// OpenCL ICD probe. Uses the vendored Khronos OpenCL-Headers
// (third_party/OpenCL-Headers/) for type definitions and constant values; the
// runtime is dlopen'd at runtime, so no link-time vendor SDK is required.

#include "probe.hpp"

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_function_types.h>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../identity.hpp"
#include "../platform/lib_loader.hpp"
#include "../search_paths.hpp"
#include "device_identity.hpp"

namespace gpgpu::backends {

namespace {

// AMD topology layout (cl_amd_device_topology is declared in cl_ext.h as a
// union; we use a POD struct here for trivial init and copy).
struct DeviceTopologyAmd {
    cl_uint type;
    char    unused[17];
    char    bus;
    char    device;
    char    function;
};

// NVIDIA-specific OpenCL extension query keys (cl_nv_device_attribute_query).
// These are not part of the Khronos headers - NVIDIA documents them in their
// OpenCL Best Practices Guide but does not contribute them upstream.
constexpr cl_device_info CL_DEVICE_PCI_BUS_ID_NV    = 0x4008;
constexpr cl_device_info CL_DEVICE_PCI_SLOT_ID_NV   = 0x4009;
constexpr cl_device_info CL_DEVICE_PCI_DOMAIN_ID_NV = 0x400A;

constexpr cl_uint CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD = 1;

std::string info_string(clGetDeviceInfo_fn fn, cl_device_id dev, cl_device_info key) {
    std::size_t size = 0;
    if (fn(dev, key, 0, nullptr, &size) != CL_SUCCESS || size == 0) return {};
    std::string out(size, '\0');
    if (fn(dev, key, size, out.data(), nullptr) != CL_SUCCESS) return {};
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

template <typename T>
bool info_scalar(clGetDeviceInfo_fn fn, cl_device_id dev, cl_device_info key, T& out) {
    return fn(dev, key, sizeof(T), &out, nullptr) == CL_SUCCESS;
}

std::string platform_info_string(clGetPlatformInfo_fn fn, cl_platform_id p, cl_platform_info key) {
    std::size_t size = 0;
    if (fn(p, key, 0, nullptr, &size) != CL_SUCCESS || size == 0) return {};
    std::string out(size, '\0');
    if (fn(p, key, size, out.data(), nullptr) != CL_SUCCESS) return {};
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    std::size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    return s.substr(i);
}

bool has_extension(const std::string& extensions, std::string_view name) {
    std::size_t pos = 0;
    while ((pos = extensions.find(name, pos)) != std::string::npos) {
        const std::size_t end = pos + name.size();
        const bool starts = pos == 0 || extensions[pos - 1] == ' ';
        const bool ends = end == extensions.size() || extensions[end] == ' ';
        if (starts && ends) return true;
        pos = end;
    }
    return false;
}

std::optional<PciAddress> read_pci_address(clGetDeviceInfo_fn fn, cl_device_id dev,
                                           const std::string& extensions) {
    if (has_extension(extensions, "cl_khr_pci_bus_info")) {
        cl_device_pci_bus_info_khr info{};
        if (info_scalar(fn, dev, CL_DEVICE_PCI_BUS_INFO_KHR, info)) {
            return PciAddress{info.pci_domain, info.pci_bus, info.pci_device, info.pci_function};
        }
    }
    if (has_extension(extensions, "cl_nv_device_attribute_query")) {
        cl_uint bus = 0, slot = 0;
        if (info_scalar(fn, dev, CL_DEVICE_PCI_BUS_ID_NV, bus) &&
            info_scalar(fn, dev, CL_DEVICE_PCI_SLOT_ID_NV, slot)) {
            // Drivers older than CL_DEVICE_PCI_DOMAIN_ID_NV report bus and slot only; domain 0
            // is assumed for them, which can only be wrong on a multi-segment host.
            cl_uint domain = 0;
            info_scalar(fn, dev, CL_DEVICE_PCI_DOMAIN_ID_NV, domain);
            return PciAddress{domain, bus, (slot >> 3) & 0x1F, slot & 0x7};
        }
    }
    if (has_extension(extensions, "cl_amd_device_attribute_query")) {
        DeviceTopologyAmd topo{};
        if (info_scalar(fn, dev, CL_DEVICE_TOPOLOGY_AMD, topo) &&
            topo.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD) {
            // cl_amd_device_topology has no domain field; domain 0 is assumed, which can only
            // be wrong on a multi-segment host.
            return PciAddress{0,
                              static_cast<unsigned char>(topo.bus),
                              static_cast<unsigned char>(topo.device),
                              static_cast<unsigned char>(topo.function)};
        }
    }
    return std::nullopt;
}

DeviceType read_device_type(clGetDeviceInfo_fn fn, cl_device_id dev) {
    cl_device_type type = 0;
    if (!info_scalar(fn, dev, CL_DEVICE_TYPE, type)) return DeviceType::Unknown;
    if (type & CL_DEVICE_TYPE_CPU) return DeviceType::Cpu;
    if (type & CL_DEVICE_TYPE_GPU) {
        cl_bool unified = CL_FALSE;
        if (!info_scalar(fn, dev, CL_DEVICE_HOST_UNIFIED_MEMORY, unified)) return DeviceType::Unknown;
        return unified ? DeviceType::Integrated : DeviceType::Discrete;
    }
    return DeviceType::Unknown;
}

} // namespace

void read_opencl_identity(clGetDeviceInfo_fn get_device_info, cl_device_id device, Device& out) {
    out.set_type(read_device_type(get_device_info, device));

    cl_uint vendor_id = 0;
    info_scalar(get_device_info, device, CL_DEVICE_VENDOR_ID, vendor_id);

    const std::string extensions = info_string(get_device_info, device, CL_DEVICE_EXTENSIONS);
    if (auto address = read_pci_address(get_device_info, device, extensions)) {
        out.set_pci_address(*address);
        // CL_DEVICE_VENDOR_ID is only specified to be the PCI vendor id for a PCI device.
        if (vendor_id <= 0xFFFF) out.set_pci_vendor_id(static_cast<std::uint16_t>(vendor_id));
    }

    if (has_extension(extensions, "cl_khr_device_uuid")) {
        std::uint8_t bytes[CL_UUID_SIZE_KHR] = {0};
        if (get_device_info(device, CL_DEVICE_UUID_KHR, sizeof(bytes), bytes, nullptr) == CL_SUCCESS) {
            if (auto uuid = identity::uuid_from(bytes)) out.set_uuid(*uuid);
        }
    }

    char backend_id[64];
    std::snprintf(backend_id, sizeof(backend_id), "ocl-%04x-%p", vendor_id,
                  static_cast<void*>(device));
    identity::assign_id(out, backend_id);
}

bool is_skipped_opencl_platform(clGetPlatformInfo_fn get_platform_info, cl_platform_id platform) {
    const std::string name = platform_info_string(get_platform_info, platform, CL_PLATFORM_NAME);
    const std::string version = platform_info_string(get_platform_info, platform, CL_PLATFORM_VERSION);
    return name.find("OpenCLOn12") != std::string::npos ||
           version.find("D3D12 Implementation") != std::string::npos;
}

ProbeResult probe_opencl() {
    ProbeResult out;

    std::string loader_err;
    auto lib = platform::try_load(search::candidates(BackendId::OpenCL), loader_err);
    if (!lib.is_open()) {
        out.backend = make_missing(BackendId::OpenCL, std::move(loader_err));
        return out;
    }
    const std::string lib_path = lib.path();

    auto clGetPlatformIDs  = platform::resolve_as<clGetPlatformIDs_fn>(lib, "clGetPlatformIDs");
    auto clGetPlatformInfo = platform::resolve_as<clGetPlatformInfo_fn>(lib, "clGetPlatformInfo");
    auto clGetDeviceIDs    = platform::resolve_as<clGetDeviceIDs_fn>(lib, "clGetDeviceIDs");
    auto clGetDeviceInfo   = platform::resolve_as<clGetDeviceInfo_fn>(lib, "clGetDeviceInfo");

    if (!clGetPlatformIDs || !clGetDeviceIDs || !clGetDeviceInfo || !clGetPlatformInfo) {
        out.backend = Backend(BackendId::OpenCL, /*version*/ {}, lib_path,
                              BackendStatus::InitFailed,
                              std::string("required OpenCL ICD symbols missing"));
        return out;
    }

    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) {
        out.backend = Backend(BackendId::OpenCL, /*version*/ {}, lib_path,
                              BackendStatus::NoDevices, std::nullopt);
        return out;
    }

    std::vector<cl_platform_id> platforms(n_platforms);
    clGetPlatformIDs(n_platforms, platforms.data(), nullptr);

    std::string best_version; // pick the highest OpenCL version string seen.

    for (cl_platform_id p : platforms) {
        std::string pver = platform_info_string(clGetPlatformInfo, p, CL_PLATFORM_VERSION);

        if (is_skipped_opencl_platform(clGetPlatformInfo, p)) continue;
        // CL_PLATFORM_VERSION is shaped "OpenCL <major>.<minor> ...". Extract.
        const std::string prefix = "OpenCL ";
        std::string short_ver;
        if (pver.rfind(prefix, 0) == 0) {
            auto rest = pver.substr(prefix.size());
            auto sp = rest.find(' ');
            short_ver = sp == std::string::npos ? rest : rest.substr(0, sp);
        } else {
            short_ver = pver;
        }
        if (short_ver > best_version) best_version = short_ver;

        cl_uint n_devs = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &n_devs) != CL_SUCCESS || n_devs == 0)
            continue;
        std::vector<cl_device_id> devs(n_devs);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, n_devs, devs.data(), nullptr);

        for (cl_device_id dev : devs) {
            Device d;
            d.set_name(trim(info_string(clGetDeviceInfo, dev, CL_DEVICE_NAME)));

            std::string vendor_str = trim(info_string(clGetDeviceInfo, dev, CL_DEVICE_VENDOR));
            d.set_vendor_string(vendor_str);

            cl_uint vendor_id = 0;
            info_scalar(clGetDeviceInfo, dev, CL_DEVICE_VENDOR_ID, vendor_id);
            Vendor v = classify_vendor_pci(vendor_id);
            if (v == Vendor::Unknown) v = classify_vendor(vendor_str);
            d.set_vendor(v);

            cl_ulong mem = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_GLOBAL_MEM_SIZE, mem) && mem > 0)
                d.set_memory(mem);

            cl_ulong l2 = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, l2) && l2 > 0)
                d.set_l2_cache_bytes(l2);

            cl_ulong local_mem = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_LOCAL_MEM_SIZE, local_mem) && local_mem > 0)
                d.set_shared_memory_per_block(static_cast<std::uint32_t>(local_mem));

            cl_uint cu = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_MAX_COMPUTE_UNITS, cu) && cu > 0)
                d.set_compute_units(cu);

            std::size_t max_wg = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, max_wg) && max_wg > 0)
                d.set_max_workgroup_size(static_cast<std::uint32_t>(max_wg));

            cl_uint freq = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, freq) && freq > 0)
                d.set_frequency(freq);

            cl_uint native_half = 0;
            info_scalar(clGetDeviceInfo, dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, native_half);
            if (native_half > 0) d.set_fp16(true);

            cl_uint native_double = 0;
            info_scalar(clGetDeviceInfo, dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, native_double);
            if (native_double > 0) d.set_fp64(true);

            cl_bool unified = 0;
            if (info_scalar(clGetDeviceInfo, dev, CL_DEVICE_HOST_UNIFIED_MEMORY, unified)) {
                d.set_integrated(unified != 0);
            }

            std::string exts = info_string(clGetDeviceInfo, dev, CL_DEVICE_EXTENSIONS);
            if (exts.find("cl_khr_fp16") != std::string::npos) d.set_fp16(true);
            if (exts.find("cl_khr_fp64") != std::string::npos) d.set_fp64(true);
            if (exts.find("cl_khr_integer_dot_product") != std::string::npos) d.set_int8(true);

            d.set_driver_version(info_string(clGetDeviceInfo, dev, CL_DRIVER_VERSION));

            read_opencl_identity(clGetDeviceInfo, dev, d);

            out.devices.push_back(std::move(d));
        }
    }

    BackendStatus status = out.devices.empty() ? BackendStatus::NoDevices
                                                : BackendStatus::Available;
    out.backend = Backend(BackendId::OpenCL, std::move(best_version), lib_path,
                          status, std::nullopt);
    return out;
}

} // namespace gpgpu::backends
