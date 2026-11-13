// Metal probe (macOS only). Enumerates all Metal-capable GPUs and fills in
// device characteristics that the Metal API exposes.

#if defined(__APPLE__)

#include "probe.hpp"

#include "gpgpu/vendor.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#if __has_feature(objc_arc)
#error "metal.mm manages Objective-C references manually and must be compiled without ARC"
#endif

#include <cstdint>
#include <cstdio>
#include <string>

#include "../identity.hpp"

namespace gpgpu::backends {

namespace {

std::string nsstr(NSString* s) {
    if (!s) return {};
    const char* c = [s UTF8String];
    return c ? std::string(c) : std::string();
}

Vendor classify_metal_device(id<MTLDevice> d) {
    std::string n = nsstr([d name]);
    Vendor v = classify_vendor(n);
    if (v != Vendor::Unknown) return v;
    // Apple-silicon GPUs always identify as Apple.
    if ([d respondsToSelector:@selector(supportsFamily:)] &&
        [d supportsFamily:MTLGPUFamilyApple1]) {
        return Vendor::Apple;
    }
    return Vendor::Unknown;
}

// Metal exposes neither a PCI address nor a UUID, so every Metal device has a backend-specific id.
void read_metal_identity(id<MTLDevice> dev, std::size_t index, Device& out) {
    if ([dev respondsToSelector:@selector(hasUnifiedMemory)]) {
        out.set_type([dev hasUnifiedMemory] ? DeviceType::Integrated : DeviceType::Discrete);
    }
    std::string backend_id = "metal-" + std::to_string(index);
    if ([dev respondsToSelector:@selector(registryID)]) {
        char id_buf[40];
        std::snprintf(id_buf, sizeof(id_buf), "metal-%llx", (unsigned long long)[dev registryID]);
        backend_id = id_buf;
    }
    identity::assign_id(out, backend_id);
}

} // namespace

ProbeResult probe_metal() {
    ProbeResult out;

    NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
    if (!devs || [devs count] == 0) {
        out.backend = Backend(BackendId::Metal, /*version*/ {},
                              "/System/Library/Frameworks/Metal.framework/Metal",
                              BackendStatus::NoDevices, std::nullopt);
        [devs release];
        return out;
    }

    NSOperatingSystemVersion ver = [[NSProcessInfo processInfo] operatingSystemVersion];
    char version_buf[32];
    std::snprintf(version_buf, sizeof(version_buf), "%ld.%ld.%ld",
                  (long)ver.majorVersion, (long)ver.minorVersion, (long)ver.patchVersion);

    for (id<MTLDevice> dev in devs) {
        Device d;
        d.set_name(nsstr([dev name]));
        Vendor v = classify_metal_device(dev);
        d.set_vendor(v);
        d.set_vendor_string(std::string(to_string(v)));

        if ([dev respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
            std::uint64_t mem = (std::uint64_t)[dev recommendedMaxWorkingSetSize];
            if (mem > 0) d.set_memory(mem);
        }
        if ([dev respondsToSelector:@selector(maxThreadsPerThreadgroup)]) {
            MTLSize tg = [dev maxThreadsPerThreadgroup];
            std::uint64_t total = (std::uint64_t)tg.width * tg.height * tg.depth;
            if (total > 0) d.set_max_workgroup_size((std::uint32_t)total);
        }
        if ([dev respondsToSelector:@selector(maxThreadgroupMemoryLength)]) {
            std::uint64_t sm = (std::uint64_t)[dev maxThreadgroupMemoryLength];
            if (sm > 0) d.set_shared_memory_per_block((std::uint32_t)sm);
        }

        read_metal_identity(dev, out.devices.size(), d);

        d.set_driver_version(version_buf);

        d.set_fp16(true);
        d.set_fp64(false); // Apple GPUs don't expose hardware FP64.
        d.set_int8(true);

        out.devices.push_back(std::move(d));
    }
    [devs release];

    out.backend = Backend(BackendId::Metal, std::string(version_buf),
                          "/System/Library/Frameworks/Metal.framework/Metal",
                          BackendStatus::Available, std::nullopt);
    return out;
}

} // namespace gpgpu::backends

namespace gpgpu::vendor {

id<MTLDevice> find_metal_device(const Device& target) noexcept {
    NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
    id<MTLDevice> found = nil;
    std::size_t index = 0;
    for (id<MTLDevice> dev in devs) {
        Device candidate;
        backends::read_metal_identity(dev, index++, candidate);
        if (candidate.id() == target.id()) {
            found = [[dev retain] autorelease];
            break;
        }
    }
    [devs release];
    return found;
}

} // namespace gpgpu::vendor

#endif // __APPLE__
