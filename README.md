# gpu_runtime

A C++ library that detects available (GP)GPU backends and loads them in
runtime. It also provides Device information for detected GPUs and their
backends.

## Building

```bash
# After clone, fetch the third-party header submodules:
git submodule update --init --recursive

cmake -B build -DBUILD_SAMPLE_APP=1 -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/app/gpu_runtime_query/gpu_runtime_query
```

The unit tests (GoogleTest, fetched when not installed) build by default when
gpu_runtime is the top-level project; set `-DGPGPU_BUILD_TESTS=OFF` to skip
them. They need the default static build.

```bash
ctest --test-dir build
```

## Device identity

`gpgpu::Runtime::query()` returns one `Setup` per (device, backend) pair.
`Device::id()` identifies the physical device, by the same rule on every
backend:

1. `pci-DDDD:BB:DD.F` when the backend reports the PCI bus address
   (CUDA attributes, `hipDeviceGetPCIBusId`, `cl_khr_pci_bus_info` or the
   NVIDIA / AMD OpenCL extensions, `VK_EXT_pci_bus_info`,
   `zeDevicePciGetPropertiesExt`);
2. otherwise `uuid-<8-4-4-4-12>` when it reports a device UUID;
3. otherwise a backend-specific id (`cuda-<n>`, `hip-<n>`, `ze-VVVV:DDDD-<n>`,
   `ocl-VVVV-<handle>`, `vk-VVVV:DDDD-<pipelineCacheUUID>`, `metal-<registryID>`).

A GPU seen by several backends therefore has one id, so `Report::devices()`
lists it once and `Report::preferred()` returns one setup for it: the
vendor-native backend (`Native`), else OpenCL, else Vulkan (`Fallback`).
Only the PCI address is comparable across backends; UUIDs and backend ids are
not, so a device without a PCI address may appear once per backend.

The identity is also available typed:

| Accessor | Meaning |
|---|---|
| `pci_address()` | `std::optional<PciAddress>` (domain, bus, device, function); `to_string()` gives `DDDD:BB:DD.F` |
| `pci_vendor_id()`, `pci_device_id()` | `std::optional<std::uint16_t>`; absent for non-PCI devices or when the API does not report them (CUDA, ROCm and OpenCL give no device id) |
| `uuid()` | `std::optional<DeviceUuid>`; only stable within one backend |
| `type()` | `DeviceType::Discrete`, `Integrated`, `Virtual`, `Cpu` or `Unknown` |

`type()` is the only integrated-vs-discrete flag. `DeviceType::Cpu` marks
software implementations such as Mesa's llvmpipe or pocl. They stay in the
report and in `preferred()`; filter on `type()` to skip them.

Within a report, a PCI vendor id, PCI device id or type (when `Unknown`) that
one backend cannot report is filled from another backend's `Device` with the
same id (the first that reports it, in probe order): an NVIDIA GPU's CUDA
setup carries the PCI device id its Vulkan setup reports. `uuid()` is never
filled this way; it only holds what the setup's own backend reported, since
UUIDs differ between backends.

A backend can expose one physical device twice, for example two OpenCL
platforms or two Vulkan drivers (RADV and AMDVLK) for the same GPU. Both
setups share the id. Only the first, in enumeration order, can be `Native` or
`Fallback`; the later ones are `Other`, because the lookups below always
resolve the first device with that identity on that backend.

### Limits

- OpenCL drivers that predate `CL_DEVICE_PCI_DOMAIN_ID_NV` (NVIDIA) and AMD's
  `cl_amd_device_topology`, which has no domain field, report no PCI domain;
  domain 0 is assumed for them. On a host with more than one PCI segment this
  can give two GPUs the same id and merge them.
- CUDA reports no PCI function number (it is 0); CUDA, ROCm and OpenCL report
  no PCI device id; ROCm reports no device type. Other backends of the same
  device fill these in, as described above.
- The Vulkan lookup needs an instance created with `apiVersion` 1.1 or later.

## Calling the vendor runtimes yourself

`include/gpgpu/vendor.hpp` exposes the same run-time binding the probes use, so
a consumer can drive OpenCL, Vulkan or Level Zero without linking `libOpenCL` /
`libvulkan` / `libze_loader`: `gpgpu::vendor::opencl()`, `gpgpu::vendor::vulkan()`
and `gpgpu::vendor::level_zero()` dlopen the loader once per process and return
a table of resolved entry points, or `nullptr` when the runtime is absent.
Vulkan instance- and device-level tables are filled with `load_instance_fns` /
`load_device_fns` after the instance and device exist. The vendored Khronos and
Level Zero headers are on the library's public include path for the types, and
`VK_NO_PROTOTYPES` is defined publicly so a stray direct `vk*` call fails to
compile.

To run on a `Setup` from the report, find the backend's native handle with the
lookup helpers in the same header instead of rebuilding an id string:
`find_vulkan_device`, `find_level_zero_device`, `find_opencl_device`,
`find_cuda_device` / `find_hip_device` (device ordinals) and, from
Objective-C++, `find_metal_device` (returned autoreleased). Each derives every
native device's id with the probe's own code and returns the first device on
that backend whose id equals `Device::id()`. The Vulkan lookup needs an
instance created with `apiVersion` 1.1 or later.

## Third-party headers

Three GPGPU API header sets are pulled in as git submodules under `external/`.
None of the vendor runtimes is required at build time - the libraries are
`dlopen`'d at runtime.

| Submodule | Upstream | License |
|---|---|---|
| `external/OpenCL-Headers` | [KhronosGroup/OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers) | Apache-2.0 |
| `external/Vulkan-Headers` | [KhronosGroup/Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | Apache-2.0 (a few files MIT) |
| `external/level-zero`     | [oneapi-src/level-zero](https://github.com/oneapi-src/level-zero) | MIT (a few files Apache-2.0) |

Each submodule ships its own `LICENSE`. To pin a different upstream revision,
`cd` into the submodule, `git checkout <ref>`, then commit the submodule SHA
update in this repository.

### Not vendored

- **NVIDIA CUDA** headers: proprietary license, redistribution not permitted.
  The CUDA backend (`src/backends/cuda.cpp`) declares the small Driver-API
  subset it uses inline.
- **AMD HIP / ROCm** headers: MIT-licensed, but `hip_runtime_api.h` is 10 K+
  lines and transitively pulls in `nvidia_detail/` and `amd_detail/` paths
  that require full CUDA / ROCm SDK installations. The HIP backend
  (`src/backends/rocm.cpp`) declares the small Runtime-API subset it uses
  inline.
- **Apple Metal**: `<Metal/Metal.h>` is part of the macOS system SDK and is
  `#import`ed directly when building on Apple platforms.
