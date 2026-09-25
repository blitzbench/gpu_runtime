#pragma once

#include <span>

#include <gpgpu/setup.hpp>

#include "../../result.hpp"

namespace example {

// Run c = a + b on a OpenCL setup. Always built; the OpenCL runtime is bound
// at run time through gpgpu::vendor. The lib target sets
// EXAMPLE_VS_HAVE_OPENCL which gates inclusion from main.cpp.
RunResult run_vector_add_opencl(const gpgpu::Setup& setup,
                              std::span<const float> a,
                              std::span<const float> b,
                              std::span<float>       c);

} // namespace example
