// OpenCL vector-add runner.
//
// <CL/cl.h> supplies the types only; every call goes through the entry-point
// table from gpgpu::vendor::opencl(), so the binary carries no link-time
// dependency on the ICD loader and starts on a machine without it.

#include "opencl_runner.hpp"

#include <CL/cl.h>
#include <gpgpu/vendor.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace example {

namespace {

using OpenClFns = gpgpu::vendor::OpenClFns;

constexpr const char* kKernelSource = R"CL(
__kernel void vector_add(
    __global const float* a,
    __global const float* b,
    __global       float* c,
    const uint n)
{
    uint i = get_global_id(0);
    if (i < n) c[i] = a[i] + b[i];
}
)CL";

constexpr std::size_t kLocalSize = 256;

std::string info_string(const OpenClFns& cl, cl_device_id dev, cl_device_info key) {
    std::size_t n = 0;
    if (cl.clGetDeviceInfo(dev, key, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
    std::string s(n, '\0');
    if (cl.clGetDeviceInfo(dev, key, n, s.data(), nullptr) != CL_SUCCESS) return {};
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

template <typename T>
T info_scalar(const OpenClFns& cl, cl_device_id dev, cl_device_info key) {
    T v{};
    cl.clGetDeviceInfo(dev, key, sizeof(T), &v, nullptr);
    return v;
}

cl_device_id find_matching_device(const OpenClFns& cl, const gpgpu::Device& target) {
    cl_uint n_platforms = 0;
    if (cl.clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) return nullptr;
    std::vector<cl_platform_id> platforms(n_platforms);
    cl.clGetPlatformIDs(n_platforms, platforms.data(), nullptr);
    for (cl_platform_id p : platforms) {
        cl_uint n_devs = 0;
        if (cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &n_devs) != CL_SUCCESS) continue;
        if (n_devs == 0) continue;
        std::vector<cl_device_id> devs(n_devs);
        cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, n_devs, devs.data(), nullptr);
        for (cl_device_id d : devs) {
            const std::string name = info_string(cl, d, CL_DEVICE_NAME);
            const cl_ulong   mem   = info_scalar<cl_ulong>(cl, d, CL_DEVICE_GLOBAL_MEM_SIZE);
            const bool name_ok = (name == target.name());
            const bool mem_ok  = (!target.memory().has_value() || mem == *target.memory());
            if (name_ok && mem_ok) return d;
        }
    }
    return nullptr;
}

std::chrono::duration<double> event_span(const OpenClFns& cl, cl_event start, cl_event end) {
    cl_ulong t0 = 0, t1 = 0;
    cl.clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
    cl.clGetEventProfilingInfo(end,   CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr);
    if (t1 <= t0) return {};
    return std::chrono::duration<double>{(t1 - t0) / 1.0e9};
}

std::string build_log(const OpenClFns& cl, cl_program prog, cl_device_id dev) {
    std::size_t n = 0;
    if (cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n) != CL_SUCCESS || n == 0)
        return {};
    std::string log(n, '\0');
    cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    if (!log.empty() && log.back() == '\0') log.pop_back();
    return log;
}

} // namespace

RunResult run_vector_add_opencl(const gpgpu::Setup&    setup,
                                std::span<const float> a,
                                std::span<const float> b,
                                std::span<float>       c) {
    RunResult r;
    if (a.size() != b.size() || a.size() != c.size()) {
        r.error = "vector size mismatch"; return r;
    }
    const std::size_t n     = a.size();
    const std::size_t bytes = n * sizeof(float);
    r.timings.copy_h2d_size = 2 * bytes;
    r.timings.copy_d2h_size = bytes;

    const OpenClFns* cl_fns = gpgpu::vendor::opencl();
    if (!cl_fns) { r.error = "OpenCL loader unavailable"; return r; }
    const OpenClFns& cl = *cl_fns;

    cl_device_id device = find_matching_device(cl, setup.device);
    if (!device) { r.error = "no OpenCL device matched " + setup.device.id(); return r; }

    cl_int err = CL_SUCCESS;
    cl_context ctx = cl.clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!ctx) { r.error = "clCreateContext err=" + std::to_string(err); return r; }

    const cl_queue_properties qprops[] = {
        CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(CL_QUEUE_PROFILING_ENABLE),
        0
    };
    cl_command_queue queue = cl.clCreateCommandQueueWithProperties(ctx, device, qprops, &err);
    if (!queue) {
        cl.clReleaseContext(ctx);
        r.error = "clCreateCommandQueueWithProperties err=" + std::to_string(err);
        return r;
    }

    cl_mem buf_a = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes, nullptr, &err);
    cl_mem buf_b = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes, nullptr, &err);
    cl_mem buf_c = cl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    const char*       src     = kKernelSource;
    const std::size_t src_len = std::strlen(kKernelSource);
    cl_program program = cl.clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
    if (cl.clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr) != CL_SUCCESS) {
        r.error = "clBuildProgram failed: " + build_log(cl, program, device);
        cl.clReleaseProgram(program);
        cl.clReleaseMemObject(buf_a); cl.clReleaseMemObject(buf_b); cl.clReleaseMemObject(buf_c);
        cl.clReleaseCommandQueue(queue); cl.clReleaseContext(ctx);
        return r;
    }
    cl_kernel kernel = cl.clCreateKernel(program, "vector_add", &err);

    cl_uint n_arg = static_cast<cl_uint>(n);
    cl.clSetKernelArg(kernel, 0, sizeof(cl_mem),  &buf_a);
    cl.clSetKernelArg(kernel, 1, sizeof(cl_mem),  &buf_b);
    cl.clSetKernelArg(kernel, 2, sizeof(cl_mem),  &buf_c);
    cl.clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_arg);

    const std::size_t global = ((n + kLocalSize - 1) / kLocalSize) * kLocalSize;
    const std::size_t local  = kLocalSize;

    cl_event ev_h2d_a = nullptr, ev_h2d_b = nullptr, ev_kernel = nullptr, ev_d2h = nullptr;
    const auto t_total_start = std::chrono::steady_clock::now();

    cl.clEnqueueWriteBuffer(queue, buf_a, CL_FALSE, 0, bytes, a.data(), 0, nullptr, &ev_h2d_a);
    cl.clEnqueueWriteBuffer(queue, buf_b, CL_FALSE, 0, bytes, b.data(), 0, nullptr, &ev_h2d_b);

    const auto t_launch_start = std::chrono::steady_clock::now();
    cl_int launch_rc = cl.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local,
                                                 0, nullptr, &ev_kernel);
    r.timings.kernel_launch = std::chrono::steady_clock::now() - t_launch_start;

    if (launch_rc != CL_SUCCESS) {
        r.error = "clEnqueueNDRangeKernel err=" + std::to_string(launch_rc);
    } else {
        cl.clEnqueueReadBuffer(queue, buf_c, CL_FALSE, 0, bytes, c.data(), 0, nullptr, &ev_d2h);
        cl.clFinish(queue);
    }

    r.timings.total = std::chrono::steady_clock::now() - t_total_start;

    if (ev_h2d_a && ev_h2d_b) r.timings.copy_h2d       = event_span(cl, ev_h2d_a, ev_h2d_b);
    if (ev_kernel)            r.timings.kernel_compute = event_span(cl, ev_kernel, ev_kernel);
    if (ev_d2h)               r.timings.copy_d2h       = event_span(cl, ev_d2h, ev_d2h);

    if (ev_h2d_a)  cl.clReleaseEvent(ev_h2d_a);
    if (ev_h2d_b)  cl.clReleaseEvent(ev_h2d_b);
    if (ev_kernel) cl.clReleaseEvent(ev_kernel);
    if (ev_d2h)    cl.clReleaseEvent(ev_d2h);

    cl.clReleaseKernel(kernel);
    cl.clReleaseProgram(program);
    cl.clReleaseMemObject(buf_a); cl.clReleaseMemObject(buf_b); cl.clReleaseMemObject(buf_c);
    cl.clReleaseCommandQueue(queue);
    cl.clReleaseContext(ctx);

    if (!r.error.empty()) return r;

    for (std::size_t i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        if (std::fabs(c[i] - expected) > 1e-4f * std::fabs(expected) + 1e-5f) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "first mismatch at i=%zu: c=%g expected=%g", i, c[i], expected);
            r.error = buf;
            r.correct = false;
            return r;
        }
    }
    r.correct = true;
    return r;
}

} // namespace example
