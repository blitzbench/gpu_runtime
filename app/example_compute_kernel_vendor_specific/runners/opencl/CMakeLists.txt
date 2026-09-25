# Headers come from gpu_runtime's vendored OpenCL-Headers and every entry point is
# resolved at run time (gpgpu/vendor.hpp), so no OpenCL SDK is needed to build.
add_library(example_vs_opencl STATIC opencl_runner.cpp)
target_link_libraries(example_vs_opencl PUBLIC gpu_runtime)
target_include_directories(example_vs_opencl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_compile_features(example_vs_opencl PUBLIC cxx_std_20)
target_compile_definitions(example_vs_opencl PUBLIC
    CL_TARGET_OPENCL_VERSION=300
    CL_USE_DEPRECATED_OPENCL_1_2_APIS
)
set_target_properties(example_vs_opencl PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(example_vs_opencl INTERFACE EXAMPLE_VS_HAVE_OPENCL)
set(EXAMPLE_VS_HAVE_OPENCL TRUE PARENT_SCOPE)
