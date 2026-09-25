# Level Zero runner CMake. Requires the shared shader build. Headers come from
# gpu_runtime's vendored Level Zero headers and every entry point is resolved at
# run time (gpgpu/vendor.hpp), so no Level Zero SDK is needed to build. Built
# only with ONEAPI_GLSL_SPIRV=ON (see the parent CMakeLists.txt).

if(NOT ONEAPI_GLSL_SPIRV)
    set(EXAMPLE_VS_HAVE_ONEAPI FALSE PARENT_SCOPE)
    return()
endif()
if(NOT EXAMPLE_VS_HAVE_SHADER)
    message(STATUS "    ONEAPI: skipped (shader build unavailable)")
    set(EXAMPLE_VS_HAVE_ONEAPI FALSE PARENT_SCOPE)
    return()
endif()

add_library(example_vs_oneapi STATIC oneapi_runner.cpp)
add_dependencies(example_vs_oneapi example_vs_shader)
target_link_libraries(example_vs_oneapi PUBLIC gpu_runtime)
target_include_directories(example_vs_oneapi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${EXAMPLE_VS_SHADER_INCDIR}
)
target_compile_features(example_vs_oneapi PUBLIC cxx_std_20)
set_target_properties(example_vs_oneapi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(example_vs_oneapi INTERFACE EXAMPLE_VS_HAVE_ONEAPI)
set(EXAMPLE_VS_HAVE_ONEAPI TRUE PARENT_SCOPE)
