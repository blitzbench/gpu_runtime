# Vulkan runner CMake. Requires the shared shader build (driven by
# app/.../shaders/CMakeLists.txt, processed before this subdir at the parent
# CMakeLists level). Headers come from gpu_runtime's vendored Vulkan-Headers
# (VK_NO_PROTOTYPES) and every entry point is resolved at run time
# (gpgpu/vendor.hpp); only the shader build needs a tool from the Vulkan SDK.

if(NOT EXAMPLE_VS_HAVE_SHADER)
    message(STATUS "    VULKAN: skipped (shader build unavailable)")
    set(EXAMPLE_VS_HAVE_VULKAN FALSE PARENT_SCOPE)
    return()
endif()

add_library(example_vs_vulkan STATIC vulkan_runner.cpp)
add_dependencies(example_vs_vulkan example_vs_shader)
target_link_libraries(example_vs_vulkan PUBLIC gpu_runtime)
target_include_directories(example_vs_vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${EXAMPLE_VS_SHADER_INCDIR}
)
target_compile_features(example_vs_vulkan PUBLIC cxx_std_20)
set_target_properties(example_vs_vulkan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
)
target_compile_definitions(example_vs_vulkan INTERFACE EXAMPLE_VS_HAVE_VULKAN)
set(EXAMPLE_VS_HAVE_VULKAN TRUE PARENT_SCOPE)
