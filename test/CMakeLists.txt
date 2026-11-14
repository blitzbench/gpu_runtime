include(FetchContent)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)
endif()
include(GoogleTest)

# The tests reach the library's internal headers (src/), so they link the static library.
add_executable(gpu_runtime_tests
        capability_test.cpp
        identity_test.cpp
        lookup_test.cpp
        report_test.cpp
)
target_include_directories(gpu_runtime_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(gpu_runtime_tests PRIVATE gpu_runtime GTest::gtest_main)
target_compile_features(gpu_runtime_tests PRIVATE cxx_std_17)
gtest_discover_tests(gpu_runtime_tests DISCOVERY_MODE PRE_TEST)
