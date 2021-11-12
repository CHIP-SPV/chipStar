enable_testing()
add_executable(tests tests/TestCHIPDriver.cc tests/TestCHIPBackend.cc tests/TestCHIPBackendOpenCL.cc)
target_compile_definitions(tests PRIVATE -D__HIP_PLATFORM_SPIRV__ -D__HIP_PLATFORM_CLANG__)
target_link_libraries(tests PRIVATE CHIP ${CHIP-SPV_LIBRARIES})
