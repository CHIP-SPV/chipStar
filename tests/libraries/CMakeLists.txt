if (CHIP_BUILD_HIPBLAS)
    add_hip_test(hipBLAS-Sasum.cpp)
    target_link_libraries(hipBLAS-Sasum hipblas)
endif()