# # Hip2SyCL

# add_hiplz_sycl_test(hip2syclMM hiplz_sycl_interop.cpp)
add_subdirectory(onemkl_gemm_wrapper)
add_chip_binary(hiplz_sycl_interop hiplz_sycl_interop.cpp)
target_include_directories(hiplz_sycl_interop PRIVATE onemkl_gemm_wrapper)
target_link_libraries(hiplz_sycl_interop onemkl_gemm_wrapper)


# add_chip_library(hipMatrixMul hipMatrixMul.cpp)
# set(CMAKE_CXX_STANDARD_REQUIRED OFF)
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
# add_subdirectory(sycl_hip_interop_driver)


