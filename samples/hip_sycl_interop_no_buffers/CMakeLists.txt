add_subdirectory(onemkl_gemm_wrapper_no_buffers)

add_chip_binary(hip_sycl_interop_no_buffers hip_sycl_interop.cpp)
add_dependencies(samples hip_sycl_interop_no_buffers)
target_link_options(hip_sycl_interop_no_buffers PRIVATE -fsycl -L${CMAKE_BINARY_DIR} ${INTEL_LIBS} -Wl,-rpath=${CMAKE_BINARY_DIR}:${CMAKE_CURRENT_BINARY_DIR}/onemkl_gemm_wrapper_no_buffers:${ICPX_CORE_LIBDIR}:${ICPX_SYCL_LIBDIR})
target_link_libraries(hip_sycl_interop_no_buffers onemkl_gemm_wrapper_no_buffers)

add_test(NAME "hip_sycl_interop_no_buffers"
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hip_sycl_interop_no_buffers" 
         )

set_tests_properties("hip_sycl_interop_no_buffers" PROPERTIES
         PASS_REGULAR_EXPRESSION "${TEST_PASS}"
         TIMEOUT 60
         )