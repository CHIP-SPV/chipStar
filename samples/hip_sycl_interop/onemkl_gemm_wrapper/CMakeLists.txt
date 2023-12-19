set(CMAKE_CXX_COMPILER ${ICPX_EXECUTABLE})
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations")
set(CMAKE_VERBOSE_MAKEFILE ON)
set(POSITION_INDEPENDENT_CODE ON)

add_library(onemkl_gemm_wrapper SHARED ./onemkl_gemm_wrapper.cpp)

target_link_libraries(onemkl_gemm_wrapper PUBLIC mkl_sycl mkl_intel_ilp64 mkl_sequential mkl_core)
target_include_directories(onemkl_gemm_wrapper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. "$<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>")
#target_link_options(onemkl_gemm_wrapper PRIVATE "$<LINK_ONLY:MKL::MKL>")
target_compile_options(onemkl_gemm_wrapper PRIVATE "$<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>")
