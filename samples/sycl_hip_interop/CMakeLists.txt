# SyCL2Hip
add_chip_library(hipMatrixMul hipMatrixMul.cpp)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
add_subdirectory(sycl_hip_interop_driver)
