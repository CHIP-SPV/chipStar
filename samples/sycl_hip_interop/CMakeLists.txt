# SyCL2Hip
# set(CMAKE_CXX_STANDARD_REQUIRED OFF)
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
# add_chip_library(hipMatrixMul hipMatrixMul.cpp)
# add_executable(sycl_hiplz_interop sycl_hiplz_interop.cpp)
# set(CMAKE_CXX_COMPILER icpx)
# set(CMAKE_CXX_FLAGS "")
# #message(FATAL_ERROR ${CMAKE_PROJECT_INCLUDE_BEFORE})
# #set(CMAKE_CXX_FLAGS "-fsycl -I${CHIP_DIR_}/include -lOpenCL")
# target_link_options(sycl_hiplz_interop PRIVATE -fsycl  -lze_loader hipMatrixMul )
# target_compile_options(sycl_hiplz_interop PRIVATE -fsycl  -Wno-deprecated-declarations)
#target_compile_options(sycl_hiplz_interop PRIVATE -fsycl)
# TODO : overwrite flags / build proper chain custonm 
#set_target_properties(sycl_hiplz_interop 
#PROPERTIES COMPILE_FLAGS "-lol ")

#target_include_directories(sycl_hiplz_interop PRIVATE /soft/restricted/CNDA/sdk/2021.10.30.001/oneapi/compiler/2021.4.0/linux/include/sycl)

#icpx CMakeFiles/sycl_hiplz_interop.dir/sycl_hiplz_interop.cpp.o -o sycl_hiplz_interop -g -fsycl -lze_loader ./libhipMatrixMul.a -L/home/pvelesko/CHIP-SPV/build -lCHIP -lOpenCL 