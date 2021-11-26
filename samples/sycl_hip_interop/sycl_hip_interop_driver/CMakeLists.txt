add_executable(sycl_hiplz_interop sycl_hiplz_interop.cpp)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS "")
# # #message(FATAL_ERROR ${CMAKE_PROJECT_INCLUDE_BEFORE})
# # #set(CMAKE_CXX_FLAGS "-fsycl -I${CHIP_DIR_}/include -lOpenCL")
# #add_dependencies(sycl_hiplz_interop hipMatrixMul)
target_link_options(sycl_hiplz_interop PRIVATE -fsycl  -lze_loader )
target_link_libraries(sycl_hiplz_interop PRIVATE -L${CHIP_DIR_}/build/samples/sycl_hip_interop -lhipMatrixMul  -L${CHIP_DIR_}/build -lCHIP -lOpenCL)
#target_link_libraries(sycl_hiplz_interop PUBLIC hipMatrixMul)
target_compile_options(sycl_hiplz_interop PRIVATE -fsycl  -Wno-deprecated-declarations)
install(TARGETS sycl_hiplz_interop
RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}")

#target_include_directories(sycl_hiplz_interop PRIVATE /soft/restricted/CNDA/sdk/2021.10.30.001/oneapi/compiler/2021.4.0/linux/include/sycl)

#icpx CMakeFiles/sycl_hiplz_interop.dir/sycl_hiplz_interop.cpp.o -o sycl_hiplz_interop -g -fsycl -lze_loader ./libhipMatrixMul.a -L/home/pvelesko/CHIP-SPV/build -lCHIP -lOpenCL 