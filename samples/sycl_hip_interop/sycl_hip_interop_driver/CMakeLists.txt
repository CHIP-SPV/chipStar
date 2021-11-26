set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS "")

add_executable(sycl_hiplz_interop sycl_hiplz_interop.cpp)
target_link_options(sycl_hiplz_interop PRIVATE -fsycl  -lze_loader )
target_link_libraries(sycl_hiplz_interop PRIVATE -L${CHIP_DIR_}/build/samples/sycl_hip_interop -lhipMatrixMul  -L${CHIP_DIR_}/build -lCHIP -lOpenCL)
target_compile_options(sycl_hiplz_interop PRIVATE -fsycl  -Wno-deprecated-declarations)
install(TARGETS sycl_hiplz_interop
RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}")

add_executable(sycl_hiplz_interop_usm sycl_hiplz_interop.cpp)
target_link_options(sycl_hiplz_interop_usm PRIVATE -fsycl  -lze_loader )
target_link_libraries(sycl_hiplz_interop_usm PRIVATE -L${CHIP_DIR_}/build/samples/sycl_hip_interop -lhipMatrixMul  -L${CHIP_DIR_}/build -lCHIP -lOpenCL)
target_compile_options(sycl_hiplz_interop_usm PRIVATE -fsycl  -DUSM -Wno-deprecated-declarations)
install(TARGETS sycl_hiplz_interop_usm
RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}")