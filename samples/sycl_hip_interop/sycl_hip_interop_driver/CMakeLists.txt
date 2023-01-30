set(CMAKE_CXX_COMPILER ${ICPX_EXECUTABLE})
set(CMAKE_CXX_FLAGS "")

add_executable(sycl_chip_interop sycl_chip_interop.cpp)
add_dependencies(sycl_chip_interop CHIP hipMatrixMul)
target_include_directories(sycl_chip_interop PRIVATE ${MKL_INCLUDE})

# Warning the use of manual linking here using -lCHIP is on purpose due to conflicting opencl headers that get loaded otherwise.
target_link_options(sycl_chip_interop PRIVATE -fsycl -L${CMAKE_BINARY_DIR} -Wl,-rpath=${CMAKE_BINARY_DIR}:${ICPX_CORE_LIBDIR})
target_link_libraries(sycl_chip_interop PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../libhipMatrixMul.a -L${CMAKE_BINARY_DIR} -lCHIP ${OpenCL_LIBRARY} ${LevelZero_LIBRARY} -pthread)
target_compile_options(sycl_chip_interop PRIVATE -fsycl -Wno-deprecated-declarations)
install(TARGETS sycl_chip_interop
    RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}")

add_test(NAME "sycl_chip_interop"
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/sycl_chip_interop"
)

set_tests_properties("sycl_chip_interop" PROPERTIES
    PASS_REGULAR_EXPRESSION "PASSED"
    SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST"
    TIMEOUT 60
)

add_executable(sycl_chip_interop_usm sycl_chip_interop.cpp)
add_dependencies(sycl_chip_interop_usm CHIP hipMatrixMul)
target_include_directories(sycl_chip_interop_usm PRIVATE ${MKL_INCLUDE})

# Warning the use of manual linking here using -lCHIP is on purpose due to conflicting opencl headers that get loaded otherwise.
target_link_options(sycl_chip_interop_usm PRIVATE -fsycl -L${CMAKE_BINARY_DIR} -Wl,-rpath=${CMAKE_BINARY_DIR}:${ICPX_CORE_LIBDIR})
target_link_libraries(sycl_chip_interop_usm PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../libhipMatrixMul.a -lCHIP ${OpenCL_LIBRARY} ${LevelZero_LIBRARY} -pthread)
target_compile_options(sycl_chip_interop_usm PRIVATE -fsycl -DUSM -Wno-deprecated-declarations)
install(TARGETS sycl_chip_interop_usm
    RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}")

add_dependencies(samples sycl_chip_interop sycl_chip_interop_usm)
add_test(NAME "sycl_chip_interop_usm"
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/sycl_chip_interop_usm"
)

set_tests_properties("sycl_chip_interop_usm" PROPERTIES
    PASS_REGULAR_EXPRESSION "PASSED"
    SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST"
    TIMEOUT 60
)
