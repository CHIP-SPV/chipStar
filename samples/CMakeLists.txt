
option(SAVE_TEMPS "Save temporary compilation products" OFF)
option(VERBOSE "Verbose compilation" OFF)

if(SAVE_TEMPS)
  add_compile_options("--save-temps")
endif()

if(VERBOSE)
  add_compile_options("-v")
endif()

# ARGN = test args
function(add_hiplz_test EXEC_NAME TEST_NAME TEST_PASS SOURCE)

    set(TEST_EXEC_ARGS ${ARGN})
    set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCE})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)
    
    target_link_libraries("${EXEC_NAME}" "${SANITIZER_LIBS}" "hiplz" "OpenCL" "ze_loader")

    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${HIPLZ_SAMPLE_BINDIR}")

    add_test(NAME "${TEST_NAME}"
             COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME}" ${TEST_EXEC_ARGS}
             )

    set_tests_properties("${TEST_NAME}" PROPERTIES
             PASS_REGULAR_EXPRESSION "${TEST_PASS}")


endfunction()


# ARGN = sources
function(add_hiplz_binary EXEC_NAME)

    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCES})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" "${SANITIZER_LIBS}" "hiplz" "OpenCL" "ze_loader")

    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${HIPLZ_SAMPLE_BINDIR}")

endfunction()

# ARGN = sources
function(add_hiplz_binary_device_link EXEC_NAME)
    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCES})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" "${SANITIZER_LIBS}" "hiplz" "OpenCL" "ze_loader")

    target_compile_options("${EXEC_NAME}" PRIVATE "-fgpu-rdc")

    target_link_options("${EXEC_NAME}" PRIVATE
        "-fgpu-rdc"
        "--hip-link"
        "$<INSTALL_INTERFACE:--hip-llvm-pass-path=${HIPLZ_LLVM_DIR}>"
        "$<BUILD_INTERFACE:--hip-llvm-pass-path=${CMAKE_BINARY_DIR}/llvm_passes>"
        "$<INSTALL_INTERFACE:--hip-device-lib-path=${HIPLZ_DATA_DIR}>"
        "$<BUILD_INTERFACE:--hip-device-lib-path=${CMAKE_BINARY_DIR}>"
        "--hip-device-lib=kernellib.bc")

    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${HIPLZ_SAMPLE_BINDIR}")

endfunction()

# ARGN = sources
function(add_hiplz_binary_lib LIB_NAME)
    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)

    add_library("${LIB_NAME}" SHARED ${SOURCES})

    set_target_properties("${LIB_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_compile_options("hiplz" PRIVATE "-fPIC")

    target_link_options("hiplz" PRIVATE "-shared")

    target_link_libraries("${LIB_NAME}" "${SANITIZER_LIBS}" "hiplz" "OpenCL" "ze_loader")

    install(TARGETS "${LIB_NAME}"
            RUNTIME DESTINATION "${HIPLZ_SAMPLE_BINDIR}")
endfunction()

# ARGN = sources
function(add_hiplz_sycl_test EXEC_NAME)
    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)

    add_library( onemkl_gemm_wrapper SHARED IMPORTED )
    set_target_properties( onemkl_gemm_wrapper PROPERTIES IMPORTED_LOCATION "${HIPLZ_SYCL_TEST_LIB}" )
    
    add_executable("${EXEC_NAME}" ${SOURCES})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" "${SANITIZER_LIBS}" "hiplz" "OpenCL" "ze_loader" "onemkl_gemm_wrapper")

    target_link_options("${EXEC_NAME}" PRIVATE "${HIPLZ_SYCL_TEST_LIB}")
     
    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${HIPLZ_SAMPLE_BINDIR}")

endfunction()

set(SAMPLES
    hipmath
    hiptest
    bit_extract
    hcc_dialects
    fp16
    0_MatrixTranspose
    0_MatrixMultiply
    1_hipEvent
    2_vecadd
    3_shared_memory
    4_shfl
    5_2dshfl
    6_dynamic_shared
    hipInfo
    7_streams
    9_unroll
    10_memcpy3D
    11_device
    hipSymbol
    hipDeviceLink
    sycl_hip_interop
    #hip_sycl_interop
)

foreach (SAMPLE ${SAMPLES})
  add_subdirectory(${SAMPLE})
endforeach()

add_subdirectory(hip-cuda)
