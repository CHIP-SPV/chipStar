#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2021-22 chipStar developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

#include(/home/pvelesko/chipStar/install/cmake/CHIP/CHIPTargets.cmake)
option(SAVE_TEMPS "Save temporary compilation products" OFF)
option(VERBOSE "Verbose compilation" OFF)

set(CHIP_SAMPLE_BINDIR bin/chip_spv_samples)

if(SAVE_TEMPS)
  add_compile_options("--save-temps")
endif()

if(VERBOSE)
  add_compile_options("-v")
endif()

# ARGN = test args
function(add_chip_test EXEC_NAME TEST_NAME TEST_PASS SOURCE)
    message(STATUS "chipStar Sample test: ${EXEC_NAME} ${SOURCE}")
    set(TEST_EXEC_ARGS ${ARGN})
    set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCE})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" CHIP deviceInternal)
    target_include_directories("${EXEC_NAME}" PUBLIC ${CHIP_SRC_DIR}/HIP/include ${CHIP_SRC_DIR}/include/hip ${CHIP_SRC_DIR}/include)
    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}")

    add_test(NAME "${TEST_NAME}"
             COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME}" ${TEST_EXEC_ARGS}
             )

    set_tests_properties("${TEST_NAME}" PROPERTIES
             PASS_REGULAR_EXPRESSION "${TEST_PASS}"
             LABELS internal)

endfunction()


# ARGN = sources
function(add_chip_binary EXEC_NAME)

    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCES})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)
    target_include_directories("${EXEC_NAME}" PUBLIC ${CHIP_SRC_DIR}/HIP/include ${CHIP_SRC_DIR}/include)
    target_link_libraries("${EXEC_NAME}" CHIP deviceInternal)

    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}")

endfunction()

function(add_chip_library EXEC_NAME)

    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)

    add_library("${EXEC_NAME}" STATIC ${SOURCES})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" CHIP deviceInternal)
    target_include_directories("${EXEC_NAME}" PUBLIC ${CHIP_SRC_DIR}/HIP/include ${CHIP_SRC_DIR}/include)

endfunction()

set(SAMPLES
    abort
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
    7_streams
    9_unroll
    10_memcpy3D
    11_device
    hipStreamSemantics
    hipKernelLaunchIsNonBlocking
    hipMultiThreadAddCallback
    hipInfo
    hipSymbol
    hip_async_interop
    hip-cuda
    printf
    shuffles
    simple_kernel
    clock
    graphMatrixMultiply
    graphs
    ccompat
    hipComplex
    hipHostMallocSample
)
# We skip this test on macOS, because macho does not support EXCLUDE section flags (used by clang-offload-bundler)
if (NOT APPLE)
set(SAMPLES
    ${SAMPLES}
    hipDeviceLink
)
endif()

include(mkl_and_icpx)
# Add samples that depend on Intel's oneAPI compiler - if available.
if (SYCL_AVAILABLE)
  list(APPEND SAMPLES sycl_hip_interop)
  if (LevelZero_LIBRARY)
      list(APPEND SAMPLES hip_sycl_interop)
      list(APPEND SAMPLES hip_sycl_interop_no_buffers)
  endif()

else()
  message(STATUS "icpx or MKL not found: Not building SYCL interop samples.")
endif()

add_custom_target(samples DEPENDS ${SAMPLES})
foreach (SAMPLE ${SAMPLES})
  add_subdirectory(${SAMPLE})
endforeach()

add_subdirectory(cuda_samples)
