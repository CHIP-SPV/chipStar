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

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/BC")

set(EXTRA_FLAGS)
# Ugly fix for interactions between clang13+ and igc
list(APPEND EXTRA_FLAGS "-cl-no-stdinc")

# disable opaque pointers for LLVM 15 only
if(NOT CLANG_VERSION_LESS_15 AND CLANG_VERSION_LESS_16)
  list(APPEND EXTRA_FLAGS ${DISABLE_OPAQUE_PTRS_OPT})
endif()

if(CHIP_ENABLE_NON_COMPLIANT_DEVICELIB_CODE)
  list(APPEND EXTRA_FLAGS "-DCHIP_ENABLE_NON_COMPLIANT_DEVICELIB_CODE=1")
endif()

if(CHIP_EXT_FLOAT_ATOMICS)
  list(APPEND EXTRA_FLAGS "-DCHIP_EXT_FLOAT_ATOMICS")
endif()

if("${LLVM_VERSION}" VERSION_LESS 14.0)
  # Definitions for pre-upstreamed HIP-Clang.
  set(BC_TRIPLE "spirv64")
  set(BC_FILE "devicelib.bc")
  set(BC_DESTINATION share)
else()
  # HIP-Clang automatically looks for hipspv-<triple>.bc in
  # <bc-destination> directory in HIP compilation mode.
  set(BC_TRIPLE "spirv64")
  set(BC_FILE "hipspv-${BC_TRIPLE}.bc")
  set(BC_DESTINATION lib/hip-device-lib)
endif()

set(BITCODE_CL_COMPILE_FLAGS
  -Xclang -finclude-default-header -O2 -x cl -cl-std=CL2.0
  --target=${BC_TRIPLE} -DDEFAULT_WARP_SIZE=${DEFAULT_WARP_SIZE} -emit-llvm
  ${EXTRA_FLAGS})
set(BITCODE_C_COMPILE_FLAGS
  --target=${BC_TRIPLE} -O2 -x c -DDEFAULT_WARP_SIZE=${DEFAULT_WARP_SIZE}
  -emit-llvm ${EXTRA_FLAGS})

# non-OCML sources
set(NON_OCML_SOURCES "devicelib" "_cl_print_str" "texture") # "printf_support"

foreach(SOURCE IN LISTS NON_OCML_SOURCES)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}.cl"
    COMMAND "${CMAKE_CXX_COMPILER}" ${BITCODE_CL_COMPILE_FLAGS}
    -o "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc"
    -c "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}.cl"
    COMMENT "Building ${SOURCE}.bc"
    VERBATIM)
  list(APPEND DEPEND_LIST "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc")
endforeach()

# required by ROCm-Device-Libs
set(ROCM_DEVICELIB_STANDALONE_BUILD ON) 
add_subdirectory(ROCm-Device-Libs EXCLUDE_FROM_ALL)

# ROCm-Device-Libs provides OCML and its dependencies (OCLC, OCKL, etc.)
# Since these targets don't seem to get exported as normal targets, we have to link this way.
set(OCML_LIBS
  ocml
  oclc_finite_only_off
  oclc_unsafe_math_off
  oclc_correctly_rounded_sqrt_off
  oclc_daz_opt_off
  oclc_isa_version_803
)

set(OCML_LIB_PATHS
  ${CMAKE_BINARY_DIR}/bitcode/ROCm-Device-Libs/ocml/ocml.lib.bc 
  ${CMAKE_BINARY_DIR}/bitcode/ROCm-Device-Libs/oclc/oclc_finite_only_off.lib.bc
  ${CMAKE_BINARY_DIR}/bitcode/ROCm-Device-Libs/oclc/oclc_unsafe_math_off.lib.bc
  ${CMAKE_BINARY_DIR}/bitcode/ROCm-Device-Libs/oclc/oclc_correctly_rounded_sqrt_off.lib.bc
  ${CMAKE_BINARY_DIR}/bitcode/ROCm-Device-Libs/oclc/oclc_daz_opt_off.lib.bc
  ${CMAKE_BINARY_DIR}/bitcode/ROCm-Device-Libs/oclc/oclc_isa_version_803.lib.bc
)

if(CHIP_ENABLE_NON_COMPLIANT_DEVICELIB_CODE)
    # add support for calling certain LLVM builtins that are not supported by HIP/CUDA
    add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/BC/c_to_opencl.bc"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/c_to_opencl.c"
    COMMAND "${CMAKE_CXX_COMPILER}" ${BITCODE_C_COMPILE_FLAGS}
    -o "${CMAKE_CURRENT_BINARY_DIR}/BC/c_to_opencl.bc"
    -c "${CMAKE_CURRENT_SOURCE_DIR}/c_to_opencl.c"
    COMMENT "Building c_to_opencl.bc"
    VERBATIM)
    list(APPEND DEPEND_LIST "${CMAKE_CURRENT_BINARY_DIR}/BC/c_to_opencl.bc")
endif()

# devicelib
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}"
  DEPENDS ${DEPEND_LIST} ${OCML_LIBS}
  COMMAND ${CMAKE_COMMAND} -E make_directory
  "${CMAKE_BINARY_DIR}/${BC_DESTINATION}"
  COMMAND "${LLVM_LINK}"
  -o "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}"
  ${DEPEND_LIST} 
  ${OCML_LIB_PATHS}
  COMMENT "Linking device library bitcode '${BC_FILE}'"
  VERBATIM)

add_custom_target("devicelib_bc"
  DEPENDS "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}")

install(FILES "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}"
  DESTINATION ${BC_DESTINATION})
