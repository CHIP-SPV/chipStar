#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2021-23 chipStar developers
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

# Compiles SOURCE treated as OpenCL to LLVM bitcode.
function(add_opencl_bitcode SOURCE OUTPUT)
  add_custom_command(
    OUTPUT "${OUTPUT}"
    DEPENDS "${SOURCE}"
    COMMAND "${CMAKE_CXX_COMPILER}" ${BITCODE_CL_COMPILE_FLAGS}
    -o "${OUTPUT}" -c "${SOURCE}"
    COMMENT "Building ${SOURCE}"
    VERBATIM)
endfunction()

foreach(SOURCE IN LISTS NON_OCML_SOURCES)
  add_opencl_bitcode(
    "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc")
  list(APPEND DEPEND_LIST "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc")
endforeach()

# required by ROCm-Device-Libs
set(ROCM_DEVICELIB_STANDALONE_BUILD ON)

set(SAVED_WARN_DEPRECATED ${CMAKE_WARN_DEPRECATED})
set(CMAKE_WARN_DEPRECATED OFF)
add_subdirectory(ROCm-Device-Libs EXCLUDE_FROM_ALL)
set(CMAKE_WARN_DEPRECATED ${SAVED_WARN_DEPRECATED})

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

# =============================================================================
# Runtime device library (rtdevlib)
#
# A collection of SPIR-V modules which are linked into user's device
# programs based on target capabilities at runtime during JIT compilation.
#
# For example, On OpenCL HIP/CUDA's atomicAdd(float*, float) may be
# implemented with corresponding OpenCL atomic operation if the target
# supports cl_ext_float_atomics extension. In that case the runtime
# would link in a module which implements the atomicAdd(float*, float)
# with the extension's atomic operations. Otherwise, the runtime links
# in a slower, emulated version.
#
# RTDEVLIB_SOURCES defines OpenCL C sources for the rtdevlib. They are
# compiled to SPIR-V binary and embedded into the CHIP
# library. <build-dir>/bitcode/rtdevlib-modules.h declares the
# embedded modules as 'std::array <basename-of-the-source>'.

# Use only characters allowed in C/C++/OpenCL-C language for the
# source names.
set(RTDEVLIB_SOURCES
  atomicAddFloat_native atomicAddFloat_emulation
  atomicAddDouble_native atomicAddDouble_emulation)

# Alias for add_opencl_bitcode.
foreach(SOURCE IN LISTS RTDEVLIB_SOURCES)
 add_opencl_bitcode(
   "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}.cl"
   "${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.bc")
 list(APPEND RTDEVLIB_BITCODES "${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.bc")
endforeach()

add_custom_target("rtdevlib-bitcodes" DEPENDS ${RTDEVLIB_BITCODES})

# Compile LLVM bitcode to SPIR-V binary which is then embedded into
# std::array<unsigned char, N> <ARRAY_NAME>.
function(embed_spirv_in_cpp ARRAY_NAME BC_SOURCE OUTPUT_SOURCE OUTPUT_HEADER)
  set(SPIRV_EXTENSIONS "+SPV_EXT_shader_atomic_float_add")
  get_filename_component(SOURCE_BASENAME "${BC_SOURCE}" NAME_WLE)
  # Name of the intermediate SPIR-V binary. The name of the C array will be
  # based on this filepath (with punctuation replaced with "_").
  set(SPIR_BINARY ${SOURCE_BASENAME}.spv)
  add_custom_command(
    OUTPUT "${OUTPUT_SOURCE}" "${OUTPUT_HEADER}"
    DEPENDS "${BC_SOURCE}"
    BYPRODUCTS "${SPIR_BINARY}"
    COMMAND "${LLVM_SPIRV}" --spirv-ext=${SPIRV_EXTENSIONS}
    "${BC_SOURCE}" -o "${SPIR_BINARY}"
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/embed-binary-in-cpp.bash
    ${ARRAY_NAME} ${SPIR_BINARY} ${OUTPUT_SOURCE} ${OUTPUT_HEADER}
    COMMENT "Generating embedded SPIR-V binary: ${OUTPUT_SOURCE}"
    VERBATIM
  )
endfunction()

foreach(SOURCE IN LISTS RTDEVLIB_SOURCES)
  embed_spirv_in_cpp(
    ${SOURCE}
    "${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.bc"
    "${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.h")
  list(APPEND RTDEVLIB_SOURCES
    "${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.cc")
  list(APPEND RTDEVLIB_HEADERS
    "${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.h")
endforeach()

# Generate header that declares all the arrays of the embedded SPIR-V binaries.
set(RTDEVLIB_HEADER "rtdevlib-modules.h")
string(REPLACE ";" " " RTDEVLIB_HEADERS_WITH_SPACES "${RTDEVLIB_HEADERS}")
add_custom_command(
  OUTPUT ${RTDEVLIB_HEADER}
  DEPENDS ${RTDEVLIB_HEADERS}
  COMMAND bash -c "cat ${RTDEVLIB_HEADERS_WITH_SPACES} > ${RTDEVLIB_HEADER}"
  COMMENT "Generating rtdevlib header."
  VERBATIM
)

add_custom_target(rtdevlib-header DEPENDS ${RTDEVLIB_HEADER})

add_library(rtdevlib OBJECT ${RTDEVLIB_SOURCES} ${RTDEVLIB_HEADER})
