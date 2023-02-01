#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2021-22 CHIP-SPV developers
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
if(NOT CLANG_VERSION_LESS_13)
  list(APPEND EXTRA_FLAGS "-cl-no-stdinc")
endif()

# disable opaque pointers for LLVM 15 only
if(NOT CLANG_VERSION_LESS_15 AND CLANG_VERSION_LESS_16)
  list(APPEND EXTRA_FLAGS ${DISABLE_OPAQUE_PTRS_OPT})
endif()

if("${LLVM_VERSION}" VERSION_LESS 14.0)
  # Definitions for pre-upstreamed HIP-Clang.
  set(BC_TRIPLE "spir64-unknown-unknown")
  set(BC_FILE "devicelib.bc")
  set(BC_DESTINATION share)
else()
  # HIP-Clang automatically looks for hipspv-<triple>.bc in
  # <bc-destination> directory in HIP compilation mode.
  set(BC_TRIPLE "spirv64")
  set(BC_FILE "hipspv-${BC_TRIPLE}.bc")
  set(BC_DESTINATION lib/hip-device-lib)
endif()

set(BITCODE_COMPILE_FLAGS
  -Xclang -finclude-default-header -O2 -x cl -cl-std=CL2.0
  --target=${BC_TRIPLE} -DDEFAULT_WARP_SIZE=${DEFAULT_WARP_SIZE} -emit-llvm ${EXTRA_FLAGS})

# non-OCML sources
set(NON_OCML_SOURCES "devicelib" "_cl_print_str" "texture") # "printf_support"

foreach(SOURCE IN LISTS NON_OCML_SOURCES)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}.cl"
    COMMAND "${CMAKE_CXX_COMPILER}" ${BITCODE_COMPILE_FLAGS}
    -o "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc"
    -c "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}.cl"
    COMMENT "Building ${SOURCE}.bc"
    VERBATIM)
  list(APPEND DEPEND_LIST "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc")
endforeach()

# OCML sources
set(OCML_SOURCES erfcinvD erfcinvF erfcxD erfcxF erfinvD erfinvF
    i0D i0F i1D i1F j0D j0F j1D j1F ncdfD ncdfF ncdfinvD ncdfinvF
    nearbyintD nearbyintF rcbrtD rcbrtF rhypotF rhypotD
    rlen3D rlen3F rlen4D rlen4F scalbD scalbF scalbnD scalbnF
    tables y0D y0F y1D y1F ba0F ba0D ba1F ba1D bp0F bp0D bp1F bp1D
    cosbF cosbD sinbF sinbD
    trigredD trigredF
    sincosred2D sincosred2F sincosredD sincosredF
    trigredlargeD trigredlargeF trigredsmallD trigredsmallF
    trig_preop64)

foreach(SOURCE IN LISTS OCML_SOURCES)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/OCML/${SOURCE}.cl"
    COMMAND "${CMAKE_CXX_COMPILER}" ${BITCODE_COMPILE_FLAGS}
    -o "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc"
    -c "${CMAKE_CURRENT_SOURCE_DIR}/OCML/${SOURCE}.cl"
    COMMENT "Building ${SOURCE}.bc"
    VERBATIM)
  list(APPEND DEPEND_LIST "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc")
endforeach()

# devicelib
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}"
  DEPENDS ${DEPEND_LIST}
  COMMAND ${CMAKE_COMMAND} -E make_directory
  "${CMAKE_BINARY_DIR}/${BC_DESTINATION}"
  COMMAND "${LLVM_LINK}"
  -o "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}"
  ${DEPEND_LIST}
  COMMENT "Linking device library bitcode '${BC_FILE}'"
  VERBATIM)

add_custom_target("devicelib_bc"
  DEPENDS "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}")

install(FILES "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}"
  DESTINATION ${BC_DESTINATION})
