#=============================================================================
#  CMake build system files
#
#  Copyright (c) 2022 chipStar developers
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#
#=============================================================================

add_hip_test(implicitCasts.cpp)

# Add this as a hipcc otherwise the `make all` will fail which is not convenient indicator for test failure
add_test(NAME "deviceMallocCompile"
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc
  ${CMAKE_CURRENT_SOURCE_DIR}/deviceMallocCompile.cpp)

add_hip_test(sincospifSpotTest.cc)
add_hip_test(shfl_sync.cpp)

set_tests_properties(shfl_sync PROPERTIES
  FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(shfl_sync PROPERTIES
  PASS_REGULAR_EXPRESSION "PASS")

# Verify OCML (irif.h) does not emit i128 - buggy irif uses wrong types for OpenCL
# Run after build (ocml built as part of devicelib)
add_test(NAME irif_no_i128
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../irif_no_i128.bash
    ${CMAKE_BINARY_DIR}
    ${CLANG_ROOT_PATH_BIN}/llvm-dis)
set_tests_properties(irif_no_i128 PROPERTIES
  FAIL_REGULAR_EXPRESSION "FAIL"
  PASS_REGULAR_EXPRESSION "PASS")
