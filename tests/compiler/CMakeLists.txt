#=============================================================================
#  Copyright (c) 2022 CHIP-SPV developers
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

# add_hipcc_test(<main-source>
#  [TEST_NAME <name>]
#  [HIPCC_OPTIONS <option>...])
function(add_hipcc_test MAIN_SOURCE)
  set(oneValueArgs TEST_NAME)
  set(multiValueArgs HIPCC_OPTIONS)
  cmake_parse_arguments(TESTOPT
    "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(TESTOPT_TEST_NAME)
    set(TEST_NAME ${TESTOPT_TEST_NAME})
  else()
    get_filename_component(MAIN_NAME ${MAIN_SOURCE} NAME_WLE)
    set(TEST_NAME "hipcc-${MAIN_NAME}")
  endif()

  add_test(NAME "${TEST_NAME}"
    COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc.bin
    ${TESTOPT_HIPCC_OPTIONS}
    ${CMAKE_CURRENT_SOURCE_DIR}/${MAIN_SOURCE} -o /dev/null)
endfunction()

add_hipcc_test(TestNoinlineAttrs.hip HIPCC_OPTIONS -c)
add_hipcc_test(TestAPIObjects.hip HIPCC_OPTIONS -c)

# Test the hip_runtime.h header compiles in C++ mode. If the test
# passes - the header probably is compilable by other regular C++
# compilers too (e.g. g++).
add_hipcc_test(TestHipRuntimeHeaderInclude.cpp
  TEST_NAME hip-runtime-header-cpp-mode
  HIPCC_OPTIONS -x c++ -fsyntax-only)

add_hipcc_test(TestAtomicAddNoRet.hip HIPCC_OPTIONS -fsyntax-only)
add_hipcc_test(TestFloat128Macros.hip TEST_NAME TestFloat128gnupp14
  HIPCC_OPTIONS -std=gnu++14 -fsyntax-only)
  