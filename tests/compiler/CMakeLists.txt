#=============================================================================
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
    COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc
    ${TESTOPT_HIPCC_OPTIONS}
    ${CMAKE_CURRENT_SOURCE_DIR}/${MAIN_SOURCE} -o /dev/null)
endfunction()

# add_shell_test(<script-file>)
function(add_shell_test SCRIPT)
  get_filename_component(TEST_NAME ${SCRIPT} NAME_WLE)
  configure_file(${SCRIPT} ${SCRIPT} @ONLY)
  add_test(NAME ${TEST_NAME}
    COMMAND /bin/bash ${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT})
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

add_hipcc_test(TestDoubleShuffleOverloads.hip HIPCC_OPTIONS -fsyntax-only)
add_hipcc_test(TestHipComplexInclude.hip HIPCC_OPTIONS)

add_hipcc_test(TestHipccAcceptCcFiles.cc HIPCC_OPTIONS)
add_hipcc_test(TestHipccAcceptCppFiles.cpp HIPCC_OPTIONS)

add_hipcc_test(Test513Regression.hip HIPCC_OPTIONS)

add_test(NAME "TestHipccNeedsDashO"
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc
  ${CMAKE_CURRENT_SOURCE_DIR}/TestHipccNeedDashO.cc)

add_hipcc_test(TestHipVersion.cc HIPCC_OPTIONS)

add_test(NAME "TestHipccHalfConversions"
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc -D__HIP_NO_HALF_CONVERSIONS__=1
  ${CMAKE_CURRENT_SOURCE_DIR}/TestHipccHalfConversions.cc)

add_test(NAME "TestHipccHalfOperators"
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc -D__HIP_NO_HALF_OPERATORS__=1
  ${CMAKE_CURRENT_SOURCE_DIR}/TestHipccHalfConversions.cc)

add_hipcc_test(TestZeroLenArrayTypes.hip TEST_NAME TestZeroLenArrayTypes-O0
  HIPCC_OPTIONS -O0 -c)
add_hipcc_test(TestZeroLenArrayTypes.hip TEST_NAME TestZeroLenArrayTypes-O1
  HIPCC_OPTIONS -O1 -c)
add_hipcc_test(TestPrintfLonePercent.hip HIPCC_OPTIONS -c)

add_shell_test(TestHipccCompileAndLink.bash)
add_shell_test(TestHipccCompileThenLink.bash)

add_shell_test(TestHipccAcceptCFiles.bash)
add_shell_test(TestRDCWithSingleHipccCmd.bash)
add_shell_test(TestRDCWithMultipleHipccCmds.bash)
add_shell_test(TestWholeProgramCompilation.bash)

add_shell_test(TestHipccDashX.bash)

add_test(NAME "TestHipccMultiSource" COMMAND 
  ${CMAKE_BINARY_DIR}/bin/hipcc ${CMAKE_CURRENT_SOURCE_DIR}/TestHipccCompileThenLinkMain.cpp ${CMAKE_CURRENT_SOURCE_DIR}/TestHipccCompileThenLinkKernel.cpp -o TestHipccMultiSource)

add_hipcc_test(TestLdg.hip HIPCC_OPTIONS -fsyntax-only)
add_hipcc_test(TestSwitchCase.hip HIPCC_OPTIONS -O1 -c)
add_hipcc_test(TestHostSideHIPVectors.hip HIPCC_OPTIONS -fsyntax-only)
