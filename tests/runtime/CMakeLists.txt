#=============================================================================
#  CMake build system files
#
#  Copyright (c) 2023 chipStar developers
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

function(add_hip_runtime_test MAIN_SOURCE)
  get_filename_component(EXEC_NAME ${MAIN_SOURCE} NAME_WLE)
  set_source_files_properties(${MAIN_SOURCE} PROPERTIES LANGUAGE CXX)
  add_executable("${EXEC_NAME}" ${MAIN_SOURCE})
  set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)
  target_link_libraries("${EXEC_NAME}" CHIP deviceInternal)

  # Inherit all include directories of the CHIP including the private
  # ones for internal testing purposes only.
  target_include_directories("${EXEC_NAME}"
    PRIVATE $<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>)

  add_test(NAME ${EXEC_NAME} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME})

  set_tests_properties("${EXEC_NAME}" PROPERTIES
    SKIP_REGULAR_EXPRESSION "SKIP")
endfunction()

# add_shell_test(<script-file>)
function(add_shell_test SCRIPT)
  get_filename_component(TEST_NAME ${SCRIPT} NAME_WLE)
  configure_file(${SCRIPT} ${SCRIPT} @ONLY)
  add_test(NAME ${TEST_NAME}
    COMMAND /bin/bash ${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT})
endfunction()

# TODO - this should check if immediate cmd lists are being used
add_hip_runtime_test(TestRecordEventBlocking.cpp)
add_hip_runtime_test(TestLazyModuleInit.cpp)
add_hip_runtime_test(TestKernelArgs.hip)
add_hip_runtime_test(TestUndefKernelArg.hip)
add_hip_runtime_test(RegressionTest302.hip)
add_hip_runtime_test(TestLargeGlobalVar.hip)
add_hip_runtime_test(TestCompileError.hip)
add_hip_runtime_test(TestGlobalVarInit.hip)
add_hip_runtime_test(TestArgVisitors.cpp)
add_hip_runtime_test(TestLargeKernelArgLists.hip)
add_hip_runtime_test(TestStlFunctions.hip)
add_hip_runtime_test(TestStlFunctionsDouble.hip)
add_hip_runtime_test(TestHIPMathFunctions.hip)
add_hip_runtime_test(TestAtomics.hip)
add_hip_runtime_test(TestIndirectMappedHostAlloc.hip)

add_shell_test(TestAssert.bash)
add_shell_test(TestAssertFail.bash)
add_shell_test(TestForgottenModuleUnload.bash)

add_hip_runtime_test(TestIndirectCall.hip)
add_hip_runtime_test(TestStructWithFnPtr.hip)
add_hip_runtime_test(TestHipInit.hip)

add_shell_test(TestRuntimeWarnings.bash)
