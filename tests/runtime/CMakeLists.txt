#======================================================================#  CMake build system files
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
#======================================================================
set(CHIP_SKIP_TEST 77)

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

  target_compile_definitions("${EXEC_NAME}"
    PRIVATE CHIP_SKIP_TEST=${CHIP_SKIP_TEST})

  add_test(NAME ${EXEC_NAME} COMMAND ${HIP_PROFILE_TESTS_COMMAND} ${SKIP_DOUBLE_TESTS} ${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME})

  set_tests_properties("${EXEC_NAME}" PROPERTIES
    SKIP_RETURN_CODE ${CHIP_SKIP_TEST})
  set_tests_properties("${EXEC_NAME}" PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS")
  set_tests_properties("${EXEC_NAME}" PROPERTIES
    FAIL_REGULAR_EXPRESSION "FAIL")
  set_tests_properties("${EXEC_NAME}" PROPERTIES
    SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")

endfunction()

# add_shell_test(<script-file>)
function(add_shell_test SCRIPT)
  get_filename_component(TEST_NAME ${SCRIPT} NAME_WLE)
  configure_file(${SCRIPT} ${SCRIPT} @ONLY)
  add_test(NAME ${TEST_NAME}
    COMMAND /bin/bash ${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT})
endfunction()

# Tests are registered by globbing Test*.hip and Test*.cpp, so adding one is a
# matter of dropping the file into this directory. CONFIGURE_DEPENDS re-runs the
# glob during the build when the set of matching files changes.
#
# Anything that needs more than a plain add_hip_runtime_test() call is listed
# here and registered by hand below instead.
set(RUNTIME_TESTS_MANUAL
  TestBufferDevAddr.hip           # guarded on OpenCL_LIBRARY
  TestEnvVars.hip                 # driven by ../run_testenvvars.sh, not by ctest
  TestFix1393ExitDuringModuleBuild.hip # run under an interposer by a driver script
  TestFixLlvmUsedAddrspace.hip    # not registered as a test
  TestModuleCacheInvalidation.hip # run several times through a driver script
  TestStreamWaitEventOrdering.cpp # not registered as a test
  TestTemplateKernelModules.hip   # multi-TU, linked from hand-built objects
)

file(GLOB RUNTIME_TEST_SOURCES CONFIGURE_DEPENDS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/Test*.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/Test*.cpp)
list(REMOVE_ITEM RUNTIME_TEST_SOURCES ${RUNTIME_TESTS_MANUAL})
list(SORT RUNTIME_TEST_SOURCES)
foreach(RUNTIME_TEST_SOURCE IN LISTS RUNTIME_TEST_SOURCES)
  add_hip_runtime_test(${RUNTIME_TEST_SOURCE})
endforeach()

# Names that the Test* glob does not match.
add_hip_runtime_test(RegressionTest302.hip)
add_hip_runtime_test(host-math-funcs.hip)

# CatchMemLeak1 monitors RSS growth which is unreliable on macOS
# (different ru_maxrss units, PoCL JIT memory behavior varies)
if(NOT APPLE)
  add_hip_runtime_test(CatchMemLeak1.hip)
endif()
if(OpenCL_LIBRARY) # Depends on OpenCL
  add_hip_runtime_test(TestBufferDevAddr.hip)
endif()

add_shell_test(TestAssert.bash)
add_shell_test(TestAssertFail.bash)
add_shell_test(TestForgottenModuleUnload.bash)
add_shell_test(TestModuleLoadFromFile.bash)
add_shell_test(TestLinkHiprtc.bash)
add_shell_test(TestRuntimeWarnings.bash)

find_program(SPIRV_DIS spirv-dis)
find_program(
  CLANG_OFFLOAD_BUNDLER
  NAMES clang-offload-bundler-${LLVM_VERSION_MAJOR} clang-offload-bundler
  PATHS ${HIP_CLANG_PATH})
if(SPIRV_DIS AND (NOT USE_NEW_OFFLOAD_DRIVER OR CLANG_OFFLOAD_BUNDLER))
  add_shell_test(TestBoolKernelParamSPIRV.bash)
endif()

add_shell_test(../run_testenvvars.sh)
set_tests_properties(run_testenvvars PROPERTIES
  TIMEOUT 60)

######################################
# Per-test properties, for tests registered above.
######################################

# This test checks if zeEventQuery is a blocking operation
# TODO - this should check if immediate cmd lists are being used
set_tests_properties(TestRecordEventBlocking PROPERTIES
  TIMEOUT 60)

# TestCompileError intentionally triggers ZE_RESULT_ERROR_MODULE_LINK_FAILURE
# so we need to clear the default FAIL regex which would match "FAILURE"
set_tests_properties(TestCompileError PROPERTIES FAIL_REGULAR_EXPRESSION "")

# Reproducer for issue #632: __syncthreads() must fence global memory.
set_tests_properties(TestSyncThreadsGlobalMem PROPERTIES TIMEOUT 300)

set_tests_properties(TestIGCCaching PROPERTIES
  TIMEOUT 60
  ENVIRONMENT "CHIP_MODULE_CACHE_DIR=/tmp/chipstar_test_cache_igc")

# Test hipLaunchHostFunc with multi-stream and events (from PR #1071)
set_tests_properties(TestHipLaunchHostFuncMultiStream PROPERTIES
  TIMEOUT 30)

set_tests_properties(TestEventResetWarnFlood PROPERTIES
  ENVIRONMENT "CHIP_LOGLEVEL=warn"
  FAIL_REGULAR_EXPRESSION "FAIL;reset\\(\\) called while event is recording")

# Reproducer for #1360 (sneaky_snake miscompile). Force -O2: the
# __clz(0)-poison miscompile only manifests at -O1 and above.
target_compile_options(TestSnakeMiscompileO2 PRIVATE -O2)

######################################
# Extra ctest invocations of already-registered executables.
######################################

add_test(NAME TestDefaultBackend_NoBE
  COMMAND env -u CHIP_BE -u CHIP_DEVICE_TYPE -u CHIP_PLATFORM -u CHIP_DEVICE
    ${CMAKE_CURRENT_BINARY_DIR}/TestDefaultBackend)
set_tests_properties(TestDefaultBackend_NoBE PROPERTIES
  PASS_REGULAR_EXPRESSION "PASS")

# Regression test for https://github.com/CHIP-SPV/chipStar/issues/1199
# hipGetBackendNativeHandles must return "opencl" or "level0" in NativeHandles[0],
# never "default", even when CHIP_BE is unset.
add_test(NAME TestNativeHandlesBackendName_NoBE
  COMMAND env -u CHIP_BE -u CHIP_DEVICE_TYPE -u CHIP_PLATFORM -u CHIP_DEVICE
    ${CMAKE_CURRENT_BINARY_DIR}/TestNativeHandlesBackendName)
set_tests_properties(TestNativeHandlesBackendName_NoBE PROPERTIES
  PASS_REGULAR_EXPRESSION "PASS"
  SKIP_RETURN_CODE ${CHIP_SKIP_TEST}
  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")

######################################
# Tests built by hand.
######################################

# Multi-TU test: same template instantiation in two separate TUs (#808)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/TemplatedKernelAddOne.o
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc -c
    ${CMAKE_CURRENT_SOURCE_DIR}/inputs/TemplatedKernelAddOne.hip
    -o ${CMAKE_CURRENT_BINARY_DIR}/TemplatedKernelAddOne.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/inputs/TemplatedKernelAddOne.hip
    hipcc.bin CHIP devicelib_bc
)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/TemplatedKernelAddTwo.o
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc -c
    ${CMAKE_CURRENT_SOURCE_DIR}/inputs/TemplatedKernelAddTwo.hip
    -o ${CMAKE_CURRENT_BINARY_DIR}/TemplatedKernelAddTwo.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/inputs/TemplatedKernelAddTwo.hip
    hipcc.bin CHIP devicelib_bc
)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/TestTemplateKernelModules.o
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc -c
    ${CMAKE_CURRENT_SOURCE_DIR}/TestTemplateKernelModules.hip
    -o ${CMAKE_CURRENT_BINARY_DIR}/TestTemplateKernelModules.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/TestTemplateKernelModules.hip
    hipcc.bin CHIP devicelib_bc
)
add_custom_target(TestTemplateKernelModules ALL
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc
    ${CMAKE_CURRENT_BINARY_DIR}/TestTemplateKernelModules.o
    ${CMAKE_CURRENT_BINARY_DIR}/TemplatedKernelAddOne.o
    ${CMAKE_CURRENT_BINARY_DIR}/TemplatedKernelAddTwo.o
    -o ${CMAKE_CURRENT_BINARY_DIR}/TestTemplateKernelModules
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/TestTemplateKernelModules.o
    ${CMAKE_CURRENT_BINARY_DIR}/TemplatedKernelAddOne.o
    ${CMAKE_CURRENT_BINARY_DIR}/TemplatedKernelAddTwo.o
  COMMENT "Linking TestTemplateKernelModules (multi-TU)"
)
add_test(NAME TestTemplateKernelModules
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/TestTemplateKernelModules)
set_tests_properties(TestTemplateKernelModules PROPERTIES
  PASS_REGULAR_EXPRESSION "PASS"
  FAIL_REGULAR_EXPRESSION "device function is already registered"
  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST"
  TIMEOUT 60)

# Module cache invalidation. The executable is built like any other runtime
# test, but it has to be run several times with different compilation inputs
# under one fresh cache directory, so it is registered through a driver script
# instead of add_hip_runtime_test's single invocation.
set_source_files_properties(TestModuleCacheInvalidation.hip PROPERTIES LANGUAGE CXX)
add_executable(TestModuleCacheInvalidation TestModuleCacheInvalidation.hip)
set_target_properties(TestModuleCacheInvalidation PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(TestModuleCacheInvalidation CHIP deviceInternal)
target_include_directories(TestModuleCacheInvalidation
  PRIVATE $<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>)
target_compile_definitions(TestModuleCacheInvalidation
  PRIVATE CHIP_SKIP_TEST=${CHIP_SKIP_TEST})
add_test(NAME TestModuleCacheInvalidation
  COMMAND ${CMAKE_COMMAND}
    -DTEST_EXECUTABLE=$<TARGET_FILE:TestModuleCacheInvalidation>
    -P ${CMAKE_SOURCE_DIR}/tests/run_module_cache_invalidation_test.cmake)
set_tests_properties(TestModuleCacheInvalidation PROPERTIES
  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")

# A device compiler that terminates the process during a module build must not
# leave the runtime deadlocked in its own exit handler (#1393). The exit is
# injected by an LD_PRELOAD interposer rather than by feeding a backend a module
# it rejects, so the test does not move with backend compiler behaviour, and a
# driver script owns the verdict because the expected outcome is a specific
# non-zero exit status.
if(UNIX AND NOT APPLE)
  set_source_files_properties(TestFix1393ExitDuringModuleBuild.hip
    PROPERTIES LANGUAGE CXX)
  add_executable(TestFix1393ExitDuringModuleBuild
    TestFix1393ExitDuringModuleBuild.hip)
  set_target_properties(TestFix1393ExitDuringModuleBuild PROPERTIES
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(TestFix1393ExitDuringModuleBuild CHIP deviceInternal)
  target_include_directories(TestFix1393ExitDuringModuleBuild
    PRIVATE $<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>)

  add_library(ExitDuringModuleBuildInterposer MODULE
    ExitDuringModuleBuildInterposer.cpp)
  target_link_libraries(ExitDuringModuleBuildInterposer ${CMAKE_DL_LIBS})
  # backtrace_symbols() names a frame only if its symbol is in a dynamic symbol
  # table, which is how the interposer recognises getOrCreateModule.
  target_link_options(ExitDuringModuleBuildInterposer PRIVATE -rdynamic)

  add_test(NAME TestFix1393ExitDuringModuleBuild
    COMMAND ${CMAKE_COMMAND}
      -DTEST_EXECUTABLE=$<TARGET_FILE:TestFix1393ExitDuringModuleBuild>
      -DINTERPOSER=$<TARGET_FILE:ExitDuringModuleBuildInterposer>
      -DEXPECTED_EXIT_CODE=93
      -DTEST_TIMEOUT=60
      -P ${CMAKE_SOURCE_DIR}/tests/run_exit_during_module_build_test.cmake)
  set_tests_properties(TestFix1393ExitDuringModuleBuild PROPERTIES
    SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST"
    TIMEOUT 120)
endif()
