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

function(add_hipcc_test_run MAIN_SOURCE)
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

  # Build and run as a single test
  set(EXEC_PATH ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
  add_test(NAME "${TEST_NAME}-build"
    COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc
    ${TESTOPT_HIPCC_OPTIONS}
    ${CMAKE_CURRENT_SOURCE_DIR}/${MAIN_SOURCE} -o ${EXEC_PATH})

  add_test(NAME "${TEST_NAME}-run"
    COMMAND ${EXEC_PATH})
  
  # Only run if build succeeds, and check for PASSED output
  set_tests_properties("${TEST_NAME}-run" PROPERTIES
    PASS_REGULAR_EXPRESSION "PASSED"
    DEPENDS "${TEST_NAME}-build")
    
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

# Calls HipPrintf builds must use their callee's convention (#1677).
add_shell_test(TestFix1677PrintfCallConv.bash)

# Test the hip_runtime.h header compiles in C++ mode. If the test
# passes - the header probably is compilable by other regular C++
# compilers too (e.g. g++).
add_hipcc_test(TestHipRuntimeHeaderInclude.cpp
  TEST_NAME hip-runtime-header-cpp-mode
  HIPCC_OPTIONS -x c++ -fsyntax-only)

# A shared memory clone must not keep memory(argmem: none) (#1679).
add_shell_test(TestFix1679DynMemCloneMemoryAttr.bash)

add_hipcc_test(TestAtomicAddNoRet.hip HIPCC_OPTIONS -fsyntax-only)
add_hipcc_test(TestFloat128Macros.hip TEST_NAME TestFloat128gnupp14
  HIPCC_OPTIONS -std=gnu++14 -fsyntax-only)

add_hipcc_test(TestDoubleShuffleOverloads.hip HIPCC_OPTIONS -fsyntax-only)
add_hipcc_test(TestFix797ShflHalf.hip HIPCC_OPTIONS -fsyntax-only)
add_hipcc_test(TestHipComplexInclude.hip HIPCC_OPTIONS)

add_shell_test(TestFix1650NonLoweredInitGlobalAddress.bash)

add_hipcc_test(TestHipccAcceptCcFiles.cc HIPCC_OPTIONS)
add_hipcc_test(TestHipccAcceptCppFiles.cpp HIPCC_OPTIONS)

# HIP device kernel + host (CPU) OpenMP in one TU must compile and link with
# -fopenmp.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  add_hipcc_test(TestHipccFopenmp.cc HIPCC_OPTIONS -fopenmp)
endif()

# -fopenmp-simd must stay host-only; on the device clang asserts (#1566).
add_hipcc_test(TestHipccAcceptCcFiles.cc TEST_NAME TestFix1566FopenmpSimd
  HIPCC_OPTIONS -fopenmp-simd)

add_hipcc_test(Test513Regression.hip HIPCC_OPTIONS)

# The spilled kernel must call the original one the way it is declared.
add_shell_test(TestFix1678SpillerCallConv.bash)

add_test(NAME "TestHipccNeedsDashO"
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc
  ${CMAKE_CURRENT_SOURCE_DIR}/TestHipccNeedDashO.cc)

# A shared memory user that calls another user must be lowered once (#1662).
add_shell_test(TestFix1662DynMemUserCallsUser.bash)

add_hipcc_test(TestHipVersion.cc HIPCC_OPTIONS)

add_test(NAME "TestHipccHalfConversions"
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc -D__HIP_NO_HALF_CONVERSIONS__=1
  ${CMAKE_CURRENT_SOURCE_DIR}/TestHipccHalfConversions.cc)

add_test(NAME "TestHipccHalfOperators"
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc -D__HIP_NO_HALF_OPERATORS__=1
  ${CMAKE_CURRENT_SOURCE_DIR}/TestHipccHalfConversions.cc)

add_hipcc_test(TestFix1248HalfHostOps.hip HIPCC_OPTIONS -fsyntax-only)

# A call to a shared memory user must pass the caller's argument (#1670).
add_shell_test(TestFix1670DynMemCallArgs.bash)

add_hipcc_test(TestHalfDeviceMathFuncs.hip HIPCC_OPTIONS -fsyntax-only)

add_hipcc_test(TestFix1582StdTraitSpecialization.hip HIPCC_OPTIONS -fsyntax-only)

add_hipcc_test(TestRocmVersionHeader.hip HIPCC_OPTIONS -fsyntax-only)

# A call rewritten for shared memory must keep how it calls the callee (#1673).
add_shell_test(TestFix1673DynMemCallConv.bash)

add_hipcc_test(TestZeroLenArrayTypes.hip TEST_NAME TestZeroLenArrayTypes-O0
  HIPCC_OPTIONS -O0 -c)
add_hipcc_test(TestZeroLenArrayTypes.hip TEST_NAME TestZeroLenArrayTypes-O1
  HIPCC_OPTIONS -O1 -c)
add_hipcc_test(TestPrintfLonePercent.hip HIPCC_OPTIONS -c)
add_hipcc_test(inputs/testIssue606.cpp HIPCC_OPTIONS -D__HIP_PLATFORM_SPIRV__ -x c++ -c)

# Lowering a module with many shared memory users must be reproducible (#1661).
add_shell_test(TestFix1661DynMemOrder.bash)

add_shell_test(TestHipccCompileAndLink.bash)
add_shell_test(TestHipccCompileThenLink.bash)
add_shell_test(TestHipcc588.bash)
add_shell_test(TestHipcc621.bash)
add_shell_test(TestHipccAcceptCFiles.bash)
add_shell_test(TestRDCWithSingleHipccCmd.bash)
add_shell_test(TestRDCWithMultipleHipccCmds.bash)
add_shell_test(TestWholeProgramCompilation.bash)
add_shell_test(TestHipccDashX.bash)
add_shell_test(TestHipccFp16Include.bash)
add_shell_test(TestHipcc692Regression.bash)
add_shell_test(TestHipccFileOrderPreservation.bash)
add_shell_test(TestHipcc945MacroSpace.bash)
add_shell_test(TestHipccArgOrder.bash)
add_shell_test(TestFix1631HipccClangFallback.bash)
# Separate compilation and -fgpu-rdc use clang-offload-bundler which calls
# llvm-objcopy with options not supported on MachO (macOS).
if(NOT APPLE)
  add_shell_test(TestSeparateCompilation.bash)
  add_shell_test(TestRDCNoDataLayoutWarning.bash)
  # Two translation units both including hip_bf16.h, device-linked with -fgpu-rdc.
  add_shell_test(TestBFloat16RDC.bash)
endif()
add_subdirectory(rdcLink)
add_test(NAME "TestHipccMultiSource" COMMAND 
  ${CMAKE_BINARY_DIR}/bin/hipcc ${CMAKE_CURRENT_SOURCE_DIR}/TestHipccCompileThenLinkMain.cpp ${CMAKE_CURRENT_SOURCE_DIR}/TestHipccCompileThenLinkKernel.cpp -o TestHipccMultiSource)
add_shell_test(TestPrintfStaticString.bash)

add_shell_test(TestFix1680DuplicatePhiPreds.bash)
# Regression test for HipCoalesceDuplicatePhiPredsPass (#1680). Translates to
# SPIR-V and runs spirv-val; skips when LLVM_SPIRV is NOT_NEEDED.
add_shell_test(TestSpirvDuplicatePhiHip.bash)
add_shell_test(TestFix1577PointerVectors.bash)
# A constant zero length copy must not reach SPIR-V emission.
add_shell_test(TestFix1634ZeroLengthCopy.bash)
add_shell_test(TestFix1640AggregateInitGlobalAddress.bash)
if(CHIP_LLVM_USE_INTERGRATED_SPIRV)
  add_shell_test(TestFix1654FPContract.bash)
  # Fails the day the backend stops forcing ContractionOff. Read its header.
  add_shell_test(CanaryFPContractionOff.bash)
endif()
# Fails the day upstream stops rejecting <N x ptr>, which is the signal to
# delete the hip-lower-pointer-vectors pass. Read its header before acting.
add_shell_test(CanaryPointerVectorsRejected.bash)
# Fails the day llvm-spirv stops rejecting one of the hint intrinsics, which is
# the signal to delete that case. Read its header before acting. Needs the
# translator, so it is not registered when LLVM_SPIRV is NOT_NEEDED.
if(NOT LLVM_SPIRV STREQUAL "NOT_NEEDED")
  add_shell_test(CanaryHintIntrinsicsRejected.bash)
  # Fails once llvm-spirv carries KhronosGroup/SPIRV-LLVM-Translator#3866; read its header.
  add_shell_test(CanaryDuplicatePhiPredecessors.bash)
endif()

add_shell_test(TestTextureUnsupportedMixedDim.bash)

add_hipcc_test(TestLdg.hip HIPCC_OPTIONS -fsyntax-only)
add_hipcc_test(TestSwitchCase.hip HIPCC_OPTIONS -O1 -c)
# Device side C++ virtual dispatch (CHIP-SPV/chipStar#1373). Compile only: -c
# already runs the device link that translates to SPIR-V, which is where the
# unfixed compiler failed. The dispatch is executed by
# tests/runtime/TestDeviceVirtualDispatch.hip on devices that can run it.
add_hipcc_test(TestDeviceVirtualFunctions.hip HIPCC_OPTIONS -c)
# Dead vtables of an explicit instantiation must not survive into the device
# module (#1382). Inspects the lowered device bitcode.
add_shell_test(TestDeadVTableVirtualBase.bash)
# Program-scope variable init kernels must use their work items (#582).
add_shell_test(TestFix582VarInitWorkItems.bash)
add_hipcc_test(TestFix1647ShadowKernelNames.hip HIPCC_OPTIONS -c)
add_hipcc_test(TestHostSideHIPVectors.hip HIPCC_OPTIONS -fsyntax-only)
add_hipcc_test(TestAlignAttr.hip HIPCC_OPTIONS -fsyntax-only)
# Check CHIP_FAST_MATH is set for -ffast-math and preprocessor guards
# using it are not hiding errors.
add_hipcc_test(TestFastMath.hip HIPCC_OPTIONS -fsyntax-only -ffast-math)
add_hipcc_test(TestDim3Cast.hip HIPCC_OPTIONS -fsyntax-only)
# TestPlacement uses device-side new[]/delete[], which require device malloc
# (the device heap program-scope global). Skip when that feature is disabled.
if(CHIP_ENABLE_DEVICE_PROGRAM_SCOPE_GLOBALS)
  add_hipcc_test(TestPlacement.hip)
endif()

# Test for math header conflicts between system cmath and HIP device math
add_hipcc_test(TestMathHeaderConsolidated.hip HIPCC_OPTIONS -fsyntax-only)

# Builtins whose intrinsics reach the device link and fail there. -O0 only:
# LLVM's LowerConstantIntrinsics folds llvm.objectsize away from -O1 up.
foreach(HINT_TEST
    TestBuiltinPrefetch
    TestBuiltinReadCycleCounter
    TestBuiltinReadSteadyCounter
    TestBuiltinFltRounds
    TestBuiltinAllowRuntimeCheck
    TestBuiltinReturnAddress
    TestBuiltinFrameAddress
    TestBuiltinMemcpyInline
    TestBuiltinObjectSize)
  add_hipcc_test(${HINT_TEST}.hip HIPCC_OPTIONS -O0 -c)
endforeach()
# A prefetch on a global pointer reaches SPIR-V as a real OpenCL prefetch.
add_shell_test(TestPrefetchLowering.bash)

# Add the promoteInt tests
add_subdirectory(promoteInt)

# Add the IR verification tests
add_subdirectory(irVerification)

# Add the 8 and 16 bit atomic lowering pass tests
add_subdirectory(subwordAtomics)

# Add the untranslatable math intrinsic (llround / llrint / ldexp) pass tests
add_subdirectory(mathIntrinsics)
# Add the volatile global access lowering pass tests
add_subdirectory(volatileAccesses)
