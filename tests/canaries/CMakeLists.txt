#=============================================================================
#  Copyright (c) 2026 chipStar developers
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

# add_shell_test comes from tests/compiler, so this directory is added after it.

if(CHIP_LLVM_USE_INTERGRATED_SPIRV)
  # Fails the day the backend accepts the shape HipOffsetBeforeCastPass
  # rewrites. Read its header before acting.
  add_shell_test(CanaryGepOverCastInitializer.bash)
  # Fails the day the backend stops taking an initializer's byte offset as an
  # array index. Read its header before acting.
  add_shell_test(CanaryByteOffsetInitializer.bash)
  # Fails the day the backend stops emitting memmove as OpCopyMemorySized (#1958).
  add_shell_test(CanaryMemmoveCopyMemorySized.bash)
endif()

# Fails the day clang stops hoisting such an initializer, which is the signal to
# delete HipSharedAddrLocalInitPass. Read its header before acting.
add_shell_test(CanarySharedAddrInitHoisted.bash)

if(CHIP_LLVM_USE_INTERGRATED_SPIRV)
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
  # Fails once llvm-spirv carries KhronosGroup/SPIRV-LLVM-Translator#3918; read its header.
  add_shell_test(CanarySitofpI1Translator.bash)
endif()
if(CHIP_LLVM_USE_INTERGRATED_SPIRV)
  # Fails once the backend carries llvm/llvm-project#209232; read its header.
  add_shell_test(CanarySitofpI1Backend.bash)
endif()
# Fails once IGC stops gathering a constant-offset shuffle; read its header.
add_shell_test(CanaryIgcShuffleRotateGather.bash)
set_tests_properties(CanaryIgcShuffleRotateGather PROPERTIES
  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")

# Fails once IGC shuffles as_int(float) correctly on an Intel GPU without
# 8-wide sub-groups; read its header.
if(OpenCL_LIBRARY)
  add_executable(CanaryIGCShuffleIntViewOfFloat EXCLUDE_FROM_ALL
    CanaryIGCShuffleIntViewOfFloat.cpp)
  target_compile_definitions(CanaryIGCShuffleIntViewOfFloat PRIVATE
    CL_TARGET_OPENCL_VERSION=300)
  target_link_libraries(CanaryIGCShuffleIntViewOfFloat ${OpenCL_LIBRARY})
  set_target_properties(CanaryIGCShuffleIntViewOfFloat PROPERTIES
    BUILD_RPATH ${OpenCL_DIR})
  add_dependencies(build_tests CanaryIGCShuffleIntViewOfFloat)
  add_test(NAME CanaryIGCShuffleIntViewOfFloat
    COMMAND CanaryIGCShuffleIntViewOfFloat)
  set_tests_properties(CanaryIGCShuffleIntViewOfFloat PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS" FAIL_REGULAR_EXPRESSION "FAIL"
    SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST" TIMEOUT 120)
endif()

# Fails once clang stops casting vtable slots to addrspace(1); read its header.
add_shell_test(CanaryVtableFunctionCastAS.bash)
# Fail once the SPIR-V producers accept what the lowering passes expand; read
# their headers. Each probes the tools it finds and skips when it finds none.
add_shell_test(CanaryRoundIntrinsicsRejected.bash)
add_shell_test(CanaryOverflowIntrinsics.bash)
set_tests_properties(CanaryRoundIntrinsicsRejected CanaryOverflowIntrinsics
  PROPERTIES SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")

if(NOT LLVM_SPIRV STREQUAL "NOT_NEEDED")
  # Fail once llvm-spirv stops tripping on what the workaround passes fix; read their headers.
  add_shell_test(CanaryOCLVersionMismatch.bash)
  add_shell_test(CanaryBoolKernelArg.bash)
  add_shell_test(CanaryDebugInfoTranslatorCycle.bash)
  add_shell_test(CanaryIgcPrivByteArrayGEP.bash)
  add_shell_test(CanaryIgcCacheControlDropA32.bash)
  set_tests_properties(CanaryDebugInfoTranslatorCycle CanaryIgcPrivByteArrayGEP
    CanaryIgcCacheControlDropA32 PROPERTIES
    SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
endif()

if(CHIP_LLVM_USE_INTERGRATED_SPIRV)
  # Fails once the backend legalizes the hint intrinsics; read its header.
  add_shell_test(CanaryHintIntrinsicsBackend.bash)
  set_tests_properties(CanaryHintIntrinsicsBackend PROPERTIES
    SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
  # LLVM 22 legalizes zero-size arrays; read the header.
  if(LLVM_VERSION_MAJOR LESS 22)
    add_shell_test(CanaryZeroLengthArrayBackend.bash)
  endif()
endif()
