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

configure_file(run_math_intrinsics_pass.bash
  ${CMAKE_CURRENT_BINARY_DIR}/run_math_intrinsics_pass.bash @ONLY)

# Every llvm.ldexp variant that can reach a device module. llvm-spirv has no
# translation for the intrinsic, so HipLowerRoundIntrinsicsPass has to rewrite
# it into a call to the OpenCL ldexp builtin.
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ldexp.ll)

# The llround / llrint half of the same pass, which shipped without a test.
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/llround.ll)

foreach(IR_FILE ${TEST_IR_FILES})
  get_filename_component(FILENAME ${IR_FILE} NAME)
  get_filename_component(BASENAME ${IR_FILE} NAME_WE)
  configure_file(${IR_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)

  add_test(
    NAME hipMathIntrinsics-${BASENAME}
    COMMAND env
      LLVM_OPT=${LLVM_TOOLS_BINARY_DIR}/opt
      LLVM_DIS=${LLVM_TOOLS_BINARY_DIR}/llvm-dis
      LLVM_SPIRV=${LLVM_TOOLS_BINARY_DIR}/llvm-spirv
      HIP_SPV_PASSES_LIB=${CMAKE_BINARY_DIR}/lib/libLLVMHipSpvPasses.so
      ${CMAKE_CURRENT_BINARY_DIR}/run_math_intrinsics_pass.bash
      ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}
  )
endforeach()
