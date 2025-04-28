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

# Configure environment for the test script
set(CHIPSTAR_LLVM_BIN_DIR "${LLVM_TOOLS_BINARY_DIR}")

# Create directory for test outputs
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_outputs)

# Configure and copy the test script
configure_file(run_promote_int_pass.bash ${CMAKE_CURRENT_BINARY_DIR}/run_promote_int_pass.bash @ONLY)

list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/dominance.ll)
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ext-repro.ll)
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/warpCalc.ll)
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/rocprim.ll)
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/rocprim-redux.ll)
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/rocmprim-device-scan.ll)
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/signed_constant_compare.ll)
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_block_sort-hip-spirv64-generic-link.ll)
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_device_scan-hip-spirv64-generic-link.ll)
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_warp_sort-hip-spirv64-generic-lower.ll)
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/loads.ll)

# If we have this, a pass won't help
# list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/func-return-nonstd.ll)

foreach(IR_FILE ${TEST_IR_FILES})
  get_filename_component(FILENAME ${IR_FILE} NAME)
  get_filename_component(BASENAME ${IR_FILE} NAME_WE)
  configure_file(${IR_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
  
  # Check if the test should be disabled
  set(TEST_DISABLED FALSE)
  if(LLVM_VERSION_MAJOR LESS 18 AND BASENAME STREQUAL "rocprim-redux")
    set(TEST_DISABLED TRUE)
  endif()

  add_test(
    NAME hipPromoteInt-${BASENAME}
    COMMAND env 
      LLVM_OPT=${CHIPSTAR_LLVM_BIN_DIR}/opt
      LLVM_SPIRV=${CHIPSTAR_LLVM_BIN_DIR}/llvm-spirv
      LLVM_DIS=${CHIPSTAR_LLVM_BIN_DIR}/llvm-dis
      HIP_SPV_PASSES_LIB=${CMAKE_BINARY_DIR}/lib/libLLVMHipSpvPasses.so
      ${CMAKE_CURRENT_BINARY_DIR}/run_promote_int_pass.bash ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}
  )

  # Set the DISABLED property if needed
  if(TEST_DISABLED)
    set_tests_properties(hipPromoteInt-${BASENAME} PROPERTIES DISABLED TRUE)
  endif()
endforeach() 