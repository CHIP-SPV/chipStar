#=============================================================================
#  Copyright (c) 2025 chipStar developers
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

# Configure environment for the tests
set(CHIPSTAR_LLVM_BIN_DIR "${LLVM_TOOLS_BINARY_DIR}")

# Create directory for test outputs
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_outputs)

# List of IR files to test
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_valid_spir.ll)
list(APPEND TEST_IR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_problematic_ir.ll)

foreach(IR_FILE ${TEST_IR_FILES})
  get_filename_component(FILENAME ${IR_FILE} NAME)
  get_filename_component(BASENAME ${IR_FILE} NAME_WE)
  
  # Run opt with HIP passes and verification
  add_test(
    NAME hipIRVerification-${BASENAME}
    COMMAND ${CHIPSTAR_LLVM_BIN_DIR}/opt
      -load-pass-plugin ${CMAKE_BINARY_DIR}/lib/libLLVMHipSpvPasses.so
      -passes=hip-final-ir-verification
      -disable-output
      ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
  )

  # Set different pass/fail criteria based on the test file
  if(BASENAME STREQUAL "test_problematic_ir")
    # Problematic IR should fail verification - test should fail but we expect it
    set_tests_properties(hipIRVerification-${BASENAME} PROPERTIES
      PASS_REGULAR_EXPRESSION "value doesn't match function result type")
  endif()

endforeach() 