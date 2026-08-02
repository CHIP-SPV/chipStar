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

# HipCleanupPass must detach the debug info of the globals it erases, otherwise
# the compile unit keeps advertising storage that no longer exists and the
# SPIR-V producers emit a DebugGlobalVariable that crashes some consumers.
add_test(
  NAME hipCleanupDebugInfo-erased-global
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/opt
    -load-pass-plugin ${CMAKE_BINARY_DIR}/lib/libLLVMHipSpvPasses.so
    -passes=hip-cleanup
    -S
    ${CMAKE_CURRENT_SOURCE_DIR}/erased-global-debug-info.ll
)

# The erased global's DIGlobalVariable must be gone, while the retained one's
# must survive (guards against stripping debug info wholesale).
set_tests_properties(hipCleanupDebugInfo-erased-global PROPERTIES
  FAIL_REGULAR_EXPRESSION "__hip_cuid_deadbeef"
  PASS_REGULAR_EXPRESSION "kept_global")
