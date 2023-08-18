#=============================================================================
#  CMake build system files
#
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

function(add_hip_test MAIN_SOURCE)
  get_filename_component(EXEC_NAME ${MAIN_SOURCE} NAME_WLE)
  set_source_files_properties(${MAIN_SOURCE} PROPERTIES LANGUAGE CXX)
  add_executable("${EXEC_NAME}" ${MAIN_SOURCE})
  set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

  target_link_libraries("${EXEC_NAME}" CHIP deviceInternal)
  target_include_directories("${EXEC_NAME}"
    PUBLIC ${CMAKE_SOURCE_DIR}/HIP/include ${CMAKE_SOURCE_DIR}/include)

  add_test(NAME ${EXEC_NAME} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME})
endfunction()

add_hip_test(implicitCasts.cpp)

# Add this as a hipcc otherwise the `make all` will fail which is not convenient indicator for test failure
add_test(NAME "deviceMallocCompile"
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc
  ${CMAKE_CURRENT_SOURCE_DIR}/deviceMallocCompile.cpp)

add_hip_test(sincospifSpotTest.cc)
