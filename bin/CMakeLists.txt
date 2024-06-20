#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2021-22 chipStar developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

# For telling cuspvc where to look for clang.
get_filename_component(CLANG_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)

# Read includes, prepend each one with -I.
set(prop "$<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>")
set(CHIP_INCLUDES "$<$<BOOL:${prop}>:-I$<JOIN:${prop}, -I>>")

add_executable(cucc ${CMAKE_CURRENT_SOURCE_DIR}/cucc.cc)
target_include_directories(cucc PRIVATE ${CMAKE_SOURCE_DIR}/HIPCC/src)
target_link_libraries(cucc CHIP)

# Install the cucc target
install(TARGETS cucc
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create a symlink to nvcc which points to cucc in both build and install directories
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_INSTALL_PREFIX}/bin/cucc ${CMAKE_INSTALL_PREFIX}/bin/nvcc)")
file(CREATE_LINK ${HIPCC_BUILD_PATH}/cucc ${HIPCC_BUILD_PATH}/nvcc SYMBOLIC)
