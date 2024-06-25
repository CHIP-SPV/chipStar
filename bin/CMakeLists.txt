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


# copy in cucc.py to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cucc.py
               ${CMAKE_CURRENT_BINARY_DIR}/cucc
               COPYONLY)

# create a symbolic link from cucc to nvcc
file(CREATE_LINK 
  ${CMAKE_CURRENT_BINARY_DIR}/cucc
  ${CMAKE_CURRENT_BINARY_DIR}/nvcc
  SYMBOLIC
)

# Install the cucc.py file and rename it to cucc
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/cucc.py
  DESTINATION bin RENAME cucc
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Install the nvcc symlink
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink cucc nvcc
                               WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)")


# Run static type check on cucc if there is a tool available for it.
find_program(MYPY NAMES mypy)
if(MYPY)
  add_test(NAME cucc-static-type-check
    COMMAND ${MYPY} ${CMAKE_CURRENT_SOURCE_DIR}/cucc.py)
endif()
