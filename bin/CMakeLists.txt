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

file(CREATE_LINK 
  ${CMAKE_CURRENT_BINARY_DIR}/cucc.py
  ${CMAKE_CURRENT_BINARY_DIR}/nvcc
  SYMBOLIC
)

file(CREATE_LINK 
  ${CMAKE_CURRENT_BINARY_DIR}/cucc.py
  ${CMAKE_CURRENT_BINARY_DIR}/cucc
  SYMBOLIC
)

# copy in cucc.py to build directory and set execute permissions
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/cucc.py
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install the cucc.py file
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/cucc.py
  DESTINATION bin RENAME cucc)

# Install the nvcc symlink
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nvcc
  DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)




# Run static type check on cucc if there is a tool available for it.
find_program(MYPY NAMES mypy)
if(MYPY)
  add_test(NAME cucc-static-type-check
    COMMAND ${MYPY} ${CMAKE_CURRENT_SOURCE_DIR}/cucc.py)
endif()
