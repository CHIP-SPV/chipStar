#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2021-22 CHIP-SPV developers
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


# String Options
macro(option_str OPTION_NAME DOC_STRING)
  set(${OPTION_NAME} "" CACHE PATH "${DOC_STRING}")
  if(${OPTION_NAME} STREQUAL "") 
    message(FATAL_ERROR "${OPTION_NAME} must be set")
  endif()
  message(STATUS "${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()

# Find library in LD_LIBRARY_PATH
macro(find_library_dynamic libname)
  message(STATUS "\nSearching for ${libname}")
  find_library(${libname}_LIBRARY
    NAMES ${libname}
    PATHS ENV LD_LIBRARY_PATH ./
  )
  if(${${libname}_LIBRARY} STREQUAL ${libname}_LIBRARY-NOTFOUND)
    message(FATAL_ERROR "${libname} library not found in LD_LIBRARY_PATH")
  else()
    message(STATUS "Found: ${${libname}_LIBRARY}\n")
    list(APPEND CHIP-SPV_LIBRARIES ${${libname}_LIBRARY})
  endif()
endmacro()

macro(add_to_config _configfile _variable _value)
  set(${_configfile} "${${_configfile}}${_variable}=${_value}\n")
endmacro()
