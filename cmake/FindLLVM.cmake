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


if((CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang") OR
   (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
    message(FATAL_ERROR "this project requires clang >= 8.0")
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16.0.0)
    set(CLANG_VERSION_LESS_16 ON)
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.0.0)
    set(CLANG_VERSION_LESS_15 ON)
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0.0)
    message(WARNING "Deprecated clang version '${CMAKE_CXX_COMPILER_VERSION}'. \
            Support for Clang < 14.0 will be discontinued in the future.")
    set(CLANG_VERSION_LESS_14 ON)
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
    set(CLANG_VERSION_LESS_13 ON)
  endif()

else()
  message(FATAL_ERROR "this project must be compiled with clang. CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
endif()

string(REPLACE "." ";" VERSION_LIST ${CMAKE_CXX_COMPILER_VERSION})
list(GET VERSION_LIST 0 CLANG_VERSION_MAJOR)
get_filename_component(CLANG_BIN_NAME "${CMAKE_CXX_COMPILER}" NAME)
if(CLANG_BIN_NAME MATCHES "clang[+][+](-${CLANG_VERSION_MAJOR})")
  set(BINARY_VERSION_SUFFIX "-${CLANG_VERSION_MAJOR}")
else()
  set(BINARY_VERSION_SUFFIX)
endif()

#################################################################

get_filename_component(CLANG_BIN_PATH "${CMAKE_CXX_COMPILER}" DIRECTORY)

if(NOT DEFINED LLVM_CONFIG)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-config")
    set(LLVM_CONFIG "${CLANG_BIN_PATH}/llvm-config" CACHE PATH "llvm-config")
  elseif(EXISTS "${CLANG_BIN_PATH}/llvm-config${BINARY_VERSION_SUFFIX}")
    set(LLVM_CONFIG "${CLANG_BIN_PATH}/llvm-config${BINARY_VERSION_SUFFIX}" CACHE PATH "llvm-config")
  else()
    message(FATAL_ERROR "Can't find llvm-config at ${CLANG_BIN_PATH}. Please provide CMake argument -DLLVM_CONFIG=<path/to/llvm-config>")
  endif()
endif()

message(STATUS "Using llvm-config: ${LLVM_CONFIG}")

execute_process(COMMAND "${LLVM_CONFIG}" "--obj-root"
  RESULT_VARIABLE RES
  OUTPUT_VARIABLE CLANG_ROOT_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND "${LLVM_CONFIG}" "--version"
  RESULT_VARIABLE RES
  OUTPUT_VARIABLE LLVM_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

##################################################################

if(NOT DEFINED LLVM_LINK)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-link")
    set(LLVM_LINK "${CLANG_BIN_PATH}/llvm-link" CACHE PATH "llvm-link")
  elseif(EXISTS "${CLANG_BIN_PATH}/llvm-link${BINARY_VERSION_SUFFIX}")
    set(LLVM_LINK "${CLANG_BIN_PATH}/llvm-link${BINARY_VERSION_SUFFIX}" CACHE PATH "llvm-link")
  else()
    message(FATAL_ERROR "Can't find llvm-link at ${CLANG_BIN_PATH}. Please install it into that directory")
  endif()
endif()
message(STATUS "Using llvm-link: ${LLVM_LINK}")

if(NOT DEFINED LLVM_SPIRV)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-spirv")
    set(LLVM_SPIRV "${CLANG_BIN_PATH}/llvm-spirv" CACHE PATH "llvm-spirv")
  elseif(EXISTS "${CLANG_BIN_PATH}/llvm-spirv${BINARY_VERSION_SUFFIX}")
    set(LLVM_SPIRV "${CLANG_BIN_PATH}/llvm-spirv${BINARY_VERSION_SUFFIX}" CACHE PATH "llvm-spirv")
  else()
    message(FATAL_ERROR "Can't find llvm-spirv at ${CLANG_BIN_PATH}. Please install it into that directory")
  endif()
endif()
message(STATUS "Using llvm-spirv: ${LLVM_SPIRV}")

if(NOT DEFINED CLANG_OFFLOAD_BUNDLER)
  if(EXISTS "${CLANG_BIN_PATH}/clang-offload-bundler")
    set(CLANG_OFFLOAD_BUNDLER "${CLANG_BIN_PATH}/clang-offload-bundler" CACHE PATH "clang-offload-bundler")
  elseif(EXISTS "${CLANG_BIN_PATH}/clang-offload-bundler${BINARY_VERSION_SUFFIX}")
    set(CLANG_OFFLOAD_BUNDLER "${CLANG_BIN_PATH}/clang-offload-bundler${BINARY_VERSION_SUFFIX}" CACHE PATH "clang-offload-bundler")
  else()
    message(FATAL_ERROR "Can't find clang-offload-bundler at ${CLANG_BIN_PATH}. Please install it into that directory")
  endif()
endif()
message(STATUS "Using clang-offload-bundler: ${CLANG_OFFLOAD_BUNDLER}")
