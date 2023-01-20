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

if(NOT DEFINED LLVM_CONFIG)
  find_program(LLVM_CONFIG NAMES llvm-config)
  if(NOT LLVM_CONFIG)
      message(FATAL_ERROR "Can't find llvm-config. Please provide CMake argument -DLLVM_CONFIG=/path/to/llvm-config<-version>")
  endif()
endif()
message(STATUS "Using llvm-config: ${LLVM_CONFIG}")

get_filename_component(LLVM_CONFIG_BINARY_NAME ${LLVM_CONFIG} NAME)
string(REGEX MATCH "[0-9]+" LLVM_VERSION_MAJOR "${LLVM_CONFIG_BINARY_NAME}")

execute_process(COMMAND "${LLVM_CONFIG}" "--obj-root"
  RESULT_VARIABLE RES
  OUTPUT_VARIABLE CLANG_ROOT_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Using CLANG_ROOT_PATH: ${CLANG_ROOT_PATH}")
set(CLANG_ROOT_PATH_BIN ${CLANG_ROOT_PATH}/bin)

execute_process(COMMAND "${LLVM_CONFIG}" "--version"
  RESULT_VARIABLE RES
  OUTPUT_VARIABLE LLVM_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Using LLVM_VERSION: ${LLVM_VERSION}")

# Set the compilers
find_program(CMAKE_CXX_COMPILER_PATH NAMES clang++ NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH_BIN} REQUIRED)
message(STATUS "Using CMAKE_CXX_COMPILER_PATH: ${CMAKE_CXX_COMPILER_PATH}")

find_program(CMAKE_C_COMPILER_PATH NAMES clang NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH_BIN} REQUIRED)
message(STATUS "Using CMAKE_C_COMPILER_PATH: ${CMAKE_C_COMPILER_PATH}")

if(NOT CMAKE_CXX_COMPILER EQUAL ${CMAKE_CXX_COMPILER_PATH})
  message(STATUS "CMAKE_CXX_COMPILER is set to ${CMAKE_CXX_COMPILER}. Overriding with ${CMAKE_CXX_COMPILER_PATH}")
  set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER_PATH})
endif()
if(NOT CMAKE_C_COMPILER EQUAL ${CMAKE_C_COMPILER_PATH})
  message(STATUS "CMAKE_C_COMPILER is set to ${CMAKE_C_COMPILER}. Overriding with ${CMAKE_C_COMPILER_PATH}")
  set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER_PATH})
endif()
##################################################################
if(NOT DEFINED LLVM_LINK)
  find_program(LLVM_LINK NAMES llvm-link NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH_BIN} ENV PATH)
  if(NOT LLVM_LINK)
    message(FATAL_ERROR "Can't find llvm-link. Please provide CMake argument -D$LLVM_LINK=/path/to/llvm-link<-version>")
  endif()
endif()
message(STATUS "Using llvm-link: ${LLVM_LINK}")

if(NOT DEFINED LLVM_SPIRV)
  find_program(LLVM_SPIRV NAMES llvm-spirv FIND_TARGET NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH_BIN} ENV PATH)
  if(NOT LLVM_SPIRV)
    message(FATAL_ERROR "Can't find llvm-spirv. Please provide CMake argument -DLLVM_SPIRV=/path/to/llvm-spirv<-version>")
  endif()
endif()
message(STATUS "Using llvm-spirv: ${LLVM_SPIRV}")

if(NOT DEFINED CLANG_OFFLOAD_BUNDLER)
  find_program(CLANG_OFFLOAD_BUNDLER NAMES clang-offload-bundler NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH_BIN} ENV PATH)
  if(NOT CLANG_OFFLOAD_BUNDLER)
    message(FATAL_ERROR "Can't find clang-offload-bundler. Please provide CMake argument -DCLANG_OFFLOAD_BUNDLER=/path/to/clang-offload-bundler<-version>")
  endif()
endif()
message(STATUS "Using clang-offload-bundler: ${CLANG_OFFLOAD_BUNDLER}")
