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

# Ensure CMAKE_INSTALL_LIBDIR is defined
if(NOT CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR "lib")
endif()

if(DEFINED LLVM_CONFIG_BIN)
  # if it was cached to NOT_FOUND, unset it
  if(LLVM_CONFIG_BIN STREQUAL "LLVM_CONFIG_BIN-NOTFOUND")
    message(STATUS "LLVM_CONFIG_BIN was set to LLVM_CONFIG_BIN-NOTFOUND. Unsetting...")
    unset(LLVM_CONFIG_BIN CACHE)
  endif()

  # if it was set to a path, check that it exists
  if(NOT EXISTS ${LLVM_CONFIG_BIN})
    message(FATAL_ERROR "Provided LLVM_CONFIG_BIN (${LLVM_CONFIG_BIN}) does not exist")
  endif()
else() # if it was not defined, look for it
  find_program(LLVM_CONFIG_BIN NAMES llvm-config)
  if(NOT LLVM_CONFIG_BIN)
      message(FATAL_ERROR "Can't find llvm-config. Please provide CMake argument -DLLVM_CONFIG_BIN=/path/to/llvm-config<-version>")
  endif()
endif()
message(STATUS "Using llvm-config: ${LLVM_CONFIG_BIN}")

execute_process(COMMAND "${LLVM_CONFIG_BIN}" "--obj-root"
  RESULT_VARIABLE RES
  OUTPUT_VARIABLE CLANG_ROOT_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Using CLANG_ROOT_PATH: ${CLANG_ROOT_PATH}")
set(CLANG_ROOT_PATH_BIN ${CLANG_ROOT_PATH}/bin)

execute_process(COMMAND "${LLVM_CONFIG_BIN}" "--version"
  RESULT_VARIABLE RES
  OUTPUT_VARIABLE LLVM_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Using LLVM_VERSION: ${LLVM_VERSION}")
string(REGEX MATCH "[0-9]+" LLVM_VERSION_MAJOR "${LLVM_VERSION}")

# Check if the LLVM_INCLUDE_DIR is already cached
if(NOT LLVM_INCLUDE_DIRS)
  # Get the include directory
  execute_process(COMMAND "${LLVM_CONFIG_BIN}" "--includedir"
    RESULT_VARIABLE RES
    OUTPUT_VARIABLE LLVM_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Cache the include directory
  set(LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIRS}" CACHE PATH "LLVM include directory")
endif()


# Cache the include directory
include_directories(${LLVM_INCLUDE_DIRS})

# Set the compilers
find_program(CMAKE_CXX_COMPILER_PATH NAMES clang++ NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH_BIN})
if(NOT CMAKE_CXX_COMPILER_PATH)
  message(FATAL_ERROR "Could not find clang++ in ${CLANG_ROOT_PATH_BIN}. Please provide CMake argument -DCMAKE_CXX_COMPILER_PATH=/path/to/clang++<-version>")
endif()
message(STATUS "Using CMAKE_CXX_COMPILER_PATH: ${CMAKE_CXX_COMPILER_PATH}")

find_program(CMAKE_C_COMPILER_PATH NAMES clang NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH_BIN})
if(NOT CMAKE_C_COMPILER_PATH)
  message(FATAL_ERROR "Could not find clang in ${CLANG_ROOT_PATH_BIN}. Please provide CMake argument -DCMAKE_C_COMPILER_PATH=/path/to/clang<-version>")
endif()
set(CLANG_BIN ${CMAKE_C_COMPILER_PATH})
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

message(STATUS "XXX LLVM-version-major: ${LLVM_VERSION_MAJOR}") # DEBUG

if(NOT DEFINED LLVM_SPIRV)
  find_program(LLVM_SPIRV NAMES llvm-spirv-${LLVM_VERSION_MAJOR} llvm-spirv FIND_TARGET NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH_BIN} ENV PATH)
  if(NOT LLVM_SPIRV)
    message(FATAL_ERROR "Can't find llvm-spirv. Please provide CMake argument -DLLVM_SPIRV=/path/to/llvm-spirv<-version>")
  endif()
else()
  # Check if the provided LLVM_SPIRV file exists
  if(NOT EXISTS ${LLVM_SPIRV})
    message(FATAL_ERROR "Provided LLVM_SPIRV (${LLVM_SPIRV}) does not exist")
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

# Execute llvm-spirv and check for errors
execute_process(
    COMMAND ${LLVM_SPIRV} --version
    RESULT_VARIABLE LLVM_SPIRV_ERROR
    OUTPUT_VARIABLE LLVM_SPIRV_OUTPUT
)

if(LLVM_SPIRV_ERROR)
    message(FATAL_ERROR "Error executing llvm-config: ${LLVM_CONFIG_ERROR}."
    "If 'error while loading shared libraries' error is thrown, you might have to add LLVM libs to LD_LIBRARY_PATH"
    "If file is not found, you might have a versioned version (llvm-spirv-16 instead of llvm-spirv). Pass the path to the full binary via -DLLVM_SPIRV="
)
else()
    message(STATUS "llvm-spirv version: ${LLVM_SPIRV_OUTPUT}")
endif()

enable_language(C CXX)
# required by ROCm-Device-Libs, must be after project() call
find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH}/${CMAKE_INSTALL_LIBDIR}/cmake/llvm ${CLANG_ROOT_PATH}/lib/cmake/llvm)
find_package(Clang REQUIRED CONFIG NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH}/${CMAKE_INSTALL_LIBDIR}/cmake/clang ${CLANG_ROOT_PATH}/lib/cmake/clang)
