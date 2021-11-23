# Build LLVM-PASSES 
set(LLVM_DIR "" CACHE PATH "Path to LLVM Instal with SPIR-V patches")
if(LLVM_DIR STREQUAL "") 
  message(STATUS "LLVM_DIR is not set. Setting LLVM_DIR to compiler bin directory.")
  get_filename_component(LLVM_DIR "${CMAKE_CXX_COMPILER}../" DIRECTORY)
endif()

message(STATUS "Using Clang: ${CMAKE_CXX_COMPILER}")
get_filename_component(CLANG_BIN_PATH "${CMAKE_CXX_COMPILER}" DIRECTORY)
message(STATUS "Using LLVM Install: ${LLVM_DIR}")

if(NOT DEFINED LLVM_LINK)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-link")
    set(LLVM_LINK "${CLANG_BIN_PATH}/llvm-link" CACHE PATH "llvm-link")
  else()
    message(FATAL_ERROR "Can't find llvm-link at ${CLANG_BIN_PATH}. Please provide CMake argument -DLLVM_LINK=<path/to/llvm-link>")
  endif()
endif()

message(STATUS "Using llvm-link: ${LLVM_LINK}")

if(NOT DEFINED LLVM_SPIRV)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-spirv")
    set(LLVM_SPIRV "${CLANG_BIN_PATH}/llvm-spirv" CACHE PATH "llvm-spirv")
  else()
    message(FATAL_ERROR "Can't find llvm-spirv at ${CLANG_BIN_PATH}. Please copy llvm-spirv to ${CLANG_BIN_PATH}, Clang expects it there!")
  endif()
endif()

message(STATUS "Using llvm-spirv: ${LLVM_SPIRV}")

if(NOT DEFINED LLVM_CONFIG)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-config")
    set(LLVM_CONFIG "${CLANG_BIN_PATH}/llvm-config" CACHE PATH "llvm-config")
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