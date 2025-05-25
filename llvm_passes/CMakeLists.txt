#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2021-24 chipStar developers
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


set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_EXTENSIONS OFF)

######################################

add_definitions(${LLVM_DEFINITIONS})

# abort if LLVM_INCLUDE_DIRS is empty or not set
if(NOT LLVM_INCLUDE_DIRS)
  message(FATAL_ERROR "LLVM_INCLUDE_DIRS is empty or not set")
endif()
include_directories(${LLVM_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/include)

if(NOT LLVM_ENABLE_RTTI)
  add_compile_options("-fno-rtti")
endif()

if(NOT LLVM_ENABLE_EH)
  add_compile_options("-fno-exceptions")
endif()

set(EXTRA_OBJS)

# Determine if the opt tool is statically linked and import missing
# function definitions, needed by our pass plugin, because statically
# linked opt tool only includes symbols which referenced by it. A similar
# bug has been filed here: https://github.com/llvm/llvm-project/issues/148.
if(LLVM_LINK_LLVM_DYLIB OR LLVM_ENABLE_SHARED_LIBS)
  message(STATUS "LLVM built with shared libraries -> pass plugin should just work")
else()
  message(STATUS "LLVM built with static libraries -> adding workaround for issue #102")

  get_target_property(LLVM_CONF LLVMCore IMPORTED_CONFIGURATIONS)
  string(REPLACE ";.*$" "" LLVM_CONF "${LLVM_CONF}")
  message(STATUS "LLVM CONFIG: ${LLVM_CONF}")

  get_target_property(LLVMCORE_PATH LLVMCore IMPORTED_LOCATION_${LLVM_CONF})
  message(STATUS "libLLVMCore PATH: ${LLVMCORE_PATH}")
  file(REMOVE "${CMAKE_BINARY_DIR}/ReplaceConstant.cpp.o")
  execute_process(COMMAND "ar" p "${LLVMCORE_PATH}" "ReplaceConstant.cpp.o"
    OUTPUT_FILE "${CMAKE_BINARY_DIR}/ReplaceConstant.cpp.o"
    RESULT_VARIABLE RES)
  if(NOT RES EQUAL "0")
    message(FATAL_ERROR "ar command on ${LLVMCORE_PATH} failed")
  endif()

  get_target_property(LLVMTRANSFORMUTILS_PATH LLVMTransformUtils
    IMPORTED_LOCATION_${LLVM_CONF})
  message(STATUS "libLLVMTransformUtils PATH: ${LLVMTRANSFORMUTILS_PATH}")
  file(REMOVE "${CMAKE_BINARY_DIR}/LowerMemIntrinsics.cpp.o")
  execute_process(COMMAND "ar" p "${LLVMTRANSFORMUTILS_PATH}"
    "LowerMemIntrinsics.cpp.o"
    OUTPUT_FILE "${CMAKE_BINARY_DIR}/LowerMemIntrinsics.cpp.o"
    RESULT_VARIABLE RES)
  if(NOT RES EQUAL "0")
    message(FATAL_ERROR "ar command on ${LLVMTRANSFORMUTILS_PATH} failed")
  endif()

  add_library(LLVMImports OBJECT IMPORTED)
  set_property(TARGET LLVMImports PROPERTY IMPORTED_OBJECTS
    "${CMAKE_BINARY_DIR}/ReplaceConstant.cpp.o"
    "${CMAKE_BINARY_DIR}/LowerMemIntrinsics.cpp.o")
  set(EXTRA_OBJS "$<TARGET_OBJECTS:LLVMImports>")
endif()

add_library(LLVMHipDynMem MODULE HipDynMem.cpp ${EXTRA_OBJS})

if("${LLVM_VERSION}" VERSION_GREATER_EQUAL 17.0)
  list(APPEND EXTRA_OBJS SPIRVImageType.cc)
endif()

add_library(LLVMHipStripUsedIntrinsics MODULE HipStripUsedIntrinsics.cpp)
add_library(LLVMHipDefrost MODULE HipDefrost.cpp)
add_library(LLVMHipPasses MODULE HipPasses.cpp
    HipDynMem.cpp HipStripUsedIntrinsics.cpp HipDefrost.cpp
    HipPrintf.cpp HipGlobalVariables.cpp HipTextureLowering.cpp HipAbort.cpp
    HipEmitLoweredNames.cpp HipWarps.cpp HipKernelArgSpiller.cpp
    HipLowerZeroLengthArrays.cpp HipSanityChecks.cpp HipLowerSwitch.cpp
    HipLowerMemset.cpp HipIGBADetector.cpp HipPromoteInts.cpp 
    HipFinalIRVerification.cpp HipSPIRVVerificationPass.cpp
    ${EXTRA_OBJS})

# If trying to recompile with LLVM unloaded, the inlcude path is not found
target_compile_options(LLVMHipDynMem PRIVATE -I/${LLVM_INCLUDE_DIRS})
target_compile_options(LLVMHipStripUsedIntrinsics PRIVATE -I/${LLVM_INCLUDE_DIRS})
target_compile_options(LLVMHipDefrost PRIVATE -I/${LLVM_INCLUDE_DIRS})
target_compile_options(LLVMHipPasses PRIVATE -I/${LLVM_INCLUDE_DIRS})
target_compile_definitions(LLVMHipPasses PRIVATE CHIPSTAR_LLVM_BIN_DIR="${CLANG_ROOT_PATH_BIN}")

if("${LLVM_VERSION}" VERSION_GREATER_EQUAL 14.0)
  set_target_properties(LLVMHipPasses PROPERTIES
    # HIP-Clang 14+ automatically searches for libLLVMHipSpvPasses.so in
    # <HIP-PATH>/lib where <HIP-PATH> is path given to --hip-path for
    # HIP program compilation.
    OUTPUT_NAME "LLVMHipSpvPasses"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

install(TARGETS LLVMHipDynMem LLVMHipStripUsedIntrinsics LLVMHipDefrost LLVMHipPasses
        LIBRARY DESTINATION lib/llvm
        ARCHIVE DESTINATION lib/llvm
        )
