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


set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_EXTENSIONS OFF)

execute_process(COMMAND "${LLVM_CONFIG_BIN}" "--cmakedir"
		OUTPUT_VARIABLE LLVM_DIR
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE RES)

if(NOT RES EQUAL 0)
  message(FATAL_ERROR "failed to run llvm-config (${LLVM_CONFIG_BIN})")
endif()

message(STATUS "LLVM CMake directory: ${LLVM_DIR}")

find_package(LLVM REQUIRED CONFIG HINTS "${LLVM_DIR}" PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)

if(NOT DEFINED LLVM_VERSION)
  message(FATAL "Could not determine LLVM version.")
endif()

######################################

add_definitions(${LLVM_DEFINITIONS})

include_directories(${LLVM_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/include)

if(NOT LLVM_ENABLE_RTTI)
  add_compile_options("-fno-rtti")
endif()

if(NOT LLVM_ENABLE_EH)
  add_compile_options("-fno-exceptions")
endif()

set(EXTRA_OBJS)

if(LLVM_LINK_LLVM_DYLIB OR LLVM_ENABLE_SHARED_LIBS)
  message(STATUS "LLVM built with shared libraries -> pass plugin should just work")
else()
  message(STATUS "LLVM built with static libraries -> adding workaround for issue #102")

  get_target_property(LLVMCORE_CONF LLVMCore IMPORTED_CONFIGURATIONS)
  string(REPLACE ";.*$" "" LLVMCORE_CONF "${LLVMCORE_CONF}")
  get_target_property(LLVMCORE_PATH LLVMCore IMPORTED_LOCATION_${LLVMCORE_CONF})
  message(STATUS "libLLVMCore CONFIG: ${LLVMCORE_CONF}")
  message(STATUS "libLLVMCore PATH: ${LLVMCORE_PATH}")
  file(REMOVE "${CMAKE_BINARY_DIR}/ReplaceConstant.cpp.o")
  execute_process(COMMAND "ar" p "${LLVMCORE_PATH}" "ReplaceConstant.cpp.o"
    OUTPUT_FILE "${CMAKE_BINARY_DIR}/ReplaceConstant.cpp.o"
    RESULT_VARIABLE RES)
  if(NOT RES EQUAL "0")
    message(FATAL_ERROR "ar command on ${LLVMCORE_PATH} failed")
  endif()

  add_library(ReplaceConstant OBJECT IMPORTED)
  set_property(TARGET ReplaceConstant PROPERTY IMPORTED_OBJECTS
    "${CMAKE_BINARY_DIR}/ReplaceConstant.cpp.o")

  set(EXTRA_OBJS "$<TARGET_OBJECTS:ReplaceConstant>")

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
    HipLowerMemset.cpp ${EXTRA_OBJS})

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
