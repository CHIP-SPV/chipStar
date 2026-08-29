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
# One source per line, alphabetically sorted, so that two branches each adding
# a pass insert at different anchors and merge without a conflict.
#
# Order is not load-bearing, but only because every pass guards its standalone
# llvmGetPassPluginInfo with CHIP_COMBINED_PASS_PLUGIN, leaving HipPasses.cpp
# as the single definition here. A pass that forgets the guard adds a second
# LLVM_ATTRIBUTE_WEAK definition, which the link resolves silently by position
# rather than rejecting; PassPluginRegistersPipeline below catches that.
add_library(LLVMHipPasses MODULE
    HipAbort.cpp
    HipCanonicalizeGEP.cpp
    HipCleanup.cpp
    HipDefrost.cpp
    HipDynMem.cpp
    HipEmitLoweredNames.cpp
    HipFunctionPointerAS.cpp
    HipGlobalVariables.cpp
    HipIGBADetector.cpp
    HipKernelArgSpiller.cpp
    HipLowerFPAtomicMinMax.cpp
    HipLowerMemset.cpp
    HipLowerOverflowIntrinsics.cpp
    HipLowerRoundIntrinsics.cpp
    HipLowerSubwordAtomics.cpp
    HipLowerSwitch.cpp
    HipLowerZeroLengthArrays.cpp
    HipPasses.cpp
    HipPrintf.cpp
    HipPromoteInts.cpp
    HipSanityChecks.cpp
    HipSpirvFunctionReorderPass.cpp
    HipStripDebugInfo.cpp
    HipStripUsedIntrinsics.cpp
    HipTextureLowering.cpp
    HipVerify.cpp
    HipWarps.cpp
    ${EXTRA_OBJS})

# The plugin registers its pipeline through llvmGetPassPluginInfo, and the
# ordering note above explains how easily that registration is lost: the link
# still succeeds, the symbol is still exported, and only an actual opt run
# notices. Without this test the failure surfaces as every hipcc invocation
# reporting "unknown pass name" with no build error anywhere.
find_program(LLVM_OPT NAMES opt-${LLVM_VERSION_MAJOR} opt
  NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH_BIN} ENV PATH)
if(LLVM_OPT)
  set(PLUGIN_PROBE_IR ${CMAKE_CURRENT_BINARY_DIR}/plugin-probe.ll)
  file(WRITE ${PLUGIN_PROBE_IR} "define void @probe() { ret void }\n")
  add_test(NAME PassPluginRegistersPipeline
    COMMAND ${LLVM_OPT} -load-pass-plugin $<TARGET_FILE:LLVMHipPasses>
            -passes=hip-post-link-passes ${PLUGIN_PROBE_IR} -o /dev/null)
else()
  message(STATUS "opt not found; skipping the pass plugin registration test")
endif()

# macOS Linking Requirements:
# On macOS, the linker requires all symbols to be resolved at link time for
# MODULE (plugin) libraries, unlike Linux where unresolved symbols can be
# deferred to load time. This means we must explicitly link against LLVM.
#
# With dynamic LLVM (LLVM_LINK_LLVM_DYLIB=ON), we link against libLLVM.dylib.
# With static LLVM, this is problematic because:
#   1. Linking all static LLVM libs creates a massive plugin with duplicated code
#   2. The 'ar' workaround above only extracts specific .o files, not all deps
#
# Therefore, on macOS we require LLVM to be built with shared libraries.
# Use: ./scripts/configure_llvm.sh --link-type dynamic
if(APPLE)
  if(NOT LLVM_LINK_LLVM_DYLIB)
    message(FATAL_ERROR
      "macOS requires LLVM built with shared libraries (LLVM_LINK_LLVM_DYLIB=ON).\n"
      "The macOS linker cannot build MODULE libraries with unresolved symbols.\n"
      "Rebuild LLVM with: ./scripts/configure_llvm.sh --link-type dynamic\n"
      "See MacOS-Issues.md for more details.")
  endif()
  
  message(STATUS "macOS: Linking pass plugins against libLLVM.dylib")
  target_link_libraries(LLVMHipPasses PRIVATE LLVM)
  target_link_libraries(LLVMHipDynMem PRIVATE LLVM)
  target_link_libraries(LLVMHipStripUsedIntrinsics PRIVATE LLVM)
  target_link_libraries(LLVMHipDefrost PRIVATE LLVM)
endif()

# In Debug builds, default HipVerify to "failures" mode so the IR/SPIR-V
# validation pipeline runs automatically (see HipVerify.cpp::getVerificationMode).
# Release builds keep it off due to the compile-time cost (CHIP-SPV/chipStar#1047).
target_compile_definitions(LLVMHipPasses PRIVATE
  $<$<CONFIG:Debug>:CHIP_DEBUG_BUILD>
  $<$<BOOL:${CHIP_KEEP_KERNEL_DEBUG_INFO}>:CHIP_KEEP_KERNEL_DEBUG_INFO>)

# If trying to recompile with LLVM unloaded, the inlcude path is not found
target_compile_options(LLVMHipDynMem PRIVATE -I/${LLVM_INCLUDE_DIRS})
target_compile_options(LLVMHipStripUsedIntrinsics PRIVATE -I/${LLVM_INCLUDE_DIRS})
target_compile_options(LLVMHipDefrost PRIVATE -I/${LLVM_INCLUDE_DIRS})
target_compile_options(LLVMHipPasses PRIVATE -I/${LLVM_INCLUDE_DIRS})

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
target_compile_definitions(LLVMHipPasses PRIVATE CHIP_COMBINED_PASS_PLUGIN)
