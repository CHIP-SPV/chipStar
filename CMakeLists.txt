# =============================================================================
# CMake build system files
#
# Copyright (c) 2021-22 CHIP-SPV developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# =============================================================================

# temporary
set(CMAKE_WARN_DEPRECATED OFF)
add_compile_options(-Wno-format-extra-args)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-duplicate-decl-specifier \
-Wno-tautological-constant-compare  -Wno-c++20-extensions -Wno-unused-result \
-Wno-delete-abstract-non-virtual-dtor -Wno-deprecated-declarations -Wunused-command-line-argument -gdwarf-4")

# end temporary
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(CHIP-SPV
  VERSION 1.0
  DESCRIPTION "HIP implementation for runtimes that support SPIR-V"
  LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(chip_spv_cmake_macros)
enable_testing()

# =============================================================================
# CHIP-SPV CMAKE DEPENDENCIES
include(FindLLVM)

if(NOT DEFINED OpenCL_LIBRARY)
  message(STATUS "OpenCL_LIBRARY was not set. Searching for libOpenCL.so in LD_LIBRARY_PATH")
  find_library(OpenCL_LIBRARY NAMES OpenCL PATHS ENV LD_LIBRARY_PATH ./ NO_CACHE)
  get_filename_component(OpenCL_DIR ${OpenCL_LIBRARY} DIRECTORY CACHE)
  message(STATUS "Setting OpenCL_DIR to ${OpenCL_LIBRARY}")
endif()

if(NOT DEFINED LevelZero_LIBRARY)
  message(STATUS "LevelZero_LIBRARY was not set. Searching for ze_loader.so in LD_LIBRARY_PATH")
  find_library(LevelZero_LIBRARY NAMES ze_loader PATHS ENV LD_LIBRARY_PATH ./ NO_CACHE)
  get_filename_component(LevelZero_DIR ${LevelZero_LIBRARY} DIRECTORY CACHE)
  message(STATUS "Setting LevelZero_DIR to ${LevelZero_DIR}")
endif()

message(STATUS "OpenCL_LIBRARY: ${OpenCL_LIBRARY}")
message(STATUS "LevelZero_LIBRARY: ${LevelZero_LIBRARY}")

if(NOT OpenCL_LIBRARY AND NOT LevelZero_LIBRARY)
  message(FATAL_ERROR "At least one of OpenCL,Level0 libraries must be available")
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
set(PTHREAD_LIBRARY Threads::Threads)

include(CheckIncludeFileCXX)
check_include_file_cxx("filesystem" HAS_FILESYSTEM)

if(NOT HAS_FILESYSTEM)
  check_include_file_cxx("experimental/filesystem" HAS_EXPERIMENTAL_FILESYSTEM)
endif()

if(NOT HAS_FILESYSTEM AND NOT HAS_EXPERIMENTAL_FILESYSTEM)
  message(FATAL_ERROR "<filesystem> was not found.")
endif()

# CHIP-SPV CMAKE DEPENDENCIES
# =============================================================================

# =============================================================================
# CHIP-SPV CMAKE VARIABLBLES/OPTIONS
set(CHIP_SRC
  src/spirv.cc
  src/spirv_hiprtc.cc
  src/CHIPDriver.cc
  src/CHIPBackend.cc
  src/CHIPGraph.cc
  src/CHIPBindings.cc
  src/logging.cc
  src/Utils.cc
)

if(OpenCL_LIBRARY)
  list(APPEND CHIP_SRC
    src/backend/OpenCL/CHIPBackendOpenCL.cc
    src/backend/OpenCL/SVMemoryRegion.cc)
endif()

if(LevelZero_LIBRARY)
  list(APPEND CHIP_SRC
    src/backend/Level0/CHIPBackendLevel0.cc)
endif()

set(CHIP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "CHIP-SPV source directory")

set(CHIP_SPV_LINK_FLAGS "")
set(CHIP_SPV_COMPILE_FLAGS "")
list(APPEND CHIP_SPV_DEFINITIONS "")

# used by hipcc/hipconfig
set(HIP_COMPILER "clang" CACHE STRING "CHIP-SPV compiler")
set(HIP_RUNTIME "spirv" CACHE STRING "CHIP-SPV runtime")
set(HIP_PLATFORM "spirv" CACHE STRING "CHIP-SPV platform")
set(HIP_ARCH "spirv" CACHE STRING "CHIP-SPV architecture")
get_filename_component(HIP_CLANG_PATH ${CMAKE_CXX_COMPILER} DIRECTORY)
get_filename_component(HIP_CLANG_PATH "${HIP_CLANG_PATH}../" DIRECTORY CACHE)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# INSTALL DIRECTORY VARIABLES
set(BIN_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/bin CACHE PATH "binary install dir")
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib CACHE PATH "include install dir")
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include CACHE PATH "include install dir")
set(SHARE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/share CACHE PATH "share install dir")
set(CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/hip CACHE PATH "cmake config install dir")
set(CONFIG_LANG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/hip-lang CACHE PATH "cmake lang install dir")

# TODO: Add CHIP_ prefix
option(ERROR_IF_NOT_IMPLEMENTED "Some HIP functionality is not yet implemented in CHIP-SPV. If such a function is called, throw hipErrorNotSupported" ON)
option(LAZY_JIT "JIT kernels at the time of the first kernel invocation instead of at the start of the program execution" OFF)
option(CACHE_KERNELS "Save the compiled kernel to disk to speed up program startup" OFF)
option(VERBOSE "Verbose compilation" OFF)
option(BUILD_SHARED_LIBS "Build CHIP-SPV as a shared library" ON)
option(BUILD_DOCS "Build doxygen documentation" OFF)
option(ENFORCE_QUEUE_SYNCHRONIZATION "Enforce the correct HIP stream semantics of synchronizing queues with the default queue" ON)
option(PER_THREAD_DEFAULT_STREAM "Each thread has a thread local default stream. Default: OFF - use legacy stream" OFF)
option(LEVEL_ZERO_IMMEDIATE_QUEUES "Enable/disable the use of immediate command queues" OFF)
option(MALLOC_SHARED_WORKAROUND "Use zeMallocHost instead of zeMallocShared as it causes segfaults on some systems" OFF)
option(LLVM_USE_INTERGRATED_SPIRV "Use LLVM's intergrated SPIR-V backend for emitting device binary instead of SPIR-V translator. Requires LLVM 15" OFF)
option(SET_RPATH "Add CMAKE_INSTALL_PREFIX/lib to the RPATH for CHIP-SPV executables" ON)
option(ENABLE_FAILING_TESTS "Enable tests which are known to fail or be unreliable" OFF)
option(ENABLE_UNCOMPILABLE_TESTS "Enable tests which are known to not compile" OFF)
option(BUILD_TESTS "Build Catch2 unit tests" ON)
option(STANDALONE_TESTS "Create a separate executable for each test instead of combining tests into a shared lib by category" ON)
option(DUBIOUS_LOCKS "Enable locks that don't seem necessary but make a lot of valgrind issues go away" ON)
option(USE_EXTERNAL_HIP_TESTS "Use Catch2 tests from the hip-tests submodule" OFF)

set(CMAKE_BUILD_TYPE "DEBUG" CACHE STRING "The build type to use DEBUG, RELEASE, RELWITHDEBINFO, MINSIZEREL")
set(CHIP_SPV_DEFAULT_INSTALL_DIR "${CMAKE_SOURCE_DIR}/install" CACHE PATH "The installation directory for CHIP-SPV")
set(HIP_ENABLE_SPIRV "-D__HIP_PLATFORM_SPIRV__=" CACHE INTERNAL "Compiler difinition which enables SPIR-V defines")

# FOR EXTERNAL USE
set(HIP_OFFLOAD_COMPILE_OPTIONS "" CACHE STRING "space separated list of compiler flags needed for generating offload code")
set(HIP_OFFLOAD_LINK_OPTIONS "" CACHE STRING "space separated list of compiler flags needed for linking binaries with GPU offload")

# CHIP-SPV CMAKE VARIABLBLES/OPTIONS
# =============================================================================

# =============================================================================
# CHIP-SPV CMAKE OPTIONS PARSING
if(DUBIOUS_LOCKS)
  list(APPEND CHIP_SPV_DEFINITIONS DUBIOUS_LOCKS)
endif()

if(MALLOC_SHARED_WORKAROUND)
  list(APPEND CHIP_SPV_DEFINITIONS MALLOC_SHARED_WORKAROUND)
endif()

set(DISABLE_OPAQUE_PTRS_OPT "")

if(NOT CLANG_VERSION_LESS_15)
  # LLVM 15 switches to opaque pointer mode by default which
  # llvm-spirv currently does not support. Switch the mode off.
  #
  # NOTE: https://reviews.llvm.org/D130766 will disable the mode off
  # for SPIR-V targets but the patch has not yet landed on the
  # LLVM 15 branch.
  #
  # NOTE: Bitcode library must be compiled with the same opaque
  # pointer setting.
  set(DISABLE_OPAQUE_PTRS_OPT -Xclang -no-opaque-pointers)
endif()

if(LLVM_USE_INTERGRATED_SPIRV)
  if(CLANG_VERSION_LESS_14)
    message(FATAL_ERROR "-DLLVM_USE_INTERGRATED_SPIRV=On is not supported for LLVM 13 and less")
  else()
    set(SPIRV_EMITTER_OPTS "-fintegrated-objemitter")
  endif()
else()
  set(SPIRV_EMITTER_OPTS "")
endif()

if(ENFORCE_QUEUE_SYNCHRONIZATION)
  list(APPEND CHIP_SPV_DEFINITIONS ENFORCE_QUEUE_SYNC)
endif()

if(PER_THREAD_DEFAULT_STREAM)
  list(APPEND CHIP_SPV_DEFINITIONS HIP_API_PER_THREAD_DEFAULT_STREAM)
endif()

if(LEVEL_ZERO_IMMEDIATE_QUEUES)
  list(APPEND CHIP_SPV_DEFINITIONS L0_IMM_QUEUES)
endif()

if(VERBOSE)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  add_compile_options("-v")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  add_compile_options("-Wall")
  list(APPEND CHIP_SPV_DEFINITIONS SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
else()
  list(APPEND CHIP_SPV_DEFINITIONS SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_WARN)
endif()

if(ERROR_IF_NOT_IMPLEMENTED)
  list(APPEND CHIP_SPV_DEFINITIONS CHIP_ERROR_ON_UNIMPL)
endif()

if(BUILD_SHARED_LIBS)
  message(STATUS "Buiding CHIP-SPV as a shared library")
  add_library(CHIP SHARED ${CHIP_SRC})
  set(CHIP_LIB_NAME "libCHIP.so")
else()
  message(STATUS "Buiding CHIP-SPV as a static library")
  add_library(CHIP STATIC ${CHIP_SRC})
  set(CHIP_LIB_NAME "libCHIP.a")
endif()

set(CHIP_INTERFACE_LIBS ${PTHREAD_LIBRARY})

if(OpenCL_LIBRARY)
  list(APPEND CHIP_SPV_DEFINITIONS HAVE_OPENCL)
  list(PREPEND CHIP_INTERFACE_LIBS ${OpenCL_LIBRARY})
endif()

if(LevelZero_LIBRARY)
  list(APPEND CHIP_SPV_DEFINITIONS HAVE_LEVEL0)
  list(PREPEND CHIP_INTERFACE_LIBS ${LevelZero_LIBRARY})
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CHIP_SPV_DEFAULT_INSTALL_DIR}" CACHE PATH "Install path prefix")
endif()

message(STATUS "CHIP-SPV will be installed to: ${CMAKE_INSTALL_PREFIX}")

# CHIP-SPV CMAKE OPTIONS PARSING
# =============================================================================

# =============================================================================
# CHIP-SPV BINARIES & TESTS
add_subdirectory(llvm_passes)
add_subdirectory(bitcode)

add_subdirectory(tests/cuda)
add_subdirectory(tests/hiprtc)
add_subdirectory(tests/compiler)
add_subdirectory(./samples samples)
set(HIPCC_BUILD_PATH "${CMAKE_BINARY_DIR}/bin")
add_subdirectory(HIPCC)

# Make CHIP depend on devicelib_bc and LLVMHipPasses for
# convenience. The CHIP module itself does not depend on these but
# HIP program compilation does.
add_dependencies(CHIP devicelib_bc LLVMHipPasses)
add_dependencies(samples CHIP)
add_dependencies(CHIP hipcc.bin hipconfig.bin)

add_subdirectory(bin)

# custom target to avoid tests that are known to fail
#
# Note that this list only contains tests external to CHIP-SPV,
# such as those frome HIP's testsuite; the internal tests
# should be disabled based on value ENABLE_FAILING_TESTS option

# tests that fail only when run many times
list(APPEND FLAKY_TESTS "Unit_hipStreamCreateWithPriority_ValidateWithEvents")
list(APPEND FLAKY_TESTS "cuda-simpleCallback")
list(APPEND FLAKY_TESTS "hipMultiThreadAddCallback")

list(APPEND FAILING_TESTS "hip_sycl_interop")
list(APPEND FAILING_TESTS "hip_sycl_interop_no_buffers")

list(APPEND FAILING_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rd_float") # (Failed)
list(APPEND FAILING_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rn_float") # (Failed)
list(APPEND FAILING_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_ru_float") # (Failed)
list(APPEND FAILING_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rz_float") # (Failed)
list(APPEND FAILING_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_l_int") # (Failed)
list(APPEND FAILING_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_lc_int") # (Failed)
list(APPEND FAILING_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_r_int") # (Failed)
list(APPEND FAILING_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_rc_int") # (Failed)

list(APPEND FAILING_TESTS "Unit_hipMemset2DAsync_MultiThread") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipMemset3DASyncMulti") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipGraphAddEventRecordNode_MultipleRun") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipGraphAddEventWaitNode_MultipleRun") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_diffprio") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_CapturingMultGraphsFrom1Strm") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipMemset2DAsync_MultiThread") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipMemsetFunctional_PartialSet_3D") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipStreamCreate_MultistreamBasicFunctionalities") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetStreamPriorityRange_Default") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipHostMalloc_CoherentAccess") # (Timeout)
list(APPEND FAILING_TESTS "Unit_hipStreamAddCallback_MultipleThreads") # (Timeout)

list(APPEND FAILING_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphAddDependencies_NegTest") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraph_BasicFunctional") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipGraphClone_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphAddHostNode_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphAddHostNode_ClonedGraphwithHostNode") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemory") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalConstMemory") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemoryWithKernel") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphNodeFindInClone_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalMemory") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalConstMemory") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_MemcpyToSymbolNodeWithKernel") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphDestroyNode_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphGetNodes_Functional") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphGetNodes_CapturedStream") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphGetNodes_ParamValidation") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphGetRootNodes_Functional") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphGetRootNodes_CapturedStream") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphGetRootNodes_ParamValidation") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphHostNodeSetParams_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphAddMemcpyNode1D_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphAddChildGraphNode_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphAddChildGraphNode_CloneChildGraph") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphAddChildGraphNode_MultipleChildNodes") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphAddChildGraphNode_SingleChildNode") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphNodeGetType_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphGetEdges_Functionality") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipGraphGetEdges_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphRemoveDependencies_Func_StrmCapture") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphRemoveDependencies_ChangeComputeFunc") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphRemoveDependencies_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphInstantiate_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecUpdate_Negative_Basic") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecUpdate_Negative_TypeChange") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphExecUpdate_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_VerifyEventNotChanged") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphEventWaitNodeSetEvent_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphMemsetNodeGetParams_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphMemsetNodeSetParams_InvalidParams") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphEventRecordNodeGetEvent_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphEventRecordNodeSetEvent_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphEventWaitNodeGetEvent_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_hipStreamPerThread") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_defaultflag") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_blockingflag") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffflags") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffprio") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_multiplestrms") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_CapturingFromWithinStrms") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_DetectingInvalidCapture") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_EndingCapturewhenCaptureInProgress") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_nestedStreamCapture") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_streamReuse") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_captureComplexGraph") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamBeginCapture_captureEmptyStreams") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamIsCapturing_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamIsCapturing_Functional_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamIsCapturing_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamIsCapturing_hipStreamPerThread") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamGetCaptureInfo_BasicFunctional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamGetCaptureInfo_hipStreamPerThread") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamGetCaptureInfo_UniqueID") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamGetCaptureInfo_ArgValidation") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamEndCapture_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamEndCapture_Thread_Negative") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_SetAndVerifyMemory") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_VerifyEventNotChanged") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphAddMemsetNode_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphAddKernelNode_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphMemcpyNodeGetParams_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphMemcpyNodeGetParams_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphMemcpyNodeSetParams_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphKernelNodeGetParams_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphKernelNodeSetParams_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphKernelNodeSetParams_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphKernelNodeGetSetParams_Functional") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecKernelNodeSetParams_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecKernelNodeSetParams_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphLaunch_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphMemcpyNodeSetParams1D_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphNodeGetDependentNodes_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphNodeGetDependentNodes_ParamValidation") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphNodeGetDependencies_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphNodeGetDependencies_ParamValidation") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphHostNodeGetParams_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamGetCaptureInfo_v2_BasicFunctional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamGetCaptureInfo_v2_hipStreamPerThread") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipStreamGetCaptureInfo_v2_UniqueID") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamGetCaptureInfo_v2_ParamValidation") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipUserObjectCreate_Functional_1") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipUserObjectCreate_Functional_2") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipUserObjectCreate_Functional_3") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipUserObjectCreate_Functional_4") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipUserObjectCreate_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipUserObjectRelease_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipUserObjectRetain_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipUserObj_Negative_Test") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphRetainUserObject_Functional_1") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphRetainUserObject_Functional_2") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphRetainUserObject_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphReleaseUserObject_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphRetainUserObject_Negative_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGraphRetainUserObject_Negative_Null_Object") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostRegister_Flags - int") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostRegister_Flags - float") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostRegister_Flags - double") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostRegister_Negative - int") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostRegister_Negative - float") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostRegister_Negative - double") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostUnregister_MemoryNotAccessableAfterUnregister") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostUnregister_NullPtr") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostUnregister_NotRegisteredPointer") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipHostUnregister_AlreadyUnregisteredPointer") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipMalloc3D_ValidatePitch") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3D_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocPitch_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemAllocPitch_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostGetFlags_DifferentThreads") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipHostGetFlags_InvalidArgs") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostGetDevicePointer_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocManaged_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset_Negative_InvalidPtr") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset2D_Negative_InvalidPtr") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset2D_Negative_InvalidSizes") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset2D_Negative_OutOfBoundsPtr") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset_SetMemoryWithOffset") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemsetAsync_SetMemoryWithOffset") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostMalloc_ArgValidation") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemcpyFromToSymbol_Negative") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipPtrGetAttribute_Simple") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemPoolApi_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemPoolApi_BasicAlloc") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemPoolApi_BasicTrim") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemPoolApi_BasicReuse") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemPoolApi_Opportunistic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemPoolApi_Default") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc_ArgumentValidation") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostMalloc_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostMalloc_NonCoherent") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostMalloc_Coherent") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostMalloc_Default") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemset") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD32") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD16") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD8") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemsetFunctional_PartialSet_1D") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemsetFunctional_PartialSet_2D") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_DiffSizes") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_MultiThread") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_MaxTexture_Default - uint") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_MaxTexture_Default - int4") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_MaxTexture_Default - ushort") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_MaxTexture_Default - short2") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_MaxTexture_Default - char") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_MaxTexture_Default - char4") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_MaxTexture_Default - float2") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_MaxTexture_Default - float4") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_DifferentChannelSizes") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_NullArrayPtr") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_NullDescPtr") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_BadFlags") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float2") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float4") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_BadNumberOfBits") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_3ChannelElement") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_ChannelAfterZeroChannel") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_InvalidChannelFormat") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMallocArray_Negative_NumericLimit") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_DiffSizes") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_MultiThread") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_NullArrayPtr") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_NullDescPtr") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_ZeroHeight") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_InvalidFlags") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelLayout") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_8BitFloat") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_DifferentChannelSizes") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelSize") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_NumericLimit") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - char") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - uchar2") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - short4") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float2") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float4") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - uint8_t") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - int") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - float") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvMemcpy3D_HosttoDevice") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvMemcpy3D_ExtentValidation") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - uint8_t") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - int") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - float") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvMemcpy3DAsync_HosttoDevice") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvMemcpy3DAsync_ExtentValidation") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipPointerGetAttribute_MemoryTypes") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipPointerGetAttribute_KernelUpdation") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipPointerGetAttribute_BufferID") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipPointerGetAttribute_MappedMem") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipPointerGetAttribute_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvPtrGetAttributes_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDrvPtrGetAttributes_Functional") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemGetInfo_DifferentMallocSmall") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemGetInfo_ParaSmall") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipMemGetInfo_ParaNonDiv") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipMemGetInfo_ParaMultiSmall") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipMemGetInfo_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipFreeNegativeDev") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipFreeNegativeHost") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipFreeNegativeArray") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipFreeDoubleDevice") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipFreeDoubleHost") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipFreeDoubleArray") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTDev - char") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTDev - int") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTDev - float2") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTDev - float4") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTHost - char") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTHost - int") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTHost - float2") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTHost - float4") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTArray - char") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTArray - int") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTArray - float2") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipFreeMultiTArray - float4") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipMemsetASyncMulti") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemset2DASyncMulti") # (Failed)
list(APPEND FAILING_TESTS "Unit_HMM_OverSubscriptionTst") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipStreamGetFlags_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipEventCreate_IncompatibleFlags") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipEventRecord") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipEventIpc") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipEventSynchronize_Default_Positive") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetName_NegTst") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetPCIBusId_Negative_PartialFill") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Default") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Threaded") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_HipDeviceGetCacheConfig_Negative_Parameters") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceTotalMem_NegTst") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGetSetDeviceFlags_NullptrFlag") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGetSetDeviceFlags_InvalidFlag") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGetSetDeviceFlags_ValidFlag") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGetSetDeviceFlags_SetThenGet") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGetSetDeviceFlags_Threaded") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipGetDeviceFlags_Positive_Context") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGetSetDevice_MultiThreaded") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipSetGetDevice_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetUuid_Positive") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetUuid_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetDefaultMemPool_Positive_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetDefaultMemPool_Negative_Parameters") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceSetLimit_SetGet") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Default") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Threaded") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetSharedMemConfig_Negative_Parameters") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceReset_Positive_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceReset_Positive_Threaded") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceSetMemPool_Positive_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceSetMemPool_Negative_Parameters") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetMemPool_Positive_Default") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetMemPool_Positive_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetMemPool_Positive_Threaded") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDeviceGetMemPool_Negative_Parameters") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipInit_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipDriverGetVersion_Negative") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Reused_Memory") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipIpcCloseMemHandle_Positive_Reference_Counting") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipIpcCloseMemHandle_Negative_Close_In_Originating_Process") # (Failed)
list(APPEND FAILING_TESTS "Unit_printf_flags") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_printf_specifier") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipCreateTextureObject_ArgValidation") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipCreateTextureObject_LinearResource") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipCreateTextureObject_Pitch2DResource") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipGetChannelDesc_CreateAndGet") # (Failed)
list(APPEND FAILING_TESTS "Unit_tex1Dfetch_CheckModes") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipTextureObj1DCheckRGBAModes - array") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - array") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipTextureObj1DCheckRGBAModes - buffer") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - buffer") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipTextureObj2DCheckRGBAModes") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipTextureObj2DCheckSRGBAModes") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipMemFaultStackAllocation_Check") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # (SEGFAULT)
list(APPEND FAILING_TESTS "Unit_hipGetLastError_Positive_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipGetLastError_Positive_Threaded") # (Subprocess aborted)
list(APPEND FAILING_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # (Failed)
list(APPEND FAILING_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # (Subprocess aborted)


list(APPEND FAILING_TESTS ${FLAKY_TESTS})

string(REGEX REPLACE ";" "\$|" FAILING_TESTS_STR "${FAILING_TESTS}")
set(EXCLUDED_TESTS "${FAILING_TESTS_STR}$")

# TODO fix-254 how do I make these read from the environment?
# MULTI_TESTS_REPEAT=33 make multi_tests
# Preferably without an additional reconfigure. Every way I tried escaping ${MULTI_TESTS_REPEAT} results in something undesirable like \${MULTI_TESTS_REPEAT}
set(FLAKY_TESTS_REPEAT 100)
set(MULTI_TESTS_REPEAT 10)
set(PARALLEL_TESTS 1)

set(TEST_OPTIONS -j ${PARALLEL_TESTS} --timeout 120 --output-on-failure)
add_custom_target(flaky_tests COMMAND ${CMAKE_CTEST_COMMAND} ${TEST_OPTIONS} -R ${FLAKY_TESTS} --repeat until-fail:${FLAKY_TESTS_REPEAT} USES_TERMINAL VERBATIM)
add_custom_target(multi_tests COMMAND ${CMAKE_CTEST_COMMAND} ${TEST_OPTIONS} -R "[Aa]sync|[Mm]ulti[Tt]hread|[Mm]ulti[Ss]tream|[Tt]hread|[Ss]tream" --repeat until-fail:${MULTI_TESTS_REPEAT} USES_TERMINAL VERBATIM)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} ${TEST_OPTIONS} -E ${EXCLUDED_TESTS} VERBATIM)

# CHIP-SPV BINARIES & TESTS
# =============================================================================

# =============================================================================
# CHIP-SPV COMPILATION SETUP
# For building on top of HIP-COMMON
target_compile_options(CHIP PRIVATE ${HIP_ENABLE_SPIRV} ${CHIP_SPV_COMPILE_FLAGS})
target_compile_definitions(CHIP PRIVATE ${CHIP_SPV_DEFINITIONS})

target_link_libraries(CHIP INTERFACE ${CHIP_INTERFACE_LIBS})

if(HAS_EXPERIMENTAL_FILESYSTEM)
  target_link_libraries(CHIP PUBLIC stdc++fs)
endif()

# Previously these were set to SYSTEM which resulted in CMake picking up
# OpenCL headers from the system where the version might differ resulting in errors.
target_include_directories(CHIP
  PUBLIC
  "$<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>"
  "$<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}/cuspv>"
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/HIP/include>"
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/cuspv>"
  PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}"
)

# CHIP-SPV COMPILATION SETUP
# =============================================================================

# =============================================================================
# HIP OFFLOAD FLAGS
set(HIP_OFFLOAD_COMPILE_ONLY_OPTIONS_ -x hip --target=x86_64-linux-gnu
  ${SPIRV_EMITTER_OPTS} ${DISABLE_OPAQUE_PTRS_OPT})

# hipDeviceLink sample requires --offload=spirv64 flag at link time. Why?
if(NOT CLANG_VERSION_LESS_14)
  set(HIP_OFFLOAD_COMPILE_ONLY_OPTIONS_ ${HIP_OFFLOAD_COMPILE_ONLY_OPTIONS_}
    --offload=spirv64

    # By default RocmInstallationDetector::AddHIPIncludeArgs in the
    # HIP-Clang inserts include wrapper headers which cause compile
    # errors when compiling HIP sources for SPIR-V target.  This flag
    # excludes the wrappers.
    "-nohipwrapperinc")
endif()

if(CLANG_VERSION_LESS_14)
  set(HIP_LINK_DEVICELIB_BUILD --hip-device-lib-path=${CMAKE_BINARY_DIR}/share --hip-device-lib=devicelib.bc --hip-llvm-pass-path=${CMAKE_BINARY_DIR}/llvm_passes)
  set(HIP_LINK_DEVICELIB_INSTALL --hip-device-lib-path=${SHARE_INSTALL_DIR} --hip-device-lib=devicelib.bc --hip-llvm-pass-path=${LIB_INSTALL_DIR}/llvm)
else()
  set(HIP_LINK_DEVICELIB_BUILD --hip-path=${CMAKE_BINARY_DIR})
  set(HIP_LINK_DEVICELIB_INSTALL --hip-path=${CMAKE_INSTALL_PREFIX})
endif()

# For use by hipcc
set(HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL_
  ${HIP_ENABLE_SPIRV}
  ${HIP_OFFLOAD_COMPILE_ONLY_OPTIONS_}
  ${HIP_LINK_DEVICELIB_INSTALL})

set(HIP_OFFLOAD_COMPILE_OPTIONS_BUILD_
  ${HIP_ENABLE_SPIRV}
  ${HIP_OFFLOAD_COMPILE_ONLY_OPTIONS_}
  ${HIP_LINK_DEVICELIB_BUILD})

# HIP applications need to link against libCHIP.so; add it to rpath
list(APPEND HIP_OFFLOAD_LINK_OPTIONS_INSTALL_ "-L${LIB_INSTALL_DIR}" "-lCHIP")
list(APPEND HIP_OFFLOAD_LINK_OPTIONS_BUILD_ "-L${CMAKE_BINARY_DIR}" "-lCHIP")

if(OpenCL_LIBRARY)
  list(APPEND HIP_OFFLOAD_LINK_OPTIONS_INSTALL_ "-L${OpenCL_DIR}" "-lOpenCL")
  list(APPEND HIP_OFFLOAD_LINK_OPTIONS_BUILD_ "-L${OpenCL_DIR}" "-lOpenCL")
endif()

if(LevelZero_LIBRARY)
  list(APPEND HIP_OFFLOAD_LINK_OPTIONS_INSTALL_ "-L${LevelZero_DIR}" "-lze_loader")
  list(APPEND HIP_OFFLOAD_LINK_OPTIONS_BUILD_ "-L${LevelZero_DIR}" "-lze_loader")
endif()

if(SET_RPATH)
  list(APPEND HIP_OFFLOAD_LINK_OPTIONS_INSTALL_ "-Wl,-rpath,${LIB_INSTALL_DIR}")
  list(APPEND HIP_OFFLOAD_LINK_OPTIONS_BUILD_ "-Wl,-rpath,${CMAKE_BINARY_DIR}")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# Workaround istead of trying to generate the CMake generator expression
string(REPLACE ";" " " HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL "${HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL_}")
string(REPLACE ";" " " HIP_OFFLOAD_COMPILE_OPTIONS_BUILD "${HIP_OFFLOAD_COMPILE_OPTIONS_BUILD_}")
string(REPLACE ";" " " HIP_OFFLOAD_LINK_OPTIONS_INSTALL "${HIP_OFFLOAD_LINK_OPTIONS_INSTALL_}")
string(REPLACE ";" " " HIP_OFFLOAD_LINK_OPTIONS_BUILD "${HIP_OFFLOAD_LINK_OPTIONS_BUILD_}")
message(STATUS "Generated HIP_OFFLOAD_COMPILE_OPTIONS: ${HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL}")
message(STATUS "Generated HIP_OFFLOAD_LINK_OPTIONS: ${HIP_OFFLOAD_LINK_OPTIONS_INSTALL}")

# Must separate these targets into deviceInteral to be used for building samples,etc.
# and device for external export. Have to do this because generator expressions don't seem to work here.
# TODO: Is there a better way to do this?
add_library(deviceInternal INTERFACE)
target_compile_options(deviceInternal INTERFACE
  ${HIP_ENABLE_SPIRV}
  ${HIP_OFFLOAD_COMPILE_ONLY_OPTIONS_}
  ${HIP_LINK_DEVICELIB_BUILD})
target_link_options(deviceInternal INTERFACE
  ${HIP_LINK_DEVICELIB_BUILD})
target_link_libraries(deviceInternal INTERFACE CHIP)

add_library(device INTERFACE)
target_compile_options(device INTERFACE
  ${HIP_ENABLE_SPIRV}
  ${HIP_OFFLOAD_COMPILE_ONLY_OPTIONS_}
  ${HIP_LINK_DEVICELIB_INSTALL})
target_link_options(device INTERFACE
  ${HIP_LINK_DEVICELIB_INSTALL})
target_link_libraries(device INTERFACE CHIP)

# same as device on CHIP-SPV but provides compatibility with AMD
add_library(host INTERFACE)
target_link_libraries(host INTERFACE device)

# HIP OFFLOAD FLAGS
# =============================================================================

# =============================================================================
# PROJECT CONFIGURATION HEADER
set(CHIP_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(CHIP_BUILD_DIR ${CMAKE_BINARY_DIR})
set(CHIP_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(CHIP_CLANG_PATH ${CLANG_BIN_PATH})
configure_file(Config.hh.in Config.hh @ONLY)

# PROJECT CONFIGURATION HEADER
# =============================================================================

# =============================================================================
# CHIP-SPV INSTALLATION AND PACKAGING
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Setup .hipVersion
# Install .hipVersion for hipcc. The file is used to detect a HIP
# runtime and clang extracts HIP version from it.
set(_versionInfo "# Auto-generated by cmake\n")
set(HIP_VERSION_MAJOR 5)
set(HIP_VERSION_MINOR 1)
set(HIP_VERSION_PATCH 0)
set(HIP_VERSION_PATCH_GITHASH 0) # TODO

add_to_config(_versionInfo HIP_VERSION_MAJOR "${HIP_VERSION_MAJOR}")
add_to_config(_versionInfo HIP_VERSION_MINOR "${HIP_VERSION_MINOR}")
add_to_config(_versionInfo HIP_VERSION_PATCH "${HIP_VERSION_PATCH}")
add_to_config(_versionInfo HIP_VERSION_PATCH_GITHASH "${HIP_VERSION_PATCH_GITHASH}")

# Generate and install .hipVersion file.
file(WRITE ${PROJECT_BINARY_DIR}/bin/.hipVersion ${_versionInfo})
install(FILES ${PROJECT_BINARY_DIR}/bin/.hipVersion DESTINATION bin)

# Setup .hipInfo. One for install and another for build directory.
string(TIMESTAMP _timestamp UTC)
set(_hipInfo_install "# Auto-generated by cmake on ${_timestamp} UTC\n")
message(STATUS "HIP_PATH: " ${CMAKE_INSTALL_PREFIX})
message(STATUS "HIP Runtime: " ${HIP_RUNTIME})
message(STATUS "HIP Compiler: " ${HIP_COMPILER})
add_to_config(_hipInfo_install HIP_PATH "${CMAKE_INSTALL_PREFIX}")
add_to_config(_hipInfo_install HIP_RUNTIME "${HIP_RUNTIME}")
add_to_config(_hipInfo_install HIP_COMPILER "${HIP_COMPILER}")
add_to_config(_hipInfo_install HIP_ARCH "${HIP_ARCH}")
add_to_config(_hipInfo_install HIP_CLANG_PATH "${HIP_CLANG_PATH}")
add_to_config(_hipInfo_install
  HIP_OFFLOAD_COMPILE_OPTIONS "${HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL}")
add_to_config(_hipInfo_install
  HIP_OFFLOAD_LINK_OPTIONS "${HIP_OFFLOAD_LINK_OPTIONS_INSTALL}")

# Build version
set(_hipInfo_build "# Auto-generated by cmake on ${_timestamp} UTC\n")
add_to_config(_hipInfo_build HIP_PATH "${CMAKE_BINARY_DIR}")
add_to_config(_hipInfo_build HIP_RUNTIME "${HIP_RUNTIME}")
add_to_config(_hipInfo_build HIP_COMPILER "${HIP_COMPILER}")
add_to_config(_hipInfo_build HIP_ARCH "${HIP_ARCH}")
add_to_config(_hipInfo_build HIP_CLANG_PATH "${HIP_CLANG_PATH}")
add_to_config(_hipInfo_build
  HIP_OFFLOAD_COMPILE_OPTIONS "${HIP_OFFLOAD_COMPILE_OPTIONS_BUILD}")
add_to_config(_hipInfo_build
  HIP_OFFLOAD_LINK_OPTIONS "${HIP_OFFLOAD_LINK_OPTIONS_BUILD}")

INSTALL(TARGETS CHIP host device
  EXPORT hip-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
INSTALL(EXPORT hip-targets DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR} NAMESPACE hip::)

# hip-config.cmake generation
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake/hip-config-in.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/hip-config.cmake
  INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
  PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR SHARE_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/hip-config-version.cmake
  VERSION "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_VERSION_GITDATE}"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/hip-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/hip-config-version.cmake
  DESTINATION
  ${CONFIG_PACKAGE_INSTALL_DIR}
)
install(EXPORT hip-targets FILE hip-targets.cmake NAMESPACE hip:: DESTINATION lib/cmake/hip COMPONENT binary)

# Generate .hipInfo
file(WRITE "${PROJECT_BINARY_DIR}/share/.hipInfo_install" ${_hipInfo_install})
file(WRITE "${PROJECT_BINARY_DIR}/share/.hipInfo" ${_hipInfo_build})
install(FILES ${PROJECT_BINARY_DIR}/share/.hipInfo_install
  DESTINATION ${SHARE_INSTALL_DIR} RENAME .hipInfo)

# Copy over includes for hipcc to work in the build dir
file(COPY ${CMAKE_SOURCE_DIR}/include DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/HIP/include/hip DESTINATION ${CMAKE_BINARY_DIR}/include)

# Copy hipconfig, hipvars, etc to bin
install(FILES ${CMAKE_BINARY_DIR}/bin/hipcc.bin DESTINATION ${BIN_INSTALL_DIR} RENAME hipcc PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
install(FILES ${CMAKE_BINARY_DIR}/bin/hipconfig.bin DESTINATION ${BIN_INSTALL_DIR} RENAME hipconfig PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)

set(CHIP_CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
set(CHIP_CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS CHIP
  EXPORT CHIPTargets
  LIBRARY DESTINATION ${CHIP_CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CHIP_CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${PROJECT_BINARY_DIR}/${CHIP_LIB_NAME} DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HIP/include DESTINATION . USE_SOURCE_PERMISSIONS)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION . USE_SOURCE_PERMISSIONS)

install(EXPORT CHIPTargets
  FILE CHIPTargets.cmake
  NAMESPACE CHIP::
  DESTINATION cmake/CHIP)

# CHIP-SPV INSTALLATION AND PACKAGING
# =============================================================================
#
if(BUILD_TESTS)
  SET(HIP_PATH ${CMAKE_BINARY_DIR})
  if(USE_EXTERNAL_HIP_TESTS)
    add_subdirectory(hip-tests/catch catch)
  else()
    add_subdirectory(HIP/tests/catch catch)
  endif()
  add_dependencies(build_tests CHIP samples)
endif()

if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_GENERATE_MAN NO)
  set(DOXYGEN_OUTPUT_DIRECTORY doxygen)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
  doxygen_add_docs(gendocs README.md src llvm_passes
    COMMENT "generate documentation for source")
endif()
