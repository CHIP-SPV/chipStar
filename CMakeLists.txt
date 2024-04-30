# =============================================================================
# CMake build system files
#
# Copyright (c) 2021-22 chipStar developers
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
add_compile_options(-Wno-format-extra-args -mf16c)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-duplicate-decl-specifier \
-Wno-tautological-constant-compare  -Wno-c++20-extensions -Wno-unused-result \
-Wno-delete-abstract-non-virtual-dtor -Wno-deprecated-declarations -Wunused-command-line-argument -gdwarf-4")

# end temporary
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CHIPSTAR_MAJOR_VERSION 1)
set(CHIPSTAR_MINOR_VERSION 1)
set(CHIPSTAR_PATCH_VERSION 00)
set(CHIPSTAR_VERSION "${CHIPSTAR_MAJOR_VERSION}.${CHIPSTAR_MINOR_VERSION}.${CHIPSTAR_PATCH_VERSION}")

project(chipStar
  VERSION ${CHIPSTAR_VERSION}
  DESCRIPTION "HIP implementation for runtimes that support SPIR-V"
  LANGUAGES NONE)

include(FindLLVM) # Enables C, CXX
include(LLVMCheck)
include(chip_spv_cmake_macros)
enable_testing()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

set(REQUIRED_DIRECTORIES
    HIPCC
    HIP
    bitcode/ROCm-Device-Libs
    hip-tests
    hip-testsuite
)

foreach(dir ${REQUIRED_DIRECTORIES})
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${dir}")
        message(FATAL_ERROR "Directory '${dir}' is missing. Please execute: git submodule update --init")
    endif()
endforeach()

# =============================================================================
# chipStar CMAKE DEPENDENCIES
if(NOT DEFINED OpenCL_LIBRARY)
  message(STATUS "OpenCL_LIBRARY was not set. Searching for libOpenCL.so in LD_LIBRARY_PATH")
  find_library(OpenCL_LIBRARY NAMES OpenCL PATHS ENV LD_LIBRARY_PATH ./ NO_CACHE)
endif()

if(OpenCL_LIBRARY)
  get_filename_component(OpenCL_DIR ${OpenCL_LIBRARY} DIRECTORY CACHE)
  message(STATUS "Setting OpenCL_DIR to ${OpenCL_DIR}")
else()
  message(STATUS "OpenCL not found")
endif()

if(NOT DEFINED LevelZero_LIBRARY)
  message(STATUS "LevelZero_LIBRARY was not set. Searching for ze_loader.so in LD_LIBRARY_PATH")
  find_library(LevelZero_LIBRARY NAMES ze_loader PATHS ENV LD_LIBRARY_PATH NO_CACHE)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(LevelZero_LIBRARY)
  get_filename_component(LevelZeroLib_DIR ${LevelZero_LIBRARY} DIRECTORY CACHE)
  # Check if the library is in the system path
  if("${LevelZeroLib_DIR}" STREQUAL "/usr/lib/x86_64-linux-gnu")
    set(LevelZeroInclude_DIR "/usr/include/level_zero")
  else()
    # get the path to the LevelZero directory
    set(LevelZeroInclude_DIR "${LevelZeroLib_DIR}/../include/level_zero")
  endif()

  message(STATUS "Setting LevelZeroInclude_DIR to ${LevelZeroInclude_DIR}")
  message(STATUS "Setting LevelZeroLib_DIR to ${LevelZeroLib_DIR}")

  # make sure that ze_api.h exists in LevelZeroInclude_DIR
  if(NOT EXISTS "${LevelZeroInclude_DIR}/ze_api.h")
    message(FATAL_ERROR "ze_api.h was not found in ${LevelZeroInclude_DIR}")
  endif()

  # Re-enable this check once more deployments reach 1.6
  # # grep ze_api.h for @version and make sure it's at least 1.6
  # execute_process(COMMAND grep "@version" "${LevelZeroInclude_DIR}/ze_api.h" OUTPUT_VARIABLE ZE_API_VERSION)
  # string(REGEX MATCHALL "[0-9]+" ZE_API_VERSION "${ZE_API_VERSION}")
  # list(GET ZE_API_VERSION 0 ZE_API_MAJOR)
  # list(GET ZE_API_VERSION 1 ZE_API_MINOR)
  # if(ZE_API_MAJOR LESS 1 OR (ZE_API_MAJOR EQUAL 1 AND ZE_API_MINOR LESS 6))
  #   message(FATAL_ERROR "LevelZero version must be at least 1.6")
  # else()
  #   message(STATUS "LevelZero version: ${ZE_API_MAJOR}.${ZE_API_MINOR}")
  # endif()
  
else()
  message(STATUS "Level Zero not found")
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

# chipStar CMAKE DEPENDENCIES
# =============================================================================

# =============================================================================
# chipStar CMAKE VARIABLBLES/OPTIONS
set(CHIP_SRC
  src/spirv.cc
  src/spirv_hiprtc.cc
  src/CHIPDriver.cc
  src/CHIPBackend.cc
  src/SPVRegister.cc
  src/CHIPGraph.cc
  src/CHIPBindings.cc
  src/logging.cc
  src/Utils.cc
  src/SPIRVFuncInfo.cc
)

if(OpenCL_LIBRARY)
  list(APPEND CHIP_SRC
    src/backend/OpenCL/CHIPBackendOpenCL.cc
    src/backend/OpenCL/MemoryManager.cc)
endif()

if(LevelZero_LIBRARY)
  list(APPEND CHIP_SRC
    src/backend/Level0/CHIPBackendLevel0.cc)
endif()

set(CHIP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "chipStar source directory")

list(APPEND CHIP_SPV_DEFINITIONS "")

# used by hipcc/hipconfig
set(HIP_COMPILER "clang" CACHE STRING "chipStar compiler")
set(HIP_RUNTIME "spirv" CACHE STRING "chipStar runtime")
set(HIP_PLATFORM "spirv" CACHE STRING "chipStar platform")
set(HIP_ARCH "spirv" CACHE STRING "chipStar architecture")
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

# HIP options
option(STANDALONE_TESTS "Create a separate executable for each test instead of combining tests into a shared lib by category" ON)

# chipStar options
option(CHIP_ERROR_IF_NOT_IMPLEMENTED "Some HIP functionality is not yet implemented in chipStar. If such a function is called, throw hipErrorNotSupported" ON)
option(CHIP_VERBOSE "Verbose compilation" OFF)
option(CHIP_BUILD_SHARED_LIBS "Build chipStar as a shared library" ON)
option(CHIP_BUILD_DOCS "Build doxygen documentation" OFF)
option(CHIP_LLVM_USE_INTERGRATED_SPIRV "Use LLVM's intergrated SPIR-V backend for emitting device binary instead of SPIR-V translator. Requires LLVM 15" OFF)
option(CHIP_SET_RPATH "Add CMAKE_INSTALL_PREFIX/lib to the RPATH for chipStar executables" ON)
option(CHIP_ENABLE_UNCOMPILABLE_TESTS "Enable tests which are known to not compile" OFF)
option(CHIP_BUILD_TESTS "Enable build_tests target" ON)
option(CHIP_BUILD_SAMPLES "Build samples" ON)
option(CHIP_DUBIOUS_LOCKS "Enable locks that don't seem necessary but make a lot of valgrind issues go away" OFF)
option(CHIP_USE_EXTERNAL_HIP_TESTS "Use Catch2 tests from the hip-tests submodule" OFF)
option(CHIP_ENABLE_NON_COMPLIANT_DEVICELIB_CODE "Enable non-compliant devicelib code such as calling LLVM builtins from inside kernel code. Enables certain unsigned long devicelib func variants" OFF)
option(CHIP_FAST_MATH "Use native_ OpenCL functions which are fast but their precision is implementation defined" OFF)
option(CHIP_USE_INTEL_USM "When enabled, cl_intel_unified_shared_memory extension, when available, will be used for HIP allocations in the OpenCL backend" ON)
option(CHIP_BUILD_HIPBLAS "When enabled, hipBLAS will be built and installed along chipStar" OFF)
option(CATCH2_DISCOVER_TESTS_COMPILE_TIME "Discover the tests at compile time" ON)

# This mitigation might be necessary on some systems with an older runtime. 
# This mitigation makes memory resident (disable swapping) on the GPU
# This has a significant impact on the cost of a GPU malloc 
# If your application does a lot of small allocations, it might be slow
# https://github.com/CHIP-SPV/chipStar/issues/458 https://github.com/intel/compute-runtime/issues/631
option(CHIP_L0_FIRST_TOUCH "First-touch workaround for Level Zero." OFF)
set(CHIP_DEFAULT_JIT_FLAGS "-cl-kernel-arg-info -cl-std=CL3.0")

option(CHIP_PROFILE_TESTS "Attach iprof to each test under ctest" OFF)
if (CHIP_PROFILE_TESTS)
  set(HIP_PROFILE_TESTS_COMMAND "iprof" "-m" "full" "--")
  else()
  set(HIP_PROFILE_TESTS_COMMAND "")
endif()

option(OCML_BASIC_ROUNDED_OPERATIONS "Use OCML implementations for devicelib functions with explicit rounding mode such as __dadd_rd. Otherwise, rounding mode will be ignored" OFF)

# (old) ARM Mali GPU driver is known to fail to process valid SPIR-V
# modules with NoRwrite and NoReadWrite parameter attributes. Enabling
# this option will strip the such attributes.
option(CHIP_MALI_GPU_WORKAROUNDS "Apply work-arounds for avoiding SPIR-V \
consumption issues in ARM Mali GPU driver." OFF)

if(CHIP_EXT_FLOAT_ATOMICS)
  message(DEPRECATION "-DCHIP_EXT_FLOAT_ATOMICS is no longer effective.")
endif()

# Warpsize would optimally be a device-specific, queried and made
# effective at runtime. However, we need to fix the warpsize since SPIR-Vs need
# to be portable across multiple devices. It should be more portable to
# fix it to 32 instead of 64, with a trade-off of potential underutilization
# with GPUs with 64 wide warps.
set(DEFAULT_WARP_SIZE 32 CACHE STRING "The warp size to use.")

# Set the default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "The build type to use Debug, Release, RelWithDebInfo, MinRelSize" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CHIP_SPV_DEFAULT_INSTALL_DIR "${CMAKE_SOURCE_DIR}/install" CACHE PATH "The installation directory for chipStar")
set(HIP_ENABLE_SPIRV "-D__HIP_PLATFORM_SPIRV__=" CACHE INTERNAL "Compiler difinition which enables SPIR-V defines")

# FOR EXTERNAL USE
set(HIP_OFFLOAD_COMPILE_OPTIONS "" CACHE STRING "space separated list of compiler flags needed for generating offload code")
set(HIP_OFFLOAD_LINK_OPTIONS "" CACHE STRING "space separated list of compiler flags needed for linking binaries with GPU offload")

# chipStar CMAKE VARIABLBLES/OPTIONS
# =============================================================================

# =============================================================================
# chipStar CMAKE OPTIONS PARSING


set(DISABLE_OPAQUE_PTRS_OPT "")

if(NOT CLANG_VERSION_LESS_15 AND CLANG_VERSION_LESS_16)
  # LLVM 15 switches to opaque pointer mode by default which
  # llvm-spirv currently does not support. Switch the mode off.
  #
  # NOTE: https://reviews.llvm.org/D130766 will disable the mode off
  # for SPIR-V targets but the patch has not yet landed on the
  # LLVM 15 branch.
  #
  # For LLVM 16+ we swtich to use opaque pointers.
  #
  # NOTE: Bitcode library must be compiled with the same opaque
  # pointer setting.
  set(DISABLE_OPAQUE_PTRS_OPT -Xclang -no-opaque-pointers)
endif()

set(SPIRV_EMITTER_OPTS "")
if(CHIP_LLVM_USE_INTERGRATED_SPIRV)
  set(SPIRV_EMITTER_OPTS "-fintegrated-objemitter")
endif()

if(CHIP_VERBOSE)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  add_compile_options("-v")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  add_compile_options(-Wall -O1)
  list(APPEND CHIP_SPV_DEFINITIONS SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
else()
  list(APPEND CHIP_SPV_DEFINITIONS SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_WARN)
endif()

if(CHIP_BUILD_SHARED_LIBS)
  message(STATUS "Buiding chipStar as a shared library")
  add_library(CHIP SHARED ${CHIP_SRC})
  set(CHIP_LIB_NAME "libCHIP.so")
else()
  message(STATUS "Buiding chipStar as a static library")
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

message(STATUS "chipStar will be installed to: ${CMAKE_INSTALL_PREFIX}")

# =============================================================================
# chipStar BINARIES & TESTS
add_subdirectory(llvm_passes)
add_subdirectory(bitcode)

# Embed HIP device built-in implementations which are linked into users' device
# programs at runtime based on device capabilities.
target_sources(CHIP PRIVATE $<TARGET_OBJECTS:rtdevlib>)

set(HIPCC_BUILD_PATH "${CMAKE_BINARY_DIR}/bin")
add_subdirectory(HIPCC)

# Make CHIP depend on devicelib_bc and LLVMHipPasses for
# convenience. The CHIP module itself does not depend on these but
# HIP program compilation does.
add_dependencies(CHIP devicelib_bc LLVMHipPasses)
add_dependencies(CHIP hipcc.bin hipconfig.bin)

file(MAKE_DIRECTORY ${HIPCC_BUILD_PATH})
file(CREATE_LINK hipcc.bin ${HIPCC_BUILD_PATH}/hipcc SYMBOLIC)
file(CREATE_LINK hipconfig.bin ${HIPCC_BUILD_PATH}/hipconfig SYMBOLIC)

add_subdirectory(bin)

# once hipcc.bin is built, execute hipcc ./samples/hipSample.cpp
add_custom_target(hipInfoBin ALL
  COMMAND ${HIPCC_BUILD_PATH}/hipcc ${CMAKE_SOURCE_DIR}/samples/hipInfo/hipInfo.cpp -o ${CMAKE_BINARY_DIR}/hipInfo
  DEPENDS hipcc.bin CHIP devicelib_bc)

execute_process(
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/manage_known_failures.py 
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/known_failures.yaml
  --generate ${CMAKE_BINARY_DIR}/test_lists
  RESULT_VARIABLE result_var
  ERROR_VARIABLE error_var
)
message("Result: ${result_var}")
# if the script fails, print the error and abort
if(NOT result_var EQUAL 0)
  message(FATAL_ERROR "manage_known_failures.py: ${error_var}")
endif()
# chipStar BINARIES & TESTS
# =============================================================================

# =============================================================================
# chipStar COMPILATION SETUP
# For building on top of HIP-COMMON
target_compile_definitions(
  CHIP PRIVATE ${HIP_ENABLE_SPIRV} ${CHIP_SPV_DEFINITIONS})

target_link_libraries(CHIP PUBLIC ${CHIP_INTERFACE_LIBS})

if(HAS_EXPERIMENTAL_FILESYSTEM)
  target_link_libraries(CHIP PUBLIC stdc++fs)
endif()

# Previously these were set to SYSTEM which resulted in CMake picking up
# OpenCL headers from the system where the version might differ resulting in errors.
target_include_directories(CHIP
  PUBLIC
  "$<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>"
  "$<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}/cuspv>"
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/bitcode>"
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/HIP/include>"
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/cuspv>"
  PRIVATE
  "${CMAKE_BINARY_DIR}/include" # for chipStarConfig.hh, should be first
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/bitcode"
)


# =============================================================================
# HIP OFFLOAD FLAGS
set(HIP_OFFLOAD_COMPILE_ONLY_OPTIONS_
  --offload=spirv64 ${SPIRV_EMITTER_OPTS} ${DISABLE_OPAQUE_PTRS_OPT}
  # By default RocmInstallationDetector::AddHIPIncludeArgs in the
  # HIP-Clang inserts include wrapper headers which cause compile
  # errors when compiling HIP sources for SPIR-V target.  This flag
  # excludes the wrappers.
  -nohipwrapperinc)

# Include a header for applying fixups before any user or system includes.
set(HIP_FIXUPS_HEADER_BUILD
  -include ${CMAKE_SOURCE_DIR}/include/hip/spirv_fixups.h)
set(HIP_FIXUPS_HEADER_INSTALL
  -include ${CMAKE_INSTALL_PREFIX}/include/hip/spirv_fixups.h)

# Flags needed additionally for linking phase with -fgpu-rdc.
set(HIP_RDC_SUPPLEMENT_LINK_FLAGS_
  # Infors clang the type of the code object inputs (which are different than
  # in regular host C/C++ linking)
  --hip-link
  # Required for selecting HIPSPV toolchain instead of AMD's one in clang.
  --offload=spirv64
  # --hip-path is also needed but not included here (different option
  # value is needed for build and installation).
)

if (NOT CLANG_VERSION_LESS_15)
  list(APPEND HIP_RDC_SUPPLEMENT_LINK_FLAGS_
    -no-hip-rt) # Clang 15+: Prevents linking amdhip64 library.
endif()

# For use by hipcc
execute_process(COMMAND ${LLVM_CONFIG_BIN} --host-target
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE HOST_ARCH)
set(HOST_ARCH "--target=${HOST_ARCH}")

set(HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL_
  ${HIP_ENABLE_SPIRV}
  ${HIP_OFFLOAD_COMPILE_ONLY_OPTIONS_}
  --hip-path=${CMAKE_INSTALL_PREFIX}
  ${HOST_ARCH}
  ${HIP_FIXUPS_HEADER_INSTALL})

set(HIP_OFFLOAD_COMPILE_OPTIONS_BUILD_
  ${HIP_ENABLE_SPIRV}
  ${HIP_OFFLOAD_COMPILE_ONLY_OPTIONS_}
  --hip-path=${CMAKE_BINARY_DIR}
  ${HOST_ARCH}
  ${HIP_FIXUPS_HEADER_BUILD}
  -I${CMAKE_SOURCE_DIR}/include
  -I${CMAKE_SOURCE_DIR}/HIP/include
  -I${CMAKE_BINARY_DIR}/include)

# HIP applications need to link against libCHIP.so; add it to rpath
#
# -no-hip-rt: Prevents linking amdhip64 library implicitly when doing
#             combined compile and link or when the --hip-link is
#             present at the link step.
list(APPEND HIP_OFFLOAD_LINK_OPTIONS_INSTALL_
  "-L${LIB_INSTALL_DIR}" "-lCHIP" "-no-hip-rt")
list(APPEND HIP_OFFLOAD_LINK_OPTIONS_BUILD_
  "-L${CMAKE_BINARY_DIR}" "-lCHIP" "-no-hip-rt")

if(OpenCL_LIBRARY)
  target_link_options(CHIP PUBLIC -Wl,-rpath,${OpenCL_DIR})
  target_link_directories(CHIP PUBLIC ${OpenCL_DIR})
  target_link_libraries(CHIP PUBLIC OpenCL)
endif()

if(LevelZero_LIBRARY)
  include_directories(CHIP PUBLIC ${LevelZeroInclude_DIR})
  target_link_options(CHIP PUBLIC -Wl,-rpath,${LevelZeroLib_DIR})
  target_link_directories(CHIP PUBLIC ${LevelZeroLib_DIR})
  target_link_libraries(CHIP PUBLIC ze_loader)
endif()

if(CHIP_SET_RPATH)
  list(APPEND HIP_OFFLOAD_LINK_OPTIONS_INSTALL_ "-Wl,-rpath,${LIB_INSTALL_DIR}")
  list(APPEND HIP_OFFLOAD_LINK_OPTIONS_BUILD_ "-Wl,-rpath,${CMAKE_BINARY_DIR}")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# Workaround istead of trying to generate the CMake generator expression
string(REPLACE ";" " " HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL "${HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL_}")
string(REPLACE ";" " " HIP_OFFLOAD_COMPILE_OPTIONS_BUILD "${HIP_OFFLOAD_COMPILE_OPTIONS_BUILD_}")
string(REPLACE ";" " " HIP_OFFLOAD_LINK_OPTIONS_INSTALL "${HIP_OFFLOAD_LINK_OPTIONS_INSTALL_}")
string(REPLACE ";" " " HIP_OFFLOAD_LINK_OPTIONS_BUILD "${HIP_OFFLOAD_LINK_OPTIONS_BUILD_}")
string(REPLACE ";" " " HIP_RDC_SUPPLEMENT_LINK_FLAGS
  "${HIP_RDC_SUPPLEMENT_LINK_FLAGS_}")
message(STATUS "Generated HIP_OFFLOAD_COMPILE_OPTIONS: ${HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL}")
message(STATUS "Generated HIP_OFFLOAD_LINK_OPTIONS: ${HIP_OFFLOAD_LINK_OPTIONS_INSTALL}")

# Must separate these targets into deviceInteral to be used for building samples,etc.
# and device for external export. Have to do this because generator expressions don't seem to work here.
# TODO: Is there a better way to do this?
add_library(deviceInternal INTERFACE)
target_compile_options(deviceInternal INTERFACE
  -x hip ${HIP_OFFLOAD_COMPILE_OPTIONS_BUILD_})
target_link_libraries(deviceInternal INTERFACE CHIP)

add_library(device INTERFACE)
target_compile_options(device INTERFACE
  -x hip ${HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL_})
target_link_libraries(device INTERFACE CHIP)

# same as device on chipStar but provides compatibility with AMD
add_library(host INTERFACE)
target_compile_definitions(host INTERFACE ${HIP_ENABLE_SPIRV})
target_link_libraries(host INTERFACE CHIP)

# Interface libraries for compiling HIP sources conveniently in RDC
# (relocatable device code) mode.
add_library(deviceRDCInternal INTERFACE)
target_compile_options(deviceRDCInternal INTERFACE -fgpu-rdc)
target_link_options(deviceRDCInternal INTERFACE
  -fgpu-rdc ${HIP_RDC_SUPPLEMENT_LINK_FLAGS_} --hip-path=${CMAKE_BINARY_DIR})
target_link_libraries(deviceRDCInternal INTERFACE deviceInternal)

add_library(deviceRDC INTERFACE)
target_compile_options(deviceRDC INTERFACE -fgpu-rdc)
target_link_options(deviceRDC INTERFACE
  -fgpu-rdc ${HIP_RDC_SUPPLEMENT_LINK_FLAGS_}
  --hip-path=${CMAKE_INSTALL_PREFIX})
target_link_libraries(deviceRDC INTERFACE device)




# HIP OFFLOAD FLAGS
# =============================================================================

# =============================================================================
# PROJECT CONFIGURATION HEADER
set(CHIP_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(CHIP_BUILD_DIR ${CMAKE_BINARY_DIR})
set(CHIP_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(CHIP_CLANG_PATH ${CLANG_BIN_PATH})
set(CHIP_DEFAULT_WARP_SIZE ${DEFAULT_WARP_SIZE})
set(CHIP_DEBUG_BUILD 0)
if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CHIP_DEBUG_BUILD 1)
endif()
configure_file(chipStarConfig.hh.in include/chipStarConfig.hh @ONLY)

# PROJECT CONFIGURATION HEADER
# =============================================================================

# =============================================================================
# chipStar INSTALLATION AND PACKAGING
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Setup .hipVersion
# Install .hipVersion for hipcc. The file is used to detect a HIP
# runtime and clang extracts HIP version from it.
set(_versionInfo "# Auto-generated by cmake\n")
set(HIP_VERSION_MAJOR 6)
set(HIP_VERSION_MINOR 0)
set(HIP_VERSION_PATCH 0)
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_SHORT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(HIP_VERSION_GITHASH ${GIT_SHORT_HASH})
set(HIP_VERSION_PATCH_GITHASH "${HIP_VERSION_PATCH}.${HIP_VERSION_GITHASH}")
set(HIP_VERSION "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_VERSION_PATCH}-${HIP_VERSION_GITHASH}")

add_to_config(_versionInfo HIP_VERSION_MAJOR "${HIP_VERSION_MAJOR}")
add_to_config(_versionInfo HIP_VERSION_MINOR "${HIP_VERSION_MINOR}")
add_to_config(_versionInfo HIP_VERSION_PATCH "${HIP_VERSION_PATCH}")
add_to_config(_versionInfo HIP_VERSION_GITHASH "${HIP_VERSION_GITHASH}")

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
add_to_config(_hipInfo_install
  HIP_OFFLOAD_RDC_SUPPLEMENT_LINK_OPTIONS
  "${HIP_RDC_SUPPLEMENT_LINK_FLAGS} --hip-path=${CMAKE_INSTALL_PREFIX}")

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
add_to_config(_hipInfo_build
  HIP_OFFLOAD_RDC_SUPPLEMENT_LINK_OPTIONS
  "${HIP_RDC_SUPPLEMENT_LINK_FLAGS} --hip-path=${CMAKE_BINARY_DIR}")

INSTALL(TARGETS CHIP host device deviceRDC
  EXPORT hip-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
INSTALL(EXPORT hip-targets DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR} NAMESPACE hip::)
# export(EXPORT hip-targets FILE "${CHIP_BUILD_DIR}/hip-targets.cmake" NAMESPACE hip::)

add_library(hip::host ALIAS host)
add_library(hip::device ALIAS device)

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

# Generate hip_version.h
set(_pchStatus 0)
set(_versionInfoHeader
"// Auto-generated by cmake\n
#ifndef HIP_VERSION_H
#define HIP_VERSION_H\n
#define HIP_VERSION_MAJOR ${HIP_VERSION_MAJOR}
#define HIP_VERSION_MINOR ${HIP_VERSION_MINOR}
#define HIP_VERSION_PATCH ${HIP_VERSION_PATCH}
#define HIP_VERSION_GITHASH \"${HIP_VERSION_GITHASH}\"
#define HIP_VERSION_BUILD_ID ${HIP_VERSION_BUILD_ID}
#define HIP_VERSION_BUILD_NAME \"${HIP_VERSION_BUILD_NAME}\"
#define HIP_VERSION    (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)\n
#define __HIP_HAS_GET_PCH ${_pchStatus}\n
#endif\n
")
file(WRITE "${PROJECT_BINARY_DIR}/include/hip/hip_version.h" ${_versionInfoHeader})
install(FILES "${PROJECT_BINARY_DIR}/include/hip/hip_version.h"
  DESTINATION ${INCLUDE_INSTALL_DIR}/hip)

# Generate .hipInfo
file(WRITE "${PROJECT_BINARY_DIR}/share/.hipInfo_install" ${_hipInfo_install})
file(WRITE "${PROJECT_BINARY_DIR}/share/.hipInfo" ${_hipInfo_build})
install(FILES ${PROJECT_BINARY_DIR}/share/.hipInfo_install
  DESTINATION ${SHARE_INSTALL_DIR} RENAME .hipInfo)

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

install(FILES ${CMAKE_BINARY_DIR}/include/chipStarConfig.hh DESTINATION ${INCLUDE_INSTALL_DIR})
install(FILES ${PROJECT_BINARY_DIR}/${CHIP_LIB_NAME} DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HIP/include DESTINATION . USE_SOURCE_PERMISSIONS)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION . USE_SOURCE_PERMISSIONS)

install(EXPORT CHIPTargets
  FILE CHIPTargets.cmake
  NAMESPACE CHIP::
  DESTINATION cmake/CHIP)

# chipStar INSTALLATION AND PACKAGING
# =============================================================================
#
# enable the build_tests target
SET(HIP_PATH ${CMAKE_BINARY_DIR}) # pick up build sources, not install sources

if (CHIP_BUILD_HIPBLAS)
  add_subdirectory(H4I-MKLShim)
  add_subdirectory(H4I-HipBLAS)
  add_dependencies(hipblas CHIP MKLShim)
endif()

set(SAVED_WARN_DEPRECATED ${CMAKE_WARN_DEPRECATED})
set(CMAKE_WARN_DEPRECATED OFF)
if(CHIP_BUILD_TESTS)
  if(CHIP_USE_EXTERNAL_HIP_TESTS)
    add_subdirectory(hip-tests/catch catch)
  else()
    add_subdirectory(HIP/tests/catch catch)
  endif()
  add_subdirectory(tests)
endif()

if(CHIP_BUILD_SAMPLES)
  add_subdirectory(./samples samples)
  add_dependencies(samples CHIP devicelib_bc)
endif()

if(CHIP_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_GENERATE_MAN NO)
  set(DOXYGEN_OUTPUT_DIRECTORY doxygen)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
  doxygen_add_docs(gendocs README.md src llvm_passes
    COMMENT "generate documentation for source")
endif()

# Short Summary 
# print if Level Zero or OpenCL are enabbled
if(OpenCL_LIBRARY)
  message(STATUS "OpenCL is enabled: ${OpenCL_LIBRARY}")
endif()

if(LevelZero_LIBRARY)
  message(STATUS "Level Zero is enabled: ${LevelZero_LIBRARY}")
endif()
