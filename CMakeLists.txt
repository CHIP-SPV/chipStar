# temporary
set(CMAKE_WARN_DEPRECATED OFF)
add_compile_options(-Wno-format-extra-args)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-duplicate-decl-specifier \
-Wno-tautological-constant-compare  -Wno-c++20-extensions -Wno-unused-result \
-Wno-delete-abstract-non-virtual-dtor -Wno-deprecated-declarations")
# end temporary

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(CHIP-SPV
  VERSION 0.1.0
  DESCRIPTION "HIP implementation for runtimes that support SPIR-V"
  LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(chip_spv_cmake_macros)
include(FindLLVM)



###############################################################################
# CHIP-SPV CMAKE VARIABLBLES/OPTIONS
set(CHIP_SRC
  src/spirv.cc
  src/spirv_hiprtc.cc
  src/CHIPDriver.cc
  src/CHIPBackend.cc
  src/CHIPBindings.cc
  src/backend/Level0/CHIPBackendLevel0.cc
  src/backend/OpenCL/CHIPBackendOpenCL.cc
  src/backend/OpenCL/SVMemoryRegion.cc
  src/logging.cc
)

set(CHIP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "CHIP-SPV source directory")
set(CLANG_VERSION_LESS_14 OFF CACHE INTERNAL "Clang version < 14.0")

set(CHIP_SPV_LINK_FLAGS "")
set(CHIP_SPV_COMPILE_FLAGS "")
list(APPEND CHIP_SPV_DEFINITIONS "")
set(HIP_PLATFORM spirv)
set(HIP_COMPILER clang)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# INSTALL DIRECTORY VARIABLES
set(BIN_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/bin)
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include)
set(SHARE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/share)
set(CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/hip)
set(CONFIG_LANG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/hip-lang)

option(ERROR_IF_NOT_IMPLEMENTED "Some HIP functionality is not yet implemented in CHIP-SPV. If such a function is called, throw hipErrorNotSupported" ON)
option(LAZY_JIT "JIT kernels at the time of the first kernel invocation instead of at the start of the program execution" OFF)
option(CACHE_KERNELS "Save the compiled kernel to disk to speed up program startup" OFF)
option(VERBOSE "Verbose compilation" OFF)
option(BUILD_SHARED_LIBS "Build CHIP-SPV as a shared library" ON)
option(ENFORCE_QUEUE_SYNCHRONIZATION "Enforce the correct HIP stream semantics of synchronizing queues with the default queue" ON)
option(PER_THREAD_DEFAULT_STREAM "Each thread has a thread local default stream. Default: OFF - use legacy stream" OFF)
option(LEVEL_ZERO_IMMEDIATE_QUEUES "Enable/disable the use of immediate command queues" OFF)

set(CMAKE_BUILD_TYPE "DEBUG" CACHE STRING "The build type to use DEBUG, RELEASE, RELWITHDEBINFO, MINSIZEREL")
set(CHIP_SPV_INSTALL_DIR "${CMAKE_SOURCE_DIR}/install" CACHE PATH "The installation directory for CHIP-SPV")

# FOR EXTERNAL USE
set(HIP_ENABLE_SPIRV "-D__HIP_PLATFORM_SPIRV__=" CACHE INTERNAL "Compiler difinition which enables SPIR-V defines")
set(HIP_OFFLOAD_COMPILE_OPTIONS "" CACHE STRING "Semicolon separated list of compiler flags needed for generating offload code")
set(HIP_PATH HIP) #TODO WHAT DOES IT DO
# CHIP-SPV CMAKE VARIABLBLES/OPTIONS
###############################################################################

###############################################################################
# CHIP-SPV CMAKE OPTIONS PARSING
if((CMAKE_C_COMPILER_ID MATCHES "[Cc]lang") OR(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
    message(FATAL_ERROR "this project requires clang >= 8.0")
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0.0)
    message(WARNING "Deprecated clang version '${CMAKE_CXX_COMPILER_VERSION}'. \
            Support for Clang < 14.0 will be discontinued in the future.")
    set(CLANG_VERSION_LESS_14 ON)
  endif()
else()
  message(FATAL_ERROR "this project must be compiled with clang. CMAKE_C_COMPILER_ID = ${CMAKE_C_COMPILER_ID}")
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

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CHIP_SPV_INSTALL_DIR}" CACHE PATH "Install path prefix" FORCE)
endif()
message(STATUS "CHIP-SPV will be installed to: ${CMAKE_INSTALL_PREFIX}")
# CHIP-SPV CMAKE OPTIONS PARSING
###############################################################################

###############################################################################
# CHIP-SPV CMAKE DEPENDENCIES
if(NOT DEFINED OpenCL_LIBRARY)
  message(STATUS "OpenCL_LIBRARY was not set. Searching for libOpenCL.so in LD_LIBRARY_PATH")
  find_library_dynamic(OpenCL)
endif()
find_library_dynamic(ze_loader)
message(STATUS "\n\nCHIP-SPV_LIBRARIES=${CHIP-SPV_LIBRARIES}\n\n")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
set(PTHREAD_LIBRARY Threads::Threads)
# CHIP-SPV CMAKE DEPENDENCIES
###############################################################################

###############################################################################
# CHIP-SPV COMPILATION SETUP
set(HIP_OFFLOAD_COMPILE_FLAGS -x hip --target=x86_64-linux-gnu)
if(NOT CLANG_VERSION_LESS_14)
  set(HIP_OFFLOAD_COMPILE_FLAGS ${HIP_OFFLOAD_COMPILE_FLAGS}
      --offload=spirv64
      # By default RocmInstallationDetector::AddHIPIncludeArgs in the
      # HIP-Clang inserts include wrapper headers which cause compile
      # errors when compiling HIP sources for SPIR-V target.  This flag
      # excludes the wrappers.
      "-nohipwrapperinc")
endif()

if(CLANG_VERSION_LESS_14 )
  set(HIP_LINK_DEVICELIB_BUILD --hip-device-lib-path=${CMAKE_BINARY_DIR}/share --hip-device-lib=devicelib.bc --hip-llvm-pass-path=${CMAKE_BINARY_DIR}/llvm_passes)
  set(HIP_LINK_DEVICELIB_INSTALL --hip-device-lib-path=${SHARE_INSTALL_DIR} --hip-device-lib=devicelib.bc --hip-llvm-pass-path=${LIB_INSTALL_DIR}/llvm)
else()
  set(HIP_LINK_DEVICELIB_BUILD --hip-path=${CMAKE_BINARY_DIR})
  set(HIP_LINK_DEVICELIB_INSTALL --hip-path=${CMAKE_INSTALL_PREFIX})
endif()

# for exporting
set(HIP_OFFLOAD_COMPILE_OPTIONS_
  ${HIP_ENABLE_SPIRV} 
  ${HIP_OFFLOAD_COMPILE_FLAGS} 
  ${HIP_LINK_DEVICELIB_INSTALL})
# Workaround istead of trying to generate the CMake generator expression
string(REPLACE ";" " " HIP_OFFLOAD_COMPILE_OPTIONS "${HIP_OFFLOAD_COMPILE_OPTIONS_}")
message(STATUS "Generated HIP_OFFLOAD_COMPILE_OPTIONS: ${HIP_OFFLOAD_COMPILE_OPTIONS}")

add_library(device INTERFACE)
target_compile_options(device INTERFACE 
  ${HIP_ENABLE_SPIRV} 
  ${HIP_OFFLOAD_COMPILE_FLAGS} 
  ${HIP_LINK_DEVICELIB_BUILD})


# target_link_libraries(device INTERFACE ${HIP_LINK_DEVICELIB_BUILD})
# target_include_directories(device INTERFACE ${INCLUDE_INSTALL_DIR})
# target_link_libraries(device INTERFACE CHIP)

# same as device on CHIP-SPV but provides compatibility with AMD
add_library(host INTERFACE)
# target_link_libraries(host INTERFACE device)


# For building on top of HIP-COMMON
target_compile_options(CHIP PUBLIC ${HIP_ENABLE_SPIRV} ${CHIP_SPV_COMPILE_FLAGS})
target_compile_definitions(CHIP PUBLIC ${CHIP_SPV_DEFINITIONS})

set(CHIP_SPV_LINK_FLAGS ${OpenCL_LIBRARY} ${CHIP-SPV_LIBRARIES})

# TODO: Check if pthread flag is needed for hipcc
target_link_libraries(CHIP PUBLIC ${CHIP_SPV_LINK_FLAGS} ${PTHREAD_LIBRARY})

# Previously these were set to SYSTEM which resulted in CMake picking up
# OpenCL headers from the system where the version might differ resulting in errors.
target_include_directories(CHIP
  PUBLIC
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/HIP/include;${CMAKE_SOURCE_DIR}/include>"
  PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/include"
)

# CHIP-SPV COMPILATION SETUP
###############################################################################

###############################################################################
# CHIP-SPV INSTALLATION AND PACKAGING
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_subdirectory(llvm_passes)
add_subdirectory(bitcode)

enable_testing()
add_subdirectory(HIP/tests/catch catch)

add_subdirectory(./samples samples)

# Make CHIP depend on devicelib_bc and LLVMHipPasses for
# convenience. The CHIP module itself does not depend on these but
# HIP program compilation does.
add_dependencies(CHIP devicelib_bc LLVMHipPasses)
add_dependencies(samples CHIP)

# ####################
# Setup .hipVersion #
# ####################
# Install .hipVersion for hipcc. The file is used to detect a HIP
# runtime and clang extracts HIP version from it.
set(_versionInfo "# Auto-generated by cmake\n")
set(HIP_VERSION_MAJOR 5)
set(HIP_VERSION_MINOR 1)
set(HIP_VERSION_PATCH 0)
add_to_config(_versionInfo HIP_VERSION_MAJOR)
add_to_config(_versionInfo HIP_VERSION_MINOR)
add_to_config(_versionInfo HIP_VERSION_PATCH)

# Generate and install .hipVersion file.
file(WRITE ${PROJECT_BINARY_DIR}/bin/.hipVersion ${_versionInfo})
install(FILES ${PROJECT_BINARY_DIR}/bin/.hipVersion DESTINATION bin)

# ####################
# Setup .hipInfo #
# ####################
set(HIP_COMPILER clang)
set(HIP_RUNTIME spirv)
set(HIP_PLATFORM spirv)

string(TIMESTAMP _timestamp UTC)
set(_buildInfo "# Auto-generated by cmake on ${_timestamp} UTC\n")
message(STATUS "HIP Runtime: " ${HIP_RUNTIME})
message(STATUS "HIP Compiler: " ${HIP_COMPILER})
add_to_config(_buildInfo HIP_RUNTIME)
add_to_config(_buildInfo HIP_COMPILER)
add_to_config(_buildInfo HIP_OFFLOAD_COMPILE_OPTIONS)



INSTALL(TARGETS CHIP host device
  EXPORT hip-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
INSTALL(EXPORT hip-targets DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR} NAMESPACE hip::)

# ############################
# hip-config
# ############################
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake/hip-config-in.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/hip-config.cmake
  INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
  PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR
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
install ( EXPORT hip-targets  FILE hip-targets.cmake NAMESPACE hip:: DESTINATION lib/cmake/hip  COMPONENT binary)
# Convert absolute path to library into -L<...> and -l<...> components
set(CHIP_SPV_LINK_FLAGS_STR "")

foreach(_library ${CHIP_SPV_LINK_FLAGS})
  get_filename_component(libpath ${_library} DIRECTORY)
  get_filename_component(libname ${_library} NAME)
  string(REGEX MATCH "lib(.*)\.(so*|a)" libname_stripped ${libname})
  set(libname_stripped ${CMAKE_MATCH_1})
  set(CHIP_SPV_LINK_FLAGS_STR "${CHIP_SPV_LINK_FLAGS_STR} -L${libpath} -l${libname_stripped}")
endforeach()

# Add link directory fir libCHIP.so
set(CHIP_SPV_LINK_FLAGS_STR "${CHIP_SPV_LINK_FLAGS_STR}" " -L${LIB_INSTALL_DIR} -lCHIP")

# Add an rpath for libCHIP.so so that you don't need to add <CHIP_INSTALL_DIR> to LD_LIBRARY_PATH
set(CHIP_SPV_LINK_FLAGS_STR "${CHIP_SPV_LINK_FLAGS_STR}" " -Wl,-rpath,${LIB_INSTALL_DIR}")
set(HIP_LINK_OPTIONS ${CHIP_SPV_LINK_FLAGS_STR})
add_to_config(_buildInfo HIP_LINK_OPTIONS)

# Generate .hipInfo
file(WRITE "${PROJECT_BINARY_DIR}/lib/.hipInfo" ${_buildInfo})
install(FILES ${PROJECT_BINARY_DIR}/lib/.hipInfo DESTINATION lib)

# ####################
# Copy hipconfig, hipvars, etc to bin
# ####################
file(COPY "${CMAKE_SOURCE_DIR}/HIP/bin" DESTINATION "${PROJECT_BINARY_DIR}")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HIP/bin DESTINATION . USE_SOURCE_PERMISSIONS)

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
###############################################################################