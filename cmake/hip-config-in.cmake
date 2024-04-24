cmake_minimum_required(VERSION 3.3)

@PACKAGE_INIT@
include(CheckCXXCompilerFlag)
include(CMakeFindDependencyMacro OPTIONAL RESULT_VARIABLE _CMakeFindDependencyMacro_FOUND)
if (NOT _CMakeFindDependencyMacro_FOUND)
  macro(find_dependency dep)
    if (NOT ${dep}_FOUND)
      set(cmake_fd_version)
      if (${ARGC} GREATER 1)
        set(cmake_fd_version ${ARGV1})
      endif()
      set(cmake_fd_exact_arg)
      if(${CMAKE_FIND_PACKAGE_NAME}_FIND_VERSION_EXACT)
        set(cmake_fd_exact_arg EXACT)
      endif()
      set(cmake_fd_quiet_arg)
      if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
        set(cmake_fd_quiet_arg QUIET)
      endif()
      set(cmake_fd_required_arg)
      if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
        set(cmake_fd_required_arg REQUIRED)
      endif()
      find_package(${dep} ${cmake_fd_version}
          ${cmake_fd_exact_arg}
          ${cmake_fd_quiet_arg}
          ${cmake_fd_required_arg}
      )
      string(TOUPPER ${dep} cmake_dep_upper)
      if (NOT ${dep}_FOUND AND NOT ${cmake_dep_upper}_FOUND)
        set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE "${CMAKE_FIND_PACKAGE_NAME} could not be found because dependency ${dep} could not be found.")
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND False)
        return()
      endif()
      set(cmake_fd_version)
      set(cmake_fd_required_arg)
      set(cmake_fd_quiet_arg)
      set(cmake_fd_exact_arg)
    endif()
  endmacro()
endif()

set(_HIP_SHELL "SHELL:")
if(CMAKE_VERSION VERSION_LESS 3.12)
  set(_HIP_SHELL "")
endif()

function(hip_add_interface_compile_flags TARGET)
  set_property(TARGET ${TARGET} APPEND PROPERTY
    INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${_HIP_SHELL}${ARGN}>"
  )
endfunction()

function(hip_add_interface_link_flags TARGET)
  if(CMAKE_VERSION VERSION_LESS 3.20)
    set_property(TARGET ${TARGET} APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES "${ARGN}"
    )
  else()
    set_property(TARGET ${TARGET} APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES "$<$<LINK_LANGUAGE:CXX>:${ARGN}>"
    )
  endif()
endfunction()

#Number of parallel jobs by default is 1
if(NOT DEFINED HIP_CLANG_NUM_PARALLEL_JOBS)
  set(HIP_CLANG_NUM_PARALLEL_JOBS 1)
endif()

if(WIN32)
  message(FATAL_ERROR "Windows not yet supported for chipStar")
endif()

set(HIP_PATH "@HIP_PATH@" CACHE PATH "Path to the chipStar installation")
set(HIP_COMPILER "@HIP_COMPILER@" CACHE STRING "C++ compiler")
set(HIP_RUNTIME "@HIP_RUNTIME@" CACHE STRING "" FORCE)
set(HIP_PLATFORM "@HIP_PLATFORM@" CACHE STRING "" FORCE)
set(HIP_ARCH "@HIP_ARCH@" CACHE STRING "" FORCE)
set(HIP_OFFLOAD_COMPILE_OPTIONS "@HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL@"
  CACHE STRING "clang compiler variables for offload compilation")
set(HIP_OFFLOAD_LINK_OPTIONS "@HIP_OFFLOAD_LINK_OPTIONS_INSTALL@"
  CACHE STRING "HIP application linker options")
message(STATUS "hip-config.cmake chipStar:")
message(STATUS "HIP_PATH: ${HIP_PATH}")
message(STATUS "HIP_COMPILER: ${HIP_COMPILER}")
message(STATUS "HIP_RUNTIME: ${HIP_RUNTIME}")
message(STATUS "HIP_PLATFORM: ${HIP_PLATFORM}")
message(STATUS "HIP_ARCH: ${HIP_ARCH}")
message(STATUS "HIP_OFFLOAD_COMPILE_OPTIONS: ${HIP_OFFLOAD_COMPILE_OPTIONS_INSTALL}")
message(STATUS "HIP_OFFLOAD_LINK_OPTIONS: ${HIP_OFFLOAD_LINK_OPTIONS_INSTALL}")

set(hip_INCLUDE_DIR $<INSTALL_INTERFACE:@CHIP_INSTALL_DIR@>$<BUILD_INTERFACE:${CHIP_BUILD_DIR}/include> )
set(hip_INCLUDE_DIRS $<INSTALL_INTERFACE:@CHIP_INSTALL_DIR@>$<BUILD_INTERFACE:${CHIP_BUILD_DIR}/include> )
set(hip_LIB_INSTALL_DIR $<INSTALL_INTERFACE:@CHIP_INSTALL_DIR@>$<BUILD_INTERFACE:${CHIP_BUILD_DIR}/lib> )
set(hip_BIN_INSTALL_DIR $<INSTALL_INTERFACE:@CHIP_INSTALL_DIR@>$<BUILD_INTERFACE:${CHIP_BUILD_DIR}/bin> )

if(WIN32)
  message(FATAL_ERROR "Windows not yet supported for chipStar")
  #set_and_check(hip_HIPCC_EXECUTABLE "${hip_BIN_INSTALL_DIR}/hipcc.bat")
  #set_and_check(hip_HIPCONFIG_EXECUTABLE "${hip_BIN_INSTALL_DIR}/hipconfig.bat")
else()
  set(hip_HIPCC_EXECUTABLE $<INSTALL_INTERFACE:@CHIP_INSTALL_DIR@>$<BUILD_INTERFACE:${CHIP_BUILD_DIR}/bin/hipcc> )
  set(hip_HIPCONFIG_EXECUTABLE $<INSTALL_INTERFACE:@CHIP_INSTALL_DIR@>$<BUILD_INTERFACE:${CHIP_BUILD_DIR}/bin/hipconfig> )
endif()

if(NOT HIP_CXX_COMPILER)
  set(HIP_CXX_COMPILER ${CMAKE_CXX_COMPILER})
endif()

# # Get the clang git version
# if(HIP_CXX_COMPILER MATCHES ".*hipcc" OR HIP_CXX_COMPILER MATCHES ".*clang\\+\\+")
#   execute_process(COMMAND ${HIP_CXX_COMPILER} --version
#                   OUTPUT_STRIP_TRAILING_WHITESPACE
#                   OUTPUT_VARIABLE HIP_CXX_COMPILER_VERSION_OUTPUT)
#   # Capture the repo, branch and patch level details of the HIP CXX Compiler.
#   # Ex. clang version 13.0.0 (https://github.com/ROCm-Developer-Tools/HIP main 12345 COMMIT_HASH)
#   # HIP_CLANG_REPO: https://github.com/ROCm-Developer-Tools/HIP
#   # HIP_CLANG_BRANCH: main
#   # HIP_CLANG_PATCH_LEVEL: 12345
#   if(${HIP_CXX_COMPILER_VERSION_OUTPUT} MATCHES "clang version [0-9]+\\.[0-9]+\\.[0-9]+ \\(([^ \n]*) ([^ \n]*) ([^ \n]*)")
#     set(HIP_CLANG_REPO ${CMAKE_MATCH_1})
#     set(HIP_CLANG_BRANCH ${CMAKE_MATCH_2})
#     set(HIP_CLANG_PATCH_LEVEL ${CMAKE_MATCH_3})
#   endif()
# endif()

# if(HIP_CXX_COMPILER MATCHES ".*hipcc")
#   if(HIP_CXX_COMPILER_VERSION_OUTPUT MATCHES "InstalledDir:[ \t]*([^\n]*)")
#     get_filename_component(HIP_CLANG_ROOT "${CMAKE_MATCH_1}" DIRECTORY)
#   endif()
# elseif (HIP_CXX_COMPILER MATCHES ".*clang\\+\\+")
#   get_filename_component(_HIP_CLANG_REAL_PATH "${HIP_CXX_COMPILER}" REALPATH)
#   get_filename_component(_HIP_CLANG_BIN_PATH "${_HIP_CLANG_REAL_PATH}" DIRECTORY)
#   get_filename_component(HIP_CLANG_ROOT "${_HIP_CLANG_BIN_PATH}" DIRECTORY)
# endif()
# file(GLOB HIP_CLANG_INCLUDE_SEARCH_PATHS ${HIP_CLANG_ROOT}/lib/clang/*/include)
# find_path(HIP_CLANG_INCLUDE_PATH stddef.h
#     HINTS
#         ${HIP_CLANG_INCLUDE_SEARCH_PATHS}
#     NO_DEFAULT_PATH)

include( "${CMAKE_CURRENT_LIST_DIR}/hip-targets.cmake" )

#Using find_dependency to locate the dependency for the packages
#This makes the cmake generated file xxxx-targets to supply the linker libraries
# without worrying other transitive dependencies
if(NOT WIN32)
  find_dependency(Threads)
endif()

set( hip_LIBRARIES hip::host hip::device)
set( hip_LIBRARY ${hip_LIBRARIES})

set(HIP_INCLUDE_DIR ${hip_INCLUDE_DIR})
set(HIP_INCLUDE_DIRS ${hip_INCLUDE_DIRS})
set(HIP_LIB_INSTALL_DIR ${hip_LIB_INSTALL_DIR})
set(HIP_BIN_INSTALL_DIR ${hip_BIN_INSTALL_DIR})
set(HIP_LIBRARIES ${hip_LIBRARIES})
set(HIP_LIBRARY ${hip_LIBRARY})
set(HIP_HIPCC_EXECUTABLE ${hip_HIPCC_EXECUTABLE})
set(HIP_HIPCONFIG_EXECUTABLE ${hip_HIPCONFIG_EXECUTABLE})
