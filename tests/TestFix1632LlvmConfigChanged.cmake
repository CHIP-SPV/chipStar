# Reproduces #1632: reconfiguring an existing build directory with a different
# LLVM_CONFIG_BIN kept the first LLVM's tools, CMake packages and compilers.
# find_program results (CMAKE_CXX_COMPILER_PATH, LLVM_LINK,
# CLANG_OFFLOAD_BUNDLER) and Clang_DIR are cached and never re-searched, and
# the compiler cannot be moved at all: CMake records it in
# CMakeFiles/<ver>/CMakeCXXCompiler.cmake on the first configure and does not
# re-run detection when those cached entries are cleared. So the second
# configure must be rejected rather than produce a build that mixes two LLVMs.
#
# Invoked as:
#   cmake -DCHIPSTAR_SOURCE_DIR=<chipStar source> -DLLVM_CONFIG_BIN=<llvm-config>
#         -DGENERATOR=<generator> -P TestFix1632LlvmConfigChanged.cmake

if(NOT CHIPSTAR_SOURCE_DIR)
  message(FATAL_ERROR "CHIPSTAR_SOURCE_DIR not set")
endif()
if(NOT GENERATOR)
  set(GENERATOR "Ninja")
endif()

# The cross-compiled trees run their tests on a different machine, where the
# builder's llvm-config does not exist.
if(NOT EXISTS "${LLVM_CONFIG_BIN}")
  message(STATUS "HIP_SKIP_THIS_TEST: no llvm-config at '${LLVM_CONFIG_BIN}'")
  return()
endif()

string(RANDOM LENGTH 12 ALPHABET "abcdefghijklmnopqrstuvwxyz0123456789" SUFFIX)
set(SCRATCH_DIR "$ENV{TMPDIR}")
if(NOT SCRATCH_DIR)
  set(SCRATCH_DIR "/tmp")
endif()
set(SCRATCH_DIR "${SCRATCH_DIR}/chipstar-llvm-config-changed-${SUFFIX}")
file(REMOVE_RECURSE "${SCRATCH_DIR}")
file(MAKE_DIRECTORY "${SCRATCH_DIR}")

# A consumer of the module: the values FindLLVM.cmake derives from llvm-config,
# in a build directory of its own.
set(CONSUMER_DIR "${SCRATCH_DIR}/consumer")
set(CONSUMER_BUILD "${SCRATCH_DIR}/build")
file(WRITE "${CONSUMER_DIR}/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.20)\n"
  "project(FindLLVMProbe NONE)\n"
  "include(${CHIPSTAR_SOURCE_DIR}/cmake/FindLLVM.cmake)\n")

# A second path to the same LLVM, so the test needs one install.
set(WRAPPER "${SCRATCH_DIR}/other-llvm/bin/llvm-config")
file(WRITE "${WRAPPER}" "#!/bin/sh\nexec \"${LLVM_CONFIG_BIN}\" \"$@\"\n")
file(CHMOD "${WRAPPER}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

execute_process(
  COMMAND ${CMAKE_COMMAND} -G "${GENERATOR}" -DCMAKE_BUILD_TYPE=Release
          -DLLVM_CONFIG_BIN=${LLVM_CONFIG_BIN}
          -S "${CONSUMER_DIR}" -B "${CONSUMER_BUILD}"
  RESULT_VARIABLE RC OUTPUT_VARIABLE OUT ERROR_VARIABLE ERR)
if(NOT RC EQUAL 0)
  message(FATAL_ERROR "the first configure failed (exit ${RC})\n${OUT}\n${ERR}")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -DLLVM_CONFIG_BIN=${WRAPPER}
          -S "${CONSUMER_DIR}" -B "${CONSUMER_BUILD}"
  RESULT_VARIABLE RC OUTPUT_VARIABLE OUT ERROR_VARIABLE ERR)

if(RC EQUAL 0)
  file(STRINGS "${CONSUMER_BUILD}/CMakeCache.txt" STALE
    REGEX "^(CMAKE_CXX_COMPILER_PATH|LLVM_LINK|Clang_DIR|LLVM_DIR):")
  string(REPLACE ";" "\n  " STALE "${STALE}")
  message(FATAL_ERROR
    "reconfiguring with LLVM_CONFIG_BIN=${WRAPPER} was accepted; the build "
    "directory keeps the values derived from ${LLVM_CONFIG_BIN}:\n  ${STALE}\n"
    "scratch tree kept at ${SCRATCH_DIR}")
endif()

if(NOT "${ERR}" MATCHES "LLVM_CONFIG_BIN")
  message(FATAL_ERROR
    "the second configure failed without naming LLVM_CONFIG_BIN (exit ${RC})\n${OUT}\n${ERR}")
endif()

file(REMOVE_RECURSE "${SCRATCH_DIR}")
message(STATUS "TestFix1632LlvmConfigChanged passed")
