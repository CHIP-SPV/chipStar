# Reproduces #1486: the SPIRV-Tools ExternalProject installs to /usr/local.
#
# cmake/SpirvToolsExternal.cmake hands the external project its compilers as
# CMAKE_ARGS. The external's configure step re-runs on every build (its
# UPDATE_COMMAND is never up to date), and each run passes those arguments to
# cmake again. If the compiler argument is a bare name, cmake resolves it
# through PATH at every run; when PATH differs between two runs (a module
# loaded in one shell but not the other, `ninja` then `ninja install` from
# different environments) the resolved path changes, cmake declares the
# compiler changed, deletes CMakeCache.txt and re-runs configure keeping only
# the compiler entries: CMAKE_INSTALL_PREFIX, CMAKE_INSTALL_LIBDIR and every
# other CMAKE_ARGS value revert to their defaults, and `ninja install` then
# tries to write /usr/local/lib64.
#
# The check configures a tiny project that includes the module, then runs the
# configure step cmake generated for the external project twice against a stub
# source tree: once with the ambient PATH and once with a directory of gcc/g++
# symlinks prepended. Whatever the external's CMakeCache.txt says afterwards
# is what `ninja install` would use, so CMAKE_INSTALL_PREFIX must still lie
# inside the build tree and CMAKE_INSTALL_LIBDIR must still be lib. No real
# SPIRV-Tools source is downloaded, built or installed.
#
# Invoked as:
#   cmake -DCHIPSTAR_SOURCE_DIR=<chipStar source> -DGENERATOR=<generator>
#         -P TestFix1486SpirvToolsInstallPrefix.cmake

if(NOT CHIPSTAR_SOURCE_DIR)
  message(FATAL_ERROR "CHIPSTAR_SOURCE_DIR not set")
endif()
if(NOT GENERATOR)
  set(GENERATOR "Ninja")
endif()

find_program(REAL_GCC gcc)
find_program(REAL_GXX g++)
if(NOT REAL_GCC OR NOT REAL_GXX)
  message(FATAL_ERROR "gcc and g++ are required (the external project builds with them)")
endif()

string(RANDOM LENGTH 12 ALPHABET "abcdefghijklmnopqrstuvwxyz0123456789" SUFFIX)
set(SCRATCH_DIR "$ENV{TMPDIR}")
if(NOT SCRATCH_DIR)
  set(SCRATCH_DIR "/tmp")
endif()
set(SCRATCH_DIR "${SCRATCH_DIR}/chipstar-spirv-tools-external-${SUFFIX}")
file(REMOVE_RECURSE "${SCRATCH_DIR}")
file(MAKE_DIRECTORY "${SCRATCH_DIR}")

# A consumer of the module: the same ExternalProject_Add chipStar runs, with
# its install prefix under this consumer's build tree.
set(CONSUMER_DIR "${SCRATCH_DIR}/consumer")
set(CONSUMER_BUILD "${CONSUMER_DIR}/build")
file(WRITE "${CONSUMER_DIR}/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.20)\n"
  "project(SpirvToolsExternalProbe NONE)\n"
  "include(${CHIPSTAR_SOURCE_DIR}/cmake/SpirvToolsExternal.cmake)\n")
execute_process(
  COMMAND ${CMAKE_COMMAND} -G "${GENERATOR}" -DCMAKE_BUILD_TYPE=Release
          -S "${CONSUMER_DIR}" -B "${CONSUMER_BUILD}"
  RESULT_VARIABLE RC OUTPUT_VARIABLE OUT ERROR_VARIABLE ERR)
if(NOT RC EQUAL 0)
  message(FATAL_ERROR "configuring the consumer project failed (exit ${RC})\n${OUT}\n${ERR}")
endif()

# The step cmake generated for the external's configure: the exact command
# `ninja` runs, including the CMAKE_ARGS as they were spelled. The install
# prefix that command asks for is what the external's cache must still hold
# afterwards; it is taken from the generated command rather than recomputed
# here so the comparison is exact (TMPDIR on macOS ends in a slash, which
# cmake collapses in CMAKE_BINARY_DIR and a string-prefix check would not).
set(EXT_PREFIX "${CONSUMER_BUILD}/SPIRV-Tools-External-prefix")
file(READ "${EXT_PREFIX}/tmp/SPIRV-Tools-External-cfgcmd.txt" CFGCMD)
if(NOT CFGCMD MATCHES "-DCMAKE_INSTALL_PREFIX=([^;']+)")
  message(FATAL_ERROR "no -DCMAKE_INSTALL_PREFIX in the generated configure command:\n${CFGCMD}")
endif()
set(EXPECTED_PREFIX "${CMAKE_MATCH_1}")
file(GLOB CONFIGURE_SCRIPT
  "${EXT_PREFIX}/src/SPIRV-Tools-External-stamp/SPIRV-Tools-External-configure-*.cmake")
list(LENGTH CONFIGURE_SCRIPT N_SCRIPTS)
if(NOT N_SCRIPTS EQUAL 1)
  message(FATAL_ERROR
    "expected one generated configure step under ${EXT_PREFIX}, found: '${CONFIGURE_SCRIPT}'")
endif()

# Stand in for the downloaded SPIRV-Tools checkout: the smallest project that
# still runs the compiler checks the CMAKE_C_COMPILER/CMAKE_CXX_COMPILER
# arguments trigger.
set(EXT_SOURCE "${EXT_PREFIX}/src/SPIRV-Tools-External")
set(EXT_BUILD "${EXT_PREFIX}/src/SPIRV-Tools-External-build")
file(WRITE "${EXT_SOURCE}/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.20)\n"
  "project(stub C CXX)\n")
file(MAKE_DIRECTORY "${EXT_BUILD}")

# gcc and g++ reachable through a different path than the ambient PATH gives.
set(SHIM_DIR "${SCRATCH_DIR}/shim")
file(MAKE_DIRECTORY "${SHIM_DIR}")
file(CREATE_LINK "${REAL_GCC}" "${SHIM_DIR}/gcc" SYMBOLIC)
file(CREATE_LINK "${REAL_GXX}" "${SHIM_DIR}/g++" SYMBOLIC)

function(run_configure_step LABEL)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env ${ARGN} ${CMAKE_COMMAND} -P "${CONFIGURE_SCRIPT}"
    WORKING_DIRECTORY "${EXT_BUILD}"
    RESULT_VARIABLE RC OUTPUT_VARIABLE OUT ERROR_VARIABLE ERR)
  if(NOT RC EQUAL 0)
    message(FATAL_ERROR "${LABEL}: the external's configure step failed (exit ${RC})\n${OUT}\n${ERR}")
  endif()
  message(STATUS "${LABEL}: configure step ran")
endfunction()

run_configure_step("ambient PATH")
run_configure_step("shim first on PATH" "PATH=${SHIM_DIR}:$ENV{PATH}")

file(STRINGS "${EXT_BUILD}/CMakeCache.txt" PREFIX_LINE REGEX "^CMAKE_INSTALL_PREFIX:")
file(STRINGS "${EXT_BUILD}/CMakeCache.txt" LIBDIR_LINE REGEX "^CMAKE_INSTALL_LIBDIR:")
string(REGEX REPLACE "^[^=]*=" "" INSTALL_PREFIX "${PREFIX_LINE}")
string(REGEX REPLACE "^[^=]*=" "" INSTALL_LIBDIR "${LIBDIR_LINE}")
message(STATUS "external CMAKE_INSTALL_PREFIX: ${INSTALL_PREFIX}")
message(STATUS "external CMAKE_INSTALL_LIBDIR: ${INSTALL_LIBDIR}")

set(FAILURES "")
if(NOT INSTALL_PREFIX STREQUAL EXPECTED_PREFIX)
  set(FAILURES "${FAILURES}  CMAKE_INSTALL_PREFIX is '${INSTALL_PREFIX}', the configure step asked for '${EXPECTED_PREFIX}'\n")
endif()
if(NOT INSTALL_LIBDIR STREQUAL "lib")
  set(FAILURES "${FAILURES}  CMAKE_INSTALL_LIBDIR is '${INSTALL_LIBDIR}', not lib\n")
endif()

if(NOT FAILURES STREQUAL "")
  file(READ "${EXT_PREFIX}/src/SPIRV-Tools-External-stamp/SPIRV-Tools-External-configure-err.log" CONFIGURE_ERR)
  message(FATAL_ERROR
    "the SPIRV-Tools external project lost its CMAKE_ARGS across two configure runs:\n"
    "${FAILURES}"
    "configure stderr of the last run:\n${CONFIGURE_ERR}"
    "scratch tree kept at ${SCRATCH_DIR}")
endif()

file(REMOVE_RECURSE "${SCRATCH_DIR}")
message(STATUS "TestFix1486SpirvToolsInstallPrefix passed")
