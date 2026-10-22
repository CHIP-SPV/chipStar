# Reproduces #2082: the SPIRV-Tools ExternalProject must check out a pinned commit, also in a reused build dir.
#
# Invoked as:
#   cmake -DCHIPSTAR_SOURCE_DIR=<chipStar source> -DGENERATOR=<generator>
#         -P TestFix2082SpirvToolsPinned.cmake

if(NOT CHIPSTAR_SOURCE_DIR)
  message(FATAL_ERROR "CHIPSTAR_SOURCE_DIR not set")
endif()
if(NOT GENERATOR)
  set(GENERATOR "Ninja")
endif()

string(RANDOM LENGTH 12 ALPHABET "abcdefghijklmnopqrstuvwxyz0123456789" SUFFIX)
set(SCRATCH_DIR "$ENV{TMPDIR}")
if(NOT SCRATCH_DIR)
  set(SCRATCH_DIR "/tmp")
endif()
set(SCRATCH_DIR "${SCRATCH_DIR}/chipstar-spirv-tools-pinned-${SUFFIX}")
file(REMOVE_RECURSE "${SCRATCH_DIR}")
file(MAKE_DIRECTORY "${SCRATCH_DIR}")

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

set(EXT_TMP "${CONSUMER_BUILD}/SPIRV-Tools-External-prefix/tmp")
set(FAILURES "")

# The default git update step is what checks out GIT_TAG in a reused tree.
if(NOT EXISTS "${EXT_TMP}/SPIRV-Tools-External-gitupdate.cmake")
  set(FAILURES "${FAILURES}  no git update step: a reused build dir never moves to a new GIT_TAG\n")
endif()

set(CLONE_SCRIPT "${EXT_TMP}/SPIRV-Tools-External-gitclone.cmake")
if(NOT EXISTS "${CLONE_SCRIPT}")
  message(FATAL_ERROR "no generated clone script at ${CLONE_SCRIPT}")
endif()
file(READ "${CLONE_SCRIPT}" CLONE)
if(CLONE MATCHES "checkout \"([^\"]+)\"")
  set(REF "${CMAKE_MATCH_1}")
  string(LENGTH "${REF}" REF_LEN)
  if(NOT REF MATCHES "^[0-9a-f]+$" OR NOT REF_LEN EQUAL 40)
    set(FAILURES "${FAILURES}  the clone checks out '${REF}', not a 40-char commit hash\n")
  endif()
else()
  set(FAILURES "${FAILURES}  no checkout line in the clone script\n")
endif()
if(CLONE MATCHES "--depth")
  set(FAILURES "${FAILURES}  the clone is shallow (--depth), so a pinned commit that is not a branch tip cannot be checked out\n")
endif()

if(NOT FAILURES STREQUAL "")
  message(FATAL_ERROR
    "the SPIRV-Tools external project is not pinned to a commit:\n${FAILURES}"
    "scratch tree kept at ${SCRATCH_DIR}")
endif()

file(REMOVE_RECURSE "${SCRATCH_DIR}")
message(STATUS "TestFix2082SpirvToolsPinned passed")
