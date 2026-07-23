# Generic run wrapper: execute a test with a fresh, uniquely named module cache
# directory (CHIP_MODULE_CACHE_DIR) and info-level logging (CHIP_LOGLEVEL=info).
#
# Some tests must observe cache behavior from a known-empty starting state. But
# libCHIP reads CHIP_MODULE_CACHE_DIR and CHIP_LOGLEVEL at static-init time,
# before main(), so they must be set before the test process starts -- which a
# run-time wrapper allows and a fixed configure-time ENVIRONMENT value cannot
# randomize per run. A `mktemp -d` name has never been used before, so it cannot
# contain a prior run's cache entries regardless of how the runtime canonicalizes
# the path (case, symlinks, ...); the first compile is thus reliably a cold miss.
# The directory is removed afterwards.
#
# Invoked as:
#   cmake -DTEST_EXECUTABLE=<path> -P run_test_with_fresh_cache.cmake

if(NOT TEST_EXECUTABLE)
  message(FATAL_ERROR "TEST_EXECUTABLE not set")
endif()
get_filename_component(TEST_NAME "${TEST_EXECUTABLE}" NAME)

execute_process(
  COMMAND mktemp -d
  OUTPUT_VARIABLE CACHE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE MKTEMP_RC)
if(NOT MKTEMP_RC EQUAL 0)
  message(FATAL_ERROR "mktemp -d failed (${MKTEMP_RC})")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E env
    "CHIP_MODULE_CACHE_DIR=${CACHE_DIR}"
    "CHIP_LOGLEVEL=info"
    "${TEST_EXECUTABLE}"
  RESULT_VARIABLE TEST_RC)

file(REMOVE_RECURSE "${CACHE_DIR}")

if(NOT TEST_RC EQUAL 0)
  message(FATAL_ERROR "${TEST_NAME} failed (exit ${TEST_RC})")
endif()
