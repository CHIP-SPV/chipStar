# Driver for TestFix1604ClFlushErrorMap.
#
# Runs the test binary once per clFlush ordinal with the interposer preloaded,
# because how many times chipStar flushes before it reaches a checked call
# depends on the backend and the driver. Each run is one of:
#
#   the interposer never fired      -> that ordinal exercised nothing
#   the process died on a signal    -> the #1604 abort, fail
#   the injected status came back
#   as a HIP error                  -> what the fix is for
#
# A sweep in which no ordinal ever fired means this build never called clFlush,
# which is the case on a backend other than OpenCL, and the test skips.

if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  message("HIP_SKIP_THIS_TEST: LD_PRELOAD interposition is Linux only")
  return()
endif()

set(FIRED FALSE)
set(REPORTED FALSE)

foreach(ORDINAL RANGE 1 8)
  set(LOG_FILE "${TEST_EXECUTABLE}_output_${ORDINAL}.txt")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_PRELOAD=${INTERPOSER}" "CLFLUSH_FAIL_AT=${ORDINAL}"
      ${TEST_EXECUTABLE}
    OUTPUT_FILE "${LOG_FILE}"
    ERROR_FILE "${LOG_FILE}"
    TIMEOUT ${TEST_TIMEOUT}
    RESULT_VARIABLE RESULT)

  file(READ "${LOG_FILE}" TEST_OUTPUT)

  if(NOT TEST_OUTPUT MATCHES "clflush-interposer: failing call")
    continue()
  endif()
  set(FIRED TRUE)
  message("--- clFlush ordinal ${ORDINAL}, result ${RESULT}")
  message("${TEST_OUTPUT}")

  # Before the fix the unmapped status reaches std::abort(), which leaves the
  # marker the table prints and a nonzero, non-exit-code result.
  if(TEST_OUTPUT MATCHES "Unmapped API or API Error Code")
    message(FATAL_ERROR
      "FAIL: a CL_OUT_OF_RESOURCES from clFlush was not mapped, so "
      "CHIPERR_CHECK_LOG_AND_THROW_TABLE aborted the process instead of "
      "raising a HIP error. See issue #1604.")
  endif()

  if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR
      "FAIL: the process did not survive a CL_OUT_OF_RESOURCES from clFlush "
      "(result: ${RESULT}). It must come back as a HIP error. See issue #1604.")
  endif()

  if(NOT TEST_OUTPUT MATCHES "REPORT: survived")
    message(FATAL_ERROR
      "FAIL: the process exited before reporting that it survived the "
      "injected clFlush failure.")
  endif()

  # The failure has to be visible to the application, not swallowed.
  if(TEST_OUTPUT MATCHES "REPORT: hipEventRecord hipErrorOutOfMemory" OR
     TEST_OUTPUT MATCHES "REPORT: hipStreamQuery hipErrorOutOfMemory")
    set(REPORTED TRUE)
  endif()
endforeach()

if(NOT FIRED)
  message("HIP_SKIP_THIS_TEST: no clFlush call was reached, so nothing was "
          "injected; this build does not use the OpenCL backend")
  return()
endif()

if(NOT REPORTED)
  message(FATAL_ERROR
    "FAIL: the injected CL_OUT_OF_RESOURCES never reached the application as "
    "hipErrorOutOfMemory, so the failure was swallowed.")
endif()

message("PASS")
