# Driver for TestFix1543SwitchModeEventLeak.
#
# Runs the test binary with the interposer preloaded and maps the outcomes
# onto a verdict:
#
#   the process failed                        -> fail
#   no barrier event was ever enqueued        -> nothing was exercised, skip
#   every barrier event was released          -> pass
#   some barrier event still has references   -> the #1543 leak, fail

if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  message("HIP_SKIP_THIS_TEST: LD_PRELOAD interposition is Linux only")
  return()
endif()

set(LOG_FILE "${TEST_EXECUTABLE}_output.txt")

execute_process(
  COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=${INTERPOSER}" ${TEST_EXECUTABLE}
  OUTPUT_FILE "${LOG_FILE}"
  ERROR_FILE "${LOG_FILE}"
  TIMEOUT ${TEST_TIMEOUT}
  RESULT_VARIABLE RESULT)

file(READ "${LOG_FILE}" TEST_OUTPUT)
message("${TEST_OUTPUT}")

if(NOT RESULT EQUAL 0)
  message(FATAL_ERROR "test program failed (result: ${RESULT})")
endif()

# The test process reports last, so the final report is the one that counts
# should a subprocess have printed one as well.
set(REPORT_REGEX "interposer: barrier events observed: ([0-9]+), leaked: ([0-9]+)")
string(REGEX MATCHALL "${REPORT_REGEX}" REPORTS "${TEST_OUTPUT}")
if(NOT REPORTS)
  message(FATAL_ERROR "the interposer did not report; was it preloaded?")
endif()
list(GET REPORTS -1 REPORT)
string(REGEX MATCH "${REPORT_REGEX}" _unused "${REPORT}")
set(OBSERVED "${CMAKE_MATCH_1}")
set(LEAKED "${CMAKE_MATCH_2}")

if(OBSERVED EQUAL 0)
  message("HIP_SKIP_THIS_TEST: no clEnqueueBarrierWithWaitList call was made, "
          "so there was nothing to leak")
  return()
endif()

if(LEAKED EQUAL 0)
  message("PASS")
  return()
endif()

message(FATAL_ERROR
  "${LEAKED} of ${OBSERVED} barrier events still had references at exit. "
  "CHIPQueueOpenCL::switchModeTo wraps the mode switch barrier event in a "
  "CHIPEventOpenCL it never stores or deletes, so the wrapper and the "
  "cl_event reference it retained are lost. See issue #1543.")
