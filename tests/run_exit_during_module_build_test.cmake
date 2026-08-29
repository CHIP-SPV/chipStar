# Driver for TestFix1393ExitDuringModuleBuild.
#
# Runs the test binary with the interposer preloaded and maps the three
# possible outcomes onto a verdict:
#
#   the interposer never fired  -> nothing was exercised, skip
#   the process did not exit    -> the #1393 deadlock, fail
#   the process exited with the injected code -> pass
#
# The interposer prints its marker before calling exit(), so a deadlocked run
# is distinguished from a skipped one by the marker being present while the
# process still has to be killed on timeout.

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

if(NOT TEST_OUTPUT MATCHES "interposer: exiting from inside")
  message("HIP_SKIP_THIS_TEST: no module build ran with getOrCreateModule on "
          "the stack, so the exit was never injected")
  return()
endif()

if(RESULT STREQUAL "${EXPECTED_EXIT_CODE}")
  message("PASS")
  return()
endif()

message(FATAL_ERROR
  "exit() was called during the module build but the process never "
  "terminated (result: ${RESULT}). Device::getOrCreateModule is holding "
  "DeviceVarMtx across compile(), so the atexit path re-locks it on the same "
  "thread in Device::deallocateDeviceVariables. See issue #1393.")
