# Driver for TestDeviceAbortStderr.
#
# Runs the test binary with stdout and stderr captured separately and maps the
# outcome onto a verdict:
#
#   the process did not die with SIGABRT      -> the abort was not serviced, fail
#   the assertion message is not on stderr    -> the bug, fail
#   SIGABRT and the message is on stderr      -> pass
#
# stderr is what gtest death tests match against (GetCapturedStderr), so the
# message being on stdout only is exactly the failure this test guards.
#
# Invoked as:
#   cmake -DTEST_EXECUTABLE=<path> -DTEST_TIMEOUT=<seconds>
#         -P run_device_abort_stderr_test.cmake

if(NOT TEST_EXECUTABLE)
  message(FATAL_ERROR "TEST_EXECUTABLE not set")
endif()

set(EXPECTED_MESSAGE
  ":0: : Device-side assertion `Meurs, pourriture communiste !' failed.")

execute_process(
  COMMAND ${TEST_EXECUTABLE}
  OUTPUT_VARIABLE RUN_OUT
  ERROR_VARIABLE RUN_ERR
  TIMEOUT ${TEST_TIMEOUT}
  RESULT_VARIABLE RESULT)

message("result: ${RESULT}")
message("--- stdout ---\n${RUN_OUT}--- stderr ---\n${RUN_ERR}--------------")

# CMake reports a signal death as a description string rather than a number.
if(NOT RESULT MATCHES "Subprocess aborted|SIGABRT")
  message(FATAL_ERROR
    "expected the process to die with SIGABRT from the device-side abort "
    "(result: ${RESULT})")
endif()

string(FIND "${RUN_ERR}" "${EXPECTED_MESSAGE}" MESSAGE_POS)
if(MESSAGE_POS EQUAL -1)
  string(FIND "${RUN_OUT}" "${EXPECTED_MESSAGE}" STDOUT_POS)
  if(STDOUT_POS EQUAL -1)
    message(FATAL_ERROR
      "the assertion message was not reported on either stream")
  endif()
  message(FATAL_ERROR
    "the assertion message was reported on stdout only; gtest death tests "
    "(for example Kokkos hip_DeathTest.abort_from_device) match it against "
    "stderr, where ROCm reports it")
endif()

message("PASS")
