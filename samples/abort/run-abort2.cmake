execute_process(
  COMMAND ./abort2
  RESULT_VARIABLE errorcode
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

set(report "result: ${errorcode}\n--- stdout ---\n${stdout}--- stderr ---\n${stderr}")

if("${stdout}" MATCHES "Error: kernel launch failed")
  message(FATAL_ERROR "FAIL: kernel launch failed.\n${report}")
endif()

if("${stdout}" MATCHES ".*Error: abort.. was ignored!")
  message(FATAL_ERROR "FAIL: abort() was ignored.\n${report}")
endif()

# CMake reports a signal death as a description string rather than a number.
if(NOT errorcode MATCHES "Subprocess aborted|SIGABRT")
  message(FATAL_ERROR "FAIL: expected the test program to die with SIGABRT.\n${report}")
endif()

message(STATUS "PASSED")
