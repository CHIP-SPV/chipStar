execute_process(
  COMMAND ./abort2
  RESULT_VARIABLE errorcode
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

if("${stdout}" MATCHES ".*Error: abort.. was ignored!")
  message(FATAL_ERROR "FAIL: abort() was ignored.")
endif()

if(${errorcode} EQUAL 0)
  message(FATAL_ERROR "FAIL: Expected an error code from the test program.")
endif()

message(STATUS "PASSED")
