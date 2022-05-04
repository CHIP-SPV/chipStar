execute_process(
  COMMAND ./abort
  RESULT_VARIABLE errorcode
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

if(errorcode)
  message(FATAL_ERROR
    "FAIL: HIP program returned an error code '${errorcode}'.")
endif()

file(WRITE "abort-stdout.txt" "${stdout}")

execute_process(
  COMMAND ${CMAKE_COMMAND} -E compare_files "abort-stdout.txt" "abort.xstdout"
  RESULT_VARIABLE errorcode)

if(errorcode)
  message(FATAL_ERROR "FAIL: Standard output does not match 'abort.xstdout'")
endif()

message(STATUS "PASSED")
