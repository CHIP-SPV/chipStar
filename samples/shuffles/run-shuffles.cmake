execute_process(
  COMMAND ./shuffles
  RESULT_VARIABLE errorcode
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

if(errorcode)
  message(FATAL_ERROR
    "FAIL: HIP program returned an error code '${errorcode}'.")
endif()

file(WRITE "shuffles-stdout.txt" "${stdout}")

execute_process(
  COMMAND ${CMAKE_COMMAND} -E compare_files "shuffles-stdout.txt" "shuffles.xstdout"
  RESULT_VARIABLE errorcode)

if(errorcode)
  message(FATAL_ERROR "FAIL: Standard output does not match 'shuffles.xstdout'")
endif()

message(STATUS "PASSED")
