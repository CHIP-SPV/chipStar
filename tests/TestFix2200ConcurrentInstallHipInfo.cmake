# Runs install B between install A's .hipInfo configure and copy; each prefix
# must still get its own HIP_PATH.
file(REMOVE_RECURSE "${WORK_DIR}")
set(ENV{DESTDIR} "${WORK_DIR}/stage")
set(A "${WORK_DIR}/a")
set(B "${WORK_DIR}/b")

function(configure_file)
  _configure_file(${ARGN})
  if(ARGV0 MATCHES "hipInfo_install")
    execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${B}
      -P "${BUILD_DIR}/cmake_install.cmake" OUTPUT_QUIET RESULT_VARIABLE RC)
    if(NOT RC EQUAL 0)
      message(FATAL_ERROR "install B failed: ${RC}")
    endif()
  endif()
endfunction()

set(CMAKE_INSTALL_PREFIX "${A}")
include("${BUILD_DIR}/cmake_install.cmake")

foreach(P A B)
  file(STRINGS "${WORK_DIR}/stage/${${P}}/share/.hipInfo" L REGEX "^HIP_PATH=")
  if(NOT L STREQUAL "HIP_PATH=${${P}}")
    message(FATAL_ERROR "prefix ${P}: expected HIP_PATH=${${P}}, found '${L}'")
  endif()
endforeach()
if(NOT "${A}/share/.hipInfo" IN_LIST CMAKE_INSTALL_MANIFEST_FILES)
  message(FATAL_ERROR "${A}/share/.hipInfo missing from CMAKE_INSTALL_MANIFEST_FILES")
endif()
