# Installs chipStar to a scratch prefix and configures a find_package(hip) consumer.
file(REMOVE_RECURSE "${WORK_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND} --install "${BUILD_DIR}" --prefix "${WORK_DIR}/prefix"
  OUTPUT_QUIET RESULT_VARIABLE RC)
if(NOT RC EQUAL 0)
  message(FATAL_ERROR "install failed")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} -G "${GENERATOR}" -S "${CONSUMER_DIR}" -B "${WORK_DIR}/consumer"
  -DCMAKE_PREFIX_PATH=${WORK_DIR}/prefix -DCMAKE_CXX_COMPILER=${CXX_COMPILER}
  RESULT_VARIABLE RC)
if(NOT RC EQUAL 0)
  message(FATAL_ERROR "consumer configure failed")
endif()
