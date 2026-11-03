# Installs chipStar to a scratch prefix and configures a find_package(hip) consumer.
file(REMOVE_RECURSE "${WORK_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND} --install "${BUILD_DIR}" --prefix "${WORK_DIR}/prefix"
  OUTPUT_QUIET COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${GENERATOR}" -S "${CONSUMER_DIR}" -B "${WORK_DIR}/consumer"
  -DCMAKE_PREFIX_PATH=${WORK_DIR}/prefix COMMAND_ERROR_IS_FATAL ANY)
