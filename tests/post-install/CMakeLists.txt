
include(ProcessorCount)
ProcessorCount(NPROC)

configure_file(cmake-tests.bash cmake-tests.bash @ONLY)

# Runs tests on the installed chipStar.
add_custom_target(check-install
  COMMAND cmake --build ${CMAKE_BINARY_DIR} -j${NPROC}
  COMMAND cmake --install ${CMAKE_BINARY_DIR}
  COMMAND bash ${CMAKE_CURRENT_BINARY_DIR}/cmake-tests.bash)
