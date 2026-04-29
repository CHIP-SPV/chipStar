function(add_hip_test MAIN_SOURCE)
  get_filename_component(EXEC_NAME ${MAIN_SOURCE} NAME_WLE)
  set_source_files_properties(${MAIN_SOURCE} PROPERTIES LANGUAGE CXX)
  add_executable("${EXEC_NAME}" EXCLUDE_FROM_ALL ${MAIN_SOURCE})
  set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

  target_link_libraries("${EXEC_NAME}" CHIP deviceInternal)
  target_include_directories("${EXEC_NAME}"
    PUBLIC ${CMAKE_SOURCE_DIR}/HIP/include ${CMAKE_SOURCE_DIR}/include)

  add_test(NAME ${EXEC_NAME} COMMAND ${SKIP_DOUBLE_TESTS} ${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME})
  set_tests_properties("${EXEC_NAME}" PROPERTIES
    SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
  add_dependencies(build_tests "${EXEC_NAME}")
endfunction()

add_subdirectory(compiler)
add_subdirectory(cuda)
add_subdirectory(devicelib)
add_subdirectory(fromLibCeed)
add_subdirectory(hiprtc)
add_subdirectory(post-install)
add_subdirectory(runtime)
add_subdirectory(libraries)
add_subdirectory(benchmarks)
add_subdirectory(regression)
add_subdirectory(hipcc-verify)
