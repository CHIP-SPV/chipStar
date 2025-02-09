add_chip_binary(hipDeviceMalloc hipDeviceMalloc.hip)

add_test(NAME hipDeviceMalloc
  COMMAND ${SKIP_DOUBLE_TESTS} "${CMAKE_CURRENT_BINARY_DIR}/hipDeviceMalloc")

set_tests_properties(hipDeviceMalloc PROPERTIES
  PASS_REGULAR_EXPRESSION PASSED
  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
