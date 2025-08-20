
add_chip_test(hipStreamSemantics hipStreamSemantics PASSED hipStreamSemantics.cc)
set_tests_properties(hipStreamSemantics PROPERTIES
  FAIL_REGULAR_EXPRESSION "Failed"
  LABELS "internal;SAMPLE")

